"""
Enterprise telecom BSS/OSS schema (normalized, TM Forum SID-inspired).

Modeled the way a real telco stores it — not a flat table with JSON blobs:

    Plan (product catalog)
    Customer (party / account holder)  1─* BillingAccount
                                       1─* Subscription
    BillingAccount                     1─* Subscription
                                       1─* ChargingTransaction / Invoice
    Subscription (the MSISDN / line)   1─* CallDetailRecord
                                       1─* ChargingTransaction
                                       1─* PackageActivation
    Plan                               1─* Subscription / PackageActivation
    Customer / Subscription            1─* Ticket (trouble ticketing)

Every row carries ``client_slug`` for tenant isolation, but WITHIN a tenant the
entities are joined by real ``ForeignKey`` relationships. Money is stored as
``Numeric(12, 2)`` (Decimal), timestamps as ``DateTime``, with indexes on every
key/lookup column and standard human reference numbers (ACC-, TT-, ACT-, TXN-, INV-).

These models share the single ``Base`` from ``database`` so their foreign keys
resolve against the rest of the schema. ``database.init_db()`` imports this module
before ``create_all`` so the tables register.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    Numeric,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


# ---- Product catalog --------------------------------------------------------

class Plan(Base):
    """A product/offering in the client's catalog (prepaid or postpaid)."""

    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    code = Column(String, nullable=False)                    # e.g. VALUE_20
    name = Column(String, nullable=False)
    plan_type = Column(String, nullable=False, default="prepaid")  # prepaid | postpaid
    monthly_rental = Column(Numeric(12, 2), nullable=False, default=0)
    data_quota_mb = Column(Integer, nullable=False, default=0)      # 0 = unlimited / n/a
    voice_minutes = Column(Integer, nullable=False, default=0)
    sms_units = Column(Integer, nullable=False, default=0)
    validity_days = Column(Integer, nullable=False, default=30)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_plans_slug_code", "client_slug", "code"),)


# ---- Party / account hierarchy ----------------------------------------------

class Customer(Base):
    """The party / account holder. One customer → many accounts & subscriptions."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    nic = Column(String, index=True, nullable=True)          # national ID
    email = Column(String, index=True, nullable=True)
    phone = Column(String, index=True, nullable=True)
    address = Column(String, nullable=True)
    city = Column(String, nullable=True)
    customer_type = Column(String, nullable=False, default="individual")  # individual | corporate
    kyc_status = Column(String, nullable=False, default="verified")       # verified | pending
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    accounts = relationship("BillingAccount", back_populates="customer",
                            cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="customer",
                                 cascade="all, delete-orphan")


class BillingAccount(Base):
    """A billing account (postpaid balance / credit limit) owned by a customer."""

    __tablename__ = "billing_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"),
                         index=True, nullable=False)
    account_number = Column(String, index=True, nullable=False)  # ACC-100234 (unique per tenant)
    billing_cycle_day = Column(Integer, nullable=False, default=1)
    payment_method = Column(String, nullable=False, default="prepaid")  # prepaid | credit_card | direct_debit
    status = Column(String, nullable=False, default="active")           # active | suspended | closed
    currency = Column(String, nullable=False, default="LKR")
    credit_limit = Column(Numeric(12, 2), nullable=False, default=0)
    current_balance = Column(Numeric(12, 2), nullable=False, default=0)  # postpaid amount owed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="accounts")
    subscriptions = relationship("Subscription", back_populates="account")

    __table_args__ = (
        UniqueConstraint("client_slug", "account_number", name="uq_account_number_tenant"),
    )


# ---- Service (the line / MSISDN) --------------------------------------------

class Subscription(Base):
    """A subscribed line — one MSISDN on a plan. The unit customers call 'my number'."""

    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"),
                         index=True, nullable=False)
    account_id = Column(Integer, ForeignKey("billing_accounts.id", ondelete="CASCADE"),
                        index=True, nullable=False)
    plan_id = Column(Integer, ForeignKey("plans.id"), index=True, nullable=True)
    msisdn = Column(String, index=True, nullable=False)   # the phone number (unique per tenant)
    imsi = Column(String, nullable=True)
    sim_iccid = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")          # active | suspended | deactivated
    activation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    prepaid_balance = Column(Numeric(12, 2), nullable=False, default=0)  # prepaid wallet
    data_balance_mb = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="subscriptions")
    account = relationship("BillingAccount", back_populates="subscriptions")
    plan = relationship("Plan")

    __table_args__ = (
        UniqueConstraint("client_slug", "msisdn", name="uq_subscription_msisdn_tenant"),
    )


# ---- Usage — Call Detail Records --------------------------------------------

class CallDetailRecord(Base):
    """A raw usage event (voice/video/sms/data). The telco CDR."""

    __tablename__ = "call_detail_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id", ondelete="CASCADE"),
                             index=True, nullable=False)
    msisdn = Column(String, index=True, nullable=False)       # denormalized for fast filtering
    direction = Column(String, nullable=False, default="outbound")  # inbound | outbound
    other_party = Column(String, nullable=True)
    event_type = Column(String, nullable=False, default="voice")    # voice | video | sms | data
    start_time = Column(DateTime, index=True, nullable=False)
    duration_sec = Column(Integer, nullable=False, default=0)
    bytes_used = Column(Integer, nullable=False, default=0)
    rate_applied = Column(Numeric(12, 4), nullable=False, default=0)
    charged_amount = Column(Numeric(12, 2), nullable=False, default=0)
    cell_site = Column(String, nullable=True)


# ---- Financials — ledger + invoices -----------------------------------------

class ChargingTransaction(Base):
    """The money ledger — one row per charge/recharge/adjustment. Signed amount."""

    __tablename__ = "charging_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    account_id = Column(Integer, ForeignKey("billing_accounts.id", ondelete="CASCADE"),
                        index=True, nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id", ondelete="SET NULL"),
                             index=True, nullable=True)
    msisdn = Column(String, index=True, nullable=True)
    txn_type = Column(String, nullable=False)   # usage_charge | recharge | package_purchase | adjustment | invoice
    description = Column(String, nullable=True)
    amount = Column(Numeric(12, 2), nullable=False, default=0)        # + credit / − debit
    balance_after = Column(Numeric(12, 2), nullable=True)
    channel = Column(String, nullable=True)     # ussd | app | agent | chatbot | web | system
    reference = Column(String, index=True, nullable=True)            # TXN-...
    created_at = Column(DateTime, index=True, default=datetime.utcnow, nullable=False)


class Invoice(Base):
    """A postpaid billing statement for a billing account over a period."""

    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    account_id = Column(Integer, ForeignKey("billing_accounts.id", ondelete="CASCADE"),
                        index=True, nullable=False)
    invoice_number = Column(String, index=True, nullable=False)  # INV-... (unique per tenant)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    amount_due = Column(Numeric(12, 2), nullable=False, default=0)
    due_date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False, default="unpaid")   # unpaid | paid | overdue
    issued_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("client_slug", "invoice_number", name="uq_invoice_number_tenant"),
    )


# ---- Provisioning — activation history --------------------------------------

class PackageActivation(Base):
    """A provisioned package on a subscription. Written on EVERY activation."""

    __tablename__ = "package_activations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id", ondelete="CASCADE"),
                             index=True, nullable=False)
    msisdn = Column(String, index=True, nullable=False)
    plan_id = Column(Integer, ForeignKey("plans.id"), nullable=True)
    package_name = Column(String, nullable=False)
    price = Column(Numeric(12, 2), nullable=False, default=0)
    validity_days = Column(Integer, nullable=False, default=30)
    channel = Column(String, nullable=False, default="agent")   # ussd | app | agent | chatbot | web
    status = Column(String, nullable=False, default="active")   # active | expired | pending
    activated_at = Column(DateTime, index=True, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    reference = Column(String, index=True, nullable=True)       # ACT-...


# ---- Trouble ticketing (complaint management) -------------------------------

class Ticket(Base):
    """A trouble ticket / customer complaint. Managed in the admin ticket board."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_slug = Column(String, index=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="SET NULL"),
                         index=True, nullable=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id", ondelete="SET NULL"),
                             index=True, nullable=True)
    msisdn = Column(String, index=True, nullable=True)
    ticket_number = Column(String, index=True, nullable=False)  # TT-100045 (unique per tenant)
    subject = Column(String, nullable=False)
    category = Column(String, nullable=False, default="general")  # billing|network|activation|sim|complaint|general
    priority = Column(String, nullable=False, default="medium")   # low|medium|high|urgent
    status = Column(String, nullable=False, default="open", index=True)  # open|in_progress|pending|resolved|closed
    channel = Column(String, nullable=False, default="admin")     # chatbot|voice|whatsapp|admin
    assigned_to = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    resolution = Column(Text, nullable=True)
    sla_due_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, index=True, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("client_slug", "ticket_number", name="uq_ticket_number_tenant"),
    )
