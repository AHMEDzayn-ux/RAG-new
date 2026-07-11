"""
Per-client admin portal router (enterprise telecom console).

Every route is scoped to ``/api/portal/{slug}`` and gated by ``require_portal`` —
a superadmin sees any tenant; a client_admin sees ONLY the tenant its login is
bound to. This is the backend for the /portal/{slug} staff console.

Reads/writes go through services/telecom_store, the SAME layer the chatbot uses,
so admin and bot share one live database.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.models import (
    PortalOverviewResponse,
    PlanListResponse, PlanResponse,
    SubscriptionListResponse, SubscriptionResponse,
    CustomerListResponse, CustomerResponse,
    CdrListResponse, CdrResponse,
    TransactionListResponse, TransactionResponse,
    InvoiceListResponse, InvoiceResponse,
    ActivationListResponse, ActivationResponse,
    TicketListResponse, TicketResponse,
    SubscriberDetailResponse,
    SubscriptionUpdateRequest, ActivateRequest,
    TicketCreateRequest, TicketUpdateRequest,
    MessageResponse,
)
from services import telecom_store
from database import get_db
from db_models import User
from auth import require_portal
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/portal/{slug}", tags=["portal"])


@router.get("/overview", response_model=PortalOverviewResponse)
async def overview(slug: str, db: Session = Depends(get_db),
                   _u: User = Depends(require_portal)):
    return PortalOverviewResponse(**telecom_store.overview_kpis(db, slug))


# ---- Subscribers (lines) ----------------------------------------------------

@router.get("/subscriptions", response_model=SubscriptionListResponse)
async def subscriptions(slug: str, q: str = "", status: str = "",
                        skip: int = 0, limit: int = Query(50, le=200),
                        db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.search_subscriptions(db, slug, q, status, skip, limit)
    return SubscriptionListResponse(
        subscriptions=[SubscriptionResponse(**d) for d in items], total=total)


@router.get("/subscriptions/{msisdn}", response_model=SubscriberDetailResponse)
async def subscription_detail(slug: str, msisdn: str, db: Session = Depends(get_db),
                              _u: User = Depends(require_portal)):
    data = telecom_store.get_subscription_360(db, slug, msisdn)
    if data is None:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return SubscriberDetailResponse(**data)


@router.patch("/subscriptions/{msisdn}", response_model=SubscriptionResponse)
async def update_subscription(slug: str, msisdn: str, payload: SubscriptionUpdateRequest,
                              db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    sub = telecom_store.update_subscription(
        db, slug, msisdn, plan_code=payload.plan_code, status=payload.status,
        prepaid_balance=payload.prepaid_balance)
    if sub is None:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return SubscriptionResponse(**telecom_store._sub_dict(sub))


@router.post("/subscriptions/{msisdn}/activate", response_model=ActivationResponse)
async def activate(slug: str, msisdn: str, payload: ActivateRequest,
                   db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    plan_id = None
    price = payload.price
    validity = payload.validity_days
    if payload.plan_code:
        from telecom_models import Plan
        plan = db.query(Plan).filter(Plan.client_slug == slug,
                                     Plan.code == payload.plan_code).first()
        if plan:
            plan_id = plan.id
            if price is None:
                price = float(plan.monthly_rental or 0)
            if validity is None:
                validity = plan.validity_days
    act = telecom_store.activate_package(
        db, slug, msisdn, payload.package_name, price=price or 0,
        validity_days=validity or 30, channel=payload.channel or "admin", plan_id=plan_id)
    if act is None:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return ActivationResponse(**telecom_store._activation_dict(act))


# ---- Customers & catalog ----------------------------------------------------

@router.get("/customers", response_model=CustomerListResponse)
async def customers(slug: str, q: str = "", skip: int = 0, limit: int = Query(50, le=200),
                    db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.list_customers(db, slug, q, skip, limit)
    return CustomerListResponse(customers=[CustomerResponse(**d) for d in items], total=total)


@router.get("/plans", response_model=PlanListResponse)
async def plans(slug: str, db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    return PlanListResponse(plans=[PlanResponse(**d) for d in telecom_store.list_plans(db, slug)])


# ---- Usage / financials -----------------------------------------------------

@router.get("/cdrs", response_model=CdrListResponse)
async def cdrs(slug: str, msisdn: str = "", skip: int = 0, limit: int = Query(100, le=500),
               db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.list_cdrs(db, slug, msisdn, skip, limit)
    return CdrListResponse(cdrs=[CdrResponse(**d) for d in items], total=total)


@router.get("/transactions", response_model=TransactionListResponse)
async def transactions(slug: str, msisdn: str = "", txn_type: str = "",
                       skip: int = 0, limit: int = Query(100, le=500),
                       db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.list_transactions(db, slug, msisdn, txn_type, skip, limit)
    return TransactionListResponse(
        transactions=[TransactionResponse(**d) for d in items], total=total)


@router.get("/invoices", response_model=InvoiceListResponse)
async def invoices(slug: str, status: str = "", skip: int = 0, limit: int = Query(100, le=500),
                   db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.list_invoices(db, slug, status, skip, limit)
    return InvoiceListResponse(invoices=[InvoiceResponse(**d) for d in items], total=total)


@router.get("/activations", response_model=ActivationListResponse)
async def activations(slug: str, msisdn: str = "", channel: str = "",
                      skip: int = 0, limit: int = Query(100, le=500),
                      db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total = telecom_store.list_activations(db, slug, msisdn, channel, skip, limit)
    return ActivationListResponse(
        activations=[ActivationResponse(**d) for d in items], total=total)


# ---- Trouble tickets --------------------------------------------------------

@router.get("/tickets", response_model=TicketListResponse)
async def tickets(slug: str, status: str = "", priority: str = "", category: str = "",
                  skip: int = 0, limit: int = Query(100, le=500),
                  db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    items, total, open_count = telecom_store.list_tickets(
        db, slug, status, priority, category, skip, limit)
    return TicketListResponse(
        tickets=[TicketResponse(**d) for d in items], total=total, open_count=open_count)


@router.post("/tickets", response_model=TicketResponse)
async def create_ticket(slug: str, payload: TicketCreateRequest,
                        db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    t = telecom_store.log_ticket(
        db, slug, subject=payload.subject, msisdn=payload.msisdn or "",
        category=payload.category, description=payload.description or "",
        channel=payload.channel or "admin", priority=payload.priority)
    return TicketResponse(**telecom_store._ticket_dict(t))


@router.patch("/tickets/{ticket_id}", response_model=TicketResponse)
async def patch_ticket(slug: str, ticket_id: int, payload: TicketUpdateRequest,
                       db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    t = telecom_store.update_ticket(
        db, slug, ticket_id, status=payload.status, priority=payload.priority,
        assigned_to=payload.assigned_to, resolution=payload.resolution)
    if t is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketResponse(**telecom_store._ticket_dict(t))


# ---- Seed the enterprise demo dataset ---------------------------------------

@router.post("/seed", response_model=MessageResponse)
async def seed(slug: str, db: Session = Depends(get_db), _u: User = Depends(require_portal)):
    summary = telecom_store.seed_telecom_demo(db, slug)
    logger.info(f"Seeded telecom demo for {slug}: {summary}")
    detail = ", ".join(f"{v} {k}" for k, v in summary.items())
    return MessageResponse(message="Enterprise telecom demo data seeded", detail=detail)
