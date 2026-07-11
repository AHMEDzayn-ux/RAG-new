import { useState, useEffect } from 'react';
import { portalTransactions, portalInvoices } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtLKR, fmtDate, fmtDateTime } from './ui';

const BillingHistory = ({ slug }) => {
    const [txns, setTxns] = useState([]);
    const [invoices, setInvoices] = useState([]);
    const [msisdn, setMsisdn] = useState('');
    const [txnType, setTxnType] = useState('');
    const [loading, setLoading] = useState(true);

    const load = async () => {
        setLoading(true);
        try {
            const [t, i] = await Promise.all([
                portalTransactions(slug, { msisdn, txn_type: txnType, limit: 200 }),
                portalInvoices(slug, { limit: 100 }),
            ]);
            setTxns(t.transactions || []);
            setInvoices(i.invoices || []);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, txnType]);

    const onSearch = (e) => { e.preventDefault(); load(); };

    return (
        <div>
            <div className="portal-section-head">
                <h2>Billing &amp; charging</h2>
                <span className="sub">Ledger + invoices</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-filters">
                <form className="portal-search" onSubmit={onSearch}>
                    <Icon name="search" size={15} />
                    <input className="portal-input" placeholder="Filter ledger by MSISDN…"
                        value={msisdn} onChange={(e) => setMsisdn(e.target.value)} />
                </form>
                <select className="portal-select" value={txnType} onChange={(e) => setTxnType(e.target.value)}>
                    <option value="">All transaction types</option>
                    <option value="recharge">Recharge</option>
                    <option value="package_purchase">Package purchase</option>
                    <option value="usage_charge">Usage charge</option>
                    <option value="adjustment">Adjustment</option>
                </select>
            </div>

            <div className="portal-panel">
                <div className="portal-panel-head"><Icon name="file" size={16} /> Charging ledger <span className="count">{txns.length} transactions</span></div>
                {loading ? <Spinner /> : txns.length === 0 ? (
                    <EmptyState icon="file" text="No transactions match." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Reference</th><th>MSISDN</th><th>Type</th><th>Description</th><th className="num">Amount</th><th className="num">Balance after</th><th>Channel</th><th>When</th></tr>
                            </thead>
                            <tbody>
                                {txns.map((t) => (
                                    <tr key={t.id}>
                                        <td><Mono>{t.reference || '—'}</Mono></td>
                                        <td><Mono>{t.msisdn || '—'}</Mono></td>
                                        <td><Badge value={t.txn_type} /></td>
                                        <td>{t.description}</td>
                                        <td className={Number(t.amount) < 0 ? 'amt-neg num' : 'amt-pos num'}>{fmtLKR(t.amount)}</td>
                                        <td className="num">{t.balance_after != null ? fmtLKR(t.balance_after) : '—'}</td>
                                        <td><Badge value={t.channel} /></td>
                                        <td>{fmtDateTime(t.created_at)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div className="portal-panel">
                <div className="portal-panel-head"><Icon name="file" size={16} /> Invoices <span className="count">{invoices.length} statements</span></div>
                {loading ? null : invoices.length === 0 ? (
                    <EmptyState icon="file" text="No postpaid invoices." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Invoice</th><th>Account</th><th>Period</th><th className="num">Amount due</th><th>Due date</th><th>Status</th></tr>
                            </thead>
                            <tbody>
                                {invoices.map((i) => (
                                    <tr key={i.id}>
                                        <td className="strong"><Mono>{i.invoice_number}</Mono></td>
                                        <td><Mono>{i.account_number || '—'}</Mono></td>
                                        <td>{fmtDate(i.period_start)} – {fmtDate(i.period_end)}</td>
                                        <td className="num">{fmtLKR(i.amount_due)}</td>
                                        <td>{fmtDate(i.due_date)}</td>
                                        <td><Badge value={i.status} /></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default BillingHistory;
