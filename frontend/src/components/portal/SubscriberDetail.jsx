import { useState, useEffect } from 'react';
import { portalSubscription, portalPlans, updateSubscription, activatePackage } from '../../services/api';
import Icon from '../Icon';
import {
    Spinner, Badge, Mono, fmtLKR, fmtData, fmtBytes, fmtDuration, fmtDate, fmtDateTime,
} from './ui';

const Fact = ({ label, children }) => (
    <div><div className="fact-label">{label}</div><div className="fact-value">{children}</div></div>
);

const SubscriberDetail = ({ slug, msisdn, onClose, onChanged }) => {
    const [data, setData] = useState(null);
    const [plans, setPlans] = useState([]);
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState(false);
    const [planCode, setPlanCode] = useState('');
    const [statusVal, setStatusVal] = useState('');
    const [pkgCode, setPkgCode] = useState('');

    const load = async () => {
        setLoading(true);
        try {
            const [d, p] = await Promise.all([portalSubscription(slug, msisdn), portalPlans(slug)]);
            setData(d);
            setPlans(p.plans || []);
            setStatusVal(d.subscription.status);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); }, [slug, msisdn]);

    const savePlanStatus = async () => {
        setBusy(true);
        try {
            await updateSubscription(slug, msisdn, {
                plan_code: planCode || undefined,
                status: statusVal || undefined,
            });
            setPlanCode('');
            await load();
            onChanged && onChanged();
        } catch (e) { alert('Update failed: ' + (e?.response?.data?.detail || e.message)); }
        finally { setBusy(false); }
    };

    const doActivate = async () => {
        if (!pkgCode) return;
        const plan = plans.find((p) => p.code === pkgCode);
        setBusy(true);
        try {
            await activatePackage(slug, msisdn, { package_name: plan?.name || pkgCode, plan_code: pkgCode });
            setPkgCode('');
            await load();
            onChanged && onChanged();
        } catch (e) { alert('Activation failed: ' + (e?.response?.data?.detail || e.message)); }
        finally { setBusy(false); }
    };

    const sub = data?.subscription;
    const cust = data?.customer;

    return (
        <div className="drawer-overlay" onClick={onClose}>
            <div className="drawer" onClick={(e) => e.stopPropagation()}>
                <div className="drawer-head">
                    <span className="portal-logo" style={{ width: 38, height: 38 }}><Icon name="phone" size={18} /></span>
                    <div>
                        <h3>{cust?.full_name || 'Subscriber'}</h3>
                        <div className="sub"><Mono>{msisdn}</Mono>{sub && <> · <Badge value={sub.status} /></>}</div>
                    </div>
                    <button className="drawer-close" onClick={onClose}><Icon name="x" size={20} /></button>
                </div>

                <div className="drawer-body">
                    {loading || !sub ? <Spinner /> : (
                        <>
                            <div className="detail-facts">
                                <Fact label="Plan">{sub.plan_name || '—'} <Badge value={sub.plan_type} /></Fact>
                                <Fact label="Prepaid balance">{fmtLKR(sub.prepaid_balance)}</Fact>
                                <Fact label="Data balance">{fmtData(sub.data_balance_mb)}</Fact>
                                <Fact label="Account">{sub.account_number ? <Mono>{sub.account_number}</Mono> : '—'}</Fact>
                                <Fact label="Activated">{fmtDate(sub.activation_date)}</Fact>
                                <Fact label="City">{cust?.city || '—'}</Fact>
                                <Fact label="NIC">{cust?.nic || '—'}</Fact>
                                <Fact label="Email">{cust?.email || '—'}</Fact>
                            </div>

                            {/* actions */}
                            <div className="detail-section">
                                <h4><Icon name="settings" size={15} /> Manage line</h4>
                                <div className="mini-form">
                                    <select className="portal-select" value={planCode} onChange={(e) => setPlanCode(e.target.value)}>
                                        <option value="">Change plan…</option>
                                        {plans.map((p) => <option key={p.code} value={p.code}>{p.name} ({fmtLKR(p.monthly_rental)})</option>)}
                                    </select>
                                    <select className="portal-select" value={statusVal} onChange={(e) => setStatusVal(e.target.value)}>
                                        <option value="active">active</option>
                                        <option value="suspended">suspended</option>
                                        <option value="deactivated">deactivated</option>
                                    </select>
                                    <button className="btn btn-primary btn-sm" onClick={savePlanStatus} disabled={busy}>Save</button>
                                </div>
                                <div className="mini-form">
                                    <select className="portal-select" value={pkgCode} onChange={(e) => setPkgCode(e.target.value)}>
                                        <option value="">Activate package…</option>
                                        {plans.map((p) => <option key={p.code} value={p.code}>{p.name} ({fmtLKR(p.monthly_rental)})</option>)}
                                    </select>
                                    <button className="btn btn-primary btn-sm" onClick={doActivate} disabled={busy || !pkgCode}>
                                        <Icon name="check" size={14} /> Activate
                                    </button>
                                </div>
                            </div>

                            <DetailTable title="Package activations" icon="refresh" rows={data.activations}
                                cols={['Reference', 'Package', 'Channel', 'Price', 'Status', 'When']}
                                render={(a) => (<tr key={a.id}>
                                    <td><Mono>{a.reference}</Mono></td><td>{a.package_name}</td>
                                    <td><Badge value={a.channel} /></td><td className="num">{fmtLKR(a.price)}</td>
                                    <td><Badge value={a.status} /></td><td>{fmtDateTime(a.activated_at)}</td></tr>)} />

                            <DetailTable title="Recent calls & usage (CDR)" icon="phone-call" rows={data.recent_cdrs}
                                cols={['Type', 'Direction', 'Party', 'Duration', 'Data', 'Charge', 'When']}
                                render={(r) => (<tr key={r.id}>
                                    <td><Badge value={r.event_type} /></td><td>{r.direction}</td>
                                    <td><Mono>{r.other_party || '—'}</Mono></td><td>{fmtDuration(r.duration_sec)}</td>
                                    <td>{r.bytes_used ? fmtBytes(r.bytes_used) : '—'}</td>
                                    <td className="num">{fmtLKR(r.charged_amount)}</td><td>{fmtDateTime(r.start_time)}</td></tr>)} />

                            <DetailTable title="Charging ledger" icon="file" rows={data.recent_transactions}
                                cols={['Type', 'Description', 'Amount', 'Balance', 'When']}
                                render={(t) => (<tr key={t.id}>
                                    <td><Badge value={t.txn_type} /></td><td>{t.description}</td>
                                    <td className={Number(t.amount) < 0 ? 'amt-neg num' : 'amt-pos num'}>{fmtLKR(t.amount)}</td>
                                    <td className="num">{t.balance_after != null ? fmtLKR(t.balance_after) : '—'}</td>
                                    <td>{fmtDateTime(t.created_at)}</td></tr>)} />

                            <DetailTable title="Tickets" icon="ticket" rows={data.tickets}
                                cols={['Ticket', 'Subject', 'Priority', 'Status']}
                                render={(t) => (<tr key={t.id}>
                                    <td><Mono>{t.ticket_number}</Mono></td><td>{t.subject}</td>
                                    <td><Badge value={t.priority} /></td><td><Badge value={t.status} /></td></tr>)} />

                            {data.invoices?.length > 0 && (
                                <DetailTable title="Invoices" icon="file" rows={data.invoices}
                                    cols={['Invoice', 'Period', 'Amount', 'Due', 'Status']}
                                    render={(i) => (<tr key={i.id}>
                                        <td><Mono>{i.invoice_number}</Mono></td>
                                        <td>{fmtDate(i.period_start)} – {fmtDate(i.period_end)}</td>
                                        <td className="num">{fmtLKR(i.amount_due)}</td><td>{fmtDate(i.due_date)}</td>
                                        <td><Badge value={i.status} /></td></tr>)} />
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

const DetailTable = ({ title, icon, rows, cols, render }) => (
    <div className="detail-section">
        <h4><Icon name={icon} size={15} /> {title} <span className="muted">({rows?.length || 0})</span></h4>
        {rows?.length ? (
            <div className="portal-panel" style={{ marginBottom: 0 }}>
                <div className="table-scroll">
                    <table className="portal-table">
                        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                        <tbody>{rows.map(render)}</tbody>
                    </table>
                </div>
            </div>
        ) : <p className="muted" style={{ fontSize: 13, margin: 0 }}>None.</p>}
    </div>
);

export default SubscriberDetail;
