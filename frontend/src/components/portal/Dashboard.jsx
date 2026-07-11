import { useState, useEffect } from 'react';
import { portalOverview, portalSeed } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, fmtLKR, fmtNum, fmtDateTime, EmptyState } from './ui';

const Kpi = ({ icon, label, value, sub }) => (
    <div className="kpi-card">
        <div className="kpi-label"><Icon name={icon} size={14} /> {label}</div>
        <div className="kpi-value">{value}{sub && <small> {sub}</small>}</div>
    </div>
);

const Dashboard = ({ slug, isTelecom, onGoto }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [seeding, setSeeding] = useState(false);

    const load = async () => {
        setLoading(true);
        try { setData(await portalOverview(slug)); }
        catch (e) { console.error('overview failed', e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); }, [slug]);

    const seed = async () => {
        if (!window.confirm(
            'Load the enterprise telecom demo dataset?\n\nThis creates ~50 customers with subscriptions, '
            + 'call records, a charging ledger, activations and tickets, and sets this client to the telecom domain. '
            + 'It replaces any existing telecom data for this client.')) return;
        setSeeding(true);
        try {
            const res = await portalSeed(slug);
            alert(`${res.message}\n${res.detail || ''}`);
            window.location.reload(); // pick up domain=telecom + new data
        } catch (e) {
            alert('Seed failed: ' + (e?.response?.data?.detail || e.message));
        } finally { setSeeding(false); }
    };

    if (loading) return <Spinner text="Loading dashboard…" />;
    const d = data || {};
    const empty = (d.subscribers_total || 0) === 0;

    return (
        <div>
            <div className="portal-section-head">
                <h2>Dashboard</h2>
                <span className="sub">Live customer-care operations</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
                <button className="btn btn-primary" onClick={seed} disabled={seeding}>
                    <Icon name="sparkle" size={14} /> {seeding ? 'Seeding…' : (empty ? 'Load demo data' : 'Re-seed demo data')}
                </button>
            </div>

            {empty && !isTelecom && (
                <div className="portal-panel" style={{ padding: '18px 16px' }}>
                    <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 14 }}>
                        This client has no telecom data yet. Click <strong>Load demo data</strong> to populate a full
                        enterprise telecom database (subscribers, CDRs, billing, activations, tickets) and switch on the
                        complete console.
                    </p>
                </div>
            )}

            <div className="kpi-grid">
                <Kpi icon="phone" label="Subscribers" value={fmtNum(d.subscribers_total)}
                    sub={`· ${fmtNum(d.subscribers_active)} active`} />
                <Kpi icon="users" label="Customers" value={fmtNum(d.customers_total)} />
                <Kpi icon="ticket" label="Open tickets" value={fmtNum(d.tickets_open)} />
                <Kpi icon="refresh" label="Activations today" value={fmtNum(d.activations_today)} />
                <Kpi icon="chart" label="Revenue today" value={fmtLKR(d.revenue_today)} />
                <Kpi icon="file" label="Revenue (month)" value={fmtLKR(d.revenue_month)} />
            </div>

            <div className="portal-panel">
                <div className="portal-panel-head">
                    <Icon name="refresh" size={16} /> Recent activations
                    <div className="portal-head-spacer" />
                    {onGoto && <button className="btn btn-sm btn-ghost" onClick={() => onGoto('activations')}>View all</button>}
                </div>
                {(d.recent_activations || []).length === 0 ? (
                    <EmptyState icon="refresh" text="No activations yet." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead><tr><th>Reference</th><th>MSISDN</th><th>Package</th><th>Channel</th><th className="num">Price</th><th>When</th></tr></thead>
                            <tbody>
                                {d.recent_activations.map((a) => (
                                    <tr key={a.id}>
                                        <td className="strong"><Mono>{a.reference}</Mono></td>
                                        <td><Mono>{a.msisdn}</Mono></td>
                                        <td>{a.package_name}</td>
                                        <td><Badge value={a.channel} /></td>
                                        <td className="num">{fmtLKR(a.price)}</td>
                                        <td>{fmtDateTime(a.activated_at)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div className="portal-panel">
                <div className="portal-panel-head">
                    <Icon name="ticket" size={16} /> Recent tickets
                    <div className="portal-head-spacer" />
                    {onGoto && <button className="btn btn-sm btn-ghost" onClick={() => onGoto('tickets')}>View all</button>}
                </div>
                {(d.recent_tickets || []).length === 0 ? (
                    <EmptyState icon="ticket" text="No tickets yet." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead><tr><th>Ticket</th><th>Subject</th><th>Priority</th><th>Status</th><th>Channel</th><th>When</th></tr></thead>
                            <tbody>
                                {d.recent_tickets.map((t) => (
                                    <tr key={t.id}>
                                        <td className="strong"><Mono>{t.ticket_number}</Mono></td>
                                        <td>{t.subject}</td>
                                        <td><Badge value={t.priority} /></td>
                                        <td><Badge value={t.status} /></td>
                                        <td><Badge value={t.channel} /></td>
                                        <td>{fmtDateTime(t.created_at)}</td>
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

export default Dashboard;
