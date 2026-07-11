import { useState, useEffect } from 'react';
import { portalTickets, createTicket, updateTicket } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtDateTime } from './ui';

const STATUSES = ['open', 'in_progress', 'pending', 'resolved', 'closed'];
const PRIORITIES = ['low', 'medium', 'high', 'urgent'];
const CATEGORIES = ['billing', 'network', 'activation', 'sim', 'complaint', 'general'];

// ---- create modal -----------------------------------------------------------
const CreateModal = ({ slug, onClose, onCreated }) => {
    const [form, setForm] = useState({ subject: '', category: 'general', priority: 'medium', msisdn: '', description: '' });
    const [busy, setBusy] = useState(false);
    const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

    const submit = async (e) => {
        e.preventDefault();
        setBusy(true);
        try { await createTicket(slug, { ...form, channel: 'admin' }); onCreated(); }
        catch (err) { alert('Create failed: ' + (err?.response?.data?.detail || err.message)); }
        finally { setBusy(false); }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <form className="modal" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
                <div className="modal-head"><Icon name="ticket" size={18} /> New ticket</div>
                <div className="modal-body">
                    <label>Subject
                        <input value={form.subject} onChange={set('subject')} required placeholder="Short summary of the issue" autoFocus />
                    </label>
                    <label>MSISDN (optional)
                        <input value={form.msisdn} onChange={set('msisdn')} placeholder="Link to a subscriber, e.g. 0771234567" />
                    </label>
                    <div style={{ display: 'flex', gap: 12 }}>
                        <label style={{ flex: 1 }}>Category
                            <select value={form.category} onChange={set('category')}>
                                {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </label>
                        <label style={{ flex: 1 }}>Priority
                            <select value={form.priority} onChange={set('priority')}>
                                {PRIORITIES.map((p) => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </label>
                    </div>
                    <label>Description
                        <textarea value={form.description} onChange={set('description')} placeholder="What did the customer report?" />
                    </label>
                </div>
                <div className="modal-foot">
                    <button type="button" className="btn" onClick={onClose}>Cancel</button>
                    <button type="submit" className="btn btn-primary" disabled={busy || !form.subject}>
                        {busy ? 'Creating…' : 'Create ticket'}
                    </button>
                </div>
            </form>
        </div>
    );
};

// ---- edit drawer ------------------------------------------------------------
const TicketDrawer = ({ slug, ticket, onClose, onSaved }) => {
    const [status, setStatus] = useState(ticket.status);
    const [priority, setPriority] = useState(ticket.priority);
    const [assignee, setAssignee] = useState(ticket.assigned_to || '');
    const [resolution, setResolution] = useState(ticket.resolution || '');
    const [busy, setBusy] = useState(false);

    const save = async () => {
        setBusy(true);
        try {
            await updateTicket(slug, ticket.id, { status, priority, assigned_to: assignee, resolution });
            onSaved();
        } catch (e) { alert('Save failed: ' + (e?.response?.data?.detail || e.message)); }
        finally { setBusy(false); }
    };

    return (
        <div className="drawer-overlay" onClick={onClose}>
            <div className="drawer" onClick={(e) => e.stopPropagation()}>
                <div className="drawer-head">
                    <span className="portal-logo" style={{ width: 38, height: 38 }}><Icon name="ticket" size={18} /></span>
                    <div>
                        <h3><Mono>{ticket.ticket_number}</Mono></h3>
                        <div className="sub">{ticket.subject}</div>
                    </div>
                    <button className="drawer-close" onClick={onClose}><Icon name="x" size={20} /></button>
                </div>
                <div className="drawer-body">
                    <div className="detail-facts">
                        <div><div className="fact-label">Customer</div><div className="fact-value">{ticket.customer_name || '—'}</div></div>
                        <div><div className="fact-label">MSISDN</div><div className="fact-value"><Mono>{ticket.msisdn || '—'}</Mono></div></div>
                        <div><div className="fact-label">Category</div><div className="fact-value"><Badge value={ticket.category} /></div></div>
                        <div><div className="fact-label">Channel</div><div className="fact-value"><Badge value={ticket.channel} /></div></div>
                        <div><div className="fact-label">Created</div><div className="fact-value">{fmtDateTime(ticket.created_at)}</div></div>
                    </div>

                    {ticket.description && (
                        <div className="detail-section">
                            <h4><Icon name="message" size={15} /> Report</h4>
                            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 13.5 }}>{ticket.description}</p>
                        </div>
                    )}

                    <div className="detail-section">
                        <h4><Icon name="settings" size={15} /> Manage</h4>
                        <div className="mini-form">
                            <label style={{ fontSize: 12, color: 'var(--text-muted)' }}>Status</label>
                            <select className="portal-select" value={status} onChange={(e) => setStatus(e.target.value)}>
                                {STATUSES.map((s) => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
                            </select>
                            <label style={{ fontSize: 12, color: 'var(--text-muted)' }}>Priority</label>
                            <select className="portal-select" value={priority} onChange={(e) => setPriority(e.target.value)}>
                                {PRIORITIES.map((p) => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </div>
                        <div className="mini-form">
                            <input className="portal-input" style={{ flex: 1 }} placeholder="Assign to (agent name)"
                                value={assignee} onChange={(e) => setAssignee(e.target.value)} />
                        </div>
                        <div className="mini-form" style={{ display: 'block' }}>
                            <textarea className="portal-input" style={{ width: '100%', minHeight: 70, resize: 'vertical' }}
                                placeholder="Resolution notes…" value={resolution} onChange={(e) => setResolution(e.target.value)} />
                        </div>
                    </div>

                    <div className="drawer-actions">
                        <button className="btn btn-primary" onClick={save} disabled={busy}>
                            <Icon name="check" size={15} /> {busy ? 'Saving…' : 'Save changes'}
                        </button>
                        <button className="btn" onClick={onClose}>Close</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

// ---- board ------------------------------------------------------------------
const TicketBoard = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [open, setOpen] = useState(0);
    const [total, setTotal] = useState(0);
    const [filters, setFilters] = useState({ status: '', priority: '', category: '' });
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [selected, setSelected] = useState(null);

    const load = async () => {
        setLoading(true);
        try {
            const res = await portalTickets(slug, { ...filters, limit: 200 });
            setRows(res.tickets || []);
            setOpen(res.open_count || 0);
            setTotal(res.total || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, filters]);

    const setF = (k) => (e) => setFilters({ ...filters, [k]: e.target.value });

    return (
        <div>
            <div className="portal-section-head">
                <h2>Tickets</h2>
                <span className="sub">{open} open · {total} total</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
                <button className="btn btn-primary" onClick={() => setCreating(true)}><Icon name="plus" size={14} /> New ticket</button>
            </div>

            <div className="portal-filters">
                <select className="portal-select" value={filters.status} onChange={setF('status')}>
                    <option value="">All statuses</option>
                    {STATUSES.map((s) => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
                </select>
                <select className="portal-select" value={filters.priority} onChange={setF('priority')}>
                    <option value="">All priorities</option>
                    {PRIORITIES.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <select className="portal-select" value={filters.category} onChange={setF('category')}>
                    <option value="">All categories</option>
                    {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="ticket" text="No tickets match these filters." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Ticket</th><th>Subject</th><th>Customer</th><th>MSISDN</th><th>Category</th><th>Priority</th><th>Status</th><th>Channel</th><th>Assigned</th><th>Created</th></tr>
                            </thead>
                            <tbody>
                                {rows.map((t) => (
                                    <tr key={t.id} className="clickable" onClick={() => setSelected(t)}>
                                        <td className="strong"><Mono>{t.ticket_number}</Mono></td>
                                        <td>{t.subject}</td>
                                        <td>{t.customer_name || '—'}</td>
                                        <td><Mono>{t.msisdn || '—'}</Mono></td>
                                        <td><Badge value={t.category} /></td>
                                        <td><Badge value={t.priority} /></td>
                                        <td><Badge value={t.status} /></td>
                                        <td><Badge value={t.channel} /></td>
                                        <td>{t.assigned_to || <span className="muted">Unassigned</span>}</td>
                                        <td>{fmtDateTime(t.created_at)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {creating && <CreateModal slug={slug} onClose={() => setCreating(false)}
                onCreated={() => { setCreating(false); load(); }} />}
            {selected && <TicketDrawer slug={slug} ticket={selected} onClose={() => setSelected(null)}
                onSaved={() => { setSelected(null); load(); }} />}
        </div>
    );
};

export default TicketBoard;
