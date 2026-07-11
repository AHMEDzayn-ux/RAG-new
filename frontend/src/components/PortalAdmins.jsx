import { useState, useEffect } from 'react';
import { listClientAdmins, createClientAdmin, deleteClientAdmin } from '../services/api';
import Icon from './Icon';

/*
 * Operator-side manager for a client's PORTAL admin logins.
 * The operator mints a scoped login here; the client's staff then sign in at
 * /portal/{slug} and only ever see their own tenant (backend require_portal).
 */
const PortalAdmins = ({ slug }) => {
    const [admins, setAdmins] = useState([]);
    const [portalUrl, setPortalUrl] = useState(`/portal/${slug}`);
    const [loading, setLoading] = useState(true);
    const [form, setForm] = useState({ email: '', password: '', name: '' });
    const [busy, setBusy] = useState(false);
    const [copied, setCopied] = useState(false);

    const fullUrl = `${window.location.origin}${portalUrl}`;

    const load = async () => {
        setLoading(true);
        try {
            const res = await listClientAdmins(slug);
            setAdmins(res.admins || []);
            if (res.portal_url) setPortalUrl(res.portal_url);
        } catch (e) { console.error('load admins failed', e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug]);

    const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

    const create = async (e) => {
        e.preventDefault();
        setBusy(true);
        try {
            await createClientAdmin(slug, form.email.trim(), form.password, form.name.trim());
            setForm({ email: '', password: '', name: '' });
            await load();
        } catch (err) {
            alert('Create failed: ' + (err?.response?.data?.detail || err.message));
        } finally { setBusy(false); }
    };

    const remove = async (id) => {
        if (!window.confirm('Revoke this admin login?')) return;
        try { await deleteClientAdmin(slug, id); await load(); }
        catch (err) { alert('Remove failed: ' + (err?.response?.data?.detail || err.message)); }
    };

    const copy = () => {
        navigator.clipboard?.writeText(fullUrl);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };

    return (
        <div className="inbox">
            <div className="inbox-head">
                <h3><Icon name="user" size={17} /> Admin logins <span className="hint">(per-client portal access)</span></h3>
                <button className="btn-mini" onClick={load}>Refresh</button>
            </div>

            <p className="empty-state" style={{ marginBottom: 14 }}>
                Give this client their own admin portal. Staff you add here sign in at the portal URL
                below and only ever see <strong>{slug}</strong>'s data — never other clients.
            </p>

            <div className="insights-block" style={{ marginTop: 0 }}>
                <h4><Icon name="link" size={15} /> Portal URL</h4>
                <div className="req-payload" style={{ alignItems: 'center', gap: 8 }}>
                    <a href={portalUrl} target="_blank" rel="noreferrer" className="mono">{fullUrl}</a>
                    <button className="btn-mini" onClick={copy}>
                        <Icon name={copied ? 'check' : 'copy'} size={13} /> {copied ? 'Copied' : 'Copy'}
                    </button>
                </div>
            </div>

            <div className="insights-block">
                <h4><Icon name="plus" size={15} /> Create a login</h4>
                <form onSubmit={create} style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                    <input className="portal-input" type="email" placeholder="admin@client.com" required
                        value={form.email} onChange={set('email')} />
                    <input className="portal-input" type="text" placeholder="Name (optional)"
                        value={form.name} onChange={set('name')} />
                    <input className="portal-input" type="password" placeholder="Password (6+ chars)" required minLength={6}
                        value={form.password} onChange={set('password')} />
                    <button className="btn-mini resolve" type="submit" disabled={busy || !form.email || form.password.length < 6}>
                        {busy ? 'Creating…' : 'Create login'}
                    </button>
                </form>
            </div>

            <div className="insights-block">
                <h4><Icon name="users" size={15} /> Active logins</h4>
                {loading ? <p className="empty-state">Loading…</p> : admins.length === 0 ? (
                    <p className="empty-state" style={{ marginBottom: 0 }}>No portal logins yet. Create one above.</p>
                ) : (
                    <div className="acct-list">
                        {admins.map((a) => (
                            <div key={a.id} className="acct-item" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div className="acct-id">{a.email} {a.name && <span className="q-count">{a.name}</span>}</div>
                                    <div className="hint">Added {a.created_at ? new Date(a.created_at).toLocaleDateString() : ''}</div>
                                </div>
                                <button className="btn-mini" onClick={() => remove(a.id)}>Revoke</button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default PortalAdmins;
