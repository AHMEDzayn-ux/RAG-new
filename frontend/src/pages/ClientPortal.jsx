import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
    isAuthenticated, getToken, adminLogin, logout, getMe, publicGetConfig,
} from '../services/api';
import Icon from '../components/Icon';
import { Spinner } from '../components/portal/ui';
import Dashboard from '../components/portal/Dashboard';
import SubscriberTable from '../components/portal/SubscriberTable';
import CustomerTable from '../components/portal/CustomerTable';
import PlanCatalog from '../components/portal/PlanCatalog';
import TicketBoard from '../components/portal/TicketBoard';
import ActivationsTable from '../components/portal/ActivationsTable';
import CallHistory from '../components/portal/CallHistory';
import BillingHistory from '../components/portal/BillingHistory';
import './ClientPortal.css';

const TELECOM_TABS = [
    { key: 'dashboard', label: 'Dashboard', icon: 'chart' },
    { key: 'subscribers', label: 'Subscribers', icon: 'phone' },
    { key: 'customers', label: 'Customers', icon: 'users' },
    { key: 'plans', label: 'Plans', icon: 'briefcase' },
    { key: 'tickets', label: 'Tickets', icon: 'ticket' },
    { key: 'activations', label: 'Activations', icon: 'refresh' },
    { key: 'calls', label: 'Call history', icon: 'phone-call' },
    { key: 'billing', label: 'Billing', icon: 'file' },
];
const BASIC_TABS = [
    { key: 'dashboard', label: 'Dashboard', icon: 'chart' },
    { key: 'tickets', label: 'Tickets', icon: 'ticket' },
];

// ---- login gate -------------------------------------------------------------
const DEMO_PORTAL_LOGINS = {
    nexus: { email: 'admin@nexus.lk', password: 'NexusDemo@123' },
};

const PortalLogin = ({ slug, brand, onLogin }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const demoCreds = DEMO_PORTAL_LOGINS[slug];

    const submit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await adminLogin(email, password);
            await onLogin();
        } catch (err) {
            setError(err?.response?.data?.detail || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="portal-login-wrap">
            <form className="portal-login" onSubmit={submit}>
                <span className="portal-logo"><Icon name="lock" size={22} /></span>
                <h1>{brand || slug} — Admin</h1>
                <p>Staff sign-in for the {brand || slug} customer-care console.</p>
                {error && <div className="portal-login-error">{error}</div>}
                <input type="email" placeholder="Work email" value={email}
                    onChange={(e) => setEmail(e.target.value)} autoFocus />
                <input type="password" placeholder="Password" value={password}
                    onChange={(e) => setPassword(e.target.value)} />
                <button type="submit" disabled={loading || !email || !password}>
                    {loading ? 'Signing in…' : 'Sign in'}
                </button>
                {import.meta.env.DEV && demoCreds && (
                    <button
                        type="button"
                        className="portal-login-demo"
                        onClick={() => { setEmail(demoCreds.email); setPassword(demoCreds.password); setError(''); }}
                    >
                        Autofill demo login
                    </button>
                )}
                <div className="portal-login-note">
                    Access is issued by your platform operator. This portal shows only {slug}'s data.
                </div>
                <a className="portal-login-demo" href={`/c/${slug}`} target="_blank" rel="noopener noreferrer">
                    <Icon name="phone-call" size={14} /> Test the chatbot →
                </a>
            </form>
        </div>
    );
};

// ---- page -------------------------------------------------------------------
const ClientPortal = () => {
    const { slug } = useParams();
    const [status, setStatus] = useState('loading'); // loading | login | denied | ok
    const [me, setMe] = useState(null);
    const [config, setConfig] = useState(null);
    const [tab, setTab] = useState('dashboard');

    const checkAuth = useCallback(async () => {
        if (!isAuthenticated() || !getToken()) { setStatus('login'); return; }
        try {
            const who = await getMe();
            if (who.is_superadmin || who.client_slug === slug) {
                setMe(who);
                setStatus('ok');
            } else {
                setStatus('denied');
            }
        } catch {
            setStatus('login');
        }
    }, [slug]);

    useEffect(() => {
        publicGetConfig(slug).then(setConfig).catch(() => setConfig(null));
        checkAuth();
    }, [slug, checkAuth]);

    const signOut = () => { logout(); setMe(null); setStatus('login'); };

    const brand = config?.name || slug;
    const isTelecom = (config?.domain || '') === 'telecom';
    const tabs = isTelecom ? TELECOM_TABS : BASIC_TABS;

    if (status === 'loading') {
        return <div className="portal"><Spinner text="Loading portal…" /></div>;
    }
    if (status === 'login') {
        return <PortalLogin slug={slug} brand={config?.name} onLogin={checkAuth} />;
    }
    if (status === 'denied') {
        return (
            <div className="portal-login-wrap">
                <div className="portal-login">
                    <span className="portal-logo"><Icon name="alert" size={22} /></span>
                    <h1>No access</h1>
                    <p>This account isn't authorized for the {brand} portal.</p>
                    <button type="submit" onClick={signOut}>Sign in with another account</button>
                </div>
            </div>
        );
    }

    const activeTab = tabs.find((t) => t.key === tab) ? tab : 'dashboard';

    return (
        <div className="portal">
            <div className="portal-top">
                <div className="portal-top-inner">
                    <div className="portal-brand">
                        <span className="portal-logo"><Icon name="phone-call" size={20} /></span>
                        <div>
                            <h1>{brand} <span style={{ fontWeight: 500, color: 'var(--text-faint)' }}>· Admin</span></h1>
                            <p>Customer-care operations console</p>
                        </div>
                    </div>
                    <div className="portal-top-spacer" />
                    <div className="portal-user">
                        <strong>{me?.name || me?.email}</strong>
                        <span>{me?.is_superadmin ? 'Operator (superadmin)' : 'Client admin'}</span>
                    </div>
                    <a className="btn btn-sm" href={`/c/${slug}`} target="_blank" rel="noopener noreferrer">
                        <Icon name="phone-call" size={14} /> Test chatbot
                    </a>
                    <button className="btn btn-sm" onClick={signOut}>
                        <Icon name="logout" size={14} /> Sign out
                    </button>
                </div>
            </div>

            <nav className="portal-tabs">
                <div className="portal-tabs-inner">
                    {tabs.map((t) => (
                        <button key={t.key}
                            className={`portal-tab ${activeTab === t.key ? 'active' : ''}`}
                            onClick={() => setTab(t.key)}>
                            <Icon name={t.icon} size={15} /> {t.label}
                        </button>
                    ))}
                </div>
            </nav>

            <main className="portal-main">
                {activeTab === 'dashboard' && <Dashboard slug={slug} isTelecom={isTelecom} onGoto={setTab} />}
                {activeTab === 'subscribers' && <SubscriberTable slug={slug} />}
                {activeTab === 'customers' && <CustomerTable slug={slug} />}
                {activeTab === 'plans' && <PlanCatalog slug={slug} />}
                {activeTab === 'tickets' && <TicketBoard slug={slug} />}
                {activeTab === 'activations' && <ActivationsTable slug={slug} />}
                {activeTab === 'calls' && <CallHistory slug={slug} />}
                {activeTab === 'billing' && <BillingHistory slug={slug} />}
            </main>
        </div>
    );
};

export default ClientPortal;
