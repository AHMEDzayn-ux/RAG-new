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
const demoDate = new Date().toISOString();
const DEMO_OVERVIEW = {
    subscribers_total: 64,
    subscribers_active: 60,
    customers_total: 50,
    tickets_open: 33,
    activations_today: 3,
    revenue_today: 2730,
    revenue_month: 18450,
    recent_activations: [
        { id: 1, reference: 'DEMO-A101', msisdn: 'Demo line 01', package_name: 'Weekly Data', channel: 'web', price: 490, activated_at: demoDate },
        { id: 2, reference: 'DEMO-A102', msisdn: 'Demo line 02', package_name: 'Monthly Combo', channel: 'voice', price: 1290, activated_at: demoDate },
    ],
    recent_tickets: [
        { id: 1, ticket_number: 'DEMO-T101', subject: 'Help with prepaid recharge', priority: 'medium', status: 'open', channel: 'web', created_at: demoDate },
        { id: 2, ticket_number: 'DEMO-T102', subject: 'Check mobile data settings', priority: 'low', status: 'resolved', channel: 'voice', created_at: demoDate },
    ],
};

const PortalLogin = ({ slug, brand, onLogin, onDemo }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

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
                {onDemo && (
                    <button
                        type="button"
                        className="portal-login-demo"
                        onClick={onDemo}
                        disabled={loading}
                    >
                        Demo Login
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
    const [status, setStatus] = useState('loading'); // loading | login | denied | ok | demo
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

    const isDemo = status === 'demo';
    const signOut = () => {
        if (!isDemo) logout();
        setMe(null);
        setTab('dashboard');
        setStatus('login');
    };
    const openDemo = () => {
        setTab('dashboard');
        setStatus('demo');
    };

    const brand = isDemo ? 'Nexus Telecom' : (config?.name || slug);
    const isTelecom = (config?.domain || '') === 'telecom';
    const tabs = isDemo ? [] : (isTelecom ? TELECOM_TABS : BASIC_TABS);

    if (status === 'loading') {
        return <div className="portal"><Spinner text="Loading portal…" /></div>;
    }
    if (status === 'login') {
        return <PortalLogin slug={slug} brand={config?.name} onLogin={checkAuth}
            onDemo={slug === 'nexus' ? openDemo : undefined} />;
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
                            <p>{isDemo ? 'Read-only demo with sample data' : 'Customer-care operations console'}</p>
                        </div>
                    </div>
                    <div className="portal-top-spacer" />
                    {!isDemo && (
                        <div className="portal-user">
                            <strong>{me?.name || me?.email}</strong>
                            <span>{me?.is_superadmin ? 'Operator (superadmin)' : 'Client admin'}</span>
                        </div>
                    )}
                    <a className="btn btn-sm" href={`/c/${slug}`} target="_blank" rel="noopener noreferrer">
                        <Icon name="phone-call" size={14} /> Test chatbot
                    </a>
                    <button className="btn btn-sm" onClick={signOut}>
                        <Icon name="logout" size={14} /> {isDemo ? 'Exit demo' : 'Sign out'}
                    </button>
                </div>
            </div>

            {!isDemo && <nav className="portal-tabs">
                <div className="portal-tabs-inner">
                    {tabs.map((t) => (
                        <button key={t.key}
                            className={`portal-tab ${activeTab === t.key ? 'active' : ''}`}
                            onClick={() => setTab(t.key)}>
                            <Icon name={t.icon} size={15} /> {t.label}
                        </button>
                    ))}
                </div>
            </nav>}

            <main className="portal-main">
                {activeTab === 'dashboard' && <Dashboard key={isDemo ? 'demo' : slug} slug={slug}
                    isTelecom={isTelecom} onGoto={isDemo ? undefined : setTab}
                    demoData={isDemo ? DEMO_OVERVIEW : undefined} />}
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
