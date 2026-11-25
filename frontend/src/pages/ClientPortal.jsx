import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
    isAuthenticated, getToken, adminLogin, logout, getMe, publicGetConfig,
    DEMO_LOGIN, DEMO_PORTAL, DEMO_EXPIRED_EVENT, hasDemoSession, clearDemoSession, openDemoSession,
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
const PortalLogin = ({ slug, brand, onLogin, onDemo, notice }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const submit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            if (onDemo && email.trim().toLowerCase() === DEMO_LOGIN.email) {
                await onDemo({ email, password });
            } else {
                await adminLogin(email, password);
                await onLogin();
            }
        } catch (err) {
            setError(err?.response?.data?.detail || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    const fillDemoLogin = () => {
        setEmail(DEMO_LOGIN.email);
        setPassword(DEMO_LOGIN.password);
        setError('');
    };

    return (
        <div className="portal-login-wrap">
            <form className="portal-login" onSubmit={submit}>
                <span className="portal-logo"><Icon name="lock" size={22} /></span>
                <h1>{brand || slug} — Admin</h1>
                <p>Staff sign-in for the {brand || slug} customer-care console.</p>
                {(error || notice) && <div className="portal-login-error">{error || notice}</div>}
                <input type="email" autoComplete="username" placeholder="Work email" value={email}
                    onChange={(e) => setEmail(e.target.value)} autoFocus />
                <input type="password" autoComplete="current-password" placeholder="Password" value={password}
                    onChange={(e) => setPassword(e.target.value)} />
                <button type="submit" disabled={loading || !email || !password}>
                    {loading ? 'Signing in…' : 'Sign in'}
                </button>
                {onDemo && (
                    <button
                        type="button"
                        className="portal-login-demo"
                        onClick={fillDemoLogin}
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
    const [notice, setNotice] = useState('');

    const checkAuth = useCallback(async () => {
        if (slug === 'nexus' && hasDemoSession()) { setStatus('demo'); return; }
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

    useEffect(() => {
        const onDemoExpired = () => {
            setNotice('Your demo session ended. Click Demo Login to start again.');
            setTab('dashboard');
            setStatus((previous) => previous === 'demo' ? 'login' : previous);
        };
        window.addEventListener(DEMO_EXPIRED_EVENT, onDemoExpired);
        return () => window.removeEventListener(DEMO_EXPIRED_EVENT, onDemoExpired);
    }, []);

    const isDemo = status === 'demo' && slug === 'nexus';
    const signOut = () => {
        if (isDemo) clearDemoSession();
        else logout();
        setMe(null);
        setNotice('');
        setTab('dashboard');
        setStatus('login');
    };
    const openDemo = async (login) => {
        setNotice('');
        await openDemoSession(login);
        setTab('dashboard');
        setStatus('demo');
        window.scrollTo(0, 0);
    };

    const brand = isDemo ? 'Nexus Telecom' : (config?.name || slug);
    const isTelecom = isDemo || config?.domain === 'telecom';
    const tabs = isTelecom ? TELECOM_TABS : BASIC_TABS;
    const portalScope = isDemo ? DEMO_PORTAL : slug;

    if (status === 'loading' || (status === 'demo' && !isDemo)) {
        return <div className="portal"><Spinner text="Loading portal…" /></div>;
    }
    if (status === 'login') {
        return <PortalLogin slug={slug} brand={config?.name} onLogin={checkAuth} notice={notice}
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
                            <h1>{brand} <span style={{ fontWeight: 500, color: 'var(--text-faint)' }}>· {isDemo ? 'Demo' : 'Admin'}</span></h1>
                            <p>{isDemo ? 'Interactive demo with sample data' : 'Customer-care operations console'}</p>
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

            <main className="portal-main" key={isDemo ? 'demo' : slug}>
                {activeTab === 'dashboard' && <Dashboard slug={portalScope}
                    isTelecom={isTelecom} onGoto={setTab} isDemo={isDemo} />}
                {activeTab === 'subscribers' && <SubscriberTable slug={portalScope} />}
                {activeTab === 'customers' && <CustomerTable slug={portalScope} />}
                {activeTab === 'plans' && <PlanCatalog slug={portalScope} />}
                {activeTab === 'tickets' && <TicketBoard slug={portalScope} />}
                {activeTab === 'activations' && <ActivationsTable slug={portalScope} />}
                {activeTab === 'calls' && <CallHistory slug={portalScope} />}
                {activeTab === 'billing' && <BillingHistory slug={portalScope} />}
            </main>
        </div>
    );
};

export default ClientPortal;
