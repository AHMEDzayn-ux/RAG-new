/*
 * Shared UI helpers for the client portal — formatters + small presentational
 * atoms (status pills, empty/loading states) so every table stays consistent.
 */
import Icon from '../Icon';

// ---- formatters -------------------------------------------------------------
export const fmtLKR = (n) =>
    `LKR ${Number(n || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const fmtNum = (n) => Number(n || 0).toLocaleString();
export const fmtDate = (s) => (s ? new Date(s).toLocaleDateString() : '—');
export const fmtDateTime = (s) => (s ? new Date(s).toLocaleString() : '—');

export const fmtData = (mb) => {
    const m = Number(mb || 0);
    return m >= 1024 ? `${(m / 1024).toFixed(1)} GB` : `${m} MB`;
};
export const fmtBytes = (b) => {
    const n = Number(b || 0);
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
};
export const fmtDuration = (sec) => {
    const s = Number(sec || 0);
    if (!s) return '—';
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m ? `${m}m ${r}s` : `${r}s`;
};

// ---- atoms ------------------------------------------------------------------
const slug = (v) => String(v || '').toLowerCase().replace(/[^a-z0-9]+/g, '-');

export const Badge = ({ value }) =>
    value ? <span className={`pill pill-${slug(value)}`}>{String(value).replace(/_/g, ' ')}</span> : <span className="muted">—</span>;

export const Mono = ({ children }) => <span className="mono">{children}</span>;

export const EmptyState = ({ icon = 'inbox', text }) => (
    <div className="portal-empty"><Icon name={icon} size={26} /><p>{text}</p></div>
);

export const Spinner = ({ text = 'Loading…' }) => (
    <div className="portal-loading"><span className="portal-spin" />{text}</div>
);
