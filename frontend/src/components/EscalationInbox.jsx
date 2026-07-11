import { useState, useEffect } from 'react';
import { listEscalations, resolveEscalation } from '../services/api';
import Icon from './Icon';

// Mood → dot color, so operators can scan sentiment without emoji.
const EMOTION_COLOR = {
    angry: '#dc2626', frustrated: '#ea580c', confused: '#ca8a04',
    happy: '#059669', neutral: '#94a3b8',
};

const EscalationInbox = ({ slug }) => {
    const [items, setItems] = useState([]);
    const [openCount, setOpenCount] = useState(0);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState({});

    const load = async () => {
        setLoading(true);
        try {
            const data = await listEscalations(slug);
            setItems(data.escalations || []);
            setOpenCount(data.open_count || 0);
        } catch (err) {
            console.error('Failed to load escalations', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { load(); }, [slug]);

    const handleResolve = async (id) => {
        try {
            await resolveEscalation(slug, id);
            await load();
        } catch (err) {
            alert('Failed to resolve: ' + (err?.response?.data?.detail || err.message));
        }
    };

    if (loading) return <div className="inbox"><p>Loading…</p></div>;

    return (
        <div className="inbox">
            <div className="inbox-head">
                <h3><Icon name="inbox" size={17} /> Escalations {openCount > 0 && <span className="inbox-badge">{openCount} open</span>}</h3>
                <button className="btn-mini" onClick={load}>Refresh</button>
            </div>

            {items.length === 0 ? (
                <p className="empty-state">No escalations yet. Conversations handed off to a human will appear here.</p>
            ) : (
                <div className="inbox-list">
                    {items.map((e) => (
                        <div key={e.id} className={`inbox-item ${e.status}`}>
                            <div className="inbox-item-head">
                                <span className="inbox-emotion">
                                    <span className="emotion-dot" style={{ background: EMOTION_COLOR[e.emotion] || EMOTION_COLOR.neutral }} />
                                    {e.emotion || 'neutral'}{e.intensity ? ` · ${e.intensity}/5` : ''}
                                </span>
                                <span className={`inbox-status status-${e.status}`}>{e.status}</span>
                                <span className="inbox-time">{e.created_at ? new Date(e.created_at).toLocaleString() : ''}</span>
                            </div>
                            <div className="inbox-reason"><strong>{e.reason}</strong></div>
                            {e.summary && <div className="inbox-summary">{e.summary}</div>}
                            <div className="inbox-actions">
                                <button className="btn-mini" onClick={() => setExpanded((p) => ({ ...p, [e.id]: !p[e.id] }))}>
                                    {expanded[e.id] ? 'Hide' : 'View'} transcript
                                </button>
                                {e.status === 'open' && (
                                    <button className="btn-mini resolve" onClick={() => handleResolve(e.id)}>
                                        Mark resolved
                                    </button>
                                )}
                            </div>
                            {expanded[e.id] && e.transcript && (
                                <pre className="inbox-transcript">{e.transcript}</pre>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default EscalationInbox;
