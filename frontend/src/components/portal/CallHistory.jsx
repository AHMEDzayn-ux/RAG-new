import { useState, useEffect } from 'react';
import { portalCdrs } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtLKR, fmtBytes, fmtDuration, fmtDateTime } from './ui';

const PAGE = 100;

const CallHistory = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [total, setTotal] = useState(0);
    const [msisdn, setMsisdn] = useState('');
    const [skip, setSkip] = useState(0);
    const [loading, setLoading] = useState(true);

    const load = async () => {
        setLoading(true);
        try {
            const res = await portalCdrs(slug, { msisdn, skip, limit: PAGE });
            setRows(res.cdrs || []);
            setTotal(res.total || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, skip]);

    const onSearch = (e) => { e.preventDefault(); setSkip(0); load(); };

    return (
        <div>
            <div className="portal-section-head">
                <h2>Call history (CDR)</h2>
                <span className="sub">{total} usage records</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-filters">
                <form className="portal-search" onSubmit={onSearch}>
                    <Icon name="search" size={15} />
                    <input className="portal-input" placeholder="Filter by MSISDN…"
                        value={msisdn} onChange={(e) => setMsisdn(e.target.value)} />
                </form>
                <span className="sub">Tip: paste a subscriber number to see just their calls.</span>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="phone-call" text="No call records match." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>MSISDN</th><th>Type</th><th>Direction</th><th>Other party</th><th className="num">Duration</th><th className="num">Data</th><th className="num">Charge</th><th>Cell site</th><th>When</th></tr>
                            </thead>
                            <tbody>
                                {rows.map((r) => (
                                    <tr key={r.id}>
                                        <td className="strong"><Mono>{r.msisdn}</Mono></td>
                                        <td><Badge value={r.event_type} /></td>
                                        <td>{r.direction}</td>
                                        <td><Mono>{r.other_party || '—'}</Mono></td>
                                        <td className="num">{fmtDuration(r.duration_sec)}</td>
                                        <td className="num">{r.bytes_used ? fmtBytes(r.bytes_used) : '—'}</td>
                                        <td className="num">{fmtLKR(r.charged_amount)}</td>
                                        <td><Mono>{r.cell_site || '—'}</Mono></td>
                                        <td>{fmtDateTime(r.start_time)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {total > PAGE && (
                <div className="portal-filters" style={{ justifyContent: 'flex-end' }}>
                    <button className="btn btn-sm" disabled={skip === 0} onClick={() => setSkip(Math.max(0, skip - PAGE))}>
                        <Icon name="arrow-left" size={14} /> Prev
                    </button>
                    <span className="sub">{skip + 1}–{Math.min(skip + PAGE, total)} of {total}</span>
                    <button className="btn btn-sm" disabled={skip + PAGE >= total} onClick={() => setSkip(skip + PAGE)}>
                        Next <Icon name="chevron-right" size={14} />
                    </button>
                </div>
            )}
        </div>
    );
};

export default CallHistory;
