import { useState, useEffect } from 'react';
import { portalActivations } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtLKR, fmtDateTime } from './ui';

const PAGE = 100;

const ActivationsTable = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [total, setTotal] = useState(0);
    const [msisdn, setMsisdn] = useState('');
    const [channel, setChannel] = useState('');
    const [skip, setSkip] = useState(0);
    const [loading, setLoading] = useState(true);

    const load = async () => {
        setLoading(true);
        try {
            const res = await portalActivations(slug, { msisdn, channel, skip, limit: PAGE });
            setRows(res.activations || []);
            setTotal(res.total || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, channel, skip]);

    const onSearch = (e) => { e.preventDefault(); setSkip(0); load(); };

    return (
        <div>
            <div className="portal-section-head">
                <h2>Package activations</h2>
                <span className="sub">{total} records · recorded on every activation</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-filters">
                <form className="portal-search" onSubmit={onSearch}>
                    <Icon name="search" size={15} />
                    <input className="portal-input" placeholder="Filter by MSISDN…"
                        value={msisdn} onChange={(e) => setMsisdn(e.target.value)} />
                </form>
                <select className="portal-select" value={channel} onChange={(e) => { setChannel(e.target.value); setSkip(0); }}>
                    <option value="">All channels</option>
                    <option value="chatbot">Chatbot</option>
                    <option value="agent">Agent</option>
                    <option value="app">App</option>
                    <option value="ussd">USSD</option>
                    <option value="web">Web</option>
                    <option value="admin">Admin</option>
                </select>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="refresh" text="No activations match." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Reference</th><th>MSISDN</th><th>Package</th><th>Channel</th><th className="num">Price</th><th className="num">Validity</th><th>Status</th><th>Activated</th><th>Expires</th></tr>
                            </thead>
                            <tbody>
                                {rows.map((a) => (
                                    <tr key={a.id}>
                                        <td className="strong"><Mono>{a.reference}</Mono></td>
                                        <td><Mono>{a.msisdn}</Mono></td>
                                        <td>{a.package_name}</td>
                                        <td><Badge value={a.channel} /></td>
                                        <td className="num">{fmtLKR(a.price)}</td>
                                        <td className="num">{a.validity_days}d</td>
                                        <td><Badge value={a.status} /></td>
                                        <td>{fmtDateTime(a.activated_at)}</td>
                                        <td>{fmtDateTime(a.expires_at)}</td>
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

export default ActivationsTable;
