import { useState, useEffect } from 'react';
import { portalSubscriptions } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, fmtLKR, fmtData, EmptyState } from './ui';
import SubscriberDetail from './SubscriberDetail';

const PAGE = 50;

const SubscriberTable = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [total, setTotal] = useState(0);
    const [q, setQ] = useState('');
    const [status, setStatus] = useState('');
    const [skip, setSkip] = useState(0);
    const [loading, setLoading] = useState(true);
    const [selected, setSelected] = useState(null);

    const load = async () => {
        setLoading(true);
        try {
            const res = await portalSubscriptions(slug, { q, status, skip, limit: PAGE });
            setRows(res.subscriptions || []);
            setTotal(res.total || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, status, skip]);

    const onSearch = (e) => { e.preventDefault(); setSkip(0); load(); };

    return (
        <div>
            <div className="portal-section-head">
                <h2>Subscribers</h2>
                <span className="sub">{total} lines</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-filters">
                <form className="portal-search" onSubmit={onSearch}>
                    <Icon name="search" size={15} />
                    <input className="portal-input" placeholder="Search name or MSISDN…"
                        value={q} onChange={(e) => setQ(e.target.value)} />
                </form>
                <select className="portal-select" value={status} onChange={(e) => { setStatus(e.target.value); setSkip(0); }}>
                    <option value="">All statuses</option>
                    <option value="active">Active</option>
                    <option value="suspended">Suspended</option>
                    <option value="deactivated">Deactivated</option>
                </select>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="phone" text="No subscribers match." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr>
                                    <th>MSISDN</th><th>Customer</th><th>Plan</th><th>Type</th>
                                    <th className="num">Balance</th><th className="num">Data</th><th>Status</th><th>City</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows.map((s) => (
                                    <tr key={s.id} className="clickable" onClick={() => setSelected(s.msisdn)}>
                                        <td className="strong"><Mono>{s.msisdn}</Mono></td>
                                        <td>{s.customer_name}</td>
                                        <td>{s.plan_name || '—'}</td>
                                        <td><Badge value={s.plan_type} /></td>
                                        <td className="num">{fmtLKR(s.prepaid_balance)}</td>
                                        <td className="num">{fmtData(s.data_balance_mb)}</td>
                                        <td><Badge value={s.status} /></td>
                                        <td>{s.city || '—'}</td>
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

            {selected && (
                <SubscriberDetail slug={slug} msisdn={selected}
                    onClose={() => setSelected(null)} onChanged={load} />
            )}
        </div>
    );
};

export default SubscriberTable;
