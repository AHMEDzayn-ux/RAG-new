import { useState, useEffect } from 'react';
import { portalCustomers } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtDate } from './ui';

const PAGE = 50;

const CustomerTable = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [total, setTotal] = useState(0);
    const [q, setQ] = useState('');
    const [skip, setSkip] = useState(0);
    const [loading, setLoading] = useState(true);

    const load = async () => {
        setLoading(true);
        try {
            const res = await portalCustomers(slug, { q, skip, limit: PAGE });
            setRows(res.customers || []);
            setTotal(res.total || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); /* eslint-disable-next-line */ }, [slug, skip]);

    const onSearch = (e) => { e.preventDefault(); setSkip(0); load(); };

    return (
        <div>
            <div className="portal-section-head">
                <h2>Customers</h2>
                <span className="sub">{total} account holders</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-filters">
                <form className="portal-search" onSubmit={onSearch}>
                    <Icon name="search" size={15} />
                    <input className="portal-input" placeholder="Search name, NIC, email or phone…"
                        value={q} onChange={(e) => setQ(e.target.value)} />
                </form>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="users" text="No customers match." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Name</th><th>NIC</th><th>Phone</th><th>City</th><th>Type</th><th>KYC</th><th className="num">Lines</th><th>Since</th></tr>
                            </thead>
                            <tbody>
                                {rows.map((c) => (
                                    <tr key={c.id}>
                                        <td className="strong">{c.full_name}</td>
                                        <td><Mono>{c.nic || '—'}</Mono></td>
                                        <td><Mono>{c.phone || '—'}</Mono></td>
                                        <td>{c.city || '—'}</td>
                                        <td><Badge value={c.customer_type} /></td>
                                        <td><Badge value={c.kyc_status} /></td>
                                        <td className="num">{c.subscription_count}</td>
                                        <td>{fmtDate(c.created_at)}</td>
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

export default CustomerTable;
