import { useState, useEffect } from 'react';
import { portalPlans } from '../../services/api';
import Icon from '../Icon';
import { Spinner, Badge, Mono, EmptyState, fmtLKR, fmtData, fmtNum } from './ui';

const PlanCatalog = ({ slug }) => {
    const [rows, setRows] = useState([]);
    const [loading, setLoading] = useState(true);

    const load = async () => {
        setLoading(true);
        try { setRows((await portalPlans(slug)).plans || []); }
        catch (e) { console.error(e); }
        finally { setLoading(false); }
    };
    useEffect(() => { load(); }, [slug]);

    return (
        <div>
            <div className="portal-section-head">
                <h2>Plan catalog</h2>
                <span className="sub">{rows.length} products</span>
                <div className="portal-head-spacer" />
                <button className="btn" onClick={load}><Icon name="refresh" size={14} /> Refresh</button>
            </div>

            <div className="portal-panel">
                {loading ? <Spinner /> : rows.length === 0 ? (
                    <EmptyState icon="briefcase" text="No plans in the catalog yet." />
                ) : (
                    <div className="table-scroll">
                        <table className="portal-table">
                            <thead>
                                <tr><th>Code</th><th>Name</th><th>Type</th><th className="num">Rental</th><th className="num">Data</th><th className="num">Voice</th><th className="num">SMS</th><th className="num">Validity</th><th>Status</th></tr>
                            </thead>
                            <tbody>
                                {rows.map((p) => (
                                    <tr key={p.id}>
                                        <td className="strong"><Mono>{p.code}</Mono></td>
                                        <td>{p.name}</td>
                                        <td><Badge value={p.plan_type} /></td>
                                        <td className="num">{fmtLKR(p.monthly_rental)}</td>
                                        <td className="num">{p.data_quota_mb ? fmtData(p.data_quota_mb) : 'Unlimited'}</td>
                                        <td className="num">{p.voice_minutes ? fmtNum(p.voice_minutes) : '—'}</td>
                                        <td className="num">{p.sms_units ? fmtNum(p.sms_units) : '—'}</td>
                                        <td className="num">{p.validity_days}d</td>
                                        <td><Badge value={p.is_active ? 'active' : 'inactive'} /></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PlanCatalog;
