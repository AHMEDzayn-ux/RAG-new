import React, { useState, useEffect } from 'react';
import { createClient, listClients, deleteClient } from '../services/api';
import './ClientManager.css';

const ClientManager = ({ onClientSelect, selectedClient }) => {
    const [clients, setClients] = useState([]);
    const [newClientId, setNewClientId] = useState('');
    const [newClientDesc, setNewClientDesc] = useState('');
    const [loading, setLoading] = useState(false);
    const [initialLoading, setInitialLoading] = useState(true);
    const [error, setError] = useState('');
    const [page, setPage] = useState(0);
    const [hasMore, setHasMore] = useState(true);
    const [totalClients, setTotalClients] = useState(0);
    const PAGE_SIZE = 20;

    useEffect(() => {
        loadClients(true); // Load initial batch on mount
    }, []);

    const loadClients = async (reset = false) => {
        try {
            if (reset && page === 0) {
                setInitialLoading(true);
            } else {
                setLoading(true);
            }
            const currentPage = reset ? 0 : page;
            const skip = currentPage * PAGE_SIZE;

            const data = await listClients(skip, PAGE_SIZE);

            if (reset) {
                // Reset: replace all clients
                setClients(data.clients || []);
                setPage(1);
            } else {
                // Append: add more clients
                setClients(prev => [...prev, ...(data.clients || [])]);
                setPage(prev => prev + 1);
            }

            setTotalClients(data.total || 0);
            setHasMore((skip + PAGE_SIZE) < (data.total || 0));
            setError('');
        } catch (err) {
            setError('Failed to load clients: ' + err.message);
        } finally {
            setLoading(false);
            setInitialLoading(false);
        }
    };

    const loadMoreClients = () => {
        if (!loading && hasMore) {
            loadClients(false);
        }
    };

    const handleCreateClient = async (e) => {
        e.preventDefault();
        if (!newClientId.trim()) {
            setError('Client ID is required');
            return;
        }

        try {
            setLoading(true);
            await createClient(newClientId, newClientDesc);
            setNewClientId('');
            setNewClientDesc('');
            await loadClients(true); // Reset pagination
            setError('');
        } catch (err) {
            setError('Failed to create client: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteClient = async (clientId) => {
        if (!window.confirm(`Delete client "${clientId}"?`)) return;

        try {
            setLoading(true);
            await deleteClient(clientId);
            if (selectedClient === clientId) {
                onClientSelect(null);
            }
            await loadClients(true); // Reset pagination
            setError('');
        } catch (err) {
            setError('Failed to delete client: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="client-manager">
            <h2>Client Management</h2>

            {error && <div className="error-message">{error}</div>}

            <form onSubmit={handleCreateClient} className="create-client-form">
                <input
                    type="text"
                    placeholder="Client ID (e.g., acme-corp)"
                    value={newClientId}
                    onChange={(e) => setNewClientId(e.target.value)}
                    disabled={loading}
                />
                <input
                    type="text"
                    placeholder="Description (optional)"
                    value={newClientDesc}
                    onChange={(e) => setNewClientDesc(e.target.value)}
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Creating...' : 'Create Client'}
                </button>
            </form>

            <div className="clients-list">
                <h3>Existing Clients ({clients.length}{totalClients > clients.length ? ` of ${totalClients}` : ''})</h3>

                {initialLoading ? (
                    <div className="loading-skeleton">
                        <p>Loading clients...</p>
                        <div className="skeleton-cards">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="skeleton-card"></div>
                            ))}
                        </div>
                    </div>
                ) : totalClients === 0 ? (
                    <p className="empty-state">No clients yet. Create one to get started!</p>
                ) : (
                    <>
                        <div className="clients-grid">
                            {clients.map((client) => (
                                <div
                                    key={client.client_id}
                                    className={`client-card ${selectedClient === client.client_id ? 'selected' : ''}`}
                                >
                                    <div className="client-info">
                                        <h4>{client.client_id}</h4>
                                        {client.description && <p>{client.description}</p>}
                                        <small>{client.document_count || 0} documents</small>
                                    </div>
                                    <div className="client-actions">
                                        <button
                                            onClick={() => onClientSelect(client.client_id)}
                                            className="btn-select"
                                            disabled={selectedClient === client.client_id}
                                        >
                                            {selectedClient === client.client_id ? 'Selected' : 'Select'}
                                        </button>
                                        <button
                                            onClick={() => handleDeleteClient(client.client_id)}
                                            className="btn-delete"
                                            disabled={loading}
                                        >
                                            Delete
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                        {hasMore && (
                            <div className="load-more-container">
                                <button
                                    className="btn-load-more"
                                    onClick={loadMoreClients}
                                    disabled={loading}
                                >
                                    {loading ? 'Loading...' : `Load More (${totalClients - clients.length} remaining)`}
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default ClientManager;
