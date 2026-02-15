import React, { useState, useEffect } from 'react';
import { createClient, listClients, deleteClient } from '../services/api';
import './ClientManager.css';

const ClientManager = ({ onClientSelect, selectedClient }) => {
    const [clients, setClients] = useState([]);
    const [newClientId, setNewClientId] = useState('');
    const [newClientDesc, setNewClientDesc] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        loadClients();
    }, []);

    const loadClients = async () => {
        try {
            setLoading(true);
            const data = await listClients();
            setClients(data.clients || []);
            setError('');
        } catch (err) {
            setError('Failed to load clients: ' + err.message);
        } finally {
            setLoading(false);
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
            await loadClients();
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
            await loadClients();
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
                <h3>Existing Clients ({clients.length})</h3>
                {clients.length === 0 ? (
                    <p className="empty-state">No clients yet. Create one to get started!</p>
                ) : (
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
                )}
            </div>
        </div>
    );
};

export default ClientManager;
