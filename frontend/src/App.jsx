import { useState, useEffect } from 'react';
import { listClients } from './services/api';
import ChatInterface from './components/ChatInterface';
import VoiceChat from './components/VoiceChat';
import './App.css';

function App() {
    const [selectedClient, setSelectedClient] = useState(null);
    const [clients, setClients] = useState([]);
    const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'voice'

    useEffect(() => {
        loadClients();
    }, []);

    const loadClients = async () => {
        try {
            const data = await listClients(0, 100);
            setClients(data.clients || []);
        } catch (err) {
            console.error('Failed to load clients:', err);
        }
    };

    return (
        <div className="app">
            <header className="app-header">
                <div className="header-content">
                    <h1>🤖 RAG Chatbot System</h1>
                    <p>Multi-tenant AI chatbot with document-augmented responses</p>
                    <div className="client-selector">
                        <label htmlFor="client-select">Client:</label>
                        <select 
                            id="client-select"
                            value={selectedClient || ''} 
                            onChange={(e) => setSelectedClient(e.target.value || null)}
                            className="client-dropdown"
                        >
                            <option value="">Select a client...</option>
                            {clients.map((client) => (
                                <option key={client.client_id} value={client.client_id}>
                                    {client.client_id} ({client.document_count || 0} docs)
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </header>

            <main className="app-main">
                <div className="container">
                    {selectedClient && (
                        <>
                            {/* Tab Switcher */}
                            <div className="section">
                                <div className="tab-switcher">
                                    <button
                                        className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('chat')}
                                    >
                                        💬 Text Chat
                                    </button>
                                    <button
                                        className={`tab-button ${activeTab === 'voice' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('voice')}
                                    >
                                        🎙️ Voice Chat
                                    </button>
                                </div>
                            </div>

                            <div className="section chat-section">
                                {activeTab === 'chat' ? (
                                    <ChatInterface clientId={selectedClient} />
                                ) : (
                                    <VoiceChat clientId={selectedClient} />
                                )}
                            </div>
                        </>
                    )}

                    {!selectedClient && (
                        <div className="welcome-section">
                            <div className="welcome-card">
                                <h2>👋 Welcome to RAG Chatbot</h2>
                                <p>Get started in 3 easy steps:</p>
                                <ol>
                                    <li>
                                        <strong>Create a Client</strong> - Set up a new chatbot instance
                                    </li>
                                    <li>
                                        <strong>Upload Documents</strong> - Add PDF files for the AI to learn from
                                    </li>
                                    <li>
                                        <strong>Start Chatting</strong> - Ask questions and get AI-powered answers
                                    </li>
                                </ol>
                                <div className="features">
                                    <div className="feature">
                                        <span>🔒</span>
                                        <h4>Multi-tenant</h4>
                                        <p>Isolated data per client</p>
                                    </div>
                                    <div className="feature">
                                        <span>📚</span>
                                        <h4>Document RAG</h4>
                                        <p>Context-aware responses</p>
                                    </div>
                                    <div className="feature">
                                        <span>💬</span>
                                        <h4>Conversational</h4>
                                        <p>Natural dialogue flow</p>
                                    </div>
                                    <div className="feature">
                                        <span>⚡</span>
                                        <h4>Fast & Accurate</h4>
                                        <p>Powered by Groq LLM</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </main>

            <footer className="app-footer">
                <p>Built with React + Vite | Backend: FastAPI + Groq | Phase 8 Complete</p>
            </footer>
        </div>
    );
}

export default App;
