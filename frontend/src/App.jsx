import { useState } from 'react';
import ClientManager from './components/ClientManager';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';
import VoiceChat from './components/VoiceChat';
import './App.css';

function App() {
    const [selectedClient, setSelectedClient] = useState(null);
    const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'voice'

    return (
        <div className="app">
            <header className="app-header">
                <div className="header-content">
                    <h1>🤖 RAG Chatbot System</h1>
                    <p>Multi-tenant AI chatbot with document-augmented responses</p>
                </div>
            </header>

            <main className="app-main">
                <div className="container">
                    <div className="section">
                        <ClientManager
                            onClientSelect={setSelectedClient}
                            selectedClient={selectedClient}
                        />
                    </div>

                    {selectedClient && (
                        <>
                            <div className="section">
                                <DocumentUpload clientId={selectedClient} />
                            </div>

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

                            <div className="section">
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
