import React, { useState, useRef, useEffect } from 'react';
import { queryDocuments, chatWithDocuments } from '../services/api';
import './ChatInterface.css';

const ChatInterface = ({ clientId }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState('chat'); // 'chat' or 'query'
    const [showSources, setShowSources] = useState(true);
    const [expandedSources, setExpandedSources] = useState({});
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;

        const userMessage = {
            role: 'user',
            content: input.trim(),
            timestamp: new Date().toLocaleTimeString(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            let response;

            if (mode === 'query') {
                // One-shot query
                response = await queryDocuments(clientId, userMessage.content, showSources, 3);

                const assistantMessage = {
                    role: 'assistant',
                    content: response.answer,
                    sources: response.sources || [],
                    timestamp: new Date().toLocaleTimeString(),
                };

                setMessages((prev) => [...prev, assistantMessage]);
            } else {
                // Conversational chat
                const history = messages
                    .filter((m) => m.role === 'user' || m.role === 'assistant')
                    .map((m) => ({
                        role: m.role,
                        content: m.content,
                    }));

                response = await chatWithDocuments(
                    clientId,
                    userMessage.content,
                    history,
                    true,
                    3
                );

                const assistantMessage = {
                    role: 'assistant',
                    content: response.response,
                    sources: response.sources || [],
                    usedRetrieval: response.used_retrieval,
                    timestamp: new Date().toLocaleTimeString(),
                };

                setMessages((prev) => [...prev, assistantMessage]);
            }
        } catch (err) {
            const errorMessage = {
                role: 'error',
                content: 'Error: ' + err.message,
                timestamp: new Date().toLocaleTimeString(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const clearChat = () => {
        if (window.confirm('Clear chat history?')) {
            setMessages([]);
        }
    };

    if (!clientId) {
        return (
            <div className="chat-interface">
                <div className="no-client-selected">
                    <h3>💬 Chat Interface</h3>
                    <p>Please select a client and upload documents to start chatting</p>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-interface">
            <div className="chat-header">
                <h3>💬 Chat with Documents</h3>
                <div className="chat-controls">
                    <div className="mode-selector">
                        <button
                            className={mode === 'chat' ? 'active' : ''}
                            onClick={() => setMode('chat')}
                        >
                            Chat Mode
                        </button>
                        <button
                            className={mode === 'query' ? 'active' : ''}
                            onClick={() => setMode('query')}
                        >
                            Query Mode
                        </button>
                    </div>
                    <label className="sources-toggle">
                        <input
                            type="checkbox"
                            checked={showSources}
                            onChange={(e) => setShowSources(e.target.checked)}
                        />
                        Show Sources
                    </label>
                    <button onClick={clearChat} className="btn-clear-chat">
                        Clear
                    </button>
                </div>
            </div>

            <div className="chat-messages">
                {messages.length === 0 ? (
                    <div className="empty-chat">
                        <p>👋 Start a conversation!</p>
                        <small>
                            {mode === 'chat'
                                ? 'Chat mode maintains conversation context'
                                : 'Query mode answers each question independently'}
                        </small>
                    </div>
                ) : (
                    <>
                        {messages.map((msg, index) => (
                            <div key={index} className={`message message-${msg.role}`}>
                                <div className="message-header">
                                    <span className="message-role">
                                        {msg.role === 'user' ? '👤 You' : msg.role === 'assistant' ? '🤖 Assistant' : '⚠️ Error'}
                                    </span>
                                    <span className="message-time">{msg.timestamp}</span>
                                </div>
                                {msg.role === 'assistant' && msg.content && (() => {
                                    // Split response into summary and details
                                    const lines = msg.content.split('\n');
                                    const firstLine = lines[0];
                                    const restContent = lines.slice(1).join('\n').trim();

                                    return (
                                        <>
                                            <div className="message-summary">{firstLine}</div>
                                            {restContent && <div className="message-details">{restContent}</div>}
                                        </>
                                    );
                                })()}
                                {msg.role !== 'assistant' && <div className="message-content">{msg.content}</div>}
                                {msg.sources && msg.sources.length > 0 && showSources && (
                                    <div className="message-sources">
                                        <button
                                            className="sources-toggle-btn"
                                            onClick={() => setExpandedSources(prev => ({ ...prev, [index]: !prev[index] }))}
                                        >
                                            📚 {expandedSources[index] ? 'Hide' : 'View'} Sources ({msg.sources.length})
                                        </button>
                                        {expandedSources[index] && (
                                            <div className="sources-content">
                                                {msg.sources.map((source, idx) => (
                                                    <div key={idx} className="source-item">
                                                        <div className="source-text">{source.text}</div>
                                                        {source.metadata && (
                                                            <div className="source-meta">
                                                                {source.metadata.source && (
                                                                    <span>📄 {source.metadata.source}</span>
                                                                )}
                                                                {source.metadata.page && (
                                                                    <span>Page {source.metadata.page}</span>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                                {msg.usedRetrieval !== undefined && (
                                    <small className="retrieval-status">
                                        {msg.usedRetrieval ? '✓ Used document context' : 'ℹ️ No retrieval'}
                                    </small>
                                )}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </>
                )}
                {loading && (
                    <div className="message message-loading">
                        <div className="loading-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                )}
            </div>

            <div className="chat-input-container">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your question here..."
                    disabled={loading}
                    rows="2"
                />
                <button onClick={handleSend} disabled={!input.trim() || loading}>
                    {loading ? '⏳' : '📤'} Send
                </button>
            </div>
        </div>
    );
};

export default ChatInterface;
