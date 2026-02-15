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
                    confidence: response.confidence || 0,
                    is_uncertain: response.is_uncertain || false,
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
                    confidence: response.confidence || 0,
                    is_uncertain: response.is_uncertain || false,
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
                                {msg.role === 'assistant' && msg.content && (
                                    <>
                                        {/* Uncertainty Warning */}
                                        {msg.is_uncertain && (
                                            <div className="uncertainty-warning">
                                                ⚠️ <strong>Limited Information:</strong> This answer may be incomplete or require human verification.
                                            </div>
                                        )}

                                        {/* Confidence Indicator - only show if confidence is meaningful */}
                                        {msg.confidence > 0 && msg.confidence < 1 && !msg.is_uncertain && (
                                            <div className={`confidence-indicator confidence-${msg.confidence >= 0.7 ? 'high' : msg.confidence >= 0.5 ? 'medium' : 'low'}`}>
                                                <span className="confidence-label">Confidence:</span>
                                                <div className="confidence-bar">
                                                    <div
                                                        className="confidence-fill"
                                                        style={{ width: `${msg.confidence * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span className="confidence-value">{Math.round(msg.confidence * 100)}%</span>
                                            </div>
                                        )}

                                        {/* Full message content without artificial splitting */}
                                        <div className="message-content">{msg.content}</div>
                                    </>
                                )}
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
                                                        <div className="source-header">
                                                            <span className="citation-label">
                                                                [{source.citation_label || `Source ${idx + 1}`}]
                                                            </span>
                                                            <span className={`relevance-badge relevance-${(source.relevance || '').toLowerCase().replace(' ', '-')}`}>
                                                                {source.relevance || 'Relevant'}
                                                            </span>
                                                        </div>
                                                        <div className="source-text">{source.text}</div>
                                                        <div className="source-meta">
                                                            {source.source_file && (
                                                                <span className="source-file">
                                                                    📄 {source.source_file}
                                                                    {source.section && ` → ${source.section}`}
                                                                </span>
                                                            )}
                                                            {source.distance !== undefined && (
                                                                <span className="source-distance" title="Lower is more relevant">
                                                                    Distance: {source.distance.toFixed(3)}
                                                                </span>
                                                            )}
                                                        </div>
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
