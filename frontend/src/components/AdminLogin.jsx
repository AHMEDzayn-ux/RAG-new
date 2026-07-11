import { useState } from 'react';
import { adminLogin, adminRegister } from '../services/api';
import Icon from './Icon';

const AdminLogin = ({ onLogin }) => {
    const [mode, setMode] = useState('login'); // login | register
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            if (mode === 'register') {
                await adminRegister(email, password, name);
            } else {
                await adminLogin(email, password);
            }
            onLogin();
        } catch (err) {
            setError(err?.response?.data?.detail || (mode === 'register' ? 'Sign up failed' : 'Login failed'));
        } finally {
            setLoading(false);
        }
    };

    const isRegister = mode === 'register';

    return (
        <div className="login-wrap">
            <form className="login-card" onSubmit={handleSubmit}>
                <span className="login-brand" aria-hidden="true"><Icon name="lock" size={22} /></span>
                <h1>Operator Console</h1>
                <p>{isRegister ? 'Create an account to manage your assistants.' : 'Sign in to manage your assistants.'}</p>
                {isRegister && (
                    <input
                        type="text"
                        placeholder="Your name (optional)"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                    />
                )}
                <input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    autoFocus
                />
                <input
                    type="password"
                    placeholder={isRegister ? 'Choose a password (6+ chars)' : 'Password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                {error && <div className="login-error">{error}</div>}
                <button type="submit" disabled={loading || !email || !password}>
                    {loading ? 'Please wait…' : (isRegister ? 'Create account' : 'Sign in')}
                </button>
                <button
                    type="button"
                    className="login-switch"
                    onClick={() => { setMode(isRegister ? 'login' : 'register'); setError(''); }}
                >
                    {isRegister ? 'Have an account? Sign in' : 'New here? Create an account'}
                </button>
                {import.meta.env.DEV && !isRegister && (
                    <button
                        type="button"
                        className="login-switch"
                        onClick={() => { setEmail('admin@local'); setPassword('Ruzaini@123'); setError(''); }}
                    >
                        Autofill demo login
                    </button>
                )}
            </form>
        </div>
    );
};

export default AdminLogin;
