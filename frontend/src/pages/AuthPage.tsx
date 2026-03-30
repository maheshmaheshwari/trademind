import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, EyeOff, Eye, TrendingUp, BarChart3, GraduationCap, Loader2, User } from 'lucide-react';
import { registerUser, loginUser } from '../api';
import { useAuth } from '../AuthContext';

export default function AuthPage() {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [isSignup, setIsSignup] = useState(true);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!username.trim()) { setError('Please enter a username'); return; }
        if (!password.trim()) { setError('Please enter a password'); return; }
        if (password.length < 4) { setError('Password must be at least 4 characters'); return; }

        setLoading(true);
        setError('');
        try {
            if (isSignup) {
                const data = await registerUser(username.trim(), password, displayName.trim() || username.trim());
                login(data.user);
            } else {
                const data = await loginUser(username.trim(), password);
                login(data.user);
            }
            navigate('/dashboard');
        } catch (err: any) {
            setError(err.message || 'Something went wrong');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col overflow-hidden relative"
            style={{ background: 'var(--auth-bg, linear-gradient(135deg, #0c1825 0%, #101922 50%, #0a1628 100%))' }}>
            {/* Ambient glow */}
            <div className="fixed inset-0 -z-10" style={{ background: 'radial-gradient(ellipse at top right, rgba(13,127,242,0.08), transparent 60%)' }} />

            <header className="flex items-center justify-between px-8 py-6 w-full max-w-7xl mx-auto">
                <div className="flex items-center gap-3">
                    <div className="size-8 flex items-center justify-center rounded-lg bg-primary/20 text-primary"><TrendingUp className="w-5 h-5" /></div>
                    <h2 className="text-xl font-bold tracking-tight" style={{ color: 'var(--auth-heading, #fff)' }}>TradeMind AI</h2>
                </div>
            </header>

            <main className="flex-1 flex items-center justify-center px-4 py-8">
                <div className="w-full max-w-[1200px] grid lg:grid-cols-2 gap-12 items-center">
                    {/* Left: Hero */}
                    <div className="hidden lg:flex flex-col gap-6 pr-12">
                        <div className="inline-flex items-center gap-2 self-start px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold uppercase tracking-wider">
                            <span className="relative flex h-2 w-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span><span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span></span>
                            Live NSE/BSE Data
                        </div>
                        <h1 className="text-5xl font-bold leading-[1.1]" style={{ color: 'var(--auth-heading, #fff)' }}>
                            Master the Markets with{' '}
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-cyan-400">AI Precision</span>
                        </h1>
                        <p className="text-lg leading-relaxed max-w-lg" style={{ color: 'var(--auth-muted, #94a3b8)' }}>
                            Experience the future of trading. Join TradeMind AI and get a{' '}
                            <span className="font-semibold" style={{ color: 'var(--auth-heading, #fff)' }}>₹10,00,000 virtual trading account</span> to practice strategies risk-free.
                        </p>
                        <div className="grid grid-cols-2 gap-6 mt-8">
                            <div className="flex flex-col gap-2 p-4 rounded-xl border" style={{ background: 'var(--auth-card-sub, rgba(30,41,59,0.3))', borderColor: 'var(--auth-border, rgba(51,65,85,0.5))' }}>
                                <BarChart3 className="text-primary w-7 h-7" />
                                <h3 className="font-bold" style={{ color: 'var(--auth-heading, #fff)' }}>Smart Analysis</h3>
                                <p className="text-sm" style={{ color: 'var(--auth-muted, #94a3b8)' }}>Real-time sentiment analysis on 500+ stocks.</p>
                            </div>
                            <div className="flex flex-col gap-2 p-4 rounded-xl border" style={{ background: 'var(--auth-card-sub, rgba(30,41,59,0.3))', borderColor: 'var(--auth-border, rgba(51,65,85,0.5))' }}>
                                <GraduationCap className="text-primary w-7 h-7" />
                                <h3 className="font-bold" style={{ color: 'var(--auth-heading, #fff)' }}>Risk-Free Learning</h3>
                                <p className="text-sm" style={{ color: 'var(--auth-muted, #94a3b8)' }}>Paper trade with zero financial risk.</p>
                            </div>
                        </div>
                    </div>

                    {/* Right: Auth card */}
                    <div className="w-full max-w-md mx-auto">
                        <div className="rounded-2xl p-8 shadow-2xl relative overflow-hidden group border"
                            style={{
                                background: 'var(--auth-card, rgba(15,23,42,0.7))',
                                backdropFilter: 'blur(20px)',
                                borderColor: 'var(--auth-border, rgba(51,65,85,0.6))',
                            }}>
                            <div className="absolute -top-24 -right-24 w-48 h-48 bg-primary/20 rounded-full blur-3xl group-hover:bg-primary/30 transition-all duration-700" />
                            <div className="relative z-10">
                                <div className="text-center mb-8">
                                    <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--auth-heading, #fff)' }}>{isSignup ? 'Create Account' : 'Welcome Back'}</h2>
                                    <p className="text-sm" style={{ color: 'var(--auth-muted, #94a3b8)' }}>{isSignup ? 'Start with ₹10L Virtual Capital' : 'Login to your trading account'}</p>
                                </div>
                                {error && <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm text-center">{error}</div>}
                                <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-semibold ml-1" style={{ color: 'var(--auth-label, #cbd5e1)' }} htmlFor="username">Username</label>
                                        <div className="relative">
                                            <input
                                                className="w-full border text-sm rounded-xl focus:ring-2 focus:ring-primary/50 focus:border-primary p-3.5 pl-10 transition-all outline-none"
                                                style={{
                                                    background: 'var(--auth-input, rgba(2,6,23,0.5))',
                                                    borderColor: 'var(--auth-border, rgba(51,65,85,0.6))',
                                                    color: 'var(--auth-heading, #fff)',
                                                }}
                                                id="username" placeholder="Enter your username" type="text" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
                                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none" style={{ color: 'var(--auth-muted, #64748b)' }}><User className="w-5 h-5" /></div>
                                        </div>
                                    </div>
                                    {isSignup && (
                                        <div className="space-y-1.5">
                                            <label className="text-xs font-semibold ml-1" style={{ color: 'var(--auth-label, #cbd5e1)' }} htmlFor="displayName">Display Name <span style={{ color: 'var(--auth-muted, #64748b)' }}>(optional)</span></label>
                                            <div className="relative">
                                                <input
                                                    className="w-full border text-sm rounded-xl focus:ring-2 focus:ring-primary/50 focus:border-primary p-3.5 pl-10 transition-all outline-none"
                                                    style={{
                                                        background: 'var(--auth-input, rgba(2,6,23,0.5))',
                                                        borderColor: 'var(--auth-border, rgba(51,65,85,0.6))',
                                                        color: 'var(--auth-heading, #fff)',
                                                    }}
                                                    id="displayName" placeholder="Your display name" type="text" value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
                                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none" style={{ color: 'var(--auth-muted, #64748b)' }}><User className="w-5 h-5" /></div>
                                            </div>
                                        </div>
                                    )}
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-semibold ml-1" style={{ color: 'var(--auth-label, #cbd5e1)' }} htmlFor="password">Password</label>
                                        <div className="relative">
                                            <input
                                                className="w-full border text-sm rounded-xl focus:ring-2 focus:ring-primary/50 focus:border-primary p-3.5 pl-10 transition-all outline-none"
                                                style={{
                                                    background: 'var(--auth-input, rgba(2,6,23,0.5))',
                                                    borderColor: 'var(--auth-border, rgba(51,65,85,0.6))',
                                                    color: 'var(--auth-heading, #fff)',
                                                }}
                                                id="password" placeholder="••••••••" type={showPassword ? 'text' : 'password'} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={isSignup ? 'new-password' : 'current-password'} />
                                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none" style={{ color: 'var(--auth-muted, #64748b)' }}><Lock className="w-5 h-5" /></div>
                                            <button className="absolute inset-y-0 right-0 pr-3 flex items-center hover:opacity-80" type="button" onClick={() => setShowPassword(!showPassword)} style={{ color: 'var(--auth-muted, #64748b)' }}>
                                                {showPassword ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                                            </button>
                                        </div>
                                        {isSignup && <p className="text-xs ml-1" style={{ color: 'var(--auth-muted, #64748b)' }}>Must be at least 4 characters</p>}
                                    </div>
                                    <button type="submit" disabled={loading} className="w-full bg-primary hover:bg-primary/90 text-white font-bold py-3.5 rounded-xl shadow-[0_4px_14px_0_rgba(13,127,242,0.39)] hover:shadow-[0_6px_20px_rgba(13,127,242,0.23)] hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:hover:translate-y-0">
                                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>{isSignup ? 'Create Account' : 'Log In'} <span className="text-sm font-bold">→</span></>}
                                    </button>
                                </form>
                                <p className="mt-6 text-center text-sm" style={{ color: 'var(--auth-muted, #94a3b8)' }}>
                                    {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
                                    <button onClick={() => { setIsSignup(!isSignup); setError(''); }} className="text-primary hover:text-primary/80 font-semibold hover:underline">
                                        {isSignup ? 'Log in' : 'Create one'}
                                    </button>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
            <footer className="py-6 text-center text-xs" style={{ color: 'var(--auth-muted, #475569)' }}>
                <p>© 2024 TradeMind AI. NSE/BSE Market Data Powered.</p>
            </footer>
        </div>
    );
}
