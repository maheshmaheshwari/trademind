import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLoginMutation, useRegisterMutation } from '../services/tradeMindApiService';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';

function MiniSpark({ pts, color, w = 70, h = 26 }: { pts: number[]; color: string; w?: number; h?: number }) {
  const mn = Math.min(...pts), mx = Math.max(...pts), rng = mx - mn || 1;
  const xs = pts.map((_, i) => (i / (pts.length - 1)) * w);
  const ys = pts.map(v => h - ((v - mn) / rng) * (h * 0.85) - h * 0.075);
  const d = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');
  const area = `${d} L${w},${h} L0,${h} Z`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} fill="none">
      <path d={area} fill={color} fillOpacity={0.18} />
      <path d={d} stroke={color} strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function GoogleIcon({ size = 18 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size}>
      <path fill="#4285F4" d="M22.5 12.2c0-.7-.1-1.4-.2-2H12v3.9h5.9a5 5 0 0 1-2.2 3.3v2.7h3.6c2.1-2 3.2-4.8 3.2-7.9z"/>
      <path fill="#34A853" d="M12 23c2.9 0 5.4-1 7.2-2.7l-3.6-2.7c-1 .7-2.3 1-3.6 1-2.8 0-5.1-1.9-6-4.4H2.3v2.8A11 11 0 0 0 12 23z"/>
      <path fill="#FBBC05" d="M6 14.2a6.6 6.6 0 0 1 0-4.2V7.2H2.3a11 11 0 0 0 0 9.8z"/>
      <path fill="#EA4335" d="M12 5.4c1.6 0 3 .5 4.1 1.6l3.1-3.1A11 11 0 0 0 2.3 7.2L6 10c.9-2.5 3.2-4.4 6-4.4z"/>
    </svg>
  );
}

function SignalBadge({ signal }: { signal: 'BUY' | 'SELL' | 'HOLD' }) {
  const col = signal === 'BUY' ? 'var(--green)' : signal === 'SELL' ? 'var(--red)' : 'var(--gold)';
  const bg  = signal === 'BUY' ? 'var(--green-soft)' : signal === 'SELL' ? 'var(--red-soft)' : 'var(--gold-soft)';
  const arrow = signal === 'BUY' ? '↑' : signal === 'SELL' ? '↓' : '—';
  return (
    <span style={{ color: col, background: bg }} className="inline-flex items-center gap-1 h-[23px] px-[9px] rounded-[7px] text-[11.5px] font-bold tracking-[.02em]">
      {arrow} {signal}
    </span>
  );
}

const PREVIEW = [
  { symbol: 'RELIANCE', sector: 'Energy',   signal: 'BUY'  as const, conf: 91, pts: [42,45,43,47,51,49,54,58,56,60,62,65,63,67,71,69,74,78,76,81,79,83,86,84,88,91,89,93] },
  { symbol: 'ICICIBANK',sector: 'Banking',  signal: 'BUY'  as const, conf: 87, pts: [55,53,57,54,58,56,60,58,63,61,65,63,67,65,69,67,71,69,74,72,76,74,78,76,80,78,82,85] },
  { symbol: 'TATAMOTORS',sector: 'Auto',    signal: 'SELL' as const, conf: 74, pts: [82,80,83,81,85,78,76,79,77,73,75,72,70,68,71,67,65,68,63,61,64,59,57,60,55,53,56,51] },
];

function IconLock() {
  return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>;
}
function IconUser() {
  return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>;
}
function IconSun() {
  return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>;
}
function IconMoon() {
  return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>;
}
function IconSpinner() {
  return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ animation: 'spin 1s linear infinite' }}><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 4v4h-4"/></svg>;
}
function IconArrow() {
  return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3"/><path d="M16 17l5-5-5-5M21 12H9"/></svg>;
}
function IconPlus() {
  return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14"/></svg>;
}
function IconSparkle() {
  return <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/></svg>;
}

export default function AuthPage() {
  const navigate              = useNavigate();
  const { login }             = useAuth();
  const { theme, toggleTheme } = useTheme();

  const [mode,     setMode]     = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [pw,       setPw]       = useState('');
  const [name,     setName]     = useState('');
  const [err,      setErr]      = useState('');
  const [remember, setRemember] = useState(true);

  const [loginMutation,    { isLoading: loggingIn }]    = useLoginMutation();
  const [registerMutation, { isLoading: registering }]  = useRegisterMutation();
  const busy = loggingIn || registering;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) { setErr('Please enter your username'); return; }
    if (pw.length < 4)    { setErr('Password must be at least 4 characters'); return; }
    if (mode === 'register' && !name.trim()) { setErr('Please enter your full name'); return; }
    setErr('');
    try {
      const data = mode === 'register'
        ? await registerMutation({ username: username.trim(), password: pw, display_name: name.trim() }).unwrap()
        : await loginMutation({ username: username.trim(), password: pw }).unwrap();
      localStorage.setItem('trademind_token', (data as any).token);
      login((data as any).user);
      navigate('/dashboard');
    } catch (ex: unknown) {
      setErr(ex instanceof Error ? ex.message : 'Something went wrong');
    }
  };

  const inputCls = 'h-11 pl-10 pr-3 rounded-[11px] border border-line bg-surface-2 text-ink font-sans text-sm outline-none w-full box-border transition-colors focus:border-accent';

  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] bg-bg">

      {/* ── Left: Showcase ── */}
      <div
        className="hide-sm relative overflow-hidden border-r border-line p-12 flex flex-col justify-between"
        style={{ background: 'linear-gradient(160deg,var(--surface) 0%,var(--bg) 100%)' }}
      >
        {/* Ambient glow */}
        <div className="absolute pointer-events-none" style={{ top: '-12%', right: '-8%', width: 420, height: 420, borderRadius: '50%', background: 'radial-gradient(circle,rgba(59,130,246,.22),transparent 70%)' }} />

        {/* Brand */}
        <div className="relative flex items-center gap-[11px]">
          <span className="w-[34px] h-[34px] shrink-0 rounded-[10px] grid place-items-center" style={{ background: 'linear-gradient(135deg,var(--accent),#1E40AF)', boxShadow: '0 4px 14px rgba(59,130,246,.4)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></svg>
          </span>
          <span className="font-bold text-[19px] tracking-tight text-ink whitespace-nowrap">
            Trade<b className="text-accent-2">Mind</b>{' '}
            <span className="text-ink-3 font-semibold text-[13px]">AI</span>
          </span>
        </div>

        {/* Hero copy */}
        <div className="relative max-w-[440px]">
          <span className="inline-flex items-center gap-[5px] h-[26px] px-[10px] rounded-full text-[11.5px] font-semibold text-gold bg-gold-soft mb-[18px]">
            <IconSparkle /> AI-powered · Nifty 500
          </span>

          <h1 className="text-[38px] font-bold tracking-tight leading-[1.12] m-0 mb-[14px] text-ink">
            Trade smarter with<br />signals you can trust.
          </h1>
          <p className="text-[15px] text-ink-2 leading-[1.55] m-0">
            Machine-learning signals across 498 stocks, real-time sentiment, and broker-synced execution — all in one terminal.
          </p>

          {/* Signal preview cards */}
          <div className="flex flex-col gap-[9px] mt-[26px]">
            {PREVIEW.map(s => {
              const color = s.signal === 'BUY' ? 'var(--green)' : 'var(--red)';
              const sparkColor = s.signal === 'BUY' ? '#10B981' : '#EF4444';
              return (
                <div key={s.symbol} className="flex items-center justify-between px-[14px] py-[11px] border border-line rounded-[12px] backdrop-blur-sm" style={{ background: 'color-mix(in srgb,var(--surface) 70%,transparent)' }}>
                  <div className="flex items-center gap-[11px]">
                    <span className="w-8 h-8 rounded-[9px] grid place-items-center font-bold text-[12px] shrink-0" style={{ background: color + '22', color }}>
                      {s.symbol.slice(0, 2)}
                    </span>
                    <div>
                      <div className="font-semibold text-[13.5px] text-ink">{s.symbol}</div>
                      <div className="text-[11.5px] text-ink-3">{s.sector}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-[7px]">
                    <MiniSpark pts={s.pts} color={sparkColor} w={70} h={26} />
                    <SignalBadge signal={s.signal} />
                    <span className="font-mono font-bold text-[13px] text-gain">{s.conf}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Bottom stats */}
        <div className="relative flex gap-[18px] text-ink-3 text-[12.5px]">
          <span><b className="text-ink-2 tabular-nums">498</b> stocks tracked</span>
          <span><b className="text-ink-2">68%</b> signal win rate</span>
          <span><b className="text-ink-2">15min</b> refresh</span>
        </div>
      </div>

      {/* ── Right: Form ── */}
      <div className="flex flex-col px-8 py-7 relative bg-bg">

        {/* Theme toggle — top right */}
        <div className="flex justify-end">
          <button
            className="w-[38px] h-[38px] rounded-[10px] border border-line bg-transparent text-ink-2 grid place-items-center cursor-pointer shrink-0 transition-colors hover:bg-surface-hover hover:text-ink"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
          </button>
        </div>

        {/* Centered form content */}
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full max-w-[380px]">

            {/* Logo + title */}
            <div className="flex flex-col items-center gap-[14px] mb-[26px]">
              <span className="w-[52px] h-[52px] rounded-[15px] grid place-items-center" style={{ background: 'linear-gradient(135deg,var(--accent),#1E40AF)', boxShadow: '0 4px 14px rgba(59,130,246,.4)' }}>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></svg>
              </span>
              <div className="flex flex-col items-center gap-1">
                <h2 className="m-0 text-2xl font-bold tracking-tight text-ink">
                  {mode === 'login' ? 'Welcome back' : 'Create your account'}
                </h2>
                <p className="m-0 text-[13.5px] text-ink-2 text-center">
                  {mode === 'login' ? 'Sign in to your TradeMind terminal' : 'Start trading smarter in minutes'}
                </p>
              </div>
            </div>

            {/* Google button */}
            <button
              type="button"
              onClick={handleSubmit as unknown as React.MouseEventHandler}
              className="w-full h-[46px] mb-[18px] inline-flex items-center justify-center gap-2 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover hover:border-line-strong"
            >
              <GoogleIcon size={18} />Continue with Google
            </button>

            {/* OR divider */}
            <div className="flex items-center gap-[10px] mb-[18px]">
              <div className="flex-1 h-px bg-[var(--border)] my-1" />
              <span className="text-[11.5px] text-ink-3 font-medium">OR</span>
              <div className="flex-1 h-px bg-[var(--border)] my-1" />
            </div>

            {/* Error */}
            {err && (
              <div className="flex items-center gap-[7px] text-loss text-[12.5px] mb-3">
                <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/></svg>
                {err}
              </div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit}>
              {/* Full name (register only) */}
              {mode === 'register' && (
                <div className="flex flex-col gap-[7px] mb-[15px]">
                  <label className="text-[12.5px] font-semibold text-ink-2">Full name</label>
                  <div className="relative">
                    <span className="absolute left-[13px] top-[14px] text-ink-3 pointer-events-none"><IconUser /></span>
                    <input
                      className={inputCls}
                      value={name} onChange={e => setName(e.target.value)}
                      placeholder="Arjun Kapoor"
                      autoComplete="name"
                    />
                  </div>
                </div>
              )}

              {/* Username */}
              <div className="flex flex-col gap-[7px] mb-[15px]">
                <label className="text-[12.5px] font-semibold text-ink-2">Username</label>
                <div className="relative">
                  <span className="absolute left-[13px] top-[14px] text-ink-3 pointer-events-none"><IconUser /></span>
                  <input
                    className={inputCls}
                    type="text"
                    value={username} onChange={e => setUsername(e.target.value)}
                    placeholder="arjun_kapoor"
                    autoComplete="username"
                  />
                </div>
              </div>

              {/* Password */}
              <div className={`flex flex-col gap-[7px] ${mode === 'login' ? '' : 'mb-[15px]'}`}>
                <label className="text-[12.5px] font-semibold text-ink-2">Password</label>
                <div className="relative">
                  <span className="absolute left-[13px] top-[14px] text-ink-3 pointer-events-none"><IconLock /></span>
                  <input
                    className={inputCls}
                    type="password"
                    value={pw} onChange={e => setPw(e.target.value)}
                    placeholder="••••••••"
                    autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                  />
                </div>
              </div>

              {/* Remember me + Forgot password (login only) */}
              {mode === 'login' && (
                <div className="flex items-center justify-between my-[14px] mb-4">
                  <label className="flex items-center gap-[7px] text-[12.5px] text-ink-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={remember}
                      onChange={e => setRemember(e.target.checked)}
                      style={{ accentColor: 'var(--accent)', cursor: 'pointer' }}
                    />
                    Remember me
                  </label>
                  <a className="text-[12.5px] text-accent-2 cursor-pointer no-underline font-medium">
                    Forgot password?
                  </a>
                </div>
              )}

              {/* Submit */}
              <button
                type="submit"
                disabled={busy}
                className={`w-full h-[46px] inline-flex items-center justify-center gap-2 rounded-[11px] font-sans text-[13.5px] font-semibold border border-transparent bg-accent text-white transition-colors ${busy ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer hover:bg-accent-2'} ${mode === 'register' ? 'mt-1' : ''}`}
                style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}
              >
                {busy ? <IconSpinner /> : mode === 'login' ? <IconArrow /> : <IconPlus />}
                {busy ? 'Signing in…' : mode === 'login' ? 'Sign in' : 'Create account'}
              </button>
            </form>

            {/* Toggle mode */}
            <p className="text-center text-[13px] text-ink-2 mt-[22px]">
              {mode === 'login' ? 'New to TradeMind? ' : 'Already have an account? '}
              <a
                className="text-accent-2 font-semibold cursor-pointer no-underline"
                onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setErr(''); }}
              >
                {mode === 'login' ? 'Create an account' : 'Sign in'}
              </a>
            </p>

            <p className="text-center text-[11px] text-ink-3 mt-[26px]">
              Paper trading enabled by default · SEBI-registered broker integration
            </p>
          </div>
        </div>
      </div>

      {/* ── Responsive: single column below 860px ── */}
      <style>{`
        @media (max-width: 860px) {
          .hide-sm { display: none !important; }
          div[style*="grid-template-columns"] { grid-template-columns: 1fr !important; }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
