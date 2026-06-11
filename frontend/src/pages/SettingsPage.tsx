import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { useToast } from '../components/ui';
import {
  useGetRiskSettingsQuery,
  useUpdateRiskSettingsMutation,
} from '../services/tradeMindApiService';
import { User, Link, Bell, Palette, Shield, CheckCircle, Loader2, ExternalLink } from 'lucide-react';

// ── Shared primitives ────────────────────────────────────────────────────────

function Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button type="button" onClick={onToggle}
      className={`w-11 h-6 rounded-full border-none cursor-pointer p-[3px] flex items-center flex-shrink-0 transition-colors ${on ? 'bg-[var(--accent)]' : 'bg-[var(--surface-3)]'}`}>
      <span className={`block w-[18px] h-[18px] rounded-full bg-white transition-transform duration-[180ms] shadow-[0_1px_3px_rgba(0,0,0,.2)] ${on ? 'translate-x-5' : 'translate-x-0'}`} />
    </button>
  );
}

function PrefRow({ label, sub, children }: { label: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 py-[14px] border-b border-[var(--border)] last:border-b-0">
      <div>
        <div className="text-[13.5px] font-semibold text-[var(--text)]">{label}</div>
        {sub && <div className="text-[12.5px] text-[var(--text-3)] mt-[2px]">{sub}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function Seg<T extends string>({ options, value, onChange, labels }: {
  options: T[]; value: T; onChange: (v: T) => void; labels?: Partial<Record<T, string>>;
}) {
  return (
    <div className="inline-flex bg-[var(--surface-2)] border border-[var(--border)] rounded-[10px] p-[3px] gap-[2px]">
      {options.map(o => (
        <button key={o} type="button" onClick={() => onChange(o)}
          className="border-none font-sans text-[12.5px] font-semibold px-3 py-[5px] rounded-[7px] cursor-pointer transition-colors"
          style={{ background: value === o ? 'var(--accent)' : 'transparent', color: value === o ? '#fff' : 'var(--text-2)' }}>
          {labels?.[o] ?? o}
        </button>
      ))}
    </div>
  );
}

function SectionCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] overflow-hidden">
      {children}
    </div>
  );
}

function SectionHead({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="px-[calc(18px*var(--u))] py-[calc(14px*var(--u))] border-b border-[var(--border)]">
      <h3 className="m-0 text-[14.5px] font-semibold text-[var(--text)]">{title}</h3>
      {sub && <p className="m-0 text-[12.5px] text-[var(--text-3)] mt-[2px]">{sub}</p>}
    </div>
  );
}

function SectionBody({ children }: { children: React.ReactNode }) {
  return <div className="px-[calc(18px*var(--u))] py-[calc(4px*var(--u))]">{children}</div>;
}

// ── Nav tabs ─────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'profile',       label: 'Profile',       Icon: User },
  { id: 'brokers',       label: 'Brokers',        Icon: Link },
  { id: 'notifications', label: 'Notifications',  Icon: Bell },
  { id: 'appearance',    label: 'Appearance',     Icon: Palette },
  { id: 'security',      label: 'Security',       Icon: Shield },
] as const;

type TabId = typeof TABS[number]['id'];

const ACCENT_PRESETS = ['#3B82F6', '#8B5CF6', '#14B8A6', '#F59E0B', '#EC4899', '#10B981'];

const BROKERS = [
  { name: 'Angel One', desc: 'SmartAPI · Connected', logo: '🏦', connected: true },
  { name: 'Zerodha',   desc: 'Kite Connect · Not connected', logo: '📈', connected: false },
  { name: 'Upstox',    desc: 'v2 API · Not connected', logo: '🚀', connected: false },
  { name: 'Groww',     desc: 'Partner API · Not connected', logo: '🌱', connected: false },
];

const inputCls = 'h-11 px-[13px] rounded-[11px] border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)] font-sans text-sm outline-none w-full box-border transition-colors focus:border-[var(--accent)]';

// ── Tab panels ───────────────────────────────────────────────────────────────

function ProfileTab({ user }: { user: any }) {
  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Personal Information" sub="Update your display name and contact details" />
        <SectionBody>
          <div className="flex items-center gap-4 py-5 border-b border-[var(--border)]">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-[var(--accent)] grid place-items-center font-bold text-[22px] text-white flex-shrink-0">
              {user?.display_name?.split(' ')?.map((w: string) => w?.[0] ?? '').join('').slice(0, 2).toUpperCase() ?? 'U'}
            </div>
            <div>
              <p className="text-[14px] font-semibold text-[var(--text)] m-0">{user?.display_name ?? 'User'}</p>
              <p className="text-[12.5px] text-[var(--text-3)] m-0">@{user?.username}</p>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 py-5">
            {[
              ['Full Name',    user?.display_name ?? ''],
              ['Username',     user?.username ?? ''],
              ['Email',        'maheshmaheshwari983@gmail.com'],
              ['Phone',        '+91 ··········'],
            ].map(([label, val]) => (
              <div key={label} className="flex flex-col gap-[7px]">
                <label className="text-[12.5px] font-semibold text-[var(--text-2)]">{label}</label>
                <input defaultValue={val} className={inputCls} readOnly={label === 'Username'} />
              </div>
            ))}
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Trading Preferences" />
        <SectionBody>
          <PrefRow label="Default account" sub="Which account new trades are placed on">
            <Seg options={['PAPER', 'LIVE']} value="PAPER" onChange={() => {}} />
          </PrefRow>
          <PrefRow label="Currency" sub="Display currency for prices and P&L">
            <span className="inline-flex items-center h-9 px-3 rounded-[10px] bg-[var(--surface-2)] border border-[var(--border)] text-[13px] font-semibold text-[var(--text)]">
              ₹ INR
            </span>
          </PrefRow>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function BrokersTab() {
  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Connected Brokers" sub="Link your broker accounts for live order execution" />
        <SectionBody>
          <div className="flex flex-col gap-3 py-4">
            {BROKERS.map(b => (
              <div key={b.name} className="flex items-center justify-between gap-4 p-[13px] rounded-[var(--radius-sm)] bg-[var(--surface-2)] border border-[var(--border)]">
                <div className="flex items-center gap-3">
                  <div className="w-[42px] h-[42px] rounded-[11px] bg-[var(--surface)] border border-[var(--border)] grid place-items-center text-xl flex-shrink-0">
                    {b.logo}
                  </div>
                  <div>
                    <div className="text-[13.5px] font-semibold text-[var(--text)]">{b.name}</div>
                    <div className="text-[12px] text-[var(--text-3)]">{b.desc}</div>
                  </div>
                </div>
                <button
                  className={`inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border cursor-pointer transition-colors ${
                    b.connected
                      ? 'bg-[var(--green-soft)] text-[var(--green)] border-transparent hover:bg-[var(--red-soft)] hover:text-[var(--red)]'
                      : 'bg-[var(--surface)] text-[var(--text-2)] border-[var(--border)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]'
                  }`}
                >
                  {b.connected ? <><CheckCircle size={13} /> Connected</> : <><ExternalLink size={13} /> Connect</>}
                </button>
              </div>
            ))}
          </div>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function NotificationsTab() {
  const [prefs, setPrefs] = useState({
    signal_change: true, price_alert: true, trade_executed: true, news_sentiment: false,
    eod_summary: true, weekly_report: false,
  });
  const [channels, setChannels] = useState({ email: true, push: true, sms: false });

  const toggle = (k: keyof typeof prefs) => setPrefs(p => ({ ...p, [k]: !p[k] }));
  const toggleCh = (k: keyof typeof channels) => setChannels(c => ({ ...c, [k]: !c[k] }));

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Alert Types" sub="Choose which events generate notifications" />
        <SectionBody>
          {[
            ['signal_change',  'Signal Changes',    'Notify when a stock signal changes (BUY → SELL etc.)'],
            ['price_alert',    'Price Alerts',       'Trigger when price crosses your watchlist thresholds'],
            ['trade_executed', 'Trade Executed',     'Confirmation after buy/sell orders are placed'],
            ['news_sentiment', 'News Sentiment',     'Alert on significant sentiment shifts for watchlist stocks'],
            ['eod_summary',    'EOD Summary',        'Daily end-of-day portfolio and signal summary'],
            ['weekly_report',  'Weekly Report',      'Weekly performance and top signals digest'],
          ].map(([key, label, sub]) => (
            <PrefRow key={key} label={label} sub={sub}>
              <Toggle on={prefs[key as keyof typeof prefs]} onToggle={() => toggle(key as keyof typeof prefs)} />
            </PrefRow>
          ))}
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Delivery Channels" sub="How you receive notifications" />
        <SectionBody>
          <div className="flex gap-3 py-4 flex-wrap">
            {(Object.entries(channels) as [keyof typeof channels, boolean][]).map(([key, on]) => (
              <button key={key} onClick={() => toggleCh(key)}
                className={`inline-flex items-center gap-2 h-9 px-4 rounded-[10px] border text-[13px] font-semibold cursor-pointer capitalize transition-all ${
                  on
                    ? 'bg-[var(--accent-soft)] text-[var(--accent-2)] border-[var(--accent)]'
                    : 'bg-[var(--surface-2)] text-[var(--text-2)] border-[var(--border)]'
                }`}>
                {key === 'email' ? '✉️' : key === 'push' ? '🔔' : '📱'} {key}
              </button>
            ))}
          </div>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function AppearanceTab({ theme, toggleTheme, density, setDensity, signalStyle, setSignalStyle }: any) {
  const [accent, setAccent] = useState('#3B82F6');

  useEffect(() => {
    const saved = localStorage.getItem('trademind-accent');
    if (saved) setAccent(saved);
  }, []);

  function applyAccent(color: string) {
    setAccent(color);
    document.documentElement.style.setProperty('--accent', color);
    localStorage.setItem('trademind-accent', color);
  }

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Theme" sub="Choose your interface appearance" />
        <SectionBody>
          <div className="flex gap-3 py-4">
            {(['dark', 'light'] as const).map(t => (
              <button key={t} onClick={() => { if (t !== theme) toggleTheme(); }}
                className={`flex items-center gap-3 flex-1 p-4 rounded-[var(--radius-sm)] border cursor-pointer transition-all ${theme === t ? 'border-[var(--accent)] bg-[var(--accent-soft)]' : 'border-[var(--border)] bg-[var(--surface-2)]'}`}>
                <span className="text-[22px]">{t === 'dark' ? '🌙' : '☀️'}</span>
                <div className="text-left">
                  <div className="text-[13px] font-semibold text-[var(--text)] capitalize">{t}</div>
                  <div className="text-[11.5px] text-[var(--text-3)]">{t === 'dark' ? 'Easy on the eyes' : 'High contrast'}</div>
                </div>
                {theme === t && <CheckCircle size={16} className="ml-auto text-[var(--accent)]" />}
              </button>
            ))}
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Accent Colour" sub="Highlight colour used across buttons and active states" />
        <SectionBody>
          <div className="flex gap-3 py-4 flex-wrap">
            {ACCENT_PRESETS.map(c => (
              <button key={c} type="button" onClick={() => applyAccent(c)}
                className="w-9 h-9 rounded-[11px] cursor-pointer border-2 transition-all grid place-items-center"
                style={{ background: c, borderColor: accent === c ? 'var(--text)' : 'transparent', boxShadow: accent === c ? `0 0 0 2px var(--surface), 0 0 0 4px ${c}` : 'none' }}>
                {accent === c && <CheckCircle size={14} color="#fff" />}
              </button>
            ))}
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Density" sub="Controls spacing and padding throughout the interface" />
        <SectionBody>
          <div className="py-4">
            <Seg
              options={['compact', 'balanced', 'comfy'] as const}
              value={density}
              onChange={setDensity}
              labels={{ compact: 'Compact', balanced: 'Balanced', comfy: 'Comfy' }}
            />
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Signal Card Style" sub="How AI signal cards appear on the Dashboard" />
        <SectionBody>
          <div className="py-4">
            <Seg
              options={['rich', 'compact', 'bold'] as const}
              value={signalStyle}
              onChange={setSignalStyle}
              labels={{ rich: 'Rich', compact: 'Compact', bold: 'Bold' }}
            />
            <p className="text-[11.5px] text-ink-3 mt-2 m-0">
              {signalStyle === 'rich' && 'Full card with sparkline, horizon, expected return, and confidence bar.'}
              {signalStyle === 'compact' && 'Single-row list view — fits more signals on screen.'}
              {signalStyle === 'bold' && 'Left-border accent card with large expected return percentage.'}
            </p>
          </div>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function SecurityTab() {
  const [show, setShow] = useState(false);
  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Change Password" />
        <SectionBody>
          <div className="flex flex-col gap-4 py-4">
            {['Current password', 'New password', 'Confirm new password'].map(label => (
              <div key={label} className="flex flex-col gap-[7px]">
                <label className="text-[12.5px] font-semibold text-[var(--text-2)]">{label}</label>
                <input type={show ? 'text' : 'password'} placeholder="••••••••" className={inputCls} />
              </div>
            ))}
            <label className="flex items-center gap-2 text-[12.5px] text-[var(--text-2)] cursor-pointer">
              <input type="checkbox" checked={show} onChange={e => setShow(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
              Show passwords
            </label>
            <button className="self-start h-10 px-5 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border-none bg-[var(--accent)] text-white shadow-[0_4px_14px_rgba(59,130,246,.32)]">
              Update Password
            </button>
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Two-Factor Authentication" sub="Add an extra layer of security to your account" />
        <SectionBody>
          <PrefRow label="TOTP Authenticator" sub="Use an app like Google Authenticator">
            <span className="inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12px] font-semibold bg-[var(--green-soft)] text-[var(--green)]">
              <CheckCircle size={12} /> Enabled
            </span>
          </PrefRow>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Active Sessions" sub="Devices currently signed in to your account" />
        <SectionBody>
          <div className="py-3 flex items-center justify-between gap-4">
            <div>
              <div className="text-[13.5px] font-semibold text-[var(--text)]">macOS · Chrome</div>
              <div className="text-[12px] text-[var(--text-3)] mt-[2px]">Current session · Mumbai, IN</div>
            </div>
            <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-semibold bg-[var(--green-soft)] text-[var(--green)]">
              ● Active now
            </span>
          </div>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function RiskPanel({ user, toast }: { user: any; toast: any }) {
  const [maxDailyLoss, setMaxDailyLoss] = useState('10000');
  const [stopLossPct,  setStopLossPct]  = useState('7');
  const [targetPct,    setTargetPct]    = useState('15');
  const [maxPosSz,     setMaxPosSz]     = useState('50000');
  const [autoSL,       setAutoSL]       = useState(true);
  const [autoTarget,   setAutoTarget]   = useState(true);
  const [mode,         setMode]         = useState<'PAPER' | 'LIVE'>('PAPER');

  const { data: settings } = useGetRiskSettingsQuery(user?.id, { skip: !user });
  const [updateRiskSettings, { isLoading: saving }] = useUpdateRiskSettingsMutation();

  useEffect(() => {
    if (!settings) return;
    const s = settings as any;
    setMaxDailyLoss(String(s.max_daily_loss ?? 10000));
    setStopLossPct(String(s.stop_loss_pct   ?? 7));
    setTargetPct(String(s.target_pct        ?? 15));
    setMaxPosSz(String(s.max_position_size  ?? 50000));
    setAutoSL(s.auto_stop_loss !== 0);
    setAutoTarget(s.auto_target !== 0);
    if (s.mode) setMode(s.mode.toUpperCase() as 'PAPER' | 'LIVE');
  }, [settings]);

  async function handleSave() {
    if (!user) return;
    try {
      await updateRiskSettings({ userId: user.id, settings: { max_daily_loss: +maxDailyLoss, stop_loss_pct: +stopLossPct, target_pct: +targetPct, max_position_size: +maxPosSz, auto_stop_loss: autoSL ? 1 : 0, auto_target: autoTarget ? 1 : 0, mode } as any }).unwrap();
      toast({ type: 'success', title: 'Settings saved', msg: 'Risk profile updated' });
    } catch { toast({ type: 'error', title: 'Save failed' }); }
  }

  const num = (val: string, set: (v: string) => void, pre?: string, suf?: string) => (
    <div className="relative">
      {pre && <span className="absolute left-[13px] top-1/2 -translate-y-1/2 text-[var(--text-3)] text-sm pointer-events-none">{pre}</span>}
      <input type="number" value={val} onChange={e => set(e.target.value)} className={`${inputCls} ${pre ? 'pl-[26px]' : 'pl-[13px]'} ${suf ? 'pr-[30px]' : 'pr-[13px]'}`} />
      {suf && <span className="absolute right-[13px] top-1/2 -translate-y-1/2 text-[var(--text-3)] text-sm pointer-events-none">{suf}</span>}
    </div>
  );

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Risk Limits" sub="Automated guardrails to protect your capital" />
        <SectionBody>
          <PrefRow label="Max Daily Loss" sub="Halt trading if daily loss exceeds this">{num(maxDailyLoss, setMaxDailyLoss, '₹')}</PrefRow>
          <PrefRow label="Stop Loss %" sub="Default SL for new positions">{num(stopLossPct, setStopLossPct, undefined, '%')}</PrefRow>
          <PrefRow label="Target %" sub="Default profit target for new positions">{num(targetPct, setTargetPct, undefined, '%')}</PrefRow>
          <PrefRow label="Max Position Size" sub="Capital cap per single trade">{num(maxPosSz, setMaxPosSz, '₹')}</PrefRow>
          <PrefRow label="Auto Stop-Loss" sub="Place SL order automatically"><Toggle on={autoSL} onToggle={() => setAutoSL(x => !x)} /></PrefRow>
          <PrefRow label="Auto Target" sub="Place target exit automatically"><Toggle on={autoTarget} onToggle={() => setAutoTarget(x => !x)} /></PrefRow>
          <PrefRow label="Trading Mode" sub="Paper uses virtual ₹10L — no real money">
            <Seg options={['PAPER', 'LIVE']} value={mode} onChange={setMode} labels={{ PAPER: '📄 Paper', LIVE: '⚡ Live' }} />
          </PrefRow>
        </SectionBody>
      </SectionCard>
      <div className="flex justify-end">
        <button type="button" onClick={handleSave} disabled={saving}
          className={`h-10 px-6 rounded-[11px] font-sans text-[13.5px] font-semibold border-none bg-[var(--accent)] text-white inline-flex items-center gap-2 transition-opacity ${saving ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
          style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}>
          {saving ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle size={16} />}
          {saving ? 'Saving…' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const { user }   = useAuth();
  const { theme, toggleTheme, density, setDensity, signalStyle, setSignalStyle } = useTheme();
  const toast      = useToast();
  const location   = useLocation();

  const initialTab = (location.state as any)?.tab ?? 'profile';
  const [tab, setTab] = useState<TabId>(initialTab);

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div>
        <h1 className="font-bold tracking-tight m-0 text-[var(--text)] text-[calc(25px*var(--u))]">Settings</h1>
        <p className="text-[var(--text-2)] text-[13.5px] mt-1 m-0">Account, preferences &amp; integrations</p>
      </div>

      {/* ── Layout: sidebar + content ── */}
      <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-[calc(20px*var(--u))] items-start">

        {/* Sidebar nav */}
        <nav className="sticky top-0 flex flex-col gap-[3px] bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] p-[10px]">
          {TABS.map(({ id, label, Icon }) => (
            <button key={id} onClick={() => setTab(id)}
              className={`flex items-center gap-3 h-[38px] px-3 rounded-[9px] text-[13.5px] font-medium border-none cursor-pointer transition-colors w-full text-left ${
                tab === id
                  ? 'bg-[var(--accent-soft)] text-[var(--accent-2)]'
                  : 'bg-transparent text-[var(--text-2)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]'
              }`}>
              <Icon size={16} className="flex-shrink-0" />
              {label}
            </button>
          ))}
          {/* Risk settings as sub-item */}
          <div className="h-px bg-[var(--border)] my-1" />
          <button onClick={() => setTab('risk' as any)}
            className={`flex items-center gap-3 h-[38px] px-3 rounded-[9px] text-[13.5px] font-medium border-none cursor-pointer transition-colors w-full text-left ${
              tab === ('risk' as any)
                ? 'bg-[var(--accent-soft)] text-[var(--accent-2)]'
                : 'bg-transparent text-[var(--text-2)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]'
            }`}>
            <Shield size={16} className="flex-shrink-0" />
            Risk Management
          </button>
        </nav>

        {/* Content panel */}
        <div>
          {tab === 'profile'       && <ProfileTab user={user} />}
          {tab === 'brokers'       && <BrokersTab />}
          {tab === 'notifications' && <NotificationsTab />}
          {tab === 'appearance'    && <AppearanceTab theme={theme} toggleTheme={toggleTheme} density={density} setDensity={setDensity} signalStyle={signalStyle} setSignalStyle={setSignalStyle} />}
          {tab === 'security'      && <SecurityTab />}
          {tab === ('risk' as any) && <RiskPanel user={user} toast={toast} />}
        </div>
      </div>

      {/* ── Responsive: stack on small screens ── */}
      <style>{`@media(max-width:760px){.settings-layout{grid-template-columns:1fr!important;}}`}</style>
    </div>
  );
}
