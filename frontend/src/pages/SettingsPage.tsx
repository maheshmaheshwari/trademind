import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { useToast } from '../components/ui';
import {
  useGetRiskSettingsQuery,
  useUpdateRiskSettingsMutation,
  useGetMeQuery,
  useUpdateMeMutation,
  useChangePasswordMutation,
  useGetPreferencesQuery,
  useUpdatePreferencesMutation,
  useGetNotifPreferencesQuery,
  useUpdateNotifPreferencesMutation,
  useGetBrokersQuery,
  useConnectBrokerAngelOneMutation,
  useDisconnectBrokerMutation,
  useTotpSetupMutation,
  useTotpConfirmMutation,
  useTotpDisableMutation,
  useGetSessionsQuery,
  useRevokeSessionMutation,
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

const inputCls = 'h-11 px-[13px] rounded-[11px] border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)] font-sans text-sm outline-none w-full box-border transition-colors focus:border-[var(--accent)]';

// ── Tab panels ───────────────────────────────────────────────────────────────

function ProfileTab({ user }: { user: any }) {
  const toast = useToast();
  const { data: meData } = useGetMeQuery();
  const { data: prefData } = useGetPreferencesQuery();
  const [updateMe, { isLoading: savingProfile }] = useUpdateMeMutation();
  const [updatePreferences] = useUpdatePreferencesMutation();

  const [displayName, setDisplayName] = useState('');
  const [email,       setEmail]       = useState('');
  const [phone,       setPhone]       = useState('');
  const [defaultAcct, setDefaultAcct] = useState<'PAPER' | 'LIVE'>('PAPER');

  useEffect(() => {
    const src = meData ?? user;
    if (src) {
      setDisplayName(src?.display_name ?? '');
      setEmail(src?.email ?? '');
      setPhone(src?.phone ?? '');
    }
  }, [meData, user]);

  useEffect(() => {
    if (prefData?.default_account) {
      setDefaultAcct((prefData.default_account as 'PAPER' | 'LIVE') ?? 'PAPER');
    }
  }, [prefData]);

  async function handleSaveProfile() {
    try {
      await updateMe({ display_name: displayName, email, phone }).unwrap();
      toast({ type: 'success', title: 'Profile saved', msg: 'Your profile has been updated' });
    } catch {
      toast({ type: 'error', title: 'Save failed', msg: 'Could not update profile' });
    }
  }

  async function handleAcctChange(val: 'PAPER' | 'LIVE') {
    setDefaultAcct(val);
    try {
      await updatePreferences({ default_account: val }).unwrap();
      toast({ type: 'success', title: 'Preference saved' });
    } catch {
      toast({ type: 'error', title: 'Save failed' });
    }
  }

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      <SectionCard>
        <SectionHead title="Personal Information" sub="Update your display name and contact details" />
        <SectionBody>
          <div className="flex items-center gap-4 py-5 border-b border-[var(--border)]">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-[var(--accent)] grid place-items-center font-bold text-[22px] text-white flex-shrink-0">
              {(meData ?? user)?.display_name?.split(' ')?.map((w: string) => w?.[0] ?? '').join('').slice(0, 2).toUpperCase() ?? 'U'}
            </div>
            <div>
              <p className="text-[14px] font-semibold text-[var(--text)] m-0">{(meData ?? user)?.display_name ?? 'User'}</p>
              <p className="text-[12.5px] text-[var(--text-3)] m-0">@{(meData ?? user)?.username}</p>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 py-5">
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Full Name</label>
              <input value={displayName} onChange={e => setDisplayName(e.target.value)} className={inputCls} />
            </div>
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Username</label>
              <input defaultValue={(meData ?? user)?.username ?? ''} className={inputCls} readOnly />
            </div>
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Email</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} className={inputCls} placeholder="you@example.com" />
            </div>
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Phone</label>
              <input type="tel" value={phone} onChange={e => setPhone(e.target.value)} className={inputCls} placeholder="+91 9876543210" />
            </div>
          </div>
          <div className="flex justify-end pb-4">
            <button
              type="button"
              onClick={handleSaveProfile}
              disabled={savingProfile}
              className={`h-10 px-6 rounded-[11px] font-sans text-[13.5px] font-semibold border-none bg-[var(--accent)] text-white inline-flex items-center gap-2 transition-opacity ${savingProfile ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
              style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}
            >
              {savingProfile ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle size={15} />}
              {savingProfile ? 'Saving…' : 'Save Changes'}
            </button>
          </div>
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Trading Preferences" />
        <SectionBody>
          <PrefRow label="Default account" sub="Which account new trades are placed on">
            <Seg options={['PAPER', 'LIVE'] as const} value={defaultAcct} onChange={handleAcctChange} />
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

// ── Angel One connect modal ──────────────────────────────────────────────────

function AngelOneModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const toast = useToast();
  const [clientId, setClientId] = useState('');
  const [password, setPassword] = useState('');
  const [totp,     setTotp]     = useState('');
  const [connectBroker, { isLoading }] = useConnectBrokerAngelOneMutation();

  async function handleConnect(e: React.FormEvent) {
    e.preventDefault();
    if (!clientId.trim() || !password || !totp) {
      toast({ type: 'error', title: 'All fields required' }); return;
    }
    try {
      await connectBroker({ client_id: clientId.trim(), password, totp }).unwrap();
      toast({ type: 'success', title: 'Angel One connected!' });
      onSuccess();
      onClose();
    } catch {
      toast({ type: 'error', title: 'Connection failed', msg: 'Check your credentials and try again' });
    }
  }

  return (
    <div className="fixed inset-0 z-[200] grid place-items-center p-5">
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-[201] w-full max-w-[380px] bg-[var(--surface)] border border-[var(--border-strong)] rounded-[16px] shadow-[var(--shadow-lg)] p-6 flex flex-col gap-5">
        <div className="flex items-center justify-between">
          <h3 className="m-0 text-[16px] font-bold text-[var(--text)]">Connect Angel One</h3>
          <button onClick={onClose} className="w-8 h-8 rounded-[8px] border border-[var(--border)] bg-transparent text-[var(--text-2)] grid place-items-center cursor-pointer">
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>
        </div>
        <form onSubmit={handleConnect} className="flex flex-col gap-4">
          <div className="flex flex-col gap-[6px]">
            <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Client ID</label>
            <input value={clientId} onChange={e => setClientId(e.target.value)} placeholder="A1234567" className={inputCls} autoFocus />
          </div>
          <div className="flex flex-col gap-[6px]">
            <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" className={inputCls} />
          </div>
          <div className="flex flex-col gap-[6px]">
            <label className="text-[12.5px] font-semibold text-[var(--text-2)]">TOTP (from your authenticator app)</label>
            <input type="text" inputMode="numeric" maxLength={6} value={totp} onChange={e => setTotp(e.target.value.replace(/\D/g, '').slice(0, 6))} placeholder="123456" className={`${inputCls} font-mono tracking-[0.3em] text-center`} />
          </div>
          <div className="flex gap-3 pt-1">
            <button type="button" onClick={onClose} className="flex-1 h-10 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)]">Cancel</button>
            <button type="submit" disabled={isLoading} className={`flex-[2] h-10 rounded-[11px] font-sans text-[13.5px] font-semibold border-none bg-[var(--accent)] text-white inline-flex items-center justify-center gap-2 transition-opacity ${isLoading ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`} style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}>
              {isLoading ? <Loader2 size={15} className="animate-spin" /> : null}
              {isLoading ? 'Connecting…' : 'Connect'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function BrokersTab() {
  const toast = useToast();
  const { data: brokersData, refetch: refetchBrokers } = useGetBrokersQuery();
  const [disconnectBroker] = useDisconnectBrokerMutation();
  const [angelModal, setAngelModal] = useState(false);

  const brokers = (brokersData as any)?.data ?? [];

  // Fallback static list merged with API data
  const STATIC_BROKERS = [
    { name: 'Angel One', broker: 'angel', desc: 'SmartAPI', logo: '🏦' },
    { name: 'Zerodha',   broker: 'zerodha', desc: 'Kite Connect', logo: '📈' },
    { name: 'Upstox',    broker: 'upstox',  desc: 'v2 API', logo: '🚀' },
    { name: 'Groww',     broker: 'groww',   desc: 'Partner API', logo: '🌱' },
  ];

  const merged = STATIC_BROKERS.map(sb => {
    const apiEntry = (brokers ?? []).find((b: any) => b?.broker === sb.broker);
    return { ...sb, connected: apiEntry?.connected ?? false };
  });

  async function handleDisconnect(broker: string) {
    try {
      await disconnectBroker(broker).unwrap();
      toast({ type: 'success', title: 'Broker disconnected' });
      refetchBrokers();
    } catch {
      toast({ type: 'error', title: 'Disconnect failed' });
    }
  }

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      {angelModal && (
        <AngelOneModal onClose={() => setAngelModal(false)} onSuccess={refetchBrokers} />
      )}
      <SectionCard>
        <SectionHead title="Connected Brokers" sub="Link your broker accounts for live order execution" />
        <SectionBody>
          <div className="flex flex-col gap-3 py-4">
            {(merged ?? []).map(b => {
              const isZerodhaOrUpstox = b?.broker === 'zerodha' || b?.broker === 'upstox' || b?.broker === 'groww';
              return (
                <div key={b?.name} className="flex items-center justify-between gap-4 p-[13px] rounded-[var(--radius-sm)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <div className="flex items-center gap-3">
                    <div className="w-[42px] h-[42px] rounded-[11px] bg-[var(--surface)] border border-[var(--border)] grid place-items-center text-xl flex-shrink-0">
                      {b?.logo}
                    </div>
                    <div>
                      <div className="text-[13.5px] font-semibold text-[var(--text)]">{b?.name}</div>
                      <div className="text-[12px] text-[var(--text-3)]">
                        {b?.desc} · {b?.connected ? 'Connected' : 'Not connected'}
                      </div>
                    </div>
                  </div>
                  {isZerodhaOrUpstox ? (
                    <span className="inline-flex items-center h-8 px-3 rounded-[9px] text-[12px] font-semibold bg-[var(--surface-3)] text-[var(--text-3)] border border-[var(--border)]">
                      Coming Soon
                    </span>
                  ) : b?.connected ? (
                    <button
                      onClick={() => handleDisconnect(b?.broker ?? '')}
                      className="inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border cursor-pointer transition-colors bg-[var(--green-soft)] text-[var(--green)] border-transparent hover:bg-[var(--red-soft)] hover:text-[var(--red)]"
                    >
                      <CheckCircle size={13} /> Connected
                    </button>
                  ) : (
                    <button
                      onClick={() => { if (b?.broker === 'angel') setAngelModal(true); }}
                      className="inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border cursor-pointer transition-colors bg-[var(--surface)] text-[var(--text-2)] border-[var(--border)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]"
                    >
                      <ExternalLink size={13} /> Connect
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </SectionBody>
      </SectionCard>
    </div>
  );
}

function NotificationsTab() {
  const toast = useToast();
  const { data: apiPrefs } = useGetNotifPreferencesQuery();
  const [updateNotifPrefs] = useUpdateNotifPreferencesMutation();

  const DEFAULT_PREFS = {
    signal_change: true, price_alert: true, trade_executed: true,
    news_sentiment: false, eod_summary: true, weekly_report: false,
  };
  const DEFAULT_CHANNELS = { email: true, push: true, sms: false };

  const [prefs,    setPrefs]    = useState<Record<string, boolean>>(DEFAULT_PREFS);
  const [channels, setChannels] = useState<Record<string, boolean>>(DEFAULT_CHANNELS);

  useEffect(() => {
    if (!apiPrefs) return;
    const p: Record<string, boolean> = {};
    const c: Record<string, boolean> = {};
    Object.keys(DEFAULT_PREFS).forEach(k => { p[k] = (apiPrefs as any)?.[k] ?? DEFAULT_PREFS[k as keyof typeof DEFAULT_PREFS]; });
    Object.keys(DEFAULT_CHANNELS).forEach(k => { c[k] = (apiPrefs as any)?.[`channel_${k}`] ?? DEFAULT_CHANNELS[k as keyof typeof DEFAULT_CHANNELS]; });
    setPrefs(p);
    setChannels(c);
  }, [apiPrefs]);

  async function savePrefs(updated: Record<string, boolean>) {
    try {
      await updateNotifPrefs(updated).unwrap();
      toast({ type: 'success', title: 'Notification preferences saved' });
    } catch {
      toast({ type: 'error', title: 'Save failed' });
    }
  }

  function toggle(k: string) {
    const updated = { ...prefs, [k]: !prefs[k] };
    setPrefs(updated);
    const full = { ...updated, ...Object.fromEntries(Object.entries(channels).map(([ck, cv]) => [`channel_${ck}`, cv])) };
    savePrefs(full);
  }

  function toggleCh(k: string) {
    const updated = { ...channels, [k]: !channels[k] };
    setChannels(updated);
    const full = { ...prefs, ...Object.fromEntries(Object.entries(updated).map(([ck, cv]) => [`channel_${ck}`, cv])) };
    savePrefs(full);
  }

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
              <Toggle on={prefs[key] ?? false} onToggle={() => toggle(key)} />
            </PrefRow>
          ))}
        </SectionBody>
      </SectionCard>

      <SectionCard>
        <SectionHead title="Delivery Channels" sub="How you receive notifications" />
        <SectionBody>
          <div className="flex gap-3 py-4 flex-wrap">
            {(Object.entries(channels)).map(([key, on]) => (
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
  const toast = useToast();
  const { data: meData } = useGetMeQuery();
  const { data: sessionsData } = useGetSessionsQuery();
  const [changePassword, { isLoading: changingPw }] = useChangePasswordMutation();
  const [totpSetup,    { isLoading: totpSetupLoading }] = useTotpSetupMutation();
  const [totpConfirm,  { isLoading: totpConfirmLoading }] = useTotpConfirmMutation();
  const [totpDisable,  { isLoading: totpDisableLoading }] = useTotpDisableMutation();
  const [revokeSession] = useRevokeSessionMutation();

  const [show,        setShow]        = useState(false);
  const [pwCurrent,   setPwCurrent]   = useState('');
  const [pwNew,       setPwNew]       = useState('');
  const [pwConfirm,   setPwConfirm]   = useState('');

  // TOTP state
  const [totpQr,      setTotpQr]      = useState<string | null>(null);
  const [totpCode,    setTotpCode]    = useState('');
  const [totpDisCode, setTotpDisCode] = useState('');
  const [totpMode,    setTotpMode]    = useState<'idle' | 'setup' | 'disable'>('idle');

  const totpEnabled = meData?.totp_enabled ?? false;
  const sessions    = (sessionsData as any)?.data ?? [];

  async function handleChangePassword() {
    if (pwNew !== pwConfirm) { toast({ type: 'error', title: 'Passwords do not match' }); return; }
    if (!pwCurrent || !pwNew) { toast({ type: 'error', title: 'All fields required' }); return; }
    try {
      await changePassword({ current_password: pwCurrent, new_password: pwNew }).unwrap();
      toast({ type: 'success', title: 'Password updated' });
      setPwCurrent(''); setPwNew(''); setPwConfirm('');
    } catch {
      toast({ type: 'error', title: 'Password change failed', msg: 'Check your current password' });
    }
  }

  async function handleTotpSetup() {
    try {
      const result = await totpSetup().unwrap();
      setTotpQr(result?.qr_image ?? result?.qr_uri ?? null);
      setTotpMode('setup');
    } catch {
      toast({ type: 'error', title: '2FA setup failed' });
    }
  }

  async function handleTotpConfirm() {
    if (totpCode.length !== 6) { toast({ type: 'error', title: 'Enter 6-digit code' }); return; }
    try {
      await totpConfirm({ code: totpCode }).unwrap();
      toast({ type: 'success', title: '2FA enabled!' });
      setTotpMode('idle'); setTotpQr(null); setTotpCode('');
    } catch {
      toast({ type: 'error', title: 'Invalid code' });
    }
  }

  async function handleTotpDisable() {
    if (totpDisCode.length !== 6) { toast({ type: 'error', title: 'Enter 6-digit code' }); return; }
    try {
      await totpDisable({ code: totpDisCode }).unwrap();
      toast({ type: 'success', title: '2FA disabled' });
      setTotpMode('idle'); setTotpDisCode('');
    } catch {
      toast({ type: 'error', title: 'Invalid code' });
    }
  }

  async function handleRevokeSession(sessionId: string) {
    try {
      await revokeSession(sessionId).unwrap();
      toast({ type: 'success', title: 'Session revoked' });
    } catch {
      toast({ type: 'error', title: 'Revoke failed' });
    }
  }

  return (
    <div className="flex flex-col gap-[calc(16px*var(--u))]">
      {/* Change Password */}
      <SectionCard>
        <SectionHead title="Change Password" />
        <SectionBody>
          <div className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Current password</label>
              <input type={show ? 'text' : 'password'} value={pwCurrent} onChange={e => setPwCurrent(e.target.value)} placeholder="••••••••" className={inputCls} />
            </div>
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">New password</label>
              <input type={show ? 'text' : 'password'} value={pwNew} onChange={e => setPwNew(e.target.value)} placeholder="••••••••" className={inputCls} />
            </div>
            <div className="flex flex-col gap-[7px]">
              <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Confirm new password</label>
              <input type={show ? 'text' : 'password'} value={pwConfirm} onChange={e => setPwConfirm(e.target.value)} placeholder="••••••••" className={inputCls} />
            </div>
            <label className="flex items-center gap-2 text-[12.5px] text-[var(--text-2)] cursor-pointer">
              <input type="checkbox" checked={show} onChange={e => setShow(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
              Show passwords
            </label>
            <button
              type="button"
              onClick={handleChangePassword}
              disabled={changingPw}
              className={`self-start h-10 px-5 rounded-[11px] font-sans text-[13.5px] font-semibold border-none bg-[var(--accent)] text-white shadow-[0_4px_14px_rgba(59,130,246,.32)] inline-flex items-center gap-2 transition-opacity ${changingPw ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              {changingPw ? <Loader2 size={15} className="animate-spin" /> : null}
              {changingPw ? 'Updating…' : 'Update Password'}
            </button>
          </div>
        </SectionBody>
      </SectionCard>

      {/* TOTP */}
      <SectionCard>
        <SectionHead title="Two-Factor Authentication" sub="Add an extra layer of security to your account" />
        <SectionBody>
          <PrefRow label="TOTP Authenticator" sub="Use an app like Google Authenticator">
            <span className={`inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12px] font-semibold ${totpEnabled ? 'bg-[var(--green-soft)] text-[var(--green)]' : 'bg-[var(--surface-3)] text-[var(--text-3)]'}`}>
              {totpEnabled ? <><CheckCircle size={12} /> Enabled</> : 'Disabled'}
            </span>
          </PrefRow>

          {totpMode === 'idle' && (
            <div className="py-3 flex gap-3">
              {!totpEnabled ? (
                <button
                  type="button"
                  onClick={handleTotpSetup}
                  disabled={totpSetupLoading}
                  className="h-9 px-4 rounded-[10px] font-sans text-[13px] font-semibold cursor-pointer border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent-2)] inline-flex items-center gap-2 transition-opacity hover:opacity-90"
                >
                  {totpSetupLoading ? <Loader2 size={14} className="animate-spin" /> : null}
                  Enable 2FA
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setTotpMode('disable')}
                  className="h-9 px-4 rounded-[10px] font-sans text-[13px] font-semibold cursor-pointer border border-[var(--red)] bg-[var(--red-soft)] text-[var(--red)] inline-flex items-center gap-2"
                >
                  Disable 2FA
                </button>
              )}
            </div>
          )}

          {totpMode === 'setup' && (
            <div className="py-3 flex flex-col gap-4">
              {totpQr && (
                <div className="flex flex-col items-center gap-3">
                  <img src={totpQr} alt="TOTP QR Code" className="w-40 h-40 rounded-[12px] border border-[var(--border)]" />
                  <p className="text-[12px] text-[var(--text-3)] text-center m-0">Scan with your authenticator app, then enter the 6-digit code below to confirm.</p>
                </div>
              )}
              <div className="flex flex-col gap-[6px]">
                <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Verification code</label>
                <input
                  type="text" inputMode="numeric" maxLength={6}
                  value={totpCode} onChange={e => setTotpCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="123456"
                  className={`${inputCls} font-mono tracking-[0.3em] text-center`}
                />
              </div>
              <div className="flex gap-3">
                <button type="button" onClick={() => { setTotpMode('idle'); setTotpQr(null); setTotpCode(''); }} className="flex-1 h-9 rounded-[10px] font-sans text-[13px] font-semibold cursor-pointer border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)]">Cancel</button>
                <button type="button" onClick={handleTotpConfirm} disabled={totpConfirmLoading} className={`flex-[2] h-9 rounded-[10px] font-sans text-[13px] font-semibold border-none bg-[var(--accent)] text-white inline-flex items-center justify-center gap-2 transition-opacity ${totpConfirmLoading ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}>
                  {totpConfirmLoading ? <Loader2 size={14} className="animate-spin" /> : null}
                  Verify &amp; Enable
                </button>
              </div>
            </div>
          )}

          {totpMode === 'disable' && (
            <div className="py-3 flex flex-col gap-4">
              <p className="text-[13px] text-[var(--text-2)] m-0">Enter the 6-digit code from your authenticator app to disable 2FA.</p>
              <div className="flex flex-col gap-[6px]">
                <label className="text-[12.5px] font-semibold text-[var(--text-2)]">Authenticator code</label>
                <input
                  type="text" inputMode="numeric" maxLength={6}
                  value={totpDisCode} onChange={e => setTotpDisCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="123456"
                  className={`${inputCls} font-mono tracking-[0.3em] text-center`}
                />
              </div>
              <div className="flex gap-3">
                <button type="button" onClick={() => { setTotpMode('idle'); setTotpDisCode(''); }} className="flex-1 h-9 rounded-[10px] font-sans text-[13px] font-semibold cursor-pointer border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)]">Cancel</button>
                <button type="button" onClick={handleTotpDisable} disabled={totpDisableLoading} className={`flex-[2] h-9 rounded-[10px] font-sans text-[13px] font-semibold border-none bg-[var(--red)] text-white inline-flex items-center justify-center gap-2 transition-opacity ${totpDisableLoading ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}>
                  {totpDisableLoading ? <Loader2 size={14} className="animate-spin" /> : null}
                  Disable 2FA
                </button>
              </div>
            </div>
          )}
        </SectionBody>
      </SectionCard>

      {/* Sessions */}
      <SectionCard>
        <SectionHead title="Active Sessions" sub="Devices currently signed in to your account" />
        <SectionBody>
          {(sessions ?? []).length === 0 ? (
            <div className="py-3">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-[13.5px] font-semibold text-[var(--text)]">Current session</div>
                  <div className="text-[12px] text-[var(--text-3)] mt-[2px]">This device</div>
                </div>
                <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-semibold bg-[var(--green-soft)] text-[var(--green)]">
                  ● Active now
                </span>
              </div>
            </div>
          ) : (
            <div className="flex flex-col divide-y divide-[var(--border)]">
              {(sessions ?? []).map((s: any) => (
                <div key={s?.id} className="py-3 flex items-center justify-between gap-4">
                  <div>
                    <div className="text-[13.5px] font-semibold text-[var(--text)]">{s?.device ?? 'Unknown device'}</div>
                    <div className="text-[12px] text-[var(--text-3)] mt-[2px]">
                      {s?.current ? 'Current session' : `Last active: ${s?.last_active ?? '—'}`}
                      {s?.location ? ` · ${s?.location}` : ''}
                    </div>
                  </div>
                  {s?.current ? (
                    <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-semibold bg-[var(--green-soft)] text-[var(--green)]">
                      ● Active now
                    </span>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleRevokeSession(s?.id)}
                      className="h-7 px-3 rounded-[7px] font-sans text-[12px] font-semibold cursor-pointer border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-2)] hover:bg-[var(--red-soft)] hover:text-[var(--red)] hover:border-transparent transition-colors"
                    >
                      Revoke
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
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
