import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { useToast } from '../components/ui';
import { useGetRiskSettingsQuery, useUpdateRiskSettingsMutation } from '../services/tradeMindApiService';

const inputCls = 'h-11 px-[13px] rounded-[11px] border border-line bg-surface-2 text-ink font-sans text-sm outline-none w-full box-border transition-colors focus:border-accent';

function Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button type="button" onClick={onToggle}
      className="w-11 h-6 rounded-full border-0 cursor-pointer p-0 relative shrink-0 transition-colors"
      style={{ background: on ? 'var(--accent)' : 'var(--surface-3)' }}>
      <div className="w-[18px] h-[18px] rounded-full bg-white absolute top-[3px] transition-all"
        style={{ left: on ? 23 : 3, boxShadow: '0 1px 3px rgba(0,0,0,.2)' }} />
    </button>
  );
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)' }}>
      <div className="border-b border-line flex items-center gap-[9px]" style={{ padding: 'calc(15px * var(--u)) calc(18px * var(--u))' }}>
        <span className="text-ink-3 grid place-items-center">{icon}</span>
        <h2 className="m-0 text-[14.5px] font-semibold text-ink">{title}</h2>
      </div>
      <div style={{ padding: 'calc(18px * var(--u))' }}>{children}</div>
    </div>
  );
}

function Field({ label, sub, children }: { label: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 border-b border-line" style={{ paddingBottom: 'calc(16px * var(--u))', marginBottom: 'calc(16px * var(--u))' }}>
      <div className="flex-1">
        <div className="text-[13.5px] font-semibold text-ink">{label}</div>
        {sub && <div className="text-[12.5px] text-ink-3 mt-[2px]">{sub}</div>}
      </div>
      <div className="flex-1 max-w-[200px]">{children}</div>
    </div>
  );
}

function LastField({ label, sub, children }: { label: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6">
      <div className="flex-1">
        <div className="text-[13.5px] font-semibold text-ink">{label}</div>
        {sub && <div className="text-[12.5px] text-ink-3 mt-[2px]">{sub}</div>}
      </div>
      <div className="flex-1 max-w-[200px]">{children}</div>
    </div>
  );
}

function Seg<T extends string>({ options, value, onChange, labels }: {
  options: T[]; value: T; onChange: (v: T) => void; labels?: Record<T, string>;
}) {
  return (
    <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
      {options.map(o => (
        <button key={o} type="button" onClick={() => onChange(o)}
          className="border-0 font-sans text-[12.5px] font-semibold px-3 py-[5px] rounded-[7px] cursor-pointer transition-colors"
          style={{ background: value === o ? 'var(--accent)' : 'transparent', color: value === o ? '#fff' : 'var(--text-2)' }}>
          {labels ? labels[o] : o}
        </button>
      ))}
    </div>
  );
}

const ACCENT_PRESETS = ['#3B82F6', '#8B5CF6', '#14B8A6', '#F59E0B', '#EC4899', '#10B981'];

export default function RiskSettingsPage() {
  const { user } = useAuth();
  const { theme, toggleTheme, density, setDensity } = useTheme();
  const navigate = useNavigate();
  const toast    = useToast();

  const [maxDailyLoss, setMaxDailyLoss] = useState('10000');
  const [stopLossPct,  setStopLossPct]  = useState('7');
  const [targetPct,    setTargetPct]    = useState('15');
  const [maxPosSz,     setMaxPosSz]     = useState('50000');
  const [autoSL,       setAutoSL]       = useState(true);
  const [autoTarget,   setAutoTarget]   = useState(true);
  const [mode,         setMode]         = useState<'paper' | 'live'>('paper');
  const [accent,       setAccent]       = useState('#3B82F6');

  const { data: settings, isLoading: loading } = useGetRiskSettingsQuery(user?.id ?? 0, { skip: !user });
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
    if (s.mode) setMode(s.mode);
  }, [settings]);

  useEffect(() => {
    const saved = localStorage.getItem('trademind-accent');
    if (saved && /^#[0-9A-Fa-f]{6}$/.test(saved)) {
      setAccent(saved);
      document.documentElement.style.setProperty('--accent', saved);
    }
  }, []);

  function applyAccent(color: string) {
    if (!/^#[0-9A-Fa-f]{6}$/.test(color)) return;
    setAccent(color);
    document.documentElement.style.setProperty('--accent', color);
    localStorage.setItem('trademind-accent', color);
  }

  async function handleSave() {
    if (!user) return;
    // Audit M19 — these limits exist to protect the account; a zero/negative
    // value would silently disable the protection they're meant to provide.
    const dailyLoss = +maxDailyLoss, sl = +stopLossPct, target = +targetPct, posSz = +maxPosSz;
    if (!(dailyLoss > 0)) { toast({ type: 'error', title: 'Invalid input', msg: 'Max Daily Loss must be greater than 0' }); return; }
    if (!(sl > 0 && sl <= 100)) { toast({ type: 'error', title: 'Invalid input', msg: 'Stop Loss % must be between 0 and 100' }); return; }
    if (!(target > 0 && target <= 100)) { toast({ type: 'error', title: 'Invalid input', msg: 'Target % must be between 0 and 100' }); return; }
    if (!(posSz > 0)) { toast({ type: 'error', title: 'Invalid input', msg: 'Max Position Size must be greater than 0' }); return; }
    try {
      await updateRiskSettings({ userId: user.id, settings: {
        max_daily_loss:    dailyLoss,
        stop_loss_pct:     sl,
        target_pct:        target,
        max_position_size: posSz,
        auto_stop_loss:    autoSL ? 1 : 0,
        auto_target:       autoTarget ? 1 : 0,
        mode,
      } as any }).unwrap();
      toast({ type: 'success', title: 'Settings saved', msg: 'Risk profile updated successfully' });
    } catch (e: unknown) {
      toast({ type: 'error', title: 'Save failed', msg: e instanceof Error ? e.message : 'Try again' });
    }
  }

  const numInput = (value: string, onChange: (v: string) => void, prefix?: string, suffix?: string) => (
    <div className="relative">
      {prefix && <span className="absolute left-[13px] top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none text-sm">{prefix}</span>}
      <input type="number" value={value} onChange={e => onChange(e.target.value)} className={inputCls}
        style={{ paddingLeft: prefix ? 26 : 13, paddingRight: suffix ? 30 : 13 }}
        onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
        onBlur={e => e.currentTarget.style.borderColor = 'var(--border)'} />
      {suffix && <span className="absolute right-[13px] top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none text-sm">{suffix}</span>}
    </div>
  );

  if (loading) {
    return (
      <div className="flex flex-col dgap animate-page-in">
        <div className="h-7 w-[180px] rounded-[7px] bg-surface-3" />
        <div className="h-[200px] rounded-[14px] bg-surface-3" />
        <div className="h-[200px] rounded-[14px] bg-surface-3" />
      </div>
    );
  }

  return (
    <div className="flex flex-col dgap animate-page-in max-w-[720px]">
      <div>
        <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>Settings</h1>
        <p className="text-ink-2 text-[13.5px] mt-1 m-0">Risk management, trading preferences &amp; appearance</p>
      </div>

      <Section title="Risk Management" icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>}>
        <Field label="Max Daily Loss" sub="Halt all trading if daily loss exceeds this limit">{numInput(maxDailyLoss, setMaxDailyLoss, '₹')}</Field>
        <Field label="Stop Loss %" sub="Default stop loss percentage for new positions">{numInput(stopLossPct, setStopLossPct, undefined, '%')}</Field>
        <Field label="Target %" sub="Default profit target percentage for new positions">{numInput(targetPct, setTargetPct, undefined, '%')}</Field>
        <Field label="Max Position Size" sub="Maximum capital per single trade">{numInput(maxPosSz, setMaxPosSz, '₹')}</Field>
        <Field label="Auto Stop-Loss" sub="Automatically place SL orders on new trades"><Toggle on={autoSL} onToggle={() => setAutoSL(x => !x)} /></Field>
        <Field label="Auto Target" sub="Automatically place target exit orders on new trades"><Toggle on={autoTarget} onToggle={() => setAutoTarget(x => !x)} /></Field>
        <LastField label="Trading Mode" sub="Paper trading uses virtual ₹10L — no real money">
          <Seg options={['paper', 'live'] as const} value={mode} onChange={setMode} labels={{ paper: '📄 Paper', live: '⚡ Live' }} />
        </LastField>
      </Section>

      <Section title="Appearance" icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>}>
        <Field label="Theme" sub="Dark or light interface">
          <Seg options={['dark', 'light'] as const} value={theme} onChange={v => { if (v !== theme) toggleTheme(); }} labels={{ dark: '🌙 Dark', light: '☀️ Light' }} />
        </Field>
        <Field label="Density" sub="Controls spacing and padding throughout the UI">
          <Seg options={['compact', 'balanced', 'comfy'] as const} value={density} onChange={setDensity} labels={{ compact: 'Compact', balanced: 'Balanced', comfy: 'Comfy' }} />
        </Field>
        <LastField label="Accent colour" sub="Highlight colour for buttons and active states">
          <div className="flex gap-2 items-center">
            {ACCENT_PRESETS.map(c => (
              <button key={c} type="button" title={c} onClick={() => applyAccent(c)}
                className="w-[26px] h-[26px] rounded-full cursor-pointer outline-none box-border transition-[border]"
                style={{ background: c, border: accent === c ? '2.5px solid var(--text)' : '2.5px solid transparent' }} />
            ))}
          </div>
        </LastField>
      </Section>

      <Section title="Account" icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>}>
        <Field label="Username" sub="Your login identifier"><input value={user?.username ?? ''} readOnly className={`${inputCls} text-ink-3 cursor-default`} /></Field>
        <Field label="Display Name" sub="Shown in the UI"><input value={user?.display_name ?? ''} readOnly className={`${inputCls} text-ink-3 cursor-default`} /></Field>
        <LastField label="Account type" sub="Paper trading is risk-free with virtual capital">
          <span className="inline-flex items-center gap-[6px] h-8 px-3 rounded-[8px] text-[13px] font-semibold bg-gain-soft text-gain">
            <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
            Paper trading active
          </span>
        </LastField>
      </Section>

      <div className="flex justify-end gap-[10px]">
        <button type="button" onClick={() => navigate('/dashboard')}
          className="h-10 px-5 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover">
          Cancel
        </button>
        <button type="button" onClick={handleSave} disabled={saving}
          className={`h-10 px-6 rounded-[11px] font-sans text-[13.5px] font-semibold border-0 bg-accent text-white inline-flex items-center gap-2 transition-opacity ${saving ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
          style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}>
          {saving ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <CheckCircle size={16} />}
          {saving ? 'Saving…' : 'Save Settings'}
        </button>
      </div>

      <style>{`@keyframes spin{to{transform:rotate(360deg);}}`}</style>
    </div>
  );
}
