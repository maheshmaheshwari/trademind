import { useState, useEffect, useRef } from 'react';
import { X, Plus } from 'lucide-react';
import { useLazyGetStocksQuery, useExecuteSignalMutation } from '../services/tradeMindApiService';
import { useToast, symColor } from './ui';
import { useAuth } from '../AuthContext';
import type { Stock } from '../types';

interface Props {
  onClose: () => void;
}

function inr(n: number) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

const SIG_COL = { BUY: ['var(--green)', 'var(--green-soft)'], SELL: ['var(--red)', 'var(--red-soft)'], HOLD: ['var(--gold)', 'var(--gold-soft)'] };

export function AddPositionModal({ onClose }: Props) {
  const toast   = useToast();
  const { user } = useAuth();
  const [q,     setQ]     = useState('');
  const [sel,   setSel]   = useState<Stock | null>(null);
  const [qty,   setQty]   = useState('');
  const [price, setPrice] = useState('');
  const [mode,  setMode]  = useState<'paper' | 'live'>('paper');
  const [open,  setOpen]  = useState(false);
  const [busy,  setBusy]  = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [fetchStocks, { data: stockRes }] = useLazyGetStocksQuery();
  const [executeSignal] = useExecuteSignalMutation();
  const opts: Stock[] = (stockRes as any)?.data ?? [];

  useEffect(() => {
    if (!q || sel) { setOpen(false); return; }
    const t = setTimeout(() => {
      fetchStocks({ search: q, size: 5 });
      setOpen(true);
    }, 200);
    return () => clearTimeout(t);
  }, [q, sel, fetchStocks]);

  // Close dropdown on outside click
  useEffect(() => {
    function h(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, [onClose]);

  function choose(s: Stock) { setSel(s); setQ(s.symbol); setPrice(String(s.price)); setOpen(false); }

  const valid = sel !== null && +qty > 0 && +price > 0;
  const total = valid ? +qty * +price : 0;

  async function submit() {
    if (!valid || !sel || !user) return;
    setBusy(true);
    const investment = +qty * +price;
    try {
      await executeSignal({
        user_id: user.id,
        symbol: sel.symbol,
        name: sel.name,
        investment_amount: investment,
        buy_price: +price,
        target_price: sel.target_price ?? +price * 1.05,
        stop_loss: sel.stop_loss ?? +price * 0.95,
        signal: sel.signal ?? 'BUY',
        confidence: sel.confidence ?? 0,
        horizon: sel.horizon ?? 'Unknown',
        mode: mode === 'live' ? 'LIVE' : 'PAPER',
      }).unwrap();
      toast({ type: 'success', title: 'Position added', msg: `${qty} × ${sel.symbol} @ ${inr(+price)} · ${mode === 'paper' ? 'Paper' : 'Live'} account` });
      onClose();
    } catch (e: unknown) {
      toast({ type: 'error', title: 'Order failed', msg: e instanceof Error ? e.message : 'Try again' });
    } finally {
      setBusy(false);
    }
  }

  const inputS: React.CSSProperties = {
    height: 44, padding: '0 13px', borderRadius: 11,
    border: '1px solid var(--border)', background: 'var(--surface-2)',
    color: 'var(--text)', fontFamily: 'inherit', fontSize: 14, outline: 'none', width: '100%',
    boxSizing: 'border-box', transition: 'border .15s',
  };

  const segBtn = (active: boolean): React.CSSProperties => ({
    border: 'none', background: active ? 'var(--accent)' : 'transparent',
    color: active ? '#fff' : 'var(--text-2)',
    fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600,
    padding: '6px 16px', borderRadius: 7, cursor: 'pointer',
    transition: 'background .14s, color .14s',
  });

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'grid', placeItems: 'center', padding: 20 }}>
      {/* Scrim */}
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(3,6,15,.55)', backdropFilter: 'blur(2px)', zIndex: 100 }} onClick={onClose} />

      {/* Card */}
      <div style={{ width: 480, maxWidth: '100%', background: 'var(--surface)', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-lg,18px)', boxShadow: 'var(--shadow-lg)', position: 'relative', zIndex: 101, animation: 'pop .22s cubic-bezier(.4,0,.2,1) both', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }}>

        {/* Header */}
        <div style={{ padding: '20px 22px 14px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>Add Position</h3>
            <span style={{ fontSize: 13, color: 'var(--text-2)' }}>Record a holding or place a paper trade</span>
          </div>
          <button onClick={onClose} style={{ width: 34, height: 34, borderRadius: 9, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-2)', display: 'grid', placeItems: 'center', cursor: 'pointer', flexShrink: 0 }}>
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '4px 22px 8px', overflowY: 'auto' }}>

          {/* Symbol search */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 15, position: 'relative' }} ref={wrapRef}>
            <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>Symbol</label>
            <input
              style={inputS}
              value={q}
              onChange={e => { setQ(e.target.value); setSel(null); }}
              placeholder="Search e.g. RELIANCE, Infosys…"
              autoFocus
              onFocus={e => { e.currentTarget.style.borderColor = 'var(--accent)'; }}
              onBlur={e => { e.currentTarget.style.borderColor = 'var(--border)'; }}
            />

            {/* Dropdown */}
            {open && opts.length > 0 && !sel && (
              <div style={{ position: 'absolute', top: 74, left: 0, right: 0, background: 'var(--surface)', border: '1px solid var(--border-strong)', borderRadius: 11, boxShadow: 'var(--shadow-lg)', zIndex: 5, overflow: 'hidden' }}>
                {opts.map(s => {
                  const c = symColor(s?.symbol ?? '');
                  return (
                    <button
                      key={s?.symbol}
                      onMouseDown={() => choose(s)}
                      style={{ display: 'flex', alignItems: 'center', gap: 11, height: 46, padding: '0 12px', width: '100%', background: 'transparent', border: 'none', borderRadius: 0, cursor: 'pointer', fontFamily: 'inherit', transition: 'background .12s', textAlign: 'left' }}
                      onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)'}
                      onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.background = 'transparent'}
                    >
                      <span style={{ width: 28, height: 28, borderRadius: 9, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 11, background: c + '22', color: c, flexShrink: 0 }}>{(s?.symbol ?? '').slice(0, 2)}</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 0, flex: 1, minWidth: 0 }}>
                        <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--text)' }}>{s?.symbol}</span>
                        <span style={{ fontSize: 11, color: 'var(--text-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s?.name}</span>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono,monospace)', fontSize: 12.5, color: 'var(--text-2)' }}>₹{(s?.price ?? 0).toLocaleString('en-IN')}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Selected stock preview */}
          {sel && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 11, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 11, marginBottom: 15 }}>
              <span style={{ width: 32, height: 32, borderRadius: 9, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 12, background: symColor(sel?.symbol ?? '') + '22', color: symColor(sel?.symbol ?? ''), flexShrink: 0 }}>{(sel?.symbol ?? '').slice(0, 2)}</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0, flex: 1 }}>
                <span style={{ fontWeight: 600, fontSize: 13.5, color: 'var(--text)' }}>{sel?.symbol}</span>
                <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>{sel?.name}</span>
              </div>
              <span style={{ display: 'inline-flex', alignItems: 'center', height: 23, padding: '0 9px', borderRadius: 7, fontSize: 11.5, fontWeight: 700, color: (SIG_COL[sel?.signal ?? ''] ?? SIG_COL.HOLD)[0], background: (SIG_COL[sel?.signal ?? ''] ?? SIG_COL.HOLD)[1] }}>
                {sel?.signal === 'BUY' ? '↑' : sel?.signal === 'SELL' ? '↓' : '●'} {sel?.signal}
              </span>
            </div>
          )}

          {/* Qty + Price */}
          <div style={{ display: 'flex', gap: 13, marginBottom: 15 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, flex: 1 }}>
              <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>Quantity</label>
              <input type="number" style={inputS} value={qty} onChange={e => setQty(e.target.value)} placeholder="0"
                onFocus={e => { e.currentTarget.style.borderColor = 'var(--accent)'; }} onBlur={e => { e.currentTarget.style.borderColor = 'var(--border)'; }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, flex: 1 }}>
              <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>Price (₹)</label>
              <input type="number" style={inputS} value={price} onChange={e => setPrice(e.target.value)} placeholder="0.00"
                onFocus={e => { e.currentTarget.style.borderColor = 'var(--accent)'; }} onBlur={e => { e.currentTarget.style.borderColor = 'var(--border)'; }} />
            </div>
          </div>

          {/* Total preview */}
          {valid && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '11px 13px', background: 'var(--accent-soft)', borderRadius: 11, marginBottom: 6 }}>
              <span style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>Total investment</span>
              <span style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 15, color: 'var(--accent-2)' }}>{inr(total)}</span>
            </div>
          )}

          {/* Account toggle */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 6 }}>
            <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>Account</label>
            <div style={{ display: 'inline-flex', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 10, padding: 3, gap: 2, alignSelf: 'flex-start' }}>
              <button style={segBtn(mode === 'paper')} onClick={() => setMode('paper')}>📄 Paper</button>
              <button style={segBtn(mode === 'live')}  onClick={() => setMode('live')}>⚡ Live</button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '16px 22px 20px', display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8, height: 40, padding: '0 16px', borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600, cursor: 'pointer', border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text)' }}>
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!valid || busy}
            style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8, height: 40, padding: '0 16px', borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600, cursor: (valid && !busy) ? 'pointer' : 'not-allowed', border: '1px solid transparent', background: 'var(--accent)', color: '#fff', opacity: (valid && !busy) ? 1 : 0.5 }}
          >
            <Plus size={17} /> {busy ? 'Placing…' : 'Add Position'}
          </button>
        </div>
      </div>
    </div>
  );
}
