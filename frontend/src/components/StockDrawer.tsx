import { useEffect, useState } from 'react';
import { X, Bookmark, AlertTriangle } from 'lucide-react';
import { useLazyGetStockDetailQuery, useGetPositionsQuery, useExecuteSignalMutation } from '../services/tradeMindApiService';
import { useAuth } from '../AuthContext';
import { useToast, symColor } from './ui';
import { AreaChart } from './Charts';
import type { NewsItem, HorizonBreakdown, OpenPosition } from '../types';

// ─── types ────────────────────────────────────────────────────────────────────

interface StockDetail {
  symbol: string; name: string; sector: string;
  price: number; change: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; high52: number; low52: number;
  pe: number; mcap: number; volume: number;
  sentiment: number; updatedMin: number;
  spark: number[]; news: NewsItem[]; horizons: HorizonBreakdown[];
}

interface StockDrawerProps {
  symbol: string | null;
  onClose: () => void;
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function fmtAgo(m: number) { return m < 60 ? `${m}m ago` : `${Math.floor(m / 60)}h ago`; }
function signed(n: number, dec = 2) {
  return (n >= 0 ? '+' : '') + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN', { maximumFractionDigits: 0 });
}

const SENT_TAG: Record<string, [string, string, string]> = {
  pos: ['Bullish', 'var(--green)',  'var(--green-soft)'],
  neg: ['Bearish', 'var(--red)',    'var(--red-soft)'],
  neu: ['Neutral', 'var(--text-2)', 'var(--surface-3)'],
};

// ─── TradePanel ───────────────────────────────────────────────────────────────

function TradePanel({ data, position, onClose }: {
  data: StockDetail;
  position: OpenPosition | null;
  onClose: () => void;
}) {
  const { user }  = useAuth();
  const toast     = useToast();
  const [qty, setQty]     = useState('');
  const [executeSignalMut, { isLoading: busy }] = useExecuteSignalMutation();

  const balance   = user?.virtual_balance ?? 0;
  const price     = data.price;
  const maxBuy    = price > 0 ? Math.floor(balance / price) : 0;
  const qtyNum    = Math.max(0, parseInt(qty) || 0);
  const totalCost = qtyNum * price;

  // Estimated SL / Target based on signal
  const estSL     = +(price * 0.93).toFixed(2);
  const estTarget = +(price * 1.12).toFixed(2);
  const slPct     = -7;
  const tgtPct    = +12;

  // ── SELL signal — user does NOT own this stock ────────────────────────────
  if (data.signal === 'SELL' && !position) {
    return (
      <div style={{ padding: '16px 22px', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 12, padding: '13px 14px', background: 'rgba(245,158,11,.08)', border: '1px solid rgba(245,158,11,.25)', borderRadius: 11, marginBottom: 12 }}>
          <AlertTriangle size={18} style={{ color: 'var(--gold)', flexShrink: 0, marginTop: 1 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>You don't hold {data.symbol}</span>
            <span style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>
              SELL signals indicate potential downside — avoid adding this position. TradeMind doesn't support short selling. Only stocks you already own can be sold.
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={() => toast({ type: 'info', title: `${data.symbol} added to watchlist` })}
            style={ghostBtn}
          >
            <Bookmark size={17} /> Watchlist
          </button>
          <button onClick={onClose} style={{ ...ghostBtn, flex: 2 }}>Got it</button>
        </div>
      </div>
    );
  }

  // ── SELL signal — user owns the stock ─────────────────────────────────────
  if (data.signal === 'SELL' && position) {
    const maxSell  = position.qty;
    const sellQty  = Math.min(qtyNum, maxSell);
    const proceeds = sellQty * price;
    const pnlValue = (price - position.entry) * sellQty;
    const pnlPct   = ((price - position.entry) / position.entry) * 100;

    async function executeSell() {
      if (!user || sellQty <= 0) return;
      try {
        await executeSignalMut({ user_id: user.id, symbol: data.symbol, name: data.name, investment_amount: proceeds, buy_price: position!.entry, target_price: position!.target, stop_loss: position!.sl, signal: 'SELL', mode: 'PAPER' }).unwrap();
        toast({ type: pnlValue >= 0 ? 'success' : 'info', title: `Sold ${sellQty} × ${data.symbol}`, msg: `Proceeds: ${inr(proceeds)} · P&L: ${signed(pnlValue, 0)} (${signed(pnlPct, 2)}%)` });
        onClose();
      } catch (e: unknown) {
        toast({ type: 'error', title: 'Order failed', msg: e instanceof Error ? e.message : 'Try again' });
      }
    }

    return (
      <div style={{ padding: '16px 22px', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
        {/* Position summary */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '12px 13px', background: 'var(--surface-2)', borderRadius: 11, marginBottom: 13 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>You own</div>
            <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: 'var(--text)' }}>{position.qty} shares</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>Avg. cost</div>
            <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: 'var(--text)' }}>{inr(position.entry)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>Unrealised P&amp;L</div>
            <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: position.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {signed(position.pnl, 0)}
            </div>
          </div>
        </div>

        {/* Qty input */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>Qty to sell</label>
            <input
              type="number" min={1} max={maxSell} value={qty}
              onChange={e => setQty(e.target.value)}
              placeholder={`1–${maxSell}`}
              style={{ ...inputS, borderColor: qtyNum > maxSell ? 'var(--red)' : 'var(--border)' }}
              onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
              onBlur={e => e.currentTarget.style.borderColor = qtyNum > maxSell ? 'var(--red)' : 'var(--border)'}
            />
          </div>
          <button style={{ ...allBtn, alignSelf: 'flex-end' }} onClick={() => setQty(String(maxSell))}>All ({maxSell})</button>
        </div>

        {/* Validation */}
        {qtyNum > maxSell && (
          <p style={{ fontSize: 12, color: 'var(--red)', margin: '0 0 8px' }}>You only hold {maxSell} shares.</p>
        )}

        {/* Proceeds preview */}
        {qtyNum > 0 && qtyNum <= maxSell && (
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 13px', background: pnlValue >= 0 ? 'var(--green-soft)' : 'var(--red-soft)', borderRadius: 10, marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-3)' }}>Proceeds</div>
              <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: 'var(--text)' }}>{inr(proceeds)}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 11, color: 'var(--text-3)' }}>Realised P&amp;L</div>
              <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: pnlValue >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {signed(pnlValue, 0)} ({signed(pnlPct, 2)}%)
              </div>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => { setQty(''); }} style={ghostBtn}>Cancel</button>
          <button
            disabled={busy || qtyNum <= 0 || qtyNum > maxSell}
            onClick={executeSell}
            style={{ flex: 2, ...primaryBtn('#EF4444'), opacity: (busy || qtyNum <= 0 || qtyNum > maxSell) ? 0.5 : 1, cursor: busy || qtyNum <= 0 || qtyNum > maxSell ? 'not-allowed' : 'pointer' }}
          >
            {busy ? '…' : `Sell ${qtyNum > 0 ? qtyNum : ''} ${data.symbol}`}
          </button>
        </div>
      </div>
    );
  }

  // ── BUY / HOLD — buy form ──────────────────────────────────────────────────
  async function executeBuy() {
    if (!user || qtyNum <= 0) return;
    try {
      await executeSignalMut({ user_id: user.id, symbol: data.symbol, name: data.name, investment_amount: totalCost, buy_price: price, target_price: estTarget, stop_loss: estSL, signal: data.signal, confidence: data.confidence, mode: 'PAPER' }).unwrap();
      toast({ type: 'success', title: `Bought ${qtyNum} × ${data.symbol}`, msg: `Invested: ${inr(totalCost)} · SL: ${inr(estSL)} · Target: ${inr(estTarget)}` });
      onClose();
    } catch (e: unknown) {
      toast({ type: 'error', title: 'Order failed', msg: e instanceof Error ? e.message : 'Try again' });
    }
  }

  const insufficient = qtyNum > 0 && totalCost > balance;

  return (
    <div style={{ padding: '16px 22px', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
      {/* Balance + SL/Target info row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '12px 13px', background: 'var(--surface-2)', borderRadius: 11, marginBottom: 13 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>Available cash</div>
          <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 13, color: 'var(--text)' }}>{inrCompact(balance)}</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>Est. Stop Loss</div>
          <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 13, color: 'var(--red)' }}>
            {inr(estSL)} <span style={{ fontSize: 10.5, fontWeight: 500 }}>{slPct}%</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>Est. Target</div>
          <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 13, color: 'var(--green)' }}>
            {inr(estTarget)} <span style={{ fontSize: 10.5, fontWeight: 500 }}>+{tgtPct}%</span>
          </div>
        </div>
      </div>

      {/* Qty input row */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>
            Quantity <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>@ {inr(price)}</span>
          </label>
          <input
            type="number" min={1} value={qty}
            onChange={e => setQty(e.target.value)}
            placeholder="Enter qty"
            style={{ ...inputS, borderColor: insufficient ? 'var(--red)' : 'var(--border)' }}
            onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
            onBlur={e => e.currentTarget.style.borderColor = insufficient ? 'var(--red)' : 'var(--border)'}
          />
        </div>
        <button
          style={{ ...allBtn, alignSelf: 'flex-end' }}
          onClick={() => setQty(String(maxBuy))}
          title={`Max affordable: ${maxBuy} shares`}
        >
          Max ({maxBuy})
        </button>
      </div>

      {/* Validation messages */}
      {maxBuy === 0 && (
        <p style={{ fontSize: 12, color: 'var(--red)', margin: '0 0 8px' }}>Insufficient balance to buy even 1 share at current price.</p>
      )}
      {insufficient && (
        <p style={{ fontSize: 12, color: 'var(--red)', margin: '0 0 8px' }}>
          Total cost {inr(totalCost)} exceeds available balance {inrCompact(balance)}.
        </p>
      )}

      {/* Cost preview */}
      {qtyNum > 0 && !insufficient && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 13px', background: 'var(--accent-soft)', borderRadius: 10, marginBottom: 10 }}>
          <span style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>Total investment</span>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 15, color: 'var(--accent-2)' }}>{inr(totalCost)}</span>
            <span style={{ fontSize: 11.5, color: 'var(--text-3)', display: 'block' }}>
              Balance after: {inrCompact(balance - totalCost)}
            </span>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 10 }}>
        <button
          onClick={() => toast({ type: 'info', title: `${data.symbol} added to watchlist` })}
          style={ghostBtn}
        >
          <Bookmark size={17} /> Watchlist
        </button>
        <button
          disabled={busy || qtyNum <= 0 || insufficient || maxBuy === 0}
          onClick={executeBuy}
          style={{ flex: 2, ...primaryBtn('var(--accent)'), opacity: (busy || qtyNum <= 0 || insufficient || maxBuy === 0) ? 0.5 : 1, cursor: (busy || qtyNum <= 0 || insufficient || maxBuy === 0) ? 'not-allowed' : 'pointer' }}
        >
          {data.signal === 'SELL'
            ? <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M6 11l6 6 6-6"/></svg>
            : <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M6 11l6-6 6 6"/></svg>
          }
          {busy ? 'Placing order…' : `Buy ${qtyNum > 0 ? qtyNum : ''} ${data.symbol}`}
        </button>
      </div>
    </div>
  );
}

// ─── Shared button styles ─────────────────────────────────────────────────────

const ghostBtn: React.CSSProperties = {
  flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
  height: 40, borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600,
  cursor: 'pointer', border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text)',
};

function primaryBtn(bg: string): React.CSSProperties {
  return {
    flex: 2, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
    height: 40, borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600,
    border: 'none', background: bg, color: '#fff',
    boxShadow: bg === 'var(--accent)' ? '0 4px 14px rgba(59,130,246,.32)' : '0 4px 14px rgba(239,68,68,.32)',
  };
}

const inputS: React.CSSProperties = {
  height: 40, padding: '0 12px', borderRadius: 10,
  border: '1px solid var(--border)', background: 'var(--surface-2)',
  color: 'var(--text)', fontFamily: 'var(--font-mono,monospace)', fontSize: 14,
  outline: 'none', width: '100%', boxSizing: 'border-box', transition: 'border .15s',
};

const allBtn: React.CSSProperties = {
  height: 40, padding: '0 12px', borderRadius: 10, whiteSpace: 'nowrap',
  border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-2)',
  fontFamily: 'inherit', fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
};

// ─── Main Drawer ──────────────────────────────────────────────────────────────

export function StockDrawer({ symbol, onClose }: StockDrawerProps) {
  const { user } = useAuth();
  const [chartRange, setChartRange] = useState<'1D' | '1W' | '1M' | '3M' | '1Y'>('1M');

  const [fetchDetail, { data: detailRes, isLoading: loadDetail }] = useLazyGetStockDetailQuery();
  const { data: posRes, isLoading: loadPos } = useGetPositionsQuery(
    { userId: user!.id, size: 100 },
    { skip: !user || !symbol }
  );

  useEffect(() => {
    if (symbol) { setChartRange('1M'); fetchDetail(symbol); }
  }, [symbol, fetchDetail]);

  const loading  = loadDetail || loadPos;
  const data: StockDetail | null = (detailRes as any)?.data ?? null;
  const position: OpenPosition | null = ((posRes as any)?.data ?? []).find((p: OpenPosition) => p.symbol === symbol) ?? null;

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, [onClose]);

  // eslint-disable-next-line react-hooks/exhaustive-deps

  if (!symbol) return null;

  const color   = data ? symColor(data.symbol) : '#3B82F6';
  const sigCol  = data?.signal === 'BUY' ? 'var(--green)' : data?.signal === 'SELL' ? 'var(--red)' : 'var(--gold)';
  const sigBg   = data?.signal === 'BUY' ? 'var(--green-soft)' : data?.signal === 'SELL' ? 'var(--red-soft)' : 'var(--gold-soft)';
  const lineCol = data && data.change >= 0 ? '#10B981' : '#EF4444';
  const sigArrow = data?.signal === 'BUY' ? '↑' : data?.signal === 'SELL' ? '↓' : '●';

  const rangeBtn = (r: typeof chartRange): React.CSSProperties => ({
    border: 'none',
    background: chartRange === r ? 'var(--surface)' : 'transparent',
    color: chartRange === r ? 'var(--text)' : 'var(--text-2)',
    fontFamily: 'inherit', fontSize: 12.5, fontWeight: 600,
    padding: '4px 9px', borderRadius: 6, cursor: 'pointer',
    boxShadow: chartRange === r ? 'var(--shadow-sm)' : 'none',
    transition: 'background .14s, color .14s',
  });

  const h4: React.CSSProperties = {
    margin: '0 0 12px', fontSize: 13.5, fontWeight: 600, color: 'var(--text)',
    display: 'flex', alignItems: 'center', gap: 9,
  };

  return (
    <>
      {/* Scrim */}
      <div
        style={{ position: 'fixed', inset: 0, background: 'rgba(3,6,15,.55)', backdropFilter: 'blur(2px)', zIndex: 90 }}
        onClick={onClose}
      />

      {/* Panel */}
      <aside style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: 520, maxWidth: '92vw', background: 'var(--surface)', borderLeft: '1px solid var(--border)', zIndex: 95, display: 'flex', flexDirection: 'column', boxShadow: 'var(--shadow-lg)', animation: 'slideIn .3s cubic-bezier(.4,0,.2,1) both' }}>

        {/* ── Header ── */}
        <div style={{ padding: '20px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 14, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
            <span style={{ width: 42, height: 42, borderRadius: 9, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 15, background: color + '22', color, flexShrink: 0 }}>
              {symbol.slice(0, 2)}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {loading ? (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <div style={{ width: 90, height: 16, borderRadius: 6, background: 'var(--surface-3)' }} />
                  <div style={{ width: 55, height: 16, borderRadius: 6, background: 'var(--surface-3)' }} />
                </div>
              ) : data ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontWeight: 700, fontSize: 17, color: 'var(--text)' }}>{data.symbol}</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', height: 22, padding: '0 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: 'var(--surface-3)', color: 'var(--text-2)', border: '1px solid var(--border)' }}>
                    {data.sector}
                  </span>
                  {/* Show if user owns this stock */}
                  {position && (
                    <span style={{ display: 'inline-flex', alignItems: 'center', height: 22, padding: '0 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: 'var(--green-soft)', color: 'var(--green)', border: 'none' }}>
                      ✓ You own {position.qty}
                    </span>
                  )}
                </div>
              ) : <span style={{ fontWeight: 700, fontSize: 17, color: 'var(--text)' }}>{symbol}</span>}
              {data && <span style={{ fontSize: 12.5, color: 'var(--text-2)' }}>{data.name}</span>}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ width: 34, height: 34, borderRadius: 9, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-2)', display: 'grid', placeItems: 'center', cursor: 'pointer', flexShrink: 0, transition: 'background .15s' }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
          >
            <X size={18} />
          </button>
        </div>

        {/* ── Scrollable body ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 22, display: 'flex', flexDirection: 'column', gap: 22 }}>

          {/* Price section */}
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {loading
                ? <><div style={{ width: 160, height: 32, borderRadius: 7, background: 'var(--surface-3)' }} /><div style={{ width: 80, height: 14, borderRadius: 5, background: 'var(--surface-3)', marginTop: 6 }} /></>
                : data ? <>
                  <span style={{ fontFamily: 'var(--font-mono,monospace)', fontSize: 30, fontWeight: 700, letterSpacing: '-.02em', color: 'var(--text)' }}>{inr(data.price)}</span>
                  <span style={{ fontWeight: 600, fontSize: 14, color: data.change >= 0 ? 'var(--green)' : 'var(--red)', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                    {data.change >= 0 ? '↑' : '↓'} {Math.abs(data.change).toFixed(2)}%
                  </span>
                </> : null}
            </div>
            {data && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, height: 23, padding: '0 9px', borderRadius: 7, fontSize: 11.5, fontWeight: 700, color: sigCol, background: sigBg }}>
                  {sigArrow} {data.signal}
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-3)' }}>Updated {fmtAgo(data.updatedMin)}</span>
              </div>
            )}
          </div>

          {/* Chart */}
          {(loading || data) && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>Price · {chartRange}</span>
                <div style={{ display: 'inline-flex', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 9, padding: 3, gap: 1 }}>
                  {(['1D', '1W', '1M', '3M', '1Y'] as const).map(r => (
                    <button key={r} style={rangeBtn(r)} onClick={() => setChartRange(r)}>{r}</button>
                  ))}
                </div>
              </div>
              {loading
                ? <div style={{ height: 170, borderRadius: 9, background: 'var(--surface-3)' }} />
                : <AreaChart data={data!.spark} color={lineCol} h={170} />
              }
            </div>
          )}

          {/* Metric grid 3×2 */}
          <div>
            <h4 style={h4}>Key Metrics</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 1, background: 'var(--border)', border: '1px solid var(--border)', borderRadius: 9, overflow: 'hidden' }}>
              {loading
                ? Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} style={{ background: 'var(--surface)', padding: '12px 13px' }}>
                    <div style={{ width: '60%', height: 10, borderRadius: 4, background: 'var(--surface-3)', marginBottom: 6 }} />
                    <div style={{ width: '80%', height: 14, borderRadius: 4, background: 'var(--surface-3)' }} />
                  </div>
                ))
                : data ? [
                  ['Mkt Cap',  `₹${data.mcap.toLocaleString('en-IN')} Cr`],
                  ['P/E',      String(data.pe)],
                  ['Volume',   `${data.volume}M`],
                  ['52W High', inr(data.high52, 0)],
                  ['52W Low',  inr(data.low52,  0)],
                  ['Sentiment', signed(data.sentiment)],
                ].map(([label, value], i) => (
                  <div key={label} style={{ background: 'var(--surface)', padding: '12px 13px' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 3 }}>{label}</div>
                    <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: i === 5 ? (data.sentiment >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--text)' }}>{value}</div>
                  </div>
                ))
                : null}
            </div>
          </div>

          {/* Horizon breakdown */}
          <div>
            <h4 style={h4}>
              <span style={{ color: 'var(--text-3)', display: 'grid', placeItems: 'center' }}>
                <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>
              </span>
              Signal breakdown by horizon
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              {loading
                ? Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} style={{ display: 'grid', gridTemplateColumns: '42px 1fr auto', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 32, height: 12, borderRadius: 4, background: 'var(--surface-3)' }} />
                    <div style={{ height: 24, borderRadius: 7, background: 'var(--surface-3)' }} />
                    <div style={{ width: 34, height: 12, borderRadius: 4, background: 'var(--surface-3)' }} />
                  </div>
                ))
                : (data?.horizons ?? []).map(b => {
                  const bc = b.sig === 'BUY' ? 'var(--green)' : b.sig === 'SELL' ? 'var(--red)' : 'var(--gold)';
                  return (
                    <div key={b.h} style={{ display: 'grid', gridTemplateColumns: '42px 1fr auto', alignItems: 'center', gap: 12 }}>
                      <span style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 12.5, color: 'var(--text)' }}>{b.h}</span>
                      <div style={{ height: 24, borderRadius: 7, background: 'var(--surface-3)', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: b.conf + '%', borderRadius: 7, background: bc, display: 'flex', alignItems: 'center', paddingLeft: 9, fontSize: 11, fontWeight: 700, color: '#fff' }}>
                          {b.sig}
                        </div>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono,monospace)', fontSize: 12, fontWeight: 600, minWidth: 34, textAlign: 'right', color: 'var(--text)' }}>{b.conf}%</span>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* News */}
          <div>
            <h4 style={h4}>
              <span style={{ color: 'var(--text-3)', display: 'grid', placeItems: 'center' }}>
                <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h7M7 12h10M7 16h6"/></svg>
              </span>
              News &amp; sentiment
            </h4>
            {loading
              ? Array.from({ length: 3 }).map((_, i) => (
                <div key={i} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--surface-3)', marginTop: 6, flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ width: '90%', height: 12, borderRadius: 4, background: 'var(--surface-3)', marginBottom: 8 }} />
                    <div style={{ width: '45%', height: 10, borderRadius: 4, background: 'var(--surface-3)' }} />
                  </div>
                </div>
              ))
              : (data?.news ?? []).map((n, i) => {
                const [sentLabel, sentCol, sentBg] = SENT_TAG[n.sent] ?? SENT_TAG.neu;
                const dotCol = n.sent === 'pos' ? 'var(--green)' : n.sent === 'neg' ? 'var(--red)' : 'var(--text-3)';
                return (
                  <div key={i} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: dotCol, marginTop: 6, flexShrink: 0 }} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      <span style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: 'var(--text)' }}>{n.title}</span>
                      <div style={{ fontSize: 11, color: 'var(--text-3)', display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span>{n.src}</span><span>·</span><span>{n.time}</span>
                        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6, color: sentCol, background: sentBg }}>{sentLabel}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            }
          </div>
        </div>

        {/* ── Trade panel (always visible at bottom) ── */}
        {!loading && data && <TradePanel data={data} position={position} onClose={onClose} />}
        {loading && (
          <div style={{ padding: '16px 22px', borderTop: '1px solid var(--border)', flexShrink: 0, display: 'flex', gap: 10 }}>
            <div style={{ flex: 1, height: 40, borderRadius: 11, background: 'var(--surface-3)' }} />
            <div style={{ flex: 2, height: 40, borderRadius: 11, background: 'var(--surface-3)' }} />
          </div>
        )}
      </aside>
    </>
  );
}
