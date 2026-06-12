import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createPortal } from 'react-dom';
import { ArrowLeft, Bookmark, AlertTriangle } from 'lucide-react';
import {
  useLazyGetStockDetailQuery,
  useGetPositionsQuery,
  useExecuteSignalMutation,
  useAddToWatchlistMutation,
  useGetStockHistoryQuery,
} from '../services/tradeMindApiService';
import { useAuth } from '../AuthContext';
import { useToast, symColor } from '../components/ui';
import { AreaChart } from '../components/Charts';
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
  suggested_qty_per_user?: number;
  consumed_volume?: number;
  recommended_volume?: number;
  remaining_volume?: number;
}

// ─── helpers ──────────────────────────────────────────────────────────────────

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

// ─── Partial capacity modal ────────────────────────────────────────────────────

function PartialCapacityModal({ symbol, requested, available, onConfirm, onCancel }: {
  symbol: string; requested: number; available: number;
  onConfirm: (qty: number) => void; onCancel: () => void;
}) {
  return createPortal(
    <>
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(3,6,15,.65)', backdropFilter: 'blur(3px)', zIndex: 9100 }} onClick={onCancel} />
      <div style={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', zIndex: 9101, width: 360, maxWidth: '92vw', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: '24px 22px', boxShadow: 'var(--shadow-lg)', display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 13 }}>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: 'rgba(245,158,11,.12)', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
            <AlertTriangle size={20} style={{ color: 'var(--gold)' }} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text)', marginBottom: 4 }}>Your profit may be lower</div>
            <div style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.55 }}>
              Many users are buying {symbol} right now. High demand can push the stock price up before your order fills.
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[['📈','Higher entry price','You may buy at a slightly higher price than shown.'],
            ['📉','Smaller profit margin','Since the target price stays the same, paying more upfront means less profit.']
          ].map(([icon, title, desc]) => (
            <div key={title} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 10, border: '1px solid var(--border)' }}>
              <span style={{ fontSize: 16, flexShrink: 0 }}>{icon}</span>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text)', marginBottom: 2 }}>{title}</div>
                <div style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.5 }}>{desc}</div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.55, padding: '10px 13px', background: 'rgba(245,158,11,.06)', borderRadius: 9, border: '1px solid rgba(245,158,11,.18)' }}>
          💡 <strong style={{ color: 'var(--text-2)' }}>Tip:</strong> Buying the suggested quantity reduces this risk.
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={onCancel} style={ghostBtn}>Change qty</button>
          <button onClick={() => onConfirm(available ?? 0)} style={{ flex: 2, ...primaryBtn('var(--accent)') }}>
            Proceed with {available?.toLocaleString('en-IN') || 0} shares
          </button>
        </div>
      </div>
    </>,
    document.body
  );
}

// ─── Capacity meter ────────────────────────────────────────────────────────────

function CapacityMeter({ consumed, total, suggested }: { consumed: number; total: number; suggested: number }) {
  if (!total) return null;
  const pct      = Math.min(100, Math.round((consumed / total) * 100));
  const barColor = pct >= 80 ? 'var(--red)' : pct >= 50 ? 'var(--gold)' : 'var(--green)';
  const remaining = Math.max(0, total - consumed);
  return (
    <div style={{ padding: '11px 13px', background: 'var(--surface-2)', borderRadius: 11, border: '1px solid var(--border)', marginBottom: 13 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
        <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)' }}>Platform capacity</span>
        <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono,monospace)' }}>
          {remaining?.toLocaleString('en-IN') || 0} left · {pct}% used
        </span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden', marginBottom: 8 }}>
        <div style={{ height: '100%', width: `${pct}%`, borderRadius: 99, background: barColor, transition: 'width .4s ease' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Suggested for you</span>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--accent-2)', fontFamily: 'var(--font-mono,monospace)' }}>
          {suggested?.toLocaleString('en-IN') || 0} shares
        </span>
      </div>
    </div>
  );
}

// ─── Trade panel ──────────────────────────────────────────────────────────────

function TradePanel({ data, position }: { data: StockDetail; position: OpenPosition | null }) {
  const { user }  = useAuth();
  const toast     = useToast();
  const [qty, setQty]         = useState('');
  const [partialModal, setPartialModal] = useState<{ requested: number; available: number } | null>(null);
  const [executeSignalMut, { isLoading: busy }] = useExecuteSignalMutation();
  const [addToWatchlistMut] = useAddToWatchlistMutation();

  const balance    = user?.virtual_balance ?? 0;
  const price      = data?.price ?? 0;
  const maxBuy     = price > 0 ? Math.floor(balance / price) : 0;
  const qtyNum     = Math.max(0, parseInt(qty) || 0);
  const totalCost  = qtyNum * price;
  const suggestedQty = data?.suggested_qty_per_user ?? 0;
  const estSL      = +(price * 0.93).toFixed(2);
  const estTarget  = +(price * 1.12).toFixed(2);

  useEffect(() => {
    if (suggestedQty > 0 && qty === '') setQty(String(suggestedQty));
  }, [suggestedQty]);

  const consumed    = data?.consumed_volume ?? 0;
  const recommended = data?.recommended_volume ?? 0;
  const remaining   = data?.remaining_volume ?? 0;
  const showCapacity = recommended > 0;
  const insufficient = qtyNum > 0 && totalCost > balance;

  const handleAddToWatchlist = async () => {
    try {
      await addToWatchlistMut(data?.symbol ?? '').unwrap();
      toast({ type: 'success', title: `${data?.symbol ?? ''} added to watchlist` });
    } catch {
      toast({ type: 'error', title: 'Failed to add to watchlist' });
    }
  };

  async function executeBuy(overrideQty?: number) {
    const finalQty = overrideQty ?? qtyNum;
    if (!user || finalQty <= 0) return;
    const investment = finalQty * price;
    try {
      await executeSignalMut({ user_id: user.id, symbol: data?.symbol ?? '', name: data?.name ?? '', investment_amount: investment, buy_price: price, target_price: estTarget, stop_loss: estSL, signal: data?.signal ?? 'HOLD', confidence: data?.confidence ?? 0, mode: 'PAPER' }).unwrap();
      toast({ type: 'success', title: `Bought ${finalQty} × ${data.symbol}`, msg: `Invested: ${inr(investment)}` });
      setPartialModal(null);
    } catch (e: unknown) {
      const detail = (e as any)?.data?.detail;
      if (detail?.error === 'PARTIAL_CAPACITY') {
        setPartialModal({ requested: detail.requested, available: detail.available });
        return;
      }
      toast({ type: 'error', title: 'Order failed', msg: detail?.message ?? (e instanceof Error ? e.message : 'Try again') });
    }
  }

  // SELL — user doesn't own
  if (data?.signal === 'SELL' && !position) {
    return (
      <div style={{ padding: '18px', background: 'var(--surface-2)', borderRadius: 14, border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', gap: 10, padding: '12px 13px', background: 'rgba(245,158,11,.08)', border: '1px solid rgba(245,158,11,.25)', borderRadius: 11, marginBottom: 12 }}>
          <AlertTriangle size={18} style={{ color: 'var(--gold)', flexShrink: 0, marginTop: 1 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>You don't hold {data?.symbol ?? ''}</span>
            <span style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>SELL signals indicate potential downside. TradeMind doesn't support short selling.</span>
          </div>
        </div>
        <button onClick={handleAddToWatchlist} style={{ ...ghostBtn, width: '100%' }}>
          <Bookmark size={15} /> Add to Watchlist
        </button>
      </div>
    );
  }

  // SELL — user owns
  if (data?.signal === 'SELL' && position) {
    const maxSell = position?.quantity ?? 0;
    const sellQty = Math.min(qtyNum, maxSell);
    const proceeds = sellQty * price;
    const pnlValue = (price - (position?.avg_buy_price ?? 0)) * sellQty;
    const pnlPct   = ((price - (position?.avg_buy_price ?? 0)) / (position?.avg_buy_price || 1)) * 100;

    async function executeSell() {
      if (!user || sellQty <= 0) return;
      try {
        await executeSignalMut({ user_id: user.id, symbol: data?.symbol ?? '', name: data?.name ?? '', investment_amount: proceeds, buy_price: position?.avg_buy_price ?? 0, target_price: position?.target_price ?? 0, stop_loss: position?.stop_loss ?? 0, signal: 'SELL', mode: 'PAPER' }).unwrap();
        toast({ type: pnlValue >= 0 ? 'success' : 'info', title: `Sold ${sellQty} × ${data.symbol}`, msg: `P&L: ${signed(pnlValue, 0)} (${signed(pnlPct, 2)}%)` });
      } catch (e: unknown) { toast({ type: 'error', title: 'Order failed', msg: e instanceof Error ? e.message : 'Try again' }); }
    }

    return (
      <div style={{ padding: '18px', background: 'var(--surface-2)', borderRadius: 14, border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '12px', background: 'var(--surface)', borderRadius: 10 }}>
          {[['You own', `${position?.quantity ?? 0} sh`], ['Avg cost', inr(position?.avg_buy_price ?? 0)], ['P&L', signed(position?.unrealized_pnl ?? 0, 0)]].map(([l, v], i) => (
            <div key={l}><div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>{l}</div>
              <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 13, color: i === 2 ? ((position?.unrealized_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--text)' }}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>Qty to sell</label>
            <input type="number" min={1} max={maxSell} value={qty} onChange={e => setQty(e.target.value)} placeholder={`1–${maxSell}`} style={{ ...inputS, borderColor: qtyNum > maxSell ? 'var(--red)' : 'var(--border)' }} onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'} onBlur={e => e.currentTarget.style.borderColor = qtyNum > maxSell ? 'var(--red)' : 'var(--border)'} />
          </div>
          <button style={{ ...allBtn, alignSelf: 'flex-end' }} onClick={() => setQty(String(maxSell))}>All</button>
        </div>
        {qtyNum > 0 && qtyNum <= maxSell && (
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 12px', background: pnlValue >= 0 ? 'var(--green-soft)' : 'var(--red-soft)', borderRadius: 10 }}>
            <div><div style={{ fontSize: 11, color: 'var(--text-3)' }}>Proceeds</div><div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14 }}>{inr(proceeds)}</div></div>
            <div style={{ textAlign: 'right' }}><div style={{ fontSize: 11, color: 'var(--text-3)' }}>Realised P&L</div><div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: pnlValue >= 0 ? 'var(--green)' : 'var(--red)' }}>{signed(pnlValue, 0)}</div></div>
          </div>
        )}
        <button disabled={busy || qtyNum <= 0 || qtyNum > maxSell} onClick={executeSell} style={{ ...primaryBtn('#EF4444'), opacity: (busy || qtyNum <= 0 || qtyNum > maxSell) ? 0.5 : 1 }}>
          {busy ? '…' : `Sell ${qtyNum > 0 ? qtyNum : ''} ${data?.symbol ?? ''}`}
        </button>
      </div>
    );
  }

  // BUY / HOLD
  return (
    <div style={{ padding: '18px', background: 'var(--surface-2)', borderRadius: 14, border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 0 }}>
      {showCapacity && <CapacityMeter consumed={consumed} total={recommended} suggested={suggestedQty} />}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '12px', background: 'var(--surface)', borderRadius: 10, marginBottom: 12 }}>
        {[['Available', inrCompact(balance)], ['Est. SL', `${inr(estSL)} -7%`], ['Est. Target', `${inr(estTarget)} +12%`]].map(([l, v], i) => (
          <div key={l}><div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 2 }}>{l}</div>
            <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 12, color: i === 1 ? 'var(--red)' : i === 2 ? 'var(--green)' : 'var(--text)' }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', display: 'block', marginBottom: 5 }}>
            Quantity <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>@ {inr(price)}</span>
            {showCapacity && suggestedQty > 0 && <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--accent-2)', fontWeight: 500 }}>· suggested: {suggestedQty?.toLocaleString('en-IN') || 0}</span>}
          </label>
          <input type="number" min={1} value={qty} onChange={e => setQty(e.target.value)} placeholder="Enter qty" style={{ ...inputS, borderColor: insufficient ? 'var(--red)' : 'var(--border)' }} onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'} onBlur={e => e.currentTarget.style.borderColor = insufficient ? 'var(--red)' : 'var(--border)'} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignSelf: 'flex-end' }}>
          <button style={allBtn} onClick={() => setQty(String(maxBuy))}>Max ({maxBuy})</button>
          {showCapacity && suggestedQty > 0 && <button style={{ ...allBtn, color: 'var(--accent-2)', borderColor: 'var(--accent)' }} onClick={() => setQty(String(suggestedQty))}>Suggested</button>}
        </div>
      </div>

      {insufficient && <p style={{ fontSize: 12, color: 'var(--red)', margin: '0 0 8px' }}>Total cost {inr(totalCost)} exceeds balance {inrCompact(balance)}.</p>}
      {showCapacity && qtyNum > remaining && remaining > 0 && !insufficient && (
        <p style={{ fontSize: 12, color: 'var(--gold)', margin: '0 0 8px', display: 'flex', alignItems: 'center', gap: 5 }}>
          <AlertTriangle size={13} style={{ flexShrink: 0 }} /> Only {remaining?.toLocaleString('en-IN') || 0} shares of capacity left.
        </p>
      )}

      {qtyNum > 0 && !insufficient && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'var(--accent-soft)', borderRadius: 10, marginBottom: 12 }}>
          <span style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>Total investment</span>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 15, color: 'var(--accent-2)' }}>{inr(totalCost)}</span>
            <span style={{ fontSize: 11.5, color: 'var(--text-3)', display: 'block' }}>Balance after: {inrCompact(balance - totalCost)}</span>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 10 }}>
        <button onClick={handleAddToWatchlist} style={ghostBtn}><Bookmark size={15} /> Watchlist</button>
        <button disabled={busy || qtyNum <= 0 || insufficient || maxBuy === 0} onClick={() => executeBuy()} style={{ flex: 2, ...primaryBtn('var(--accent)'), opacity: (busy || qtyNum <= 0 || insufficient || maxBuy === 0) ? 0.5 : 1 }}>
          {busy ? 'Placing…' : `Buy ${qtyNum > 0 ? qtyNum : ''} ${data?.symbol ?? ''}`}
        </button>
      </div>

      {partialModal && (
        <PartialCapacityModal symbol={data?.symbol ?? ''} requested={partialModal.requested} available={partialModal.available} onConfirm={(qty) => executeBuy(qty)} onCancel={() => setPartialModal(null)} />
      )}
    </div>
  );
}

// ─── Shared button styles ──────────────────────────────────────────────────────

const ghostBtn: React.CSSProperties = {
  flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
  height: 40, borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600,
  cursor: 'pointer', border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)',
};
function primaryBtn(bg: string): React.CSSProperties {
  return { flex: 2, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6, height: 40, borderRadius: 11, fontFamily: 'inherit', fontSize: 13.5, fontWeight: 600, border: 'none', background: bg, color: '#fff', boxShadow: bg === 'var(--accent)' ? '0 4px 14px rgba(59,130,246,.32)' : '0 4px 14px rgba(239,68,68,.32)', cursor: 'pointer' };
}
const inputS: React.CSSProperties = {
  height: 40, padding: '0 12px', borderRadius: 10, border: '1px solid var(--border)', background: 'var(--surface)',
  color: 'var(--text)', fontFamily: 'var(--font-mono,monospace)', fontSize: 14, outline: 'none', width: '100%', boxSizing: 'border-box', transition: 'border .15s',
};
const allBtn: React.CSSProperties = {
  height: 40, padding: '0 12px', borderRadius: 10, whiteSpace: 'nowrap', border: '1px solid var(--border)',
  background: 'var(--surface)', color: 'var(--text-2)', fontFamily: 'inherit', fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
};

// ─── StockPage ─────────────────────────────────────────────────────────────────

export default function StockPage() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const navigate = useNavigate();
  const { user }  = useAuth();
  const [chartRange, setChartRange] = useState<'1D' | '1W' | '1M' | '3M' | '1Y'>('1M');

  const [fetchDetail, { data: detailRes, isLoading: loadDetail }] = useLazyGetStockDetailQuery();
  const { data: posRes } = useGetPositionsQuery({ userId: user!.id, size: 100 }, { skip: !user || !symbol });
  const { data: histRes, isFetching: loadHist } = useGetStockHistoryQuery({ symbol, range: chartRange }, { skip: !symbol });

  useEffect(() => {
    if (symbol) fetchDetail(symbol);
  }, [symbol, fetchDetail]);

  const data: StockDetail | null = (detailRes as any)?.data ?? null;
  const position: OpenPosition | null = ((posRes as any)?.data ?? []).find((p: OpenPosition) => p?.symbol === symbol) ?? null;

  const histPrices    = histRes?.prices ?? [];
  const histLabels    = histRes?.labels ?? [];
  const histChangePct = histRes?.change_pct ?? 0;
  const chartPrices   = histPrices.length > 0 ? histPrices : (data?.spark ?? []);
  const chartLabels   = histPrices.length > 0 ? histLabels : undefined;
  const chartColor    = histChangePct >= 0 ? '#10B981' : '#EF4444';

  const color    = data ? symColor(data?.symbol ?? '') : '#3B82F6';
  const sigCol   = data?.signal === 'BUY' ? 'var(--green)' : data?.signal === 'SELL' ? 'var(--red)' : 'var(--gold)';
  const sigBg    = data?.signal === 'BUY' ? 'var(--green-soft)' : data?.signal === 'SELL' ? 'var(--red-soft)' : 'var(--gold-soft)';

  const rangeBtn = (r: typeof chartRange): React.CSSProperties => ({
    border: 'none', background: chartRange === r ? 'var(--surface)' : 'transparent',
    color: chartRange === r ? 'var(--text)' : 'var(--text-2)',
    fontFamily: 'inherit', fontSize: 12.5, fontWeight: 600,
    padding: '4px 9px', borderRadius: 6, cursor: 'pointer',
    boxShadow: chartRange === r ? 'var(--shadow-sm)' : 'none', transition: 'background .14s, color .14s',
  });

  const sectionTitle: React.CSSProperties = { margin: '0 0 12px', fontSize: 13.5, fontWeight: 600, color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 8 };

  return (
    <div style={{ padding: '0 0 40px' }}>

      {/* ── Back nav + header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24, paddingTop: 4 }}>
        <button
          onClick={() => navigate(-1)}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 6, height: 36, padding: '0 14px', borderRadius: 9, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-2)', fontFamily: 'inherit', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}
        >
          <ArrowLeft size={15} /> Back
        </button>

        {/* Logo + name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
          <span style={{ width: 40, height: 40, borderRadius: 10, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 14, background: color + '22', color, flexShrink: 0 }}>
            {symbol.replace('.NS', '').slice(0, 2)}
          </span>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 18, color: 'var(--text)' }}>{data?.symbol ?? symbol}</span>
              {data?.sector && (
                <span style={{ display: 'inline-flex', alignItems: 'center', height: 20, padding: '0 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: 'var(--surface-3)', color: 'var(--text-2)', border: '1px solid var(--border)' }}>
                  {data.sector}
                </span>
              )}
              {position && (
                <span style={{ display: 'inline-flex', alignItems: 'center', height: 20, padding: '0 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: 'var(--green-soft)', color: 'var(--green)' }}>
                  ✓ You own {position?.quantity ?? 0}
                </span>
              )}
            </div>
            {data?.name && <span style={{ fontSize: 12.5, color: 'var(--text-2)' }}>{data.name}</span>}
          </div>
        </div>

        {/* Price + signal */}
        {data && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
            <span style={{ fontFamily: 'var(--font-mono,monospace)', fontSize: 26, fontWeight: 700, letterSpacing: '-.02em', color: 'var(--text)' }}>{inr(data?.price ?? 0)}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 13, color: (data?.change ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(data?.change ?? 0) >= 0 ? '↑' : '↓'} {Math.abs(data?.change ?? 0).toFixed(2)}%
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, height: 22, padding: '0 9px', borderRadius: 7, fontSize: 11.5, fontWeight: 700, color: sigCol, background: sigBg }}>
                {data?.signal === 'BUY' ? '↑' : data?.signal === 'SELL' ? '↓' : '●'} {data.signal}
              </span>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {loadDetail && !data && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
            <div style={{ width: 120, height: 28, borderRadius: 7, background: 'var(--surface-3)' }} />
            <div style={{ width: 80, height: 14, borderRadius: 5, background: 'var(--surface-3)' }} />
          </div>
        )}
      </div>

      {/* ── Two-column layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start">

        {/* ── Left column ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

          {/* Chart */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: '20px 20px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>Price</span>
                {!loadHist && histPrices.length > 0 && (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, height: 20, padding: '0 7px', borderRadius: 6, fontSize: 11.5, fontWeight: 700, color: histChangePct >= 0 ? 'var(--green)' : 'var(--red)', background: histChangePct >= 0 ? 'var(--green-soft)' : 'var(--red-soft)' }}>
                    {histChangePct >= 0 ? '▲' : '▼'} {Math.abs(histChangePct).toFixed(2)}%
                  </span>
                )}
              </div>
              <div style={{ display: 'inline-flex', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 9, padding: 3, gap: 1 }}>
                {(['1D', '1W', '1M', '3M', '1Y'] as const).map(r => (
                  <button key={r} style={rangeBtn(r)} onClick={() => setChartRange(r)}>{r}</button>
                ))}
              </div>
            </div>
            {(loadDetail || loadHist)
              ? <div style={{ height: 240, borderRadius: 9, background: 'var(--surface-3)' }} />
              : <AreaChart data={chartPrices} labels={chartLabels} color={chartColor} h={240} currency />
            }
          </div>

          {/* Key metrics */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: 20 }}>
            <h4 style={sectionTitle}>Key Metrics</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 1, background: 'var(--border)', border: '1px solid var(--border)', borderRadius: 9, overflow: 'hidden' }}>
              {loadDetail
                ? Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} style={{ background: 'var(--surface)', padding: '13px 14px' }}>
                    <div style={{ width: '60%', height: 10, borderRadius: 4, background: 'var(--surface-3)', marginBottom: 6 }} />
                    <div style={{ width: '80%', height: 14, borderRadius: 4, background: 'var(--surface-3)' }} />
                  </div>
                ))
                : data ? [
                  ['Mkt Cap',   `₹${(data?.mcap ?? 0).toLocaleString('en-IN')} Cr`],
                  ['P/E',       String(data?.pe ?? 0)],
                  ['Volume',    `${data?.volume ?? 0}M`],
                  ['52W High',  inr(data?.high52 ?? 0, 0)],
                  ['52W Low',   inr(data?.low52  ?? 0, 0)],
                  ['Sentiment', signed(data?.sentiment ?? 0)],
                ].map(([label, value], i) => (
                  <div key={label} style={{ background: 'var(--surface)', padding: '13px 14px' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 3 }}>{label}</div>
                    <div style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 14, color: i === 5 ? ((data?.sentiment ?? 0) >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--text)' }}>{value}</div>
                  </div>
                ))
                : null}
            </div>
          </div>

          {/* Signal horizons */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: 20 }}>
            <h4 style={sectionTitle}>
              <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-3)' }}><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>
              Signal breakdown by horizon
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              {loadDetail
                ? Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} style={{ display: 'grid', gridTemplateColumns: '48px 1fr auto', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 36, height: 12, borderRadius: 4, background: 'var(--surface-3)' }} />
                    <div style={{ height: 24, borderRadius: 7, background: 'var(--surface-3)' }} />
                    <div style={{ width: 36, height: 12, borderRadius: 4, background: 'var(--surface-3)' }} />
                  </div>
                ))
                : (data?.horizons ?? []).map(b => {
                  const bc = b?.sig === 'BUY' ? 'var(--green)' : b?.sig === 'SELL' ? 'var(--red)' : 'var(--gold)';
                  return (
                    <div key={b?.h} style={{ display: 'grid', gridTemplateColumns: '48px 1fr auto', alignItems: 'center', gap: 12 }}>
                      <span style={{ fontFamily: 'var(--font-mono,monospace)', fontWeight: 700, fontSize: 12.5, color: 'var(--text)' }}>{b?.h}</span>
                      <div style={{ height: 24, borderRadius: 7, background: 'var(--surface-3)', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: (b?.conf ?? 0) + '%', borderRadius: 7, background: bc, display: 'flex', alignItems: 'center', paddingLeft: 9, fontSize: 11, fontWeight: 700, color: '#fff' }}>
                          {b?.sig}
                        </div>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono,monospace)', fontSize: 12, fontWeight: 600, minWidth: 36, textAlign: 'right', color: 'var(--text)' }}>{b?.conf ?? 0}%</span>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* News */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: 20 }}>
            <h4 style={sectionTitle}>
              <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-3)' }}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h7M7 12h10M7 16h6"/></svg>
              News &amp; sentiment
            </h4>
            {loadDetail
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
                const [sentLabel, sentCol, sentBg] = SENT_TAG[n?.sent ?? ''] ?? SENT_TAG.neu;
                return (
                  <div key={i} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: n?.sent === 'pos' ? 'var(--green)' : n?.sent === 'neg' ? 'var(--red)' : 'var(--text-3)', marginTop: 6, flexShrink: 0 }} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      <span style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4, color: 'var(--text)' }}>{n?.title}</span>
                      <div style={{ fontSize: 11, color: 'var(--text-3)', display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span>{n?.src}</span><span>·</span><span>{n?.time}</span>
                        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6, color: sentCol, background: sentBg }}>{sentLabel}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            }
          </div>
        </div>

        {/* ── Right column: sticky trade panel ── */}
        <div style={{ position: 'sticky', top: 80 }}>
          {data && <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 10 }}>Updated {fmtAgo(data?.updatedMin ?? 0)}</div>}
          {data ? <TradePanel data={data} position={position} /> : (
            <div style={{ height: 300, borderRadius: 14, background: 'var(--surface-3)' }} />
          )}
        </div>
      </div>
    </div>
  );
}
