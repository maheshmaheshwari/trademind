import { useNavigate } from 'react-router-dom';
import { Bookmark, RefreshCw, ChevronRight } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { useToast } from '../components/ui';
import {
  useGetPortfolioSummaryQuery, useGetTodayPnlQuery, useGetActionableSignalsQuery,
  useGetOrdersQuery, useGetPositionsQuery, useGetMarketOverviewQuery, useRefreshSignalsMutation,
  useAddToWatchlistMutation,
} from '../services/tradeMindApiService';
import { Card, SignalBadge, Delta, Skeleton, SkeletonRows, SymbolCell, Conf, DateComponent } from '../components/ui';
import { AreaChart, Gauge, Sparkline } from '../components/Charts';
import type { Stock, IndexData, Breadth, Trade } from '../types';

function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}
function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function fmtAgo(m: number) { return m < 60 ? `${m}m ago` : `${Math.floor(m / 60)}h ago`; }

function StatCard({
  label, value, delta, deltaSuffix = '%', iconColor,
  icon, spark, sparkColor,
}: {
  label: string; value: string; delta?: number | null; deltaSuffix?: string;
  iconColor: string; icon: React.ReactNode; spark?: number[]; sparkColor?: string;
}) {
  return (
    <div className="bg-surface border border-line relative overflow-hidden" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))' }}>
      <div className="flex items-center justify-between">
        <span className="text-[12.5px] text-ink-2 font-medium">{label}</span>
        <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center" style={{ background: iconColor + '1f', color: iconColor }}>
          {icon}
        </span>
      </div>
      <div className="font-bold tracking-tight tabular-nums text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>
        {value}
      </div>
      {delta != null && (
        <span className="inline-flex items-center gap-1 text-[12.5px] font-semibold" style={{ color: delta >= 0 ? 'var(--green)' : 'var(--red)' }}>
          <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
            {delta >= 0
              ? <><path d="M7 17 17 7M8 7h9v9"/></>
              : <><path d="M12 5v14M6 13l6 6 6-6"/></>
            }
          </svg>
          {delta >= 0 ? '+' : ''}{delta}{deltaSuffix}
          <span className="text-ink-3 font-medium ml-[2px]">today</span>
        </span>
      )}
      {spark && (
        <div className="absolute right-0 bottom-0 left-0 h-[38px] opacity-55 pointer-events-none">
          <Sparkline data={spark} color={sparkColor || iconColor} w={300} h={38} />
        </div>
      )}
    </div>
  );
}

function SignalCard({ s, variant = 'rich', onClick }: { s: Stock; variant?: 'rich' | 'compact' | 'bold'; onClick: () => void }) {
  const col       = s?.signal === 'BUY' ? 'var(--green)' : s?.signal === 'SELL' ? 'var(--red)' : 'var(--gold)';
  const sparkColor = s?.signal === 'SELL' ? '#EF4444' : s?.signal === 'HOLD' ? '#F59E0B' : '#10B981';
  const expStr    = ((s?.expReturn ?? 0) >= 0 ? '+' : '') + (s?.expReturn ?? 0).toFixed(2) + '%';

  const base = "text-left font-sans text-ink bg-surface-2 border border-line cursor-pointer w-full transition-all hover:border-line-strong hover:bg-surface-hover hover:-translate-y-0.5";

  // ── compact: single row ──────────────────────────────────────────────────
  if (variant === 'compact') {
    return (
      <button onClick={onClick} className={`${base} grid grid-cols-[1.5fr_auto_1fr_90px] items-center gap-3.5 px-[15px] py-[11px]`}
        style={{ borderRadius: 'var(--radius,14px)' }}>
        <SymbolCell symbol={s?.symbol} name={s?.name} sector={s?.sector} showSector />
        <div className="flex items-center gap-1.5">
          <SignalBadge signal={s?.signal} />
          <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">{s?.horizon}</span>
        </div>
        <div className="flex flex-col items-end gap-[2px]">
          <span className="font-mono font-bold text-[13.5px]">{inr(s?.price ?? 0)}</span>
          <Delta value={s?.change ?? 0} size={11.5} />
        </div>
        <div style={{ width: 64 }}><Conf value={s?.confidence ?? 0} /></div>
      </button>
    );
  }

  // ── bold: left accent bar, large return ──────────────────────────────────
  if (variant === 'bold') {
    return (
      <button onClick={onClick} className={base}
        style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(15px * var(--u))', borderLeft: `3px solid ${col}` }}>
        <div className="flex items-center justify-between">
          <SymbolCell symbol={s?.symbol} name={s?.name} sector={s?.sector} showSector={false} />
          <SignalBadge signal={s?.signal} />
        </div>
        <div className="flex items-end justify-between mt-[14px]">
          <div className="flex flex-col gap-[1px]">
            <span className="text-[11px] text-ink-3 font-semibold uppercase tracking-[.04em]">Expected · {s?.horizon}</span>
            <span className="font-mono text-[24px] font-bold tracking-tight" style={{ color: col }}>{expStr}</span>
          </div>
          <Sparkline data={s?.spark ?? []} color={sparkColor} w={92} h={42} />
        </div>
        <div className="flex items-center justify-between mt-3">
          <span className="text-[11.5px] text-ink-2">Confidence</span>
          <span className="font-mono font-bold" style={{ color: col }}>{s?.confidence ?? 0}%</span>
        </div>
        <div className="conf-track mt-[5px]">
          <div style={{ width: (s?.confidence ?? 0) + '%', height: '100%', borderRadius: 999, background: col }} />
        </div>
      </button>
    );
  }

  // ── rich (default) ───────────────────────────────────────────────────────
  return (
    <button onClick={onClick} className={base}
      style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(15px * var(--u))' }}>
      <div className="flex items-center justify-between">
        <SymbolCell symbol={s?.symbol} name={s?.name} sector={s?.sector} showSector={false} />
        <SignalBadge signal={s?.signal} />
      </div>
      <div className="flex items-center justify-between my-[13px] mb-[11px]">
        <div className="flex flex-col gap-[1px]">
          <span className="font-mono text-[18px] font-bold">{inr(s?.price ?? 0)}</span>
          <Delta value={s?.change ?? 0} size={12} />
        </div>
        <Sparkline data={s?.spark ?? []} color={sparkColor} w={88} h={38} />
      </div>
      <div className="h-px bg-[var(--border)] mb-[11px]" />
      <div className="flex justify-between">
        <div className="flex flex-col gap-[2px]">
          <span className="text-[11px] text-ink-3">Horizon</span>
          <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">{s?.horizon}</span>
        </div>
        <div className="flex flex-col gap-[2px] items-end">
          <span className="text-[11px] text-ink-3">Exp. Return</span>
          <span className="font-mono font-bold text-[13.5px]" style={{ color: col }}>{expStr}</span>
        </div>
      </div>
      <div className="flex items-center justify-between mt-[11px] gap-[10px]">
        <span className="text-[11px] text-ink-3 whitespace-nowrap">Confidence</span>
        <div className="flex-1"><Conf value={s?.confidence ?? 0} /></div>
      </div>
    </button>
  );
}

export default function DashboardPage() {
  const { user } = useAuth();
  const { signalStyle } = useTheme();
  const navigate = useNavigate();
  const toast = useToast();

  const { data: portData,    isLoading: loadPort    } = useGetPortfolioSummaryQuery(user?.id ?? 0, { skip: !user });
  const { data: todayPnl,    isLoading: loadPnl     } = useGetTodayPnlQuery(user?.id ?? 0, { skip: !user });
  const { data: signalsData, isLoading: loadSignals } = useGetActionableSignalsQuery();
  const { data: ordersData,  isLoading: loadOrders  } = useGetOrdersQuery({ userId: user?.id ?? 0, size: 6 }, { skip: !user });
  const { data: posData,     isLoading: loadPos     } = useGetPositionsQuery({ userId: user?.id ?? 0, size: 1 }, { skip: !user });
  const { data: mktData,     isLoading: loadMkt     } = useGetMarketOverviewQuery();
  const [refreshSignals, { isLoading: refreshing }]   = useRefreshSignalsMutation();
  const [addToWatchlistMut] = useAddToWatchlistMutation();

  const loading = loadPort || loadPnl || loadSignals || loadOrders || loadPos || loadMkt;

  const signals: Stock[] = ((signalsData as any)?.data ?? []).map((t: any) => ({
    ...t,
    expReturn:    t.trade?.expected_return_pct ?? 0,
    price:        t.price?.current             ?? 0,
    horizon:      t.model?.horizon             ?? '',
    signal:       t.signal?.includes('BUY') ? 'BUY' : t.signal?.includes('SELL') ? 'SELL' : 'HOLD',
    confidence:   Math.round(t.confidence      ?? 0),
    change:       t.change                     ?? 0,
    spark:        t.spark                      ?? [],
    sector:       t.sector                     ?? '',
    target_price: t.trade?.target_price,
    stop_loss:    t.trade?.stop_loss,
  }));
  const trades: Trade[]  = (ordersData as any)?.data?.slice(0, 6) ?? [];
  const posCount: number = (posData as any)?.total ?? 0;
  const indices: IndexData[] = (mktData as any)?.indices ?? [];
  const breadth: Breadth | null = (mktData as any)?.breadth ?? null;
  const sentiment: number = (mktData as any)?.sentiment ?? 0;

  async function refresh() {
    try {
      await refreshSignals().unwrap();
      toast({ type: 'success', title: 'Signals refreshed', msg: '498 stocks re-scored' });
    } catch {
      toast({ type: 'error', title: 'Refresh failed' });
    }
  }

  async function handleSaveTopSignals() {
    if (!user) return;
    const top5 = (signals ?? []).slice(0, 5).map(s => s?.symbol).filter(Boolean) as string[];
    await Promise.all(top5.map(sym => addToWatchlistMut({ userId: user.id, symbol: sym })));
    toast({ type: 'success', title: 'Top 5 signals saved to your watchlist' });
  }

  const nifty    = indices?.[0];
  const winRate  = user ? ((user.win_count ?? 0) + (user.loss_count ?? 0) > 0 ? (((user.win_count ?? 0) / ((user.win_count ?? 0) + (user.loss_count ?? 0))) * 100).toFixed(1) : '0') : '0';
  const niftyLabels = (nifty?.spark ?? []).map((_, i) => {
    const totalMins = 9 * 60 + 15 + i * 5;
    const h = Math.floor(totalMins / 60);
    const m = totalMins % 60;
    return `${h}:${String(m).padStart(2, '0')}`;
  });

  // Dynamic greeting based on current hour
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  // Last run time from signals data
  const lastRunStr = (signalsData as any)?.generated_at
    ? fmtAgo(Math.round((Date.now() - new Date((signalsData as any).generated_at).getTime()) / 60000))
    : fmtAgo(8);

  // Today P&L % from API
  const todayPnlPct = (todayPnl as any)?.today_pnl_pct ?? null;

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>
            {greeting}, {user?.display_name?.split(' ')?.[0] || user?.username} 👋
          </h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">
            Your AI engine scanned <b className="tabular-nums">498</b> Nifty 500 stocks · last run {lastRunStr}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleSaveTopSignals}
            className="inline-flex items-center justify-center gap-2 h-10 px-4 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover"
          >
            <Bookmark size={17} /> Add to Watchlist
          </button>
          <button
            onClick={refresh}
            className="inline-flex items-center justify-center gap-2 h-10 px-4 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-transparent text-[#1A1206] transition-all hover:brightness-110"
            style={{ background: 'var(--gold)', boxShadow: '0 4px 14px rgba(245,158,11,.3)' }}
          >
            <RefreshCw size={17} style={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }} />
            Refresh Signals
          </button>
        </div>
      </div>

      {/* ── Stat cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 dgap">
        {loading ? Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u))' }}>
            <Skeleton h={12} w="50%" className="mb-3" /><Skeleton h={28} w="70%" className="mb-2" /><Skeleton h={11} w="40%" />
          </div>
        )) : (<>
          <StatCard label="Portfolio Value" value={inrCompact((portData as any)?.total_value ?? 0)} delta={todayPnlPct} iconColor="var(--accent)"
            icon={<svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="6" width="18" height="14" rx="3"/><path d="M3 10h18"/><circle cx="16.5" cy="14" r="1.3" fill="currentColor" stroke="none"/></svg>}
            spark={nifty?.spark} sparkColor="var(--accent)"
          />
          <StatCard label="Today's P&L" value={(todayPnl as any) ? ((todayPnl as any).today_pnl >= 0 ? '+' : '') + inr((todayPnl as any).today_pnl, 2).replace('₹', '') : '—'} delta={(todayPnl as any)?.today_pnl_pct ?? null} iconColor="var(--green)"
            icon={<svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></svg>}
            spark={indices?.[1]?.spark} sparkColor="var(--green)"
          />
          <StatCard label="Active Positions" value={String(posCount)} iconColor="var(--gold)"
            icon={<svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>}
          />
          <StatCard label="Win Rate" value={winRate + '%'} delta={2.3} iconColor="#8B5CF6"
            icon={<svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/></svg>}
          />
        </>)}
      </div>

      {/* ── NIFTY chart + Sentiment ── */}
      <div className="grid grid-cols-1 md:grid-cols-[1.7fr_1fr] dgap">
        <Card
          title="NIFTY 50" sub="NSE · Intraday" icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="M7 14l3-4 3 2 4-6"/></svg>}
          pad={false}
          right={nifty ? (
            <div className="flex flex-col items-end">
              <span className="font-mono text-[20px] font-bold text-ink">{nifty?.value?.toLocaleString('en-IN')}</span>
              <span className="text-[12.5px] font-semibold text-gain">+{nifty?.change} (+{nifty?.pct?.toFixed(2)}%)</span>
            </div>
          ) : undefined}
        >
          <div className="dp" style={{ paddingTop: 8 }}>
            {loading
              ? <Skeleton h={210} />
              : <AreaChart data={nifty?.spark ?? []} color="#10B981" h={210} labels={niftyLabels} />
            }
            <div className="flex gap-2 flex-wrap mt-3">
              {loading
                ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} h={32} w={110} rounded="9px" />)
                : (indices ?? []).map(ix => (
                  <div key={ix?.name} className="flex items-center gap-2 px-[11px] py-[7px] rounded-[9px] bg-surface-2 text-[12px]">
                    <span className="text-ink-2 font-semibold">{ix?.name}</span>
                    <span className="font-mono font-bold text-ink">{ix?.value?.toLocaleString('en-IN')}</span>
                    <span className="font-semibold tabular-nums" style={{ color: (ix.pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                      {((ix.pct ?? 0) >= 0 ? '+' : '') + (ix.pct ?? 0).toFixed(2) + '%'}
                    </span>
                  </div>
                ))
              }
            </div>
          </div>
        </Card>

        <Card title="Market Sentiment" sub="Driven by news + flows" icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5 3 3 0 0 0 2 4 3 3 0 0 0 5 1V4.5A2.5 2.5 0 0 0 9 4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 1 5 3 3 0 0 1-2 4 3 3 0 0 1-5 1"/></svg>}>
          <div className="dp flex flex-col items-center gap-[14px]">
            {loading ? <Skeleton w={190} h={140} rounded="12px" /> : <Gauge value={sentiment} size={200} />}
            {breadth && (
              <div className="flex justify-around w-full border-t border-line pt-[13px]">
                <div className="flex flex-col items-center gap-[1px]">
                  <span className="font-mono font-bold text-gain">{breadth.advances}</span>
                  <span className="text-[11px] text-ink-3">Advances</span>
                </div>
                <div className="flex flex-col items-center gap-[1px]">
                  <span className="font-mono font-bold text-loss">{breadth.declines}</span>
                  <span className="text-[11px] text-ink-3">Declines</span>
                </div>
                <div className="flex flex-col items-center gap-[1px]">
                  <span className="font-mono font-bold text-gold">{(breadth.advances / breadth.declines).toFixed(2)}</span>
                  <span className="text-[11px] text-ink-3">A/D Ratio</span>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* ── Top AI Signals ── */}
      <Card
        title="Top AI Signals Today"
        sub="Highest-confidence calls across Nifty 500"
        icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/></svg>}
        pad={false}
        right={
          <button onClick={() => navigate('/signals')} className="inline-flex items-center gap-[6px] h-8 px-[11px] rounded-[9px] font-sans text-[12.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover">
            View all 498 <ChevronRight size={15} />
          </button>
        }
      >
        <div className="dp">
          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} h={signalStyle === 'compact' ? 56 : 190} rounded="12px" />)}
            </div>
          ) : signalStyle === 'compact' ? (
            <div className="flex flex-col gap-2">
              {(signals ?? []).map(s => (
                <SignalCard key={s?.symbol} s={s} variant="compact" onClick={() => navigate(`/stocks/${encodeURIComponent(s?.symbol ?? '')}`)} />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {(signals ?? []).map(s => (
                <SignalCard key={s?.symbol} s={s} variant={signalStyle} onClick={() => navigate(`/stocks/${encodeURIComponent(s?.symbol ?? '')}`)} />
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* ── Recent Trades ── */}
      <Card
        title="Recent Trades"
        sub="Latest executions across your accounts"
        icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M7 8h13M17 5l3 3-3 3M17 16H4M7 13l-3 3 3 3"/></svg>}
        pad={false}
        right={
          <button onClick={() => navigate('/orders')} className="inline-flex items-center gap-[6px] h-8 px-[11px] rounded-[9px] font-sans text-[12.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover">
            All trades <ChevronRight size={15} />
          </button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                {['Symbol', 'Side', 'Price', 'Value', 'Realized P&L', 'Date'].map(h => (
                  <th key={h} className="text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 border-b border-line whitespace-nowrap sticky top-0 bg-surface z-[1]"
                    style={{ textAlign: h === 'Value' || h === 'Price' || h === 'Realized P&L' ? 'right' : 'left', padding: 'calc(11px * var(--u)) 14px' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? <SkeletonRows cols={6} rows={5} /> : trades.length === 0 ? (
                <tr><td colSpan={6} className="text-center text-ink-3 py-10 px-5">No recent trades.</td></tr>
              ) : (trades ?? []).map(t => (
                <tr key={t?.id} className="cursor-default transition-colors hover:bg-surface-2">
                  <td className="border-b border-line whitespace-nowrap" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                    <SymbolCell symbol={t?.symbol ?? ''} name={t?.name ?? ''} sector={t?.sector ?? ''} showSector={false} />
                  </td>
                  <td className="border-b border-line whitespace-nowrap" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                    <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold border border-line"
                      style={{ background: t?.order_type === 'BUY' ? 'var(--green-soft)' : 'var(--red-soft)', color: t?.order_type === 'BUY' ? 'var(--green)' : 'var(--red)' }}>
                      {t?.order_type}
                    </span>
                  </td>
                  <td className="border-b border-line whitespace-nowrap text-right font-mono tabular-nums" style={{ padding: 'calc(12px * var(--u)) 14px' }}>{inr(t?.price ?? 0)}</td>
                  <td className="border-b border-line whitespace-nowrap text-right font-mono tabular-nums" style={{ padding: 'calc(12px * var(--u)) 14px' }}>{inr(t?.value ?? 0, 0)}</td>
                  <td className="border-b border-line whitespace-nowrap text-right" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                    <Delta value={t?.pnl ?? 0} suffix="" showIcon size={12.5} />
                  </td>
                  <td className="border-b border-line whitespace-nowrap text-[12px] text-ink-3 font-mono" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                    <DateComponent inputDate={t?.created_at ?? ''} format="DD MMM" showTooltip={false} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
