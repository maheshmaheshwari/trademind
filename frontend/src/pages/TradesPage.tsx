import { useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, RefreshCw, BrainCircuit } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useToast } from '../components/ui';
import {
  useGetPositionsQuery, useGetTradesQuery, useSquareOffMutation,
  useGetGTTOrdersQuery, useSyncGTTMutation, useGetUserSignalHistoryQuery,
  useAuthorizeTradeAutoMutation, useGetAuthorizedTradesQuery,
} from '../services/tradeMindApiService';
import { Card, SymbolCell, SignalBadge, DataTable, priceColumns, type DataTableColumn } from '../components/ui';

import type { OpenPosition, TradeRow, GTTOrder } from '../types';

function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

type Tab = 'open' | 'history' | 'gtt' | 'ai_signals';
type DateRange = 'All' | '7D' | '30D' | '90D';
type SideFlt = 'All' | 'Open' | 'Closed';   // a TRADE is open or closed; BUY/SELL described a leg
const PER_PAGE = 18;

/** Trade state -> [label, text, background]. Same words the autopilot screen
 *  shows, so one trade cannot read "EXECUTED" on one page and "Running" on another. */
function tradeStatusMeta(status?: string): [string, string, string] {
  switch (status) {
    case 'OPEN':       return ['Running',    'var(--accent-2)', 'var(--accent-soft)'];
    case 'TARGET_HIT': return ['Target hit', 'var(--green)',    'var(--green-soft)'];
    case 'STOPPED':    return ['Stopped',    'var(--red)',      'var(--red-soft)'];
    default:           return ['Closed',     'var(--text-2)',   'var(--surface-3)'];
  }
}

function Pill({ color, bg, children }: { color: string; bg: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold border border-transparent" style={{ color, background: bg }}>
      {children}
    </span>
  );
}

export default function TradesPage() {
  const { user } = useAuth();
  const toast    = useToast();

  const [tab,       setTab]       = useState<Tab>('open');
  const [closed,    setClosed]    = useState<Set<string>>(new Set());
  const [dateRange, setDateRange] = useState<DateRange>('All');
  const [sideFlt,   setSideFlt]   = useState<SideFlt>('All');
  const navigate = useNavigate();

  const { data: posRes,    isLoading: loadPos, isFetching: fetchPos, isError: errPos  } = useGetPositionsQuery({ userId: user?.id ?? 0, size: 100 }, { skip: !user });
  const { data: trdRes,    isLoading: loadTrd, isFetching: fetchTrd, isError: errTrd } = useGetTradesQuery({ userId: user?.id ?? 0 }, { skip: !user });
  const { data: gttRes,    isLoading: loadGtt, isFetching: fetchGtt  } = useGetGTTOrdersQuery(user?.id ?? 0, { skip: !user });
  const { data: sigHist,   isLoading: loadSig, isFetching: fetchSig  } = useGetUserSignalHistoryQuery({ userId: user?.id ?? 0 }, { skip: !user });
  const [squareOff]                               = useSquareOffMutation();
  const [syncGTT, { isLoading: syncing }]         = useSyncGTTMutation();
  const [authorizeTradeAuto]                      = useAuthorizeTradeAutoMutation();
  const { data: authTradesRes }                   = useGetAuthorizedTradesQuery({ userId: user?.id ?? 0 }, { skip: !user });
  const [autopilotingSymbols, setAutopilotingSymbols] = useState<Set<string>>(new Set());
  const autopilotSymbolsRef = useRef<Set<string>>(new Set());

  // Symbols already managed by autopilot (PENDING or OPEN)
  const autopilotSymbolSet = new Set(
    ((authTradesRes as any)?.data ?? [])
      .filter((t: any) => t?.status === 'PENDING' || t?.status === 'OPEN')
      .map((t: any) => t?.symbol ?? '')
  );

  const loading   = loadPos || loadTrd || loadGtt;
  const positions: OpenPosition[] = (posRes as any)?.data ?? [];
  // History reads TRADES (one row per bracket). Reading `orders` here showed a
  // single trade as 3 rows, two of which were resting instructions that never
  // executed — 14 of user 2's 25 'trades' had never happened.
  const tradeRows: TradeRow[]     = (trdRes as any)?.data  ?? [];
  const gttOrders: GTTOrder[]     = (gttRes as any)?.data  ?? [];

  // Audit H12 — per-symbol in-flight guard, set synchronously (independent
  // of React's render cycle) so a fast double-click on the same row's close
  // button can't fire two square-off requests before the row re-renders.
  const closingSymbolsRef = useRef<Set<string>>(new Set());

  async function closePos(p: OpenPosition) {
    if (!user || closingSymbolsRef.current.has(p.symbol)) return;
    closingSymbolsRef.current.add(p.symbol);
    try {
      const res = await squareOff({ userId: user.id, symbol: p.symbol }).unwrap();
      setClosed(s => new Set([...s, p.symbol]));
      toast({ type: (p.unrealized_pnl ?? 0) >= 0 ? 'success' : 'info', title: `Closed ${p.symbol}`, msg: `Realized ${((p.unrealized_pnl ?? 0) >= 0 ? '+' : '') + Number((res as any)?.pnl ?? p.unrealized_pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })} (${((p.unrealized_pnl_pct ?? 0) >= 0 ? '+' : '') + (p.unrealized_pnl_pct ?? 0).toFixed(2)}%)` });
    } catch (e: unknown) { toast({ type: 'error', title: 'Close failed', msg: e instanceof Error ? e.message : 'Try again' }); }
    finally { closingSymbolsRef.current.delete(p.symbol); }
  }

  async function handleSync() {
    try { await syncGTT().unwrap(); toast({ type: 'info', title: 'Synced with Angel One', msg: 'GTT rules up to date' }); }
    catch { toast({ type: 'error', title: 'Sync failed' }); }
  }

  async function addToAutopilot(p: OpenPosition) {
    if (!user || autopilotSymbolsRef.current.has(p?.symbol ?? '')) return;
    autopilotSymbolsRef.current.add(p?.symbol ?? '');
    setAutopilotingSymbols(s => new Set([...s, p?.symbol ?? '']));
    try {
      await authorizeTradeAuto({
        user_id:    user.id,
        symbol:     p?.symbol ?? '',
        name:       p?.name ?? '',
        signal:     'BUY',
        mode:       (p as any)?.mode ?? 'PAPER',
        qty:        p?.quantity ?? 0,
        amount:     (p?.avg_buy_price ?? 0) * (p?.quantity ?? 0),
        entry:      p?.avg_buy_price ?? 0,
        target:     p?.target_price ?? 0,
        sl:         p?.stop_loss ?? 0,
        cmp:        p?.current_price ?? null,
        bracket_id: (p as any)?.bracket_id ?? undefined,
        exp_profit: p?.target_price && p?.avg_buy_price && p?.quantity
          ? Math.round((p.target_price - p.avg_buy_price) * p.quantity)
          : 0,
        max_loss:   p?.stop_loss && p?.avg_buy_price && p?.quantity
          ? Math.round((p.avg_buy_price - p.stop_loss) * p.quantity)
          : 0,
      }).unwrap();
      toast({ type: 'success', title: 'Added to Autopilot', msg: `${p?.symbol ?? ''} is now AI-managed` });
    } catch (e: unknown) {
      toast({ type: 'error', title: 'Autopilot failed', msg: e instanceof Error ? e.message : 'Try again' });
    } finally {
      autopilotSymbolsRef.current.delete(p?.symbol ?? '');
      setAutopilotingSymbols(s => { const n = new Set(s); n.delete(p?.symbol ?? ''); return n; });
    }
  }

  const today = new Date();
  const histFiltered = ([...(tradeRows ?? [])])
    .filter(t => {
      const stamp = t?.exit_at ?? t?.entry_at ?? '';
      const days = (today.getTime() - new Date(stamp).getTime()) / 86400000;
      const rangeOk = dateRange === 'All' || (dateRange === '7D' && days <= 7) || (dateRange === '30D' && days <= 30) || (dateRange === '90D' && days <= 90);
      // A trade is open or closed — "BUY/SELL" was a property of a leg, so the
      // side filter now selects by whether the trade is still running.
      const sideOk = sideFlt === 'All'
        || (sideFlt === 'Open'   && t?.status === 'OPEN')
        || (sideFlt === 'Closed' && t?.status !== 'OPEN');
      return rangeOk && sideOk;
    });  // sorting moved into DataTable


  const openPos    = (positions ?? []).filter(p => !closed.has(p?.symbol ?? ''));

  const posCols = useMemo<DataTableColumn<OpenPosition>[]>(() => [
    { id: 'symbol', header: 'Symbol',
      cell: p => (
        <span className="cursor-pointer" onClick={() => navigate(`/stocks/${encodeURIComponent(p?.symbol ?? '')}`)}>
          <SymbolCell symbol={p?.symbol ?? ''} name={p?.name ?? ''} sector={p?.sector ?? ''} />
        </span>
      ) },
    { id: 'quantity', header: 'Qty', align: 'right', mono: true,
      cell: p => <span className="text-ink-2">{p?.quantity ?? '\u2014'}</span> },
    // An open position is by definition unsold, so the Sell column always shows
    // the target as a projection here.
    ...priceColumns<OpenPosition>({
      entry:    p => p?.avg_buy_price,
      current:  p => p?.current_price,
      target:   p => p?.target_price,
      stopLoss: p => p?.stop_loss,
    }),
    { id: 'unrealized_pnl', header: 'P&L', align: 'right',
      cell: p => (
        <div className="flex flex-col items-end">
          <span className="font-mono font-semibold tabular-nums" style={{ color: (p?.unrealized_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {((p?.unrealized_pnl ?? 0) >= 0 ? '+' : '') + Number(p?.unrealized_pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </span>
          <span className="text-[11.5px] tabular-nums" style={{ color: (p?.unrealized_pnl_pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {((p?.unrealized_pnl_pct ?? 0) >= 0 ? '+' : '') + (p?.unrealized_pnl_pct ?? 0).toFixed(2)}%
          </span>
        </div>
      ) },
    { id: 'days', header: 'Days', align: 'right', mono: true,
      accessor: p => { const ts = p?.created_at ?? p?.updated_at; return ts ? new Date(ts).getTime() : 0; },
      cell: p => {
        const ts = p?.created_at ?? p?.updated_at;
        return <span className="text-ink-3">{ts ? Math.floor((Date.now() - new Date(ts).getTime()) / 86400000) + 'd' : '\u2014'}</span>;
      } },
    { id: 'actions', header: 'Actions', align: 'right', sortable: false,
      cell: p => (
        <div className="flex items-center justify-end gap-2">
          {autopilotSymbolSet.has(p?.symbol ?? '') ? (
            <span
              title="Already managed by autopilot"
              className="inline-flex items-center gap-1 h-8 px-[10px] rounded-[9px] text-[12px] font-semibold border-0 font-sans"
              style={{ background: 'var(--green-soft, #DCFCE7)', color: 'var(--green, #16A34A)' }}
            >
              <BrainCircuit size={13} /> In Autopilot
            </span>
          ) : (
            <button
              onClick={() => addToAutopilot(p)}
              disabled={autopilotingSymbols.has(p?.symbol ?? '')}
              title="Hand this position to the AI autopilot"
              className="inline-flex items-center gap-1 h-8 px-[10px] rounded-[9px] text-[12px] font-semibold cursor-pointer border-0 font-sans transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ background: 'var(--accent-soft, #EEF2FF)', color: 'var(--accent-2, #4F46E5)' }}
            >
              <BrainCircuit size={13} />
              {autopilotingSymbols.has(p?.symbol ?? '') ? '\u2026' : 'Autopilot'}
            </button>
          )}
          <button onClick={() => closePos(p)}
            className="h-8 px-[11px] rounded-[9px] text-[12.5px] font-semibold cursor-pointer border-0 bg-loss-soft text-loss transition-colors font-sans hover:bg-loss hover:text-white">
            Close
          </button>
        </div>
      ) },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [autopilotSymbolSet, autopilotingSymbols]);

  const histCols = useMemo<DataTableColumn<TradeRow>[]>(() => [
    { id: 'entry_at', header: 'Date',
      accessor: t => new Date(t?.exit_at ?? t?.entry_at ?? '').getTime(),
      cell: t => (
        <span className="text-[12.5px] text-ink-3 font-mono">
          {new Date(t?.entry_at ?? '').toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
        </span>
      ) },
    { id: 'symbol', header: 'Symbol', sortable: false,
      cell: t => <SymbolCell symbol={t?.symbol ?? ''} name={t?.name ?? ''} sector="" showSector={false} /> },
    { id: 'quantity', header: 'Qty', align: 'right', mono: true, cell: t => t?.quantity },
    // The Sell column finally has real data to show: a closed trade reports what
    // it actually sold for, an open one shows the target as a projection.
    ...priceColumns<TradeRow>({
      entry:    t => t?.entry_price,
      current:  t => t?.current_price,
      sold:     t => t?.exit_price,
      target:   t => t?.target_price,
      stopLoss: t => t?.stop_loss,
    }),
    { id: 'realized_pnl', header: 'Realized P&L', align: 'right', mono: true,
      cell: t => t?.realized_pnl == null
        ? <span className="text-ink-3">—</span>
        : (
          <span className="font-semibold" style={{ color: t.realized_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {(t.realized_pnl >= 0 ? '+' : '') + Number(t.realized_pnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </span>
        ) },
    // Trade state, in the same vocabulary the autopilot screen uses — the two
    // pages described the same trade with different words before this.
    { id: 'status', header: 'Status', align: 'right', sortable: false,
      cell: t => {
        const [label, c, bg] = tradeStatusMeta(t?.status);
        return <Pill color={c} bg={bg}>{label}</Pill>;
      } },
  ], []);

  const gttCols = useMemo<DataTableColumn<GTTOrder>[]>(() => [
    { id: 'symbol', header: 'Symbol',
      cell: g => <SymbolCell symbol={g?.symbol ?? ''} name={g?.name ?? ''} sector="" showSector={false} /> },
    { id: 'type', header: 'Type', cell: g => <Pill color="var(--text-2)" bg="var(--surface-3)">{g?.type}</Pill> },
    { id: 'side', header: 'Side',
      cell: g => <Pill color={g?.side === 'BUY' ? 'var(--green)' : 'var(--red)'} bg={g?.side === 'BUY' ? 'var(--green-soft)' : 'var(--red-soft)'}>{g?.side}</Pill> },
    { id: 'trigger', header: 'Trigger', align: 'right', mono: true,
      cell: g => <span className="font-semibold">{inr(g?.trigger ?? 0)}</span> },
    // A GTT rule is one leg of a bracket, so it carries the same four prices
    // as every other order row. `current` prefers the broker's live LTP over
    // the stored close when Angel One has given us one.
    ...priceColumns<GTTOrder>({
      entry:    g => g?.entry_price,
      current:  g => (g?.ltp || g?.current_price),
      sold:     g => (g?.sold ? g?.sell_price : null),
      target:   g => g?.target_price,
      stopLoss: g => g?.stop_loss,
    }),
    { id: 'qty', header: 'Qty', align: 'right', mono: true, cell: g => g?.qty },
    { id: 'created', header: 'Created', cell: g => <span className="text-[12.5px] text-ink-3">{g?.created}</span> },
    { id: 'status', header: 'Status', align: 'right',
      cell: g => {
        const sc  = g?.status === 'ACTIVE' ? 'var(--accent-2)' : g?.status === 'TRIGGERED' ? 'var(--green)' : 'var(--text-3)';
        const sbg = g?.status === 'ACTIVE' ? 'var(--accent-soft)' : g?.status === 'TRIGGERED' ? 'var(--green-soft)' : 'var(--surface-3)';
        return <Pill color={sc} bg={sbg}>{g?.status}</Pill>;
      } },
  ], []);

  const sigCols = useMemo<DataTableColumn<Record<string, any>>[]>(() => [
    { id: 'symbol', header: 'Symbol',
      cell: s => (
        <span className="cursor-pointer" onClick={() => navigate(`/stocks/${encodeURIComponent(s?.symbol ?? '')}`)}>
          <SymbolCell symbol={s?.symbol ?? ''} name={(s?.symbol ?? '').replace('.NS','')} sector="" showSector={false} />
        </span>
      ) },
    { id: 'signal', header: 'Signal', cell: s => <SignalBadge signal={s?.signal} /> },
    { id: 'model_horizon', header: 'Horizon',
      cell: s => (
        <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">
          {s?.model_horizon ?? '\u2014'}
        </span>
      ) },
    // A signal is a recommendation, never a holding, so there is no realised
    // sale to show — the Sell column is always the projected target here.
    ...priceColumns<Record<string, any>>({
      entry:    s => s?.buy_price,
      current:  s => s?.current_price,
      target:   s => s?.target_price,
      stopLoss: s => s?.stop_loss,
    }),
    { id: 'traded_at', header: 'Traded At',
      accessor: s => s?.traded_at ? new Date(s.traded_at).getTime() : 0,
      cell: s => (
        <span className="text-[12px] text-ink-3">
          {s?.traded_at ? new Date(s.traded_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '\u2014'}
        </span>
      ) },
    { id: 'order_status', header: 'Order Status',
      cell: s => (
        <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold border border-transparent"
          style={{
            background: s?.order_status === 'EXECUTED' ? 'var(--green-soft)' : s?.order_status === 'PENDING' ? 'var(--gold-soft)' : 'var(--surface-3)',
            color:      s?.order_status === 'EXECUTED' ? 'var(--green)'      : s?.order_status === 'PENDING' ? 'var(--gold)'      : 'var(--text-3)',
          }}>
          {s?.order_status ?? '\u2014'}
        </span>
      ) },
    { id: 'is_active', header: 'Signal Status',
      cell: s => (
        <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-full text-[11px] font-semibold border border-transparent"
          style={{
            background: s?.is_active ? 'var(--accent-soft)' : 'var(--surface-3)',
            color:      s?.is_active ? 'var(--accent-2)'    : 'var(--text-3)',
          }}>
          {s?.is_active ? '\u25CF Active' : '\u25CB Superseded'}
        </span>
      ) },
  ], []);
  const signalHist = (sigHist as any)?.data ?? [];
  const counts     = { open: openPos.length, history: tradeRows.length, gtt: gttOrders.length, ai_signals: signalHist.length };

  function exportCSV() {
    // Exports TRADES. This used to export order legs, so a CSV of "25 trades"
    // contained 14 rows for orders that never executed.
    const head = 'Entry Date,Exit Date,Symbol,Qty,Entry,Exit,Stop Loss,Target,Exit Reason,Realized P&L,Status\n';
    const body = histFiltered.map(t => [
      t?.entry_at ? new Date(t.entry_at).toISOString().slice(0, 10) : '',
      t?.exit_at  ? new Date(t.exit_at).toISOString().slice(0, 10)  : '',
      t?.symbol ?? '', t?.quantity ?? 0,
      t?.entry_price ?? '', t?.exit_price ?? '',
      t?.stop_loss ?? '', t?.target_price ?? '',
      t?.exit_reason ?? '', t?.realized_pnl ?? '', t?.status ?? '',
    ].join(',')).join('\n');
    const blob = new Blob([head + body], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a'); a.href = url; a.download = 'trademind-trades.csv'; a.click();
    URL.revokeObjectURL(url);
    toast({ type: 'success', title: 'Export complete', msg: `${histFiltered.length} trades downloaded as CSV` });
  }

  const tabCls = (active: boolean) =>
    `border-0 bg-transparent font-sans text-[14px] font-semibold px-4 py-3 cursor-pointer relative whitespace-nowrap transition-colors ${active ? 'text-accent-2 tab-active' : 'text-ink-2'}`;

  const segBtn = (active: boolean) =>
    `border-0 font-sans text-[12.5px] font-semibold px-3 py-[6px] rounded-[7px] cursor-pointer transition-colors ${active ? 'bg-accent text-white' : 'bg-transparent text-ink-2'}`;

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* Audit Low item — was indistinguishable from "no data" */}
      {(errPos || errTrd) && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-[11px] bg-[var(--red-soft)] text-[var(--red)] text-[13px] font-semibold">
          Couldn't load your {errPos && errTrd ? 'positions and trades' : errPos ? 'positions' : 'trades'}. Check your connection and try again.
        </div>
      )}

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>Trades &amp; Orders</h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">Manage open positions, review history &amp; Angel One GTT rules</p>
        </div>
        {tab === 'history' && (
          <button onClick={exportCSV} className="inline-flex items-center gap-2 h-10 px-4 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover">
            <Download size={17} /> Export CSV
          </button>
        )}
      </div>

      {/* ── Tabs ── */}
      <div className="flex gap-1 border-b border-line" style={{ marginBottom: 'calc(18px * var(--u))' }}>
        {([['open', 'Open Positions'], ['history', 'Trade History'], ['gtt', 'GTT Orders'], ['ai_signals', 'AI Signal History']] as const).map(([id, label]) => (
          <button key={id} className={tabCls(tab === id)} onClick={() => setTab(id)}>
            {label}
            <span className="ml-[7px] text-[11px] bg-surface-3 text-ink-2 px-[7px] py-[1px] rounded-full">{counts[id]}</span>
          </button>
        ))}
      </div>

      {/* ══ OPEN POSITIONS ══ */}
      {tab === 'open' && (
        <Card pad={false}>
          <DataTable
            columns={posCols}
            data={openPos}
            isLoading={loading}
            isFetching={fetchPos}
            skeletonRows={6}
            getRowId={p => p?.symbol ?? ''}
            emptyMessage="No open positions. All trades closed 🎉"
          />
        </Card>
      )}

      {/* ══ TRADE HISTORY ══ */}
      {tab === 'history' && (
        <Card pad={false}>
          <div className="flex items-center justify-between gap-3 border-b border-line flex-wrap" style={{ padding: 'calc(15px * var(--u)) calc(18px * var(--u))' }}>
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex flex-col gap-[5px]">
                <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Date Range</span>
                <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
                  {(['All', '7D', '30D', '90D'] as DateRange[]).map(r => (
                    <button key={r} className={segBtn(dateRange === r)} onClick={() => { setDateRange(r); }}>{r}</button>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-[5px]">
                <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">State</span>
                <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
                  {(['All', 'Open', 'Closed'] as SideFlt[]).map(r => (
                    <button key={r} className={segBtn(sideFlt === r)} onClick={() => { setSideFlt(r); }}>{r}</button>
                  ))}
                </div>
              </div>
            </div>
            <span className="text-[12.5px] text-ink-2"><b className="tabular-nums">{histFiltered.length}</b> trades</span>
          </div>
          <DataTable
            columns={histCols}
            data={histFiltered}
            isLoading={loadTrd}
            isFetching={fetchTrd}
            skeletonRows={9}
            initialSort={{ id: 'entry_at', desc: true }}
            pagination={{ perPage: PER_PAGE }}
            getRowId={t => t?.bracket_id ?? ''}
            emptyMessage="No trades in this range."
          />
        </Card>
      )}

      {/* ══ GTT ORDERS ══ */}
      {tab === 'gtt' && (
        <Card pad={false}>
          <div className="flex items-center justify-between gap-3 border-b border-line" style={{ padding: 'calc(15px * var(--u)) calc(18px * var(--u))' }}>
            <div className="flex flex-col gap-0">
              <h3 className="m-0 text-[14.5px] font-semibold flex items-center gap-[9px] text-ink">
                <span className="text-ink-3">
                  <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>
                </span>
                Angel One GTT Rules
              </h3>
              <span className="text-[12px] text-ink-3 mt-[2px]">Good-Till-Triggered orders synced from your broker</span>
            </div>
            <button onClick={handleSync}
              className="inline-flex items-center gap-[6px] h-8 px-[11px] rounded-[9px] font-sans text-[12.5px] font-semibold cursor-pointer border border-line bg-surface-2 text-ink transition-colors hover:bg-surface-hover">
              <RefreshCw size={15} style={{ animation: syncing ? 'spin 1s linear infinite' : 'none' }} /> Sync
            </button>
          </div>
          <DataTable
            columns={gttCols}
            data={gttOrders}
            isLoading={loading}
            isFetching={fetchGtt}
            skeletonRows={5}
            getRowId={g => String(g?.id ?? '')}
            emptyMessage="No GTT rules synced from your broker."
          />
        </Card>
      )}

      {/* ══ AI SIGNAL HISTORY ══ */}
      {tab === 'ai_signals' && (
        <Card pad={false}>
          <div className="flex items-center justify-between gap-3 border-b border-line" style={{ padding: 'calc(15px * var(--u)) calc(18px * var(--u))' }}>
            <div>
              <h3 className="m-0 text-[14.5px] font-semibold text-ink">AI Signal History</h3>
              <span className="text-[12px] text-ink-3 mt-[2px] block">AI signals you acted on — showing current status (active/superseded)</span>
            </div>
          </div>
          <DataTable
            columns={sigCols}
            data={signalHist}
            isLoading={loadSig}
            isFetching={fetchSig}
            skeletonRows={6}
            emptyMessage="No AI signals acted on yet. Execute a signal from the AI Signals page."
          />
        </Card>
      )}

      <style>{`@keyframes spin{to{transform:rotate(360deg);}}`}</style>
    </div>
  );
}
