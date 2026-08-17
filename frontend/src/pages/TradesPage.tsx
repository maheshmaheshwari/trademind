import { useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, RefreshCw, BrainCircuit } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useToast } from '../components/ui';
import {
  useGetPositionsQuery, useGetOrdersQuery, useSquareOffMutation,
  useGetGTTOrdersQuery, useSyncGTTMutation, useGetUserSignalHistoryQuery,
  useAuthorizeTradeAutoMutation, useGetAuthorizedTradesQuery,
} from '../services/tradeMindApiService';
import { Card, SymbolCell, SignalBadge, DataTable, type DataTableColumn } from '../components/ui';

import type { OpenPosition, Trade, GTTOrder } from '../types';

function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}

type Tab = 'open' | 'history' | 'gtt' | 'ai_signals';
type DateRange = 'All' | '7D' | '30D' | '90D';
type SideFlt = 'All' | 'BUY' | 'SELL';
const PER_PAGE = 18;

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
  const { data: ordRes,    isLoading: loadOrd, isFetching: fetchOrd, isError: errOrd  } = useGetOrdersQuery({ userId: user?.id ?? 0, size: 200 }, { skip: !user });
  const { data: gttRes,    isLoading: loadGtt, isFetching: fetchGtt  } = useGetGTTOrdersQuery(user?.id ?? 0, { skip: !user });
  const { data: sigHist,   isLoading: loadSig, isFetching: fetchSig  } = useGetUserSignalHistoryQuery({ userId: user?.id ?? 0 }, { skip: !user });
  const [squareOff]                               = useSquareOffMutation();
  const [syncGTT, { isLoading: syncing }]         = useSyncGTTMutation();
  const [authorizeTradeAuto]                      = useAuthorizeTradeAutoMutation();
  const { data: authTradesRes }                   = useGetAuthorizedTradesQuery({ userId: user?.id ?? 0 }, { skip: !user });
  const [autopilotingSymbols, setAutopilotingSymbols] = useState<Set<string>>(new Set());
  const autopilotSymbolsRef = useRef<Set<string>>(new Set());

  // Symbols already managed by autopilot (PENDING or EXECUTED)
  const autopilotSymbolSet = new Set(
    ((authTradesRes as any)?.data ?? [])
      .filter((t: any) => t?.status === 'PENDING' || t?.status === 'EXECUTED')
      .map((t: any) => t?.symbol ?? '')
  );

  const loading   = loadPos || loadOrd || loadGtt;
  const positions: OpenPosition[] = (posRes as any)?.data ?? [];
  const trades:    Trade[]        = (ordRes as any)?.data  ?? [];
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
  const histFiltered = ([...(trades ?? [])])
    .filter(t => {
      const days = (today.getTime() - new Date(t?.created_at ?? '').getTime()) / 86400000;
      const rangeOk = dateRange === 'All' || (dateRange === '7D' && days <= 7) || (dateRange === '30D' && days <= 30) || (dateRange === '90D' && days <= 90);
      return rangeOk && (sideFlt === 'All' || t?.order_type === sideFlt);
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
    { id: 'avg_buy_price', header: 'Entry', align: 'right', mono: true, cell: p => inr(p?.avg_buy_price ?? 0) },
    { id: 'stop_loss', header: 'SL', align: 'right', mono: true,
      cell: p => <span className="text-loss">{inr(p?.stop_loss ?? 0)}</span> },
    { id: 'target_price', header: 'Target', align: 'right', mono: true,
      cell: p => <span className="text-gain">{inr(p?.target_price ?? 0)}</span> },
    { id: 'current_price', header: 'CMP', align: 'right', mono: true, cell: p => inr(p?.current_price ?? 0) },
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

  const histCols = useMemo<DataTableColumn<Trade>[]>(() => [
    { id: 'created_at', header: 'Date',
      accessor: t => new Date(t?.created_at ?? '').getTime(),
      cell: t => (
        <span className="text-[12.5px] text-ink-3 font-mono">
          {new Date(t?.created_at ?? '').toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
        </span>
      ) },
    { id: 'symbol', header: 'Symbol', sortable: false,
      cell: t => <SymbolCell symbol={t?.symbol ?? ''} name={t?.name ?? ''} sector={t?.sector ?? ''} showSector={false} /> },
    { id: 'order_type', header: 'Type', sortable: false,
      cell: t => <Pill color={t?.order_type === 'BUY' ? 'var(--green)' : 'var(--red)'} bg={t?.order_type === 'BUY' ? 'var(--green-soft)' : 'var(--red-soft)'}>{t?.order_type}</Pill> },
    { id: 'quantity', header: 'Qty', align: 'right', mono: true, cell: t => t?.quantity },
    // Entry is the FILL, not the limit — the two differ whenever a buy limit
    // filled into a cheaper market. Falls back to `price` for orders written
    // before fill_price was recorded.
    { id: 'price', header: 'Entry', align: 'right', mono: true,
      accessor: t => t?.fill_price ?? t?.price ?? 0,
      cell: t => inr(t?.fill_price ?? t?.price ?? 0) },
    { id: 'current_price', header: 'CMP', align: 'right', mono: true,
      cell: t => t?.current_price != null
        ? <span className="text-ink-2">{inr(t.current_price)}</span>
        : <span className="text-ink-3">\u2014</span> },
    { id: 'value', header: 'Value', align: 'right', mono: true,
      cell: t => <span className="text-ink-2">{inrCompact(t?.value ?? 0)}</span> },
    { id: 'pnl', header: 'P&L', align: 'right', mono: true,
      cell: t => (
        <span className="font-semibold" style={{ color: (t?.pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {((t?.pnl ?? 0) >= 0 ? '+' : '') + Number(t?.pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </span>
      ) },
    { id: 'status', header: 'Status', align: 'right', sortable: false,
      cell: t => <Pill color="var(--green)" bg="var(--green-soft)">{t?.status}</Pill> },
  ], []);

  const gttCols = useMemo<DataTableColumn<GTTOrder>[]>(() => [
    { id: 'symbol', header: 'Symbol',
      cell: g => <SymbolCell symbol={g?.symbol ?? ''} name={g?.name ?? ''} sector="" showSector={false} /> },
    { id: 'type', header: 'Type', cell: g => <Pill color="var(--text-2)" bg="var(--surface-3)">{g?.type}</Pill> },
    { id: 'side', header: 'Side',
      cell: g => <Pill color={g?.side === 'BUY' ? 'var(--green)' : 'var(--red)'} bg={g?.side === 'BUY' ? 'var(--green-soft)' : 'var(--red-soft)'}>{g?.side}</Pill> },
    { id: 'trigger', header: 'Trigger', align: 'right', mono: true,
      cell: g => <span className="font-semibold">{inr(g?.trigger ?? 0)}</span> },
    { id: 'ltp', header: 'LTP', align: 'right', mono: true,
      cell: g => <span className="text-ink-2">{inr(g?.ltp ?? 0)}</span> },
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
    { id: 'buy_price', header: 'Buy Price', align: 'right', mono: true, cell: s => s?.buy_price ? inr(s.buy_price) : '\u2014' },
    { id: 'target_price', header: 'Target', align: 'right', mono: true,
      cell: s => <span className="text-gain">{s?.target_price ? inr(s.target_price) : '\u2014'}</span> },
    { id: 'stop_loss', header: 'SL', align: 'right', mono: true,
      cell: s => <span className="text-loss">{s?.stop_loss ? inr(s.stop_loss) : '\u2014'}</span> },
    { id: 'current_price', header: 'CMP', align: 'right', mono: true,
      cell: s => s?.current_price != null ? inr(s.current_price) : '\u2014' },
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
  const counts     = { open: openPos.length, history: trades.length, gtt: gttOrders.length, ai_signals: signalHist.length };

  function exportCSV() {
    const head = 'Date,Symbol,Side,Qty,Price,Value,Realized P&L\n';
    const body = histFiltered.map(t => `${t?.created_at ? new Date(t.created_at).toISOString().slice(0, 10) : ''},${t?.symbol ?? ''},${t?.order_type ?? ''},${t?.quantity ?? 0},${t?.price ?? 0},${t?.value ?? 0},${t?.pnl ?? 0}`).join('\n');
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
      {(errPos || errOrd) && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-[11px] bg-[var(--red-soft)] text-[var(--red)] text-[13px] font-semibold">
          Couldn't load your {errPos && errOrd ? 'positions and orders' : errPos ? 'positions' : 'orders'}. Check your connection and try again.
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
                <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Side</span>
                <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
                  {(['All', 'BUY', 'SELL'] as SideFlt[]).map(r => (
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
            isLoading={loading}
            isFetching={fetchOrd}
            skeletonRows={9}
            initialSort={{ id: 'created_at', desc: true }}
            pagination={{ perPage: PER_PAGE }}
            getRowId={t => String(t?.id ?? '')}
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
