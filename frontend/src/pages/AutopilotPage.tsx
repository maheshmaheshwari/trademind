import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BrainCircuit, Settings, CheckCircle, Clock, AlertCircle, Activity, TrendingUp, Target, Wallet, ChevronRight } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useToast } from '../components/ui';
import { Card, SignalBadge, Skeleton, SymbolCell, DataTable, priceColumns, type DataTableColumn } from '../components/ui';
import { RiskReward } from '../components/ui';
import {
  useGetAutopilotStatusQuery, useToggleAutopilotMutation,
  useGetAuthorizedTradesQuery, useRevokeAuthorizedTradeMutation,
  useGetAutopilotRecommendationsQuery, useAuthorizeTradeAutoMutation,
  type AuthorizedTrade,
} from '../services/tradeMindApiService';

// ── Formatting helpers ───────────────────────────────────────────────────────

function inr(n: number | null | undefined, dec = 2) {
  if (n == null) return '—';
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}
function signed(n: number, dec = 0) {
  const s = Math.abs(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
  return (n >= 0 ? '+₹' : '−₹') + s;
}

// ── Status metadata ──────────────────────────────────────────────────────────

type TradeStatus = AuthorizedTrade['status'];

function statusMeta(st: TradeStatus) {
  switch (st) {
    case 'EXECUTED':  return { label: 'Running',    color: 'var(--accent-2)',  bg: 'var(--accent-soft)',  Icon: Activity };
    case 'PENDING':   return { label: 'Pending',    color: 'var(--gold)',      bg: 'var(--gold-soft)',    Icon: Clock };
    case 'COMPLETED': return { label: 'Target hit', color: 'var(--green)',     bg: 'var(--green-soft)',   Icon: CheckCircle };
    case 'STOPPED':   return { label: 'Stopped',    color: 'var(--red)',       bg: 'var(--red-soft)',     Icon: AlertCircle };
    default:          return { label: st,           color: 'var(--text-2)',    bg: 'var(--surface-3)',    Icon: Target };
  }
}

const STATUS_FILTERS = ['All', 'Running', 'Pending', 'Target hit', 'Stopped'] as const;
const STATUS_MAP: Record<string, TradeStatus | undefined> = {
  Running: 'EXECUTED', Pending: 'PENDING', 'Target hit': 'COMPLETED', Stopped: 'STOPPED',
};

// ── Mini stat card ───────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color, Icon: IconComp }: {
  label: string; value: string; sub?: string; color: string; Icon: React.ElementType;
}) {
  return (
    <div className="bg-surface border border-line relative overflow-hidden"
      style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))' }}>
      <div className="flex items-center justify-between">
        <span className="text-[12.5px] text-ink-2 font-medium">{label}</span>
        <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center" style={{ background: color + '1f', color }}>
          <IconComp size={18} />
        </span>
      </div>
      <div className="font-bold tracking-tight tabular-nums text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>
        {value}
      </div>
      {sub && <span className="text-[12px] text-ink-3">{sub}</span>}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

function RecommendedTrades({ userId }: { userId: number }) {
  const toast = useToast();
  const { data, isLoading } = useGetAutopilotRecommendationsQuery(userId, { skip: !userId });
  const [authorize] = useAuthorizeTradeAutoMutation();
  const [done, setDone] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);

  const recs: any[] = data?.recommendations ?? [];

  async function authorizeOne(r: any) {
    setBusy(r.symbol);
    try {
      await authorize({
        user_id: userId, symbol: r.symbol, name: r.name ?? r.symbol,
        signal: 'BUY', mode: 'PAPER', qty: r.qty, amount: r.investment,
        entry: r.buy_price, target: r.target_price, sl: r.stop_loss,
        cmp: r.current_price ?? r.buy_price,
        exp_profit: Math.round(((r.target_price - r.buy_price) * r.qty) || 0),
        max_loss: Math.round(((r.buy_price - r.stop_loss) * r.qty) || 0),
      } as any).unwrap();
      setDone(prev => new Set(prev).add(r.symbol));
      toast({ type: 'success', title: `Authorized ${r.symbol}`, msg: `${r.qty} @ ₹${r.buy_price} · pending mandate` });
    } catch {
      toast({ type: 'error', title: `Failed to authorize ${r.symbol}` });
    } finally {
      setBusy(null);
    }
  }

  const recCols = useMemo<DataTableColumn<Record<string, any>>[]>(() => [
    { id: 'rank', header: '#', mono: true, cell: r => <span className="text-ink-3">{r?.rank}</span> },
    { id: 'symbol', header: 'Stock', cell: r => <SymbolCell symbol={r?.symbol} name={r?.name} showSector={false} /> },
    { id: 'current_price', header: 'CMP', align: 'right', mono: true, cell: r => inr(r?.current_price) },
    { id: 'confidence', header: 'Conf', mono: true,
      cell: r => <span className="font-semibold" style={{ color: 'var(--green)' }}>{r?.confidence}%</span> },
    { id: 'model_accuracy', header: 'Model acc / prec', mono: true,
      cell: r => (
        <span className="text-[12px] text-ink-2">
          {r?.model_accuracy != null ? `${r.model_accuracy}%` : '—'} / {r?.model_precision != null ? `${r.model_precision}%` : '—'}
        </span>
      ) },
    { id: 'horizon', header: 'Horizon', sortable: false,
      cell: r => <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">{r?.horizon}</span> },
    { id: 'qty', header: 'Qty', align: 'right', mono: true, cell: r => (r?.qty ?? 0).toLocaleString('en-IN') },
    { id: 'investment', header: 'Investment', align: 'right', mono: true,
      cell: r => '₹' + Math.round(r?.investment ?? 0).toLocaleString('en-IN') },
    { id: 'risk_reward', header: 'R:R', align: 'right', cell: r => <RiskReward value={r?.risk_reward} size={12.5} /> },
    { id: 'levels', header: 'Entry / Target / SL', sortable: false,
      cell: r => (
        <span className="font-mono text-[12px] text-ink-2 whitespace-nowrap">
          {inr(r?.buy_price)} · <span style={{ color: 'var(--green)' }}>{inr(r?.target_price)}</span> · <span style={{ color: 'var(--red)' }}>{inr(r?.stop_loss)}</span>
        </span>
      ) },
    { id: 'action', header: '', align: 'right', sortable: false,
      cell: r => {
        const authorized = done.has(r?.symbol);
        return (
          <button
            disabled={authorized || busy === r?.symbol}
            onClick={() => authorizeOne(r)}
            className="inline-flex items-center gap-1 h-8 px-3 rounded-[8px] text-[12.5px] font-semibold border cursor-pointer transition-colors disabled:cursor-default"
            style={authorized
              ? { color: 'var(--green)', borderColor: 'var(--line)', background: 'var(--surface-2)' }
              : { color: '#fff', borderColor: 'var(--accent)', background: 'var(--accent)' }}>
            {authorized ? <><CheckCircle size={14} /> Authorized</> : busy === r?.symbol ? 'Authorizing…' : 'Authorize'}
          </button>
        );
      } },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [done, busy]);

  if (isLoading) return <Card><Skeleton h={180} /></Card>;
  if (!recs.length) return null;

  return (
    <Card pad={false}>
      <div className="flex items-start justify-between gap-3 flex-wrap px-4 pt-4 pb-2">
        <div>
          <h3 className="text-[15px] font-bold text-ink m-0">Recommended trades — conviction-ranked</h3>
          <p className="text-[12.5px] text-ink-3 mt-0.5 m-0">
            Top STRONG BUY signals from <b>reliable models only</b> (accuracy ≥ {data?.params?.min_accuracy ?? 70}%,
            precision ≥ {data?.params?.min_precision ?? 60}%), ranked by confidence, capped at {data?.max_concurrent ?? 8} concurrent
            positions, risk-sized to your paper balance. Review and authorize — nothing is placed automatically.
            {(data?.filtered_low_quality ?? 0) > 0 && (
              <> <b className="text-ink-2">{data.filtered_low_quality}</b> confident signal{data.filtered_low_quality !== 1 ? 's' : ''} hidden for low model quality.</>
            )}
          </p>
        </div>
        <span className="text-[12px] text-ink-3 whitespace-nowrap">
          {data?.slots_available ?? 0} slot{(data?.slots_available ?? 0) !== 1 ? 's' : ''} open · ₹{Math.round(data?.cash ?? 0).toLocaleString('en-IN')} cash
        </span>
      </div>
      <DataTable
        columns={recCols}
        data={recs}
        getRowId={r => r?.symbol ?? ''}
        emptyMessage="No recommendations right now."
      />
    </Card>
  );
}

export default function AutopilotPage() {
  const { user } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<typeof STATUS_FILTERS[number]>('All');

  const userId = user?.id ?? 0;

  const { data: statusData, isLoading: loadStatus } =
    useGetAutopilotStatusQuery(userId, { skip: !userId });

  const statusFilter = filter === 'All' ? undefined : STATUS_MAP[filter];
  const { data: tradesData, isLoading: loadTrades, isFetching: fetchTrades } =
    useGetAuthorizedTradesQuery({ userId, status: statusFilter }, { skip: !userId });

  const [toggleAutopilot, { isLoading: toggling }] = useToggleAutopilotMutation();
  const [revokeTrade] = useRevokeAuthorizedTradeMutation();

  const autopilotOn = statusData?.enabled ?? false;
  const capital     = statusData?.capital ?? 0;
  const active      = statusData?.active ?? 0;
  const realizedPnl = statusData?.realized_pnl ?? 0;
  const projected   = statusData?.projected_profit ?? 0;

  const trades: AuthorizedTrade[] = (tradesData as any)?.data ?? [];
  const totalTrades = (tradesData as any)?.total ?? 0;

  async function handleToggle() {
    try {
      const res = await toggleAutopilot(userId).unwrap();
      toast({
        type: res.enabled ? 'success' : 'info',
        title: res.enabled ? 'Autopilot active' : 'Autopilot paused',
        msg: res.enabled
          ? 'AI will auto-execute authorized signals'
          : 'AI will not place new orders',
      });
    } catch {
      toast({ type: 'error', title: 'Toggle failed' });
    }
  }

  async function handleRevoke(trade: AuthorizedTrade, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await revokeTrade(trade.id).unwrap();
      toast({ type: 'info', title: 'Authorization revoked', msg: `${trade.symbol} removed from AI autopilot` });
    } catch {
      toast({ type: 'error', title: 'Revoke failed' });
    }
  }

  // Cost basis is what the entry actually FILLED at, not what was authorized.
  // A buy limit fills at the market when the market is cheaper, so the two
  // differ — and P&L measured against `entry` disagreed with the Portfolio
  // page's for the same trade. Falls back to `entry` for mandates authorized
  // before fill_price was recorded.
  const costBasis = (t: AuthorizedTrade) => t?.fill_price ?? t?.entry ?? null;

  const tradeCols = useMemo<DataTableColumn<AuthorizedTrade>[]>(() => [
    { id: 'symbol', header: 'Stock',
      cell: t => <SymbolCell symbol={t?.symbol ?? ''} name={t?.name ?? ''} sector={t?.sector ?? ''} showSector={false} /> },
    { id: 'signal', header: 'Signal', sortable: false,
      cell: t => (
        <div className="flex items-center gap-1.5">
          <SignalBadge signal={t?.signal} />
          <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">
            {t?.mode}
          </span>
        </div>
      ) },
    { id: 'amount', header: 'Authorized \u20B9', align: 'right', mono: true,
      cell: t => <span className="font-semibold text-ink">{inr(t?.amount, 0)}</span> },
    { id: 'qty', header: 'Qty', align: 'right', mono: true, cell: t => t?.qty },
    // costBasis() is the fill where we have one, else the authorized entry —
    // a buy limit fills at the market when the market is cheaper. A mandate
    // that has been squared off reports its realised price via actual_pnl
    // rather than a stored sell price, so Sell stays the projected target here.
    ...priceColumns<AuthorizedTrade>({
      entry:    t => costBasis(t),
      current:  t => (t?.status === 'EXECUTED' || t?.status === 'PENDING') ? t?.cmp : null,
      target:   t => t?.target,
      stopLoss: t => t?.sl,
    }),
    { id: 'exp_profit', header: 'Exp. Profit', align: 'right', mono: true,
      cell: t => <span className="font-semibold text-gain">+{inr(t?.exp_profit, 0)}</span> },
    { id: 'max_loss', header: 'Max Loss', align: 'right', mono: true,
      cell: t => <span className="text-loss">\u2212{inr(t?.max_loss, 0)}</span> },
    { id: 'pnl', header: 'Live / Realized P&L', align: 'right', mono: true,
      accessor: t => livePnl(t) ?? undefined,
      cell: t => {
        const pnl = livePnl(t);
        return pnl != null
          ? <span className="font-semibold" style={{ color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>{signed(pnl)}</span>
          : <span className="text-ink-3">\u2014</span>;
      } },
    { id: 'status', header: 'Status',
      cell: t => {
        const meta = statusMeta(t?.status);
        return (
          <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-semibold border-none"
            style={{ color: meta?.color, background: meta?.bg }}>
            {meta?.Icon && <meta.Icon size={11} />}
            {meta?.label}
          </span>
        );
      } },
    { id: 'action', header: 'Action', align: 'right', sortable: false,
      cell: t => {
        const canRevoke = t?.status === 'EXECUTED' || t?.status === 'PENDING';
        return canRevoke ? (
          <button
            onClick={(e) => handleRevoke(t, e)}
            className="h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border-none cursor-pointer transition-colors bg-loss-soft text-loss hover:bg-loss hover:text-white"
          >
            Revoke
          </button>
        ) : <span className="text-[12px] text-ink-3">Closed</span>;
      } },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], []);

  const livePnl = (t: AuthorizedTrade) => {
    const basis = costBasis(t);
    return t?.status === 'EXECUTED' && t?.cmp != null && basis != null
      ? (t.cmp - basis) * (t?.qty ?? 0)
      : t?.actual_pnl ?? null;
  };

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>
            AI Authorized Trades
          </h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">
            Trades you've authorized the AI to place &amp; manage automatically via <b>Angel One</b>
          </p>
        </div>

        {/* Autopilot toggle pill */}
        <button
          onClick={handleToggle}
          disabled={toggling}
          className="flex items-center gap-3 px-4 py-2 rounded-[13px] border cursor-pointer transition-all"
          style={{
            borderColor: autopilotOn ? 'color-mix(in srgb, var(--accent) 45%, transparent)' : 'var(--border)',
            background: autopilotOn ? 'var(--accent-soft)' : 'var(--surface-2)',
          }}
        >
          {/* pulse dot */}
          <span className={`w-[9px] h-[9px] rounded-full flex-shrink-0 ${autopilotOn ? 'bg-gain animate-pulse-dot' : 'bg-ink-3'}`} />
          <div className="flex flex-col gap-0 text-left" style={{ lineHeight: 1.2 }}>
            <span className="text-[13px] font-bold text-ink">AI Autopilot</span>
            <span className="text-[10.5px] text-ink-2">{autopilotOn ? 'Active · auto-executing' : 'Paused'}</span>
          </div>
          {/* toggle knob */}
          <span className={`w-11 h-6 rounded-full border-none p-[3px] flex items-center flex-shrink-0 transition-colors ${autopilotOn ? 'bg-accent' : 'bg-surface-3'}`}
            style={{ pointerEvents: 'none' }}>
            <span className={`block w-[18px] h-[18px] rounded-full bg-white shadow-sm transition-transform duration-[180ms] ${autopilotOn ? 'translate-x-5' : 'translate-x-0'}`} />
          </span>
        </button>
      </div>

      {/* ── Stat cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 dgap">
        {loadStatus ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u))' }}>
              <Skeleton h={12} w="50%" className="mb-3" /><Skeleton h={28} w="70%" className="mb-2" /><Skeleton h={11} w="40%" />
            </div>
          ))
        ) : (<>
          <StatCard label="Capital Under AI" value={inrCompact(capital)} color="var(--accent)" Icon={Wallet} />
          <StatCard label="Active Mandates" value={String(active)} color="var(--gold)" Icon={Activity} />
          <StatCard
            label="Realized P&L" sub="from closed mandates"
            value={signed(realizedPnl)}
            color={realizedPnl >= 0 ? 'var(--green)' : 'var(--red)'}
            Icon={TrendingUp}
          />
          <StatCard
            label="Projected Profit" sub="if targets hit"
            value={'+' + inrCompact(projected)}
            color="#8B5CF6"
            Icon={Target}
          />
        </>)}
      </div>

      {/* ── AI banner ── */}
      {!loadStatus && (
        <div className="flex items-center gap-3 px-4 py-[14px] rounded-[var(--radius)] border"
          style={{
            borderColor: 'color-mix(in srgb, var(--accent) 28%, transparent)',
            background: 'linear-gradient(100deg, var(--accent-soft), transparent 70%)',
          }}>
          <span className="w-10 h-10 rounded-[11px] grid place-items-center flex-shrink-0 bg-accent text-white"
            style={{ boxShadow: '0 4px 14px rgba(59,130,246,.34)' }}>
            <BrainCircuit size={20} />
          </span>
          <div className="flex flex-col gap-[1px] flex-1 min-w-0">
            <span className="text-[13.5px] font-semibold text-ink">
              AI is managing <b>{active}</b> active trade{active !== 1 ? 's' : ''} — autopilot is {autopilotOn ? 'active' : 'paused'}
            </span>
            <span className="text-[12px] text-ink-2">
              Each order respects your per-trade stop-loss &amp; target. Authorize new trades from the{' '}
              <button className="text-accent-2 font-semibold border-none bg-transparent cursor-pointer p-0 hover:underline"
                onClick={() => navigate('/signals')}>
                AI Signals
              </button>{' '}page.
            </span>
          </div>
          <button
            onClick={() => navigate('/settings', { state: { tab: 'risk' } })}
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border border-line bg-surface-2 text-ink-2 hover:bg-surface-hover hover:text-ink transition-colors flex-shrink-0 cursor-pointer"
          >
            <Settings size={14} /> Mandate rules
          </button>
        </div>
      )}

      {/* ── Recommended trades (conviction-ranked, recommend-only) ── */}
      {userId ? <RecommendedTrades userId={userId} /> : null}

      {/* ── Filters ── */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
          {(STATUS_FILTERS ?? []).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className="border-none font-sans text-[12.5px] font-semibold px-3 py-[6px] rounded-[7px] cursor-pointer transition-colors whitespace-nowrap"
              style={{ background: filter === f ? 'var(--accent)' : 'transparent', color: filter === f ? '#fff' : 'var(--text-2)' }}>
              {f}
            </button>
          ))}
        </div>
        <span className="text-[12.5px] text-ink-2">
          <b className="tabular-nums">{totalTrades}</b> mandate{totalTrades !== 1 ? 's' : ''}
        </span>
      </div>

      {/* ── Table ── */}
      <Card pad={false}>
        <DataTable
          columns={tradeCols}
          data={trades}
          isLoading={loadTrades}
          isFetching={fetchTrades}
          skeletonRows={6}
          pagination={{ perPage: 15 }}
          getRowId={t => String(t?.id ?? '')}
          emptyMessage={
            <div>
              <BrainCircuit size={34} className="mx-auto mb-3 opacity-40" />
              <p className="font-semibold text-ink m-0 mb-1">No authorized trades{filter !== 'All' ? ` in "${filter}"` : ''}</p>
              <p className="text-[12.5px] m-0 mb-4">Authorize trades from the AI Signals page to let the AI manage them.</p>
              <button
                onClick={() => navigate('/signals')}
                className="inline-flex items-center gap-2 h-9 px-4 rounded-[11px] text-[13px] font-semibold border-none bg-accent text-white cursor-pointer"
                style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}
              >
                Browse Signals <ChevronRight size={15} />
              </button>
            </div>
          }
        />

        {/* Footer count */}
        {trades.length > 0 && (
          <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-line flex-wrap">
            <span className="text-[12.5px] text-ink-2">
              Showing <b>{trades.length}</b> of <b>{totalTrades}</b> authorized trades
            </span>
          </div>
        )}
      </Card>
    </div>
  );
}
