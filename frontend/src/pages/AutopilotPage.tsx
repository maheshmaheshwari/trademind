import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BrainCircuit, Settings, CheckCircle, Clock, AlertCircle, Activity, TrendingUp, Target, Wallet, ChevronRight } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useToast } from '../components/ui';
import { Card, SignalBadge, SkeletonRows, Skeleton, SymbolCell } from '../components/ui';
import {
  useGetAutopilotStatusQuery, useToggleAutopilotMutation,
  useGetAuthorizedTradesQuery, useRevokeAuthorizedTradeMutation,
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

export default function AutopilotPage() {
  const { user } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<typeof STATUS_FILTERS[number]>('All');

  const userId = user?.id ?? 0;

  const { data: statusData, isLoading: loadStatus } =
    useGetAutopilotStatusQuery(userId, { skip: !userId });

  const statusFilter = filter === 'All' ? undefined : STATUS_MAP[filter];
  const { data: tradesData, isLoading: loadTrades } =
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

  const livePnl = (t: AuthorizedTrade) =>
    t.status === 'EXECUTED' && t.cmp != null && t.entry != null
      ? (t.cmp - t.entry) * t.qty
      : t.actual_pnl ?? null;

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
      <div className="grid max-lg:grid-cols-2" style={{ gridTemplateColumns: 'repeat(4,1fr)', gap: 'calc(16px * var(--u))' }}>
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

      {/* ── Filters ── */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
          {STATUS_FILTERS.map(f => (
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
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                {['Stock', 'Signal', 'Authorized ₹', 'Qty', 'Entry → Target', 'Exp. Profit', 'Max Loss', 'Live / Realized P&L', 'Status', 'Action'].map(h => (
                  <th key={h} className="text-left text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 border-b border-line whitespace-nowrap sticky top-0 bg-surface z-[1]"
                    style={{ padding: 'calc(11px * var(--u)) 14px', textAlign: ['Authorized ₹', 'Qty', 'Entry → Target', 'Exp. Profit', 'Max Loss', 'Live / Realized P&L', 'Action'].includes(h) ? 'right' : 'left' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loadTrades ? (
                <SkeletonRows cols={10} rows={6} />
              ) : trades.length === 0 ? (
                <tr>
                  <td colSpan={10}>
                    <div className="text-center py-14 text-ink-3">
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
                  </td>
                </tr>
              ) : trades.map(t => {
                const meta = statusMeta(t.status);
                const pnl  = livePnl(t);
                const canRevoke = t.status === 'EXECUTED' || t.status === 'PENDING';
                return (
                  <tr key={t.id} className="border-b border-line transition-colors hover:bg-surface-2">
                    <td style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      <SymbolCell symbol={t.symbol} name={t.name ?? ''} sector={t.sector ?? ''} showSector={false} />
                    </td>
                    <td style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      <div className="flex items-center gap-1.5">
                        <SignalBadge signal={t.signal} />
                        <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">
                          {t.mode}
                        </span>
                      </div>
                    </td>
                    <td className="text-right font-mono tabular-nums font-semibold text-ink" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      {inr(t.amount, 0)}
                    </td>
                    <td className="text-right font-mono tabular-nums" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      {t.qty}
                    </td>
                    <td className="text-right font-mono tabular-nums text-[12px]" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      <span className="text-ink-2">{inr(t.entry, 0)}</span>
                      <span className="text-ink-3"> → </span>
                      <span className="text-gain">{inr(t.target, 0)}</span>
                    </td>
                    <td className="text-right font-mono tabular-nums font-semibold text-gain" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      +{inr(t.exp_profit, 0)}
                    </td>
                    <td className="text-right font-mono tabular-nums text-loss" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      −{inr(t.max_loss, 0)}
                    </td>
                    <td className="text-right" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      {pnl != null ? (
                        <span className="font-mono tabular-nums font-semibold" style={{ color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                          {signed(pnl)}
                        </span>
                      ) : (
                        <span className="text-ink-3">—</span>
                      )}
                    </td>
                    <td style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-semibold border-none"
                        style={{ color: meta.color, background: meta.bg }}>
                        <meta.Icon size={11} />
                        {meta.label}
                      </span>
                    </td>
                    <td className="text-right" style={{ padding: 'calc(12px * var(--u)) 14px' }}>
                      {canRevoke ? (
                        <button
                          onClick={(e) => handleRevoke(t, e)}
                          className="h-8 px-3 rounded-[9px] text-[12.5px] font-semibold border-none cursor-pointer transition-colors bg-loss-soft text-loss hover:bg-loss hover:text-white"
                        >
                          Revoke
                        </button>
                      ) : (
                        <span className="text-[12px] text-ink-3">
                          Closed
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

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
