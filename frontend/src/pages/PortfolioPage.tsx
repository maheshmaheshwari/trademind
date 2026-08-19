import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus } from 'lucide-react';
import { useAuth } from '../AuthContext';
import {
  useGetPortfolioSummaryQuery, useGetPortfolioHistoryQuery, useGetTodayPnlQuery,
} from '../services/tradeMindApiService';
import {
  Card, SignalBadge, Delta, Skeleton, SymbolCell,
  DataTable, priceColumns, type DataTableColumn,
} from '../components/ui';
import { AreaChart, Donut } from '../components/Charts';
import { AddPositionModal } from '../components/AddPositionModal';
import { sectorColors } from '../utils/sectorColors';

import type { Holding } from '../types';

function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}

export default function PortfolioPage() {
  const { user } = useAuth();
  const [range,  setRange]  = useState<'30D' | '90D' | '1Y'>('90D');
  const [modal,  setModal]  = useState(false);
  const navigate = useNavigate();

  const { data: portData, isLoading: loading, isFetching, isError } = useGetPortfolioSummaryQuery(user?.id ?? 0, { skip: !user });
  const { data: todayPnlData } = useGetTodayPnlQuery(user?.id ?? 0, { skip: !user });
  // Its own request, not a field on the summary: it is the only thing on the
  // page that changes when the range buttons are clicked, and re-fetching the
  // whole portfolio to redraw one chart would blank the cards and the table.
  const { data: histData, isFetching: histFetching } =
    useGetPortfolioHistoryQuery({ userId: user?.id ?? 0, range }, { skip: !user });
  const raw = portData as any;

  // Sorting moved into DataTable.
  const holdings: Holding[] = raw?.positions ?? [];

  const alloc = raw?.allocation ?? [];
  const series: number[] = histData?.series ?? [];
  const pnlPct:  number  = raw?.total_pnl_pct ?? 0;   // served by the API, not divided here
  const sectors: number  = new Set((holdings ?? []).map(h => h?.sector)).size;

  // The real trading dates the points were sampled on. The axis hides these —
  // they surface in the tooltip, so hovering a point says which day it is.
  // Previously four fixed strings ("30d ago", "20d", …) were spread across
  // thirty points, so every tooltip but the first named the wrong day.
  const chartLabels = useMemo(
    () => (histData?.dates ?? []).map(d =>
      new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: '2-digit' })),
    [histData?.dates],
  );

  // Colour is assigned here, not served: it is presentation, and the palette
  // has to be resolved against the other slices in the same donut.
  const allocSlices = useMemo(() => {
    const colors = sectorColors((alloc ?? []).map((a: any) => a?.sector));
    return (alloc ?? []).map((a: any, i: number) => ({
      sector: a?.sector ?? 'Unclassified',
      val: a?.val ?? 0,
      color: colors[i],
    }));
  }, [alloc]);

  const holdingCols = useMemo<DataTableColumn<Holding>[]>(() => [
    { id: 'symbol', header: 'Symbol',
      cell: h => <SymbolCell symbol={h?.symbol ?? ''} name={h?.name ?? ''} sector={h?.sector ?? ''} /> },
    { id: 'quantity', header: 'Qty', align: 'right', mono: true, cell: h => h?.quantity },
    // A holding is open by definition, so Sell shows the target as a projection.
    ...priceColumns<Holding>({
      entry:    h => h?.avg_buy_price,
      current:  h => h?.current_price,
      target:   h => h?.target_price,
      stopLoss: h => h?.stop_loss,
    }),
    { id: 'invested_amount', header: 'Invested', align: 'right', mono: true,
      cell: h => <span className="text-ink-2">{inrCompact(h?.invested_amount ?? 0)}</span> },
    { id: 'unrealized_pnl', header: 'P&L', align: 'right', mono: true,
      cell: h => (
        <span className="font-semibold" style={{ color: (h?.unrealized_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {((h?.unrealized_pnl ?? 0) >= 0 ? '+' : '') + Number(h?.unrealized_pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </span>
      ) },
    { id: 'unrealized_pnl_pct', header: 'P&L %', align: 'right',
      cell: h => <Delta value={h?.unrealized_pnl_pct ?? 0} size={12.5} showIcon={false} /> },
    { id: 'signal', header: 'AI Signal', sortable: false, cell: h => <SignalBadge signal={h?.signal} /> },
  ], []);

  const segBtn = (active: boolean) =>
    `border-none font-sans text-[12.5px] font-semibold px-[10px] py-1 rounded-[6px] cursor-pointer transition-colors ${
      active ? 'bg-surface text-ink shadow-sm' : 'bg-transparent text-ink-2'
    }`;

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* Audit Low item — was indistinguishable from "no data" */}
      {isError && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-[11px] bg-[var(--red-soft)] text-[var(--red)] text-[13px] font-semibold">
          Couldn't load your portfolio. Check your connection and try again.
        </div>
      )}

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>Portfolio</h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">
            <b className="tabular-nums">{loading ? '—' : holdings.length}</b> holdings · diversified across{' '}
            <b className="tabular-nums">{loading ? '—' : sectors}</b> sectors
          </p>
        </div>
        <button
          onClick={() => setModal(true)}
          className="inline-flex items-center justify-center gap-2 h-10 px-4 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border-0 bg-accent text-white"
          style={{ boxShadow: '0 4px 14px rgba(59,130,246,.32)' }}
        >
          <Plus size={17} /> Add Position
        </button>
      </div>

      {/* ── 3 stat cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 dgap">
        {loading ? Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u))' }}>
            <Skeleton h={12} w="50%" className="mb-3" /><Skeleton h={28} w="70%" className="mb-2" /><Skeleton h={11} w="40%" />
          </div>
        )) : (<>
          <div className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))' }}>
            <div className="flex justify-between">
              <span className="text-[12.5px] text-ink-2 font-medium">Total Invested</span>
              <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center bg-accent-soft text-accent">
                <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="6" width="18" height="14" rx="3"/><path d="M3 10h18"/><circle cx="16.5" cy="14" r="1.3" fill="currentColor" stroke="none"/></svg>
              </span>
            </div>
            <div className="font-bold tracking-tight text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>{inrCompact(raw?.invested ?? 0)}</div>
            <span className="text-[12px] text-ink-3">across {holdings.length} stocks</span>
          </div>

          <div className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))' }}>
            <div className="flex justify-between">
              <span className="text-[12.5px] text-ink-2 font-medium">Current Value</span>
              <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center" style={{ background: '#8B5CF61f', color: '#8B5CF6' }}>
                <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M12 3a9 9 0 0 1 9 9h-9z" opacity=".4" fill="currentColor" stroke="none"/></svg>
              </span>
            </div>
            <div className="font-bold tracking-tight text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>{inrCompact(raw?.total_value ?? 0)}</div>
            {todayPnlData?.today_pnl_pct != null && (
              <span className="inline-flex items-center gap-1 text-[12.5px] font-semibold"
                    style={{ color: (todayPnlData.today_pnl_pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(todayPnlData.today_pnl_pct ?? 0) >= 0 ? '+' : ''}{(todayPnlData?.today_pnl_pct ?? 0).toFixed(2)}% today
              </span>
            )}
          </div>

          <div className="border border-line" style={{
            background: pnlPct >= 0 ? 'linear-gradient(135deg,var(--green-soft),transparent)' : 'linear-gradient(135deg,var(--red-soft),transparent)',
            borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))',
          }}>
            <div className="flex justify-between">
              <span className="text-[12.5px] text-ink-2 font-medium">Total P&amp;L</span>
              <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center"
                style={{ background: pnlPct >= 0 ? 'var(--green-soft)' : 'var(--red-soft)', color: pnlPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></svg>
              </span>
            </div>
            <div className="font-bold tracking-tight" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px', color: pnlPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {(raw?.total_pnl ?? 0) >= 0 ? '+₹' : '−₹'}{Math.abs(raw?.total_pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </div>
            <span className="font-bold tabular-nums" style={{ color: pnlPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {(pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2)}% overall
            </span>
          </div>
        </>)}
      </div>

      {/* ── Chart + Donut ── */}
      <div className="grid grid-cols-1 md:grid-cols-[1.7fr_1fr] dgap">
        <Card title="Portfolio Value" sub="Growth over time"
          icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></svg>}
          right={
            <div className="inline-flex bg-surface-2 border border-line rounded-[9px] p-[3px] gap-[1px]">
              {(['30D', '90D', '1Y'] as const).map(r => (
                <button key={r} className={segBtn(range === r)} onClick={() => setRange(r)}>{r}</button>
              ))}
            </div>
          }
        >
          <div className="dp" style={{ paddingTop: 10 }}>
            {loading || (histFetching && !series.length)
              ? <Skeleton h={230} />
              : series.length
                ? <AreaChart data={series} color="var(--accent)" h={230} labels={chartLabels} currency />
                : <div className="grid place-items-center text-[13px] text-ink-3" style={{ height: 230 }}>
                    No value history yet.
                  </div>}
          </div>
        </Card>

        <Card title="Allocation" sub="By sector"
          icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M12 3a9 9 0 0 1 9 9h-9z" opacity=".4" fill="currentColor" stroke="none"/></svg>}>
          <div className="dp">
            {loading
              ? <Skeleton h={180} />
              : allocSlices.length
                ? <Donut data={allocSlices} centerTop="Holdings"
                         centerBottom={inrCompact(allocSlices.reduce((a, s) => a + (s?.val ?? 0), 0))} size={240} />
                : <div className="grid place-items-center text-[13px] text-ink-3" style={{ height: 180 }}>
                    No open positions to allocate.
                  </div>}
          </div>
        </Card>
      </div>

      {/* ── Holdings table ── */}
      <Card title="Holdings" sub={`${holdings.length} positions`}
        icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>}
        pad={false}>
        <DataTable
          columns={holdingCols}
          data={holdings}
          isLoading={loading}
          isFetching={isFetching}
          skeletonRows={8}
          initialSort={{ id: 'invested_amount', desc: true }}
          getRowId={h => h?.symbol ?? ''}
          onRowClick={h => navigate(`/stocks/${encodeURIComponent(h?.symbol ?? '')}`)}
          emptyMessage="No holdings yet. Authorize a signal to open your first position."
        />
      </Card>

      {modal && <AddPositionModal onClose={() => setModal(false)} />}
    </div>
  );
}
