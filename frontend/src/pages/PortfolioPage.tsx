import { useState } from 'react';
import { Plus } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useGetPortfolioSummaryQuery } from '../services/tradeMindApiService';
import { Card, SignalBadge, Delta, Skeleton, SkeletonRows, SymbolCell, useSort, Th, Td } from '../components/ui';
import { AreaChart, Donut } from '../components/Charts';
import { AddPositionModal } from '../components/AddPositionModal';
import { StockDrawer } from '../components/StockDrawer';
import type { Holding, AllocSlice } from '../types';

function inrCompact(n: number) {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}
function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

export default function PortfolioPage() {
  const { user } = useAuth();
  const { sort, toggle } = useSort('current', 'desc');
  const [range,  setRange]  = useState<'30D' | '90D' | '1Y'>('90D');
  const [modal,  setModal]  = useState(false);
  const [drawer, setDrawer] = useState<string | null>(null);

  const { data: portData, isLoading: loading } = useGetPortfolioSummaryQuery(user!.id, { skip: !user });
  const raw = portData as any;

  const holdings: Holding[] = raw?.holdings ? [...raw.holdings].sort((a: Holding, b: Holding) => {
    const va = a[sort.key as keyof Holding], vb = b[sort.key as keyof Holding];
    let cmp = 0;
    if (typeof va === 'number' && typeof vb === 'number') cmp = va - vb;
    else if (typeof va === 'string' && typeof vb === 'string') cmp = va.localeCompare(vb);
    return sort.dir === 'asc' ? cmp : -cmp;
  }) : [];

  const alloc:   AllocSlice[] = raw?.allocation ?? [];
  const series:  number[]     = raw?.pnl_history?.[range] ?? [];
  const pnlPct:  number       = raw ? (raw.total_pnl / raw.total_invested) * 100 : 0;
  const sectors: number       = new Set(holdings.map(h => h.sector)).size;

  const rangeLabels: Record<string, string[]> = {
    '30D': ['30d ago', '20d', '10d', 'Today'],
    '90D': ['90d ago', '60d', '30d', 'Today'],
    '1Y':  ['1Y ago',  '9mo', '6mo', '3mo', 'Now'],
  };

  const segBtn = (active: boolean) =>
    `border-none font-sans text-[12.5px] font-semibold px-[10px] py-1 rounded-[6px] cursor-pointer transition-colors ${
      active ? 'bg-surface text-ink shadow-sm' : 'bg-transparent text-ink-2'
    }`;

  return (
    <div className="flex flex-col dgap animate-page-in">

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
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 'calc(16px * var(--u))' }}>
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
            <div className="font-bold tracking-tight text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>{inrCompact(raw?.total_invested ?? 0)}</div>
            <span className="text-[12px] text-ink-3">across {holdings.length} stocks</span>
          </div>

          <div className="bg-surface border border-line" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(17px * var(--u)) calc(18px * var(--u))' }}>
            <div className="flex justify-between">
              <span className="text-[12.5px] text-ink-2 font-medium">Current Value</span>
              <span className="w-[34px] h-[34px] rounded-[10px] grid place-items-center" style={{ background: '#8B5CF61f', color: '#8B5CF6' }}>
                <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M12 3a9 9 0 0 1 9 9h-9z" opacity=".4" fill="currentColor" stroke="none"/></svg>
              </span>
            </div>
            <div className="font-bold tracking-tight text-ink" style={{ fontSize: 'calc(27px * var(--u))', margin: '10px 0 5px' }}>{inrCompact(raw?.current_value ?? 0)}</div>
            <span className="inline-flex items-center gap-1 text-[12.5px] font-semibold text-gain">
              <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round"><path d="M7 17 17 7M8 7h9v9"/></svg>
              +1.84% today
            </span>
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
              {pnlPct >= 0 ? '+₹' : '-₹'}{Math.abs(raw?.total_pnl ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </div>
            <span className="font-bold tabular-nums" style={{ color: pnlPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {(pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2)}% overall
            </span>
          </div>
        </>)}
      </div>

      {/* ── Chart + Donut ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.7fr 1fr', gap: 'calc(16px * var(--u))' }}>
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
            {loading ? <Skeleton h={230} /> : <AreaChart data={series} color="var(--accent)" h={230} labels={rangeLabels[range]} />}
          </div>
        </Card>

        <Card title="Allocation" sub="By sector"
          icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M12 3a9 9 0 0 1 9 9h-9z" opacity=".4" fill="currentColor" stroke="none"/></svg>}>
          <div className="dp">
            {loading ? <Skeleton h={180} /> : <Donut data={alloc} centerTop="Total" centerBottom={inrCompact(raw?.current_value ?? 0)} size={240} />}
          </div>
        </Card>
      </div>

      {/* ── Holdings table ── */}
      <Card title="Holdings" sub={`${holdings.length} positions`}
        icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>}
        pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                <Th label="Symbol"   sortKey="symbol"   sort={sort} onToggle={toggle} />
                <Th label="Qty"      sortKey="qty"      sort={sort} onToggle={toggle} align="right" />
                <Th label="Avg Buy"  sortKey="avg"      sort={sort} onToggle={toggle} align="right" />
                <Th label="CMP"      sortKey="cmp"      sort={sort} onToggle={toggle} align="right" />
                <Th label="Invested" sortKey="invested" sort={sort} onToggle={toggle} align="right" />
                <Th label="P&L"      sortKey="pnl"      sort={sort} onToggle={toggle} align="right" />
                <Th label="P&L %"    sortKey="pnlPct"   sort={sort} onToggle={toggle} align="right" />
                <th style={thS}>AI Signal</th>
              </tr>
            </thead>
            <tbody>
              {loading ? <SkeletonRows cols={8} rows={8} /> : holdings.map(h => (
                <tr key={h.symbol} className="cursor-pointer transition-colors hover:bg-surface-2" onClick={() => setDrawer(h.symbol)}>
                  <Td><SymbolCell symbol={h.symbol} name={h.name} sector={h.sector} /></Td>
                  <Td align="right" mono>{h.qty}</Td>
                  <Td align="right" mono>{inr(h.avg)}</Td>
                  <Td align="right" mono>{inr(h.cmp)}</Td>
                  <Td align="right" mono><span className="text-ink-2">{inrCompact(h.invested)}</span></Td>
                  <Td align="right" mono>
                    <span className="font-semibold" style={{ color: h.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                      {(h.pnl >= 0 ? '+' : '') + Number(h.pnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                    </span>
                  </Td>
                  <Td align="right"><Delta value={h.pnlPct} size={12.5} showIcon={false} /></Td>
                  <Td><SignalBadge signal={h.signal} /></Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {modal && <AddPositionModal onClose={() => setModal(false)} />}
      <StockDrawer symbol={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

const thS: React.CSSProperties = {
  fontSize: 11, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase',
  color: 'var(--text-3)', padding: 'calc(11px * var(--u)) 14px', borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap', position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1, textAlign: 'left',
};
