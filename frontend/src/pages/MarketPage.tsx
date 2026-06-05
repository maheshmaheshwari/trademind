import { useState } from 'react';
import { useGetMarketOverviewQuery, useGetMarketSectorsQuery } from '../services/tradeMindApiService';
import { Card, SignalBadge, Delta, Skeleton, SkeletonRows, SymbolCell } from '../components/ui';
import { FlowBars, Sparkline } from '../components/Charts';
import { StockDrawer } from '../components/StockDrawer';
import type { IndexData, FIIDIIBar, HeatmapSector, Breadth, Stock } from '../types';

function pct(n: number) { return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'; }
function inr(n: number, dec = 2) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function heatColor(c: number) {
  const a = Math.min(Math.abs(c) / 3.5, 1);
  return c >= 0 ? `rgba(16,185,129,${(.12 + a * .5).toFixed(2)})` : `rgba(239,68,68,${(.12 + a * .5).toFixed(2)})`;
}

function IndexCard({ ix }: { ix: IndexData }) {
  const pos = ix.pct >= 0;
  return (
    <div className="bg-surface border border-line relative overflow-hidden" style={{ borderRadius: 'var(--radius,14px)', padding: 'calc(18px * var(--u))' }}>
      <div className="flex items-center justify-between">
        <span className="font-bold text-[13.5px] tracking-[.01em] whitespace-nowrap text-ink">{ix.name}</span>
        <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold"
          style={{ color: pos ? 'var(--green)' : 'var(--red)', background: pos ? 'var(--green-soft)' : 'var(--red-soft)' }}>
          {pct(ix.pct)}
        </span>
      </div>
      <div className="font-mono text-[23px] font-bold text-ink" style={{ margin: '9px 0 2px' }}>{ix.value.toLocaleString('en-IN')}</div>
      <div className="text-[12.5px] font-semibold tabular-nums" style={{ color: pos ? 'var(--green)' : 'var(--red)' }}>
        {(pos ? '+' : '') + ix.change}
      </div>
      <div style={{ marginTop: 10, marginLeft: -2, marginRight: -2 }}>
        <Sparkline data={ix.spark} color={pos ? '#10B981' : '#EF4444'} w={260} h={40} />
      </div>
    </div>
  );
}

export default function MarketPage() {
  const { data: mktData, isLoading: loading } = useGetMarketOverviewQuery();
  const { data: sectorsData } = useGetMarketSectorsQuery();
  const [drawer, setDrawer] = useState<string | null>(null);

  const indices: IndexData[]    = (mktData as any)?.indices  ?? [];
  const fiiDii:  FIIDIIBar[]   = (mktData as any)?.fii_dii  ?? [];
  const heatmap: HeatmapSector[] = [...(sectorsData ?? (mktData as any)?.heatmap ?? [])].sort((a: HeatmapSector, b: HeatmapSector) => b.change - a.change);
  const gainers: Stock[]        = (mktData as any)?.gainers  ?? [];
  const losers:  Stock[]        = (mktData as any)?.losers   ?? [];
  const breadth: Breadth | null = (mktData as any)?.breadth  ?? null;

  const vix = indices.find(ix => ix.name === 'INDIA VIX');
  const adRatio = breadth ? (breadth.advances / breadth.declines).toFixed(2) : '—';

  const stockRow = (s: Stock) => (
    <tr key={s.symbol} className="cursor-pointer transition-colors hover:bg-surface-2" onClick={() => setDrawer(s.symbol)}>
      <td style={tdS}><SymbolCell symbol={s.symbol} name={s.name} sector={s.sector} /></td>
      <td style={{ ...tdS, textAlign: 'right' }} className="font-mono tabular-nums">{inr(s.price)}</td>
      <td style={{ ...tdS, textAlign: 'right' }}><Delta value={s.change} size={13} showIcon={false} /></td>
      <td style={tdS}><SignalBadge signal={s.signal} /></td>
    </tr>
  );

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>Market Overview</h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">Live indices, institutional flows &amp; sector rotation · NSE</p>
        </div>
        <div className="flex items-center gap-2 h-9 px-[13px] rounded-full text-[12.5px] font-semibold border border-line text-gain bg-gain-soft">
          <span className="w-2 h-2 rounded-full bg-[var(--green)] animate-pulse-dot" />
          MARKET OPEN
          <span className="text-ink-3 font-medium font-mono text-[11.5px]">15:24:08</span>
        </div>
      </div>

      {/* ── 4 index cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 'calc(16px * var(--u))' }}>
        {loading
          ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} h={150} rounded="14px" />)
          : indices.map(ix => <IndexCard key={ix.name} ix={ix} />)
        }
      </div>

      {/* ── FII/DII + Breadth ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.7fr 1fr', gap: 'calc(16px * var(--u))' }}>
        <Card title="FII / DII Activity" sub="Net buy / sell · last 10 sessions (₹ Cr)"
          icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20v-3"/></svg>}>
          <div className="dp" style={{ paddingTop: 8 }}>
            {loading ? <Skeleton h={210} /> : <FlowBars data={fiiDii} h={210} />}
          </div>
        </Card>

        <Card title="Market Breadth" sub="Advance / Decline"
          icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></svg>}>
          <div className="dp flex flex-col gap-4">
            {loading ? <Skeleton h={120} /> : breadth ? <>
              <div className="flex h-[14px] rounded-full overflow-hidden gap-[2px]">
                <div style={{ flex: breadth.advances, background: 'var(--green)', height: '100%' }} />
                <div style={{ flex: breadth.unchanged, background: 'var(--text-3)', height: '100%' }} />
                <div style={{ flex: breadth.declines, background: 'var(--red)', height: '100%' }} />
              </div>
              <div className="flex justify-between">
                <div className="flex flex-col gap-[1px]">
                  <span className="font-mono text-[22px] font-bold text-gain">{breadth.advances}</span>
                  <span className="text-[11.5px] text-ink-3">Advancing</span>
                </div>
                <div className="flex flex-col gap-[1px] items-center">
                  <span className="font-mono text-[22px] font-bold text-ink-2">{breadth.unchanged}</span>
                  <span className="text-[11.5px] text-ink-3">Unchanged</span>
                </div>
                <div className="flex flex-col gap-[1px] items-end">
                  <span className="font-mono text-[22px] font-bold text-loss">{breadth.declines}</span>
                  <span className="text-[11.5px] text-ink-3">Declining</span>
                </div>
              </div>
              <div className="h-px bg-[var(--border)]" />
              <div className="flex justify-between items-center">
                <span className="text-[13px] text-ink-2">Advance / Decline Ratio</span>
                <span className="font-mono font-bold text-[16px] text-gain">{adRatio}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[13px] text-ink-2">India VIX</span>
                <span className="font-mono font-bold text-[16px]" style={{ color: (vix?.pct ?? 0) <= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {vix ? vix.value.toFixed(2) : '—'}
                  {vix && <span className="text-[12px]"> {pct(vix.pct)}</span>}
                </span>
              </div>
            </> : null}
          </div>
        </Card>
      </div>

      {/* ── Sector heatmap ── */}
      <Card title="Sector Heatmap" sub="12 sectors · % change today"
        icon={<svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>}
        pad={false}>
        <div className="dp">
          {loading ? <Skeleton h={170} /> : (
            <div className="grid grid-cols-4 gap-2">
              {heatmap.map(h => (
                <div key={h.sector}
                  className="flex flex-col gap-[3px] min-h-[78px] justify-between cursor-default transition-transform hover:-translate-y-0.5 border"
                  style={{ borderRadius: 'var(--radius-sm,9px)', padding: 13, borderColor: 'rgba(255,255,255,.06)', background: heatColor(h.change) }}>
                  <span className="text-[12px] font-semibold opacity-95 text-ink">{h.sector}</span>
                  <div className="flex justify-between items-end">
                    <span className="font-mono text-[17px] font-bold" style={{ color: h.change >= 0 ? 'var(--green)' : 'var(--red)' }}>{pct(h.change)}</span>
                    <span className="text-[10.5px] text-ink-3">{h.stock_count ?? '—'} stocks</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* ── Gainers / Losers ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'calc(16px * var(--u))' }}>
        {([
          ['Top Gainers', gainers, true],
          ['Top Losers',  losers,  false],
        ] as const).map(([title, list, isGain]) => (
          <Card key={title as string} title={title as string} pad={false}
            icon={isGain
              ? <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></svg>
              : <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M3 7l6 6 4-4 8 8"/><path d="M21 13v4h-4"/></svg>
            }>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-[13px]">
                <thead>
                  <tr>
                    {['Stock', 'LTP', 'Change', 'Signal'].map((h, i) => (
                      <th key={h} style={{ ...thS, textAlign: i >= 1 && i <= 2 ? 'right' : 'left' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loading ? <SkeletonRows cols={4} rows={5} /> : (list as Stock[]).map(s => stockRow(s))}
                </tbody>
              </table>
            </div>
          </Card>
        ))}
      </div>

      <StockDrawer symbol={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

const thS: React.CSSProperties = {
  fontSize: 11, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase',
  color: 'var(--text-3)', padding: 'calc(11px * var(--u)) 14px', borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap', position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1,
};
const tdS: React.CSSProperties = {
  padding: 'calc(12px * var(--u)) 14px', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap',
};
