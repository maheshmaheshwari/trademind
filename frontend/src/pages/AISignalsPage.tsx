import { useDeferredValue, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { useGetAllSignalsQuery } from '../services/tradeMindApiService';
import type { AllSignal } from '../services/tradeMindApiService';
import {
  Card, SignalBadge, SkeletonRows,
  SymbolCell, Conf, Pager, useSort, Th, PlainTh, Td,
} from '../components/ui';

const HORIZONS = ['All', '1W', '2W', '1M', '2M', '3M', '6M'] as const;
const SIGNALS  = ['All', 'BUY', 'SELL', 'HOLD'] as const;
const SECTORS  = ['All', 'IT', 'Banking', 'Financials', 'Energy', 'Auto', 'FMCG', 'Pharma', 'Metals', 'Cement', 'Infra', 'Telecom', 'Power'];
const SECTOR_COLORS: Record<string, string> = {
  IT: '#3B82F6', Banking: '#8B5CF6', Financials: '#6366F1', Energy: '#F59E0B', Auto: '#EC4899',
  FMCG: '#10B981', Pharma: '#14B8A6', Metals: '#F97316', Cement: '#A78BFA', Infra: '#0EA5E9',
  Telecom: '#EF4444', Power: '#EAB308',
};
const PER_PAGE = 12;

function fmtAgo(m: number) { return m < 60 ? `${m}m ago` : `${Math.floor(m / 60)}h ago`; }

export default function AISignalsPage() {
  const [search,       setSearch]  = useState('');
  const [sigType,      setSigType] = useState<typeof SIGNALS[number]>('All');
  const [horizon,      setHorizon] = useState<typeof HORIZONS[number]>('All');
  const [sector,       setSector]  = useState('All');
  const [conf,         setConf]    = useState(50);
  const [page,         setPage]    = useState(1);
  const navigate = useNavigate();

  const { sort, toggle } = useSort('confidence');
  const dSearch = useDeferredValue(search);

  const { data: res, isLoading: loading } = useGetAllSignalsQuery();
  const allSignals: AllSignal[] = res?.signals ?? [];

  const filtered = useMemo(() => [...allSignals]
    .filter(s =>
      (sigType  === 'All' || s.signal  === sigType)  &&
      (horizon  === 'All' || s.horizon === horizon)  &&
      (sector   === 'All' || s.sector  === sector)   &&
      (s.confidence ?? 0) >= conf &&
      (!dSearch || s?.symbol?.toLowerCase()?.includes(dSearch.toLowerCase()) ||
                   s?.name?.toLowerCase()?.includes(dSearch.toLowerCase()))
    )
    .sort((a, b) => {
      const va = a[sort.key as keyof AllSignal], vb = b[sort.key as keyof AllSignal];
      let cmp = 0;
      if (typeof va === 'number' && typeof vb === 'number') cmp = va - vb;
      else if (typeof va === 'string' && typeof vb === 'string') cmp = va.localeCompare(vb);
      return sort.dir === 'asc' ? cmp : -cmp;
    }), [allSignals, sigType, horizon, sector, conf, dSearch, sort]);

  const pages = Math.max(1, Math.ceil(filtered.length / PER_PAGE));
  const rows  = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  const counts = { BUY: 0, SELL: 0, HOLD: 0 };
  (allSignals ?? []).forEach(s => { if (s?.signal in counts) counts[s.signal as keyof typeof counts]++; });

  const segBtn = (active: boolean) =>
    `border-none font-sans text-[12.5px] font-semibold px-3 py-[6px] rounded-[7px] cursor-pointer transition-colors ${
      active ? 'bg-accent text-white' : 'bg-transparent text-ink-2'
    }`;

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-ink" style={{ fontSize: 'calc(25px * var(--u))' }}>AI Signals</h1>
          <p className="text-ink-2 text-[13.5px] mt-1 m-0">
            Machine-learning signals across all horizons · <b className="tabular-nums">{loading ? '…' : (res?.total_stocks ?? 0)}</b> stocks · <b className="tabular-nums">{loading ? '…' : (res?.count ?? 0)}</b> signals
          </p>
        </div>
        {!loading && (
          <div className="flex items-center gap-2">
            {(['BUY', 'SELL', 'HOLD'] as const).map(sig => {
              const col = sig === 'BUY' ? ['var(--green)', 'var(--green-soft)'] : sig === 'SELL' ? ['var(--red)', 'var(--red-soft)'] : ['var(--gold)', 'var(--gold-soft)'];
              const arrow = sig === 'BUY' ? '↑' : sig === 'SELL' ? '↓' : '●';
              return (
                <span key={sig} className="inline-flex items-center gap-[5px] h-[23px] px-[9px] rounded-[7px] text-[11.5px] font-bold"
                  style={{ color: col[0], background: col[1] }}>
                  {arrow} {counts[sig]} {sig}
                </span>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Filter card ── */}
      <Card pad={false}>
        <div className="flex flex-col gap-[14px]" style={{ padding: 'calc(15px * var(--u)) calc(18px * var(--u))' }}>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative flex-[1_1_220px] max-w-[280px]">
              <Search size={17} className="absolute left-[13px] top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" />
              <input
                value={search} onChange={e => { setSearch(e.target.value); setPage(1); }}
                placeholder="Search symbol or name…"
                className="w-full h-[38px] pl-10 pr-3 rounded-[10px] border border-line bg-surface-2 text-ink font-sans text-[13px] outline-none box-border focus:border-accent transition-colors"
              />
            </div>
            <div className="flex flex-col gap-[5px]">
              <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Sector</span>
              <select value={sector} onChange={e => { setSector(e.target.value); setPage(1); }}
                className="h-[38px] px-3 rounded-[10px] border border-line bg-surface-2 text-ink font-sans text-[13px] outline-none min-w-[120px] focus:border-accent transition-colors">
                {SECTORS.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex flex-col gap-[5px]">
              <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Signal</span>
              <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
                {SIGNALS.map(t => (
                  <button key={t} className={segBtn(sigType === t)} onClick={() => { setSigType(t); setPage(1); }}>{t}</button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex flex-col gap-[5px]">
              <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Horizon</span>
              <div className="inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-[2px]">
                {HORIZONS.map(h => (
                  <button key={h} className={segBtn(horizon === h)} onClick={() => { setHorizon(h); setPage(1); }}>{h}</button>
                ))}
              </div>
            </div>
            <div className="flex flex-col gap-[5px] flex-[1_1_220px] max-w-[320px]">
              <span className="text-[11px] font-semibold text-ink-3 tracking-[.03em] uppercase">Min Confidence · {conf}%</span>
              <div className="flex items-center gap-3">
                <input type="range" className="rng flex-1" min="50" max="95" value={conf} onChange={e => { setConf(+e.target.value); setPage(1); }} />
                <span className="font-mono font-bold min-w-[38px]">{conf}%</span>
              </div>
            </div>
            <div className="flex flex-col gap-[5px] justify-end">
              <span className="text-[11px] font-semibold text-transparent tracking-[.03em] uppercase">.</span>
              <span className="text-[13px] text-ink-2"><b className="tabular-nums">{loading ? '…' : filtered.length}</b> stocks match</span>
            </div>
          </div>
        </div>
      </Card>

      {/* ── Table ── */}
      <Card pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>
                <Th label="Stock"       sortKey="symbol"     sort={sort} onToggle={toggle} />
                <Th label="Sector"      sortKey="sector"     sort={sort} onToggle={toggle} />
                <PlainTh>Signal</PlainTh>
                <Th label="Confidence"  sortKey="confidence" sort={sort} onToggle={toggle} />
                <PlainTh>Horizon</PlainTh>
                <Th label="Exp. Return" sortKey="expReturn"  sort={sort} onToggle={toggle} align="right" />
                <Th label="Sentiment"   sortKey="sentiment"  sort={sort} onToggle={toggle} align="right" />
                <Th label="Updated"     sortKey="updatedMin" sort={sort} onToggle={toggle} align="right" />
              </tr>
            </thead>
            <tbody>
              {loading ? <SkeletonRows cols={8} rows={10} /> : rows.length === 0 ? (
                <tr><td colSpan={8} className="text-center py-[50px] px-5 text-ink-3">
                  No signals match your filters. Try lowering the confidence threshold.
                </td></tr>
              ) : (rows ?? []).map(s => (
                <tr key={`${s?.symbol}-${s?.horizon}`} className="cursor-pointer transition-colors hover:bg-surface-2" onClick={() => navigate(`/stocks/${encodeURIComponent(s?.symbol ?? '')}`)}>
                  <Td><SymbolCell symbol={s?.symbol} name={s?.name} sector={s?.sector} showSector={false} /></Td>
                  <Td>
                    <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 border border-line"
                      style={{ color: SECTOR_COLORS[s.sector] ?? 'var(--text-2)' }}>{s.sector}</span>
                  </Td>
                  <Td><SignalBadge signal={s.signal} /></Td>
                  <Td><div className="min-w-[130px]"><Conf value={s.confidence} /></div></Td>
                  <Td>
                    <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-surface-3 text-ink-2 border border-line">{s.horizon}</span>
                  </Td>
                  <Td align="right">
                    {s.expReturn != null ? (
                      <span className="font-mono font-semibold tabular-nums" style={{ color: s.expReturn >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {(s.expReturn >= 0 ? '+' : '') + Number(s.expReturn).toFixed(2) + '%'}
                      </span>
                    ) : <span className="text-ink-3">—</span>}
                  </Td>
                  <Td align="right">
                    {s.sentiment != null ? (
                      <span className="font-mono tabular-nums" style={{ color: s.sentiment >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {(s.sentiment >= 0 ? '+' : '') + Number(s.sentiment).toFixed(2)}
                      </span>
                    ) : <span className="text-ink-3">—</span>}
                  </Td>
                  <Td align="right"><span className="text-[12px] text-ink-3 font-mono">{fmtAgo(s.updatedMin ?? 0)}</span></Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pager page={page} pages={pages} total={filtered.length} perPage={PER_PAGE} onPage={setPage} label="signals" />
      </Card>

    </div>
  );
}
