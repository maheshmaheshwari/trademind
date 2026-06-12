import { useState, useDeferredValue } from 'react';
import { useNavigate } from 'react-router-dom';
import { LayoutGrid, List, Plus, X, Bell } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useToast } from '../components/ui';
import {
  useGetWatchlistQuery,
  useRemoveFromWatchlistMutation,
  useAddToWatchlistMutation,
  useGetStocksQuery,
} from '../services/tradeMindApiService';
import { SignalBadge, SymbolCell, Conf, Skeleton, useSort, Th, Td } from '../components/ui';
import { Sparkline } from '../components/Charts';

import type { WatchlistItem } from '../types';

function inr(n: number) {
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] p-[calc(17px*var(--u))] flex flex-col gap-1">
      <span className="text-[12.5px] font-medium text-[var(--text-2)]">{label}</span>
      <span className="font-bold tabular-nums text-[22px] tracking-tight" style={{ color: color ?? 'var(--text)' }}>{value}</span>
      {sub && <span className="text-[11.5px] text-[var(--text-3)]">{sub}</span>}
    </div>
  );
}

function WatchCard({ item, onRemove, onClick }: { item: WatchlistItem; onRemove: () => void; onClick: () => void }) {
  const sparkColor = item?.signal === 'SELL' ? '#EF4444' : item?.signal === 'HOLD' ? '#F59E0B' : '#10B981';
  const retColor   = (item?.expReturn ?? 0) >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <div
      className="bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] p-[calc(15px*var(--u))] cursor-pointer transition-all hover:border-[var(--border-strong)] hover:-translate-y-0.5 relative group"
      onClick={onClick}
    >
      {/* Remove button */}
      <button
        onClick={e => { e.stopPropagation(); onRemove(); }}
        className="absolute top-3 right-3 w-7 h-7 rounded-[8px] grid place-items-center text-[var(--text-3)] opacity-0 group-hover:opacity-100 transition-all hover:bg-[var(--red-soft)] hover:text-[var(--red)] border border-transparent"
      >
        <X size={14} />
      </button>

      {/* Top row */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <SymbolCell symbol={item?.symbol ?? ''} name={item?.name ?? ''} sector={item?.sector ?? ''} showSector={false} />
        <SignalBadge signal={item?.signal} />
      </div>

      {/* Price + sparkline */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex flex-col gap-0.5">
          <span className="font-mono text-[18px] font-bold text-[var(--text)]">{inr(item?.price ?? 0)}</span>
          <span className="text-[12.5px] font-semibold tabular-nums" style={{ color: (item?.change ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {(item?.change ?? 0) >= 0 ? '+' : ''}{(item?.change ?? 0).toFixed(2)}%
          </span>
        </div>
        <Sparkline data={item?.spark ?? []} color={sparkColor} w={88} h={36} />
      </div>

      <div className="h-px bg-[var(--border)] mb-3" />

      {/* Signal confidence */}
      <div className="mb-3">
        <div className="flex justify-between text-[11px] text-[var(--text-3)] mb-1">
          <span>Confidence</span>
          <span className="font-mono font-bold text-[var(--text)]">{item?.confidence ?? 0}%</span>
        </div>
        <Conf value={item?.confidence ?? 0} />
      </div>

      {/* Horizon + exp return */}
      <div className="flex justify-between items-center">
        <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-[var(--surface-3)] text-[var(--text-2)] border border-[var(--border)]">
          {item?.horizon}
        </span>
        <span className="font-mono font-bold text-[13px]" style={{ color: retColor }}>
          {(item?.expReturn ?? 0) >= 0 ? '+' : ''}{(item?.expReturn ?? 0).toFixed(2)}%
        </span>
      </div>

      {/* Alerts */}
      {(item?.alertAbove || item?.alertBelow) && (
        <div className="mt-3 flex gap-2 flex-wrap">
          {(item?.alertAbove ?? 0) > 0 && (
            <span className="inline-flex items-center gap-1 h-[20px] px-2 rounded-full text-[10.5px] font-semibold bg-[var(--green-soft)] text-[var(--green)]">
              <Bell size={9} /> ↑ {inr(item?.alertAbove ?? 0)}
            </span>
          )}
          {(item?.alertBelow ?? 0) > 0 && (
            <span className="inline-flex items-center gap-1 h-[20px] px-2 rounded-full text-[10.5px] font-semibold bg-[var(--red-soft)] text-[var(--red)]">
              <Bell size={9} /> ↓ {inr(item?.alertBelow ?? 0)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default function WatchlistPage() {
  const { user }  = useAuth();
  const toast     = useToast();
  const [view,    setView]    = useState<'grid' | 'table'>('grid');
  const navigate = useNavigate();
  const [addOpen, setAddOpen] = useState(false);
  const [addQ,    setAddQ]    = useState('');
  const { sort, toggle } = useSort('confidence', 'desc');

  const { data: wlRes,    isLoading } = useGetWatchlistQuery(user!.id, { skip: !user });
  const deferredAddQ = useDeferredValue(addQ);
  const { data: stockRes, isLoading: searchLoading } = useGetStocksQuery({ search: deferredAddQ, size: 6 }, { skip: !deferredAddQ || deferredAddQ.length < 2 });
  const [removeFromWatchlist] = useRemoveFromWatchlistMutation();
  const [addToWatchlist] = useAddToWatchlistMutation();

  const items: WatchlistItem[] = (wlRes as any)?.data ?? [];
  const searchResults = (stockRes as any)?.data ?? [];

  const buys   = (items ?? []).filter(i => i?.signal === 'BUY' || i?.signal === 'STRONG BUY' as any).length;
  const sells  = (items ?? []).filter(i => i?.signal === 'SELL' || i?.signal === 'STRONG SELL' as any).length;
  const alerts = (items ?? []).filter(i => (i?.alertAbove ?? 0) > 0 || (i?.alertBelow ?? 0) > 0).length;
  const profit = (items ?? []).filter(i => (i?.change ?? 0) >= 0).length;

  async function handleRemove(symbol: string) {
    if (!user) return;
    try {
      await removeFromWatchlist({ userId: user.id, symbol }).unwrap();
      toast({ type: 'info', title: `${symbol} removed from watchlist` });
    } catch { toast({ type: 'error', title: 'Remove failed' }); }
  }

  const sorted = ([...(items ?? [])]).sort((a, b) => {
    const va = a?.[sort.key as keyof WatchlistItem], vb = b?.[sort.key as keyof WatchlistItem];
    let cmp = 0;
    if (typeof va === 'number' && typeof vb === 'number') cmp = va - vb;
    else if (typeof va === 'string' && typeof vb === 'string') cmp = va.localeCompare(vb);
    return sort.dir === 'asc' ? cmp : -cmp;
  });

  return (
    <div className="flex flex-col dgap animate-page-in">

      {/* ── Header ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-bold tracking-tight m-0 text-[var(--text)] text-[calc(25px*var(--u))]">
            Watchlist
          </h1>
          <p className="text-[var(--text-2)] text-[13.5px] mt-1 m-0">
            <b className="tabular-nums">{items.length}</b> stocks ·{' '}
            <span className="text-[var(--green)] font-semibold">{buys} BUY</span> ·{' '}
            <span className="text-[var(--red)] font-semibold">{sells} SELL</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Grid/Table toggle */}
          <div className="inline-flex bg-[var(--surface-2)] border border-[var(--border)] rounded-[10px] p-[3px] gap-[2px]">
            {([['grid', LayoutGrid], ['table', List]] as const).map(([v, Icon]) => (
              <button key={v} onClick={() => setView(v)}
                className={`w-9 h-8 rounded-[7px] grid place-items-center border-none cursor-pointer transition-colors ${view === v ? 'bg-[var(--accent)] text-white' : 'bg-transparent text-[var(--text-2)] hover:text-[var(--text)]'}`}>
                <Icon size={16} />
              </button>
            ))}
          </div>
          <button
            onClick={() => setAddOpen(true)}
            className="inline-flex items-center gap-2 h-10 px-4 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border-0 bg-[var(--accent)] text-white shadow-[0_4px_14px_rgba(59,130,246,.32)]"
          >
            <Plus size={17} /> Add Stocks
          </button>
        </div>
      </div>

      {/* ── Stat cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-[calc(16px*var(--u))]">
        {isLoading ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} h={90} rounded="14px" />) : (<>
          <StatCard label="Watchlist Items" value={items.length} sub="across all sectors" />
          <StatCard label="Avg Confidence"  value={(items ?? []).length ? Math.round((items ?? []).reduce((s, i) => s + (i?.confidence ?? 0), 0) / (items ?? []).length) + '%' : '—'} color="var(--accent)" />
          <StatCard label="Price Alerts"    value={alerts} sub="active thresholds" color="var(--gold)" />
          <StatCard label="In Profit Zone"  value={profit} sub={`of ${items.length} stocks`} color="var(--green)" />
        </>)}
      </div>

      {/* ── Empty state ── */}
      {!isLoading && items.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-16 h-16 rounded-full bg-[var(--surface-2)] grid place-items-center text-[var(--text-3)]">
            <svg width={28} height={28} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
          </div>
          <div className="text-center">
            <p className="text-[15px] font-semibold text-[var(--text)] mb-1">Your watchlist is empty</p>
            <p className="text-[13.5px] text-[var(--text-2)]">Add stocks to track signals, set alerts, and monitor performance</p>
          </div>
          <button
            onClick={() => setAddOpen(true)}
            className="inline-flex items-center gap-2 h-10 px-5 rounded-[11px] font-sans text-[13.5px] font-semibold cursor-pointer border-0 bg-[var(--accent)] text-white shadow-[0_4px_14px_rgba(59,130,246,.32)]"
          >
            <Plus size={17} /> Browse Signals
          </button>
        </div>
      )}

      {/* ── Grid view ── */}
      {!isLoading && items.length > 0 && view === 'grid' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {(sorted ?? []).map(item => (
            <WatchCard key={item.symbol} item={item} onRemove={() => handleRemove(item.symbol)} onClick={() => navigate(`/stocks/${encodeURIComponent(item?.symbol ?? '')}`)} />
          ))}
        </div>
      )}

      {/* ── Table view ── */}
      {!isLoading && items.length > 0 && view === 'table' && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr>
                  <Th label="Stock"      sortKey="symbol"     sort={sort} onToggle={toggle} />
                  <Th label="LTP"        sortKey="price"      sort={sort} onToggle={toggle} align="right" />
                  <Th label="Change"     sortKey="change"     sort={sort} onToggle={toggle} align="right" />
                  <Th label="Signal"     sortKey="signal"     sort={sort} onToggle={toggle} />
                  <Th label="Confidence" sortKey="confidence" sort={sort} onToggle={toggle} />
                  <Th label="Horizon"    sortKey="horizon"    sort={sort} onToggle={toggle} />
                  <Th label="Exp. Ret"   sortKey="expReturn"  sort={sort} onToggle={toggle} align="right" />
                  <Th label="Alerts"     sortKey="alertAbove" sort={sort} onToggle={toggle} />
                  <Th label=""           sortKey="symbol"     sort={sort} onToggle={toggle} />
                </tr>
              </thead>
              <tbody>
                {(sorted ?? []).map(item => (
                  <tr key={item?.symbol} className="cursor-pointer transition-colors hover:bg-[var(--surface-2)]" onClick={() => navigate(`/stocks/${encodeURIComponent(item?.symbol ?? '')}`)}>
                    <Td><SymbolCell symbol={item?.symbol ?? ''} name={item?.name ?? ''} sector={item?.sector ?? ''} showSector={false} /></Td>
                    <Td align="right" mono>{inr(item?.price ?? 0)}</Td>
                    <Td align="right">
                      <span className="font-mono font-semibold tabular-nums text-[12.5px]" style={{ color: (item?.change ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {(item?.change ?? 0) >= 0 ? '+' : ''}{(item?.change ?? 0).toFixed(2)}%
                      </span>
                    </Td>
                    <Td><SignalBadge signal={item?.signal} /></Td>
                    <Td><div className="min-w-[110px]"><Conf value={item?.confidence ?? 0} /></div></Td>
                    <Td>
                      <span className="inline-flex items-center h-[22px] px-2 rounded-full text-[11px] font-semibold bg-[var(--surface-3)] text-[var(--text-2)] border border-[var(--border)]">
                        {item?.horizon}
                      </span>
                    </Td>
                    <Td align="right">
                      <span className="font-mono font-semibold tabular-nums text-[12.5px]" style={{ color: (item?.expReturn ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {(item?.expReturn ?? 0) >= 0 ? '+' : ''}{(item?.expReturn ?? 0).toFixed(2)}%
                      </span>
                    </Td>
                    <Td>
                      <div className="flex gap-1">
                        {(item?.alertAbove ?? 0) > 0 && <span className="text-[10px] font-semibold text-[var(--green)] bg-[var(--green-soft)] px-1.5 py-0.5 rounded-full">↑{inr(item?.alertAbove ?? 0)}</span>}
                        {(item?.alertBelow ?? 0) > 0 && <span className="text-[10px] font-semibold text-[var(--red)] bg-[var(--red-soft)] px-1.5 py-0.5 rounded-full">↓{inr(item?.alertBelow ?? 0)}</span>}
                      </div>
                    </Td>
                    <Td>
                      <button
                        onClick={e => { e.stopPropagation(); handleRemove(item?.symbol ?? ''); }}
                        className="w-7 h-7 rounded-[7px] grid place-items-center text-[var(--text-3)] hover:bg-[var(--red-soft)] hover:text-[var(--red)] transition-colors border-none bg-transparent"
                      >
                        <X size={14} />
                      </button>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Add stocks modal ── */}
      {addOpen && (
        <div className="fixed inset-0 z-[100] grid place-items-center p-5">
          <div className="fixed inset-0 bg-black/55 backdrop-blur-sm" onClick={() => { setAddOpen(false); setAddQ(''); }} />
          <div className="relative z-[101] w-full max-w-md bg-[var(--surface)] border border-[var(--border-strong)] rounded-[var(--radius-lg)] shadow-[var(--shadow-lg)] overflow-hidden">
            <div className="flex items-center justify-between p-5 border-b border-[var(--border)]">
              <h3 className="m-0 text-[17px] font-bold text-[var(--text)]">Add to Watchlist</h3>
              <button onClick={() => { setAddOpen(false); setAddQ(''); }} className="w-8 h-8 rounded-[8px] border border-[var(--border)] bg-transparent text-[var(--text-2)] grid place-items-center cursor-pointer hover:bg-[var(--surface-hover)]">
                <X size={16} />
              </button>
            </div>
            <div className="p-5">
              <input
                autoFocus
                value={addQ}
                onChange={e => setAddQ(e.target.value)}
                placeholder="Search symbol or company name…"
                className="w-full h-11 px-4 rounded-[11px] border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)] font-sans text-[13.5px] outline-none box-border focus:border-[var(--accent)] transition-colors"
              />
              <div className="mt-2 flex flex-col gap-1">
                {searchLoading && <div className="text-center py-4 text-[13px] text-[var(--text-3)]">Searching…</div>}
                {!searchLoading && (searchResults ?? []).slice(0, 6).map((s: any) => (
                  <button key={s?.symbol}
                    onClick={async () => {
                      if (!user) return;
                      try {
                        await addToWatchlist({ userId: user.id, symbol: s?.symbol ?? '' }).unwrap();
                        toast({ type: 'success', title: `${s?.symbol ?? ''} added to watchlist` });
                      } catch {
                        toast({ type: 'error', title: 'Failed to add to watchlist' });
                      }
                      setAddOpen(false); setAddQ('');
                    }}
                    className="flex items-center gap-3 h-12 px-3 rounded-[10px] bg-transparent border-none text-left cursor-pointer hover:bg-[var(--surface-2)] transition-colors w-full"
                  >
                    <span className="w-8 h-8 rounded-[8px] grid place-items-center font-bold text-[11px] flex-shrink-0 bg-[var(--accent-soft)] text-[var(--accent-2)]">
                      {(s?.symbol ?? '').slice(0, 2)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-[13px] text-[var(--text)]">{s?.symbol}</div>
                      <div className="text-[11.5px] text-[var(--text-3)] truncate">{s?.name}</div>
                    </div>
                    <SignalBadge signal={s?.signal} />
                  </button>
                ))}
                {!searchLoading && addQ.length >= 2 && searchResults.length === 0 && (
                  <div className="text-center py-4 text-[13px] text-[var(--text-3)]">No stocks found for "{addQ}"</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
