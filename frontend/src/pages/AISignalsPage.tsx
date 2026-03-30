import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Search, Brain } from 'lucide-react';
import { getLatestSignals } from '../api';
import Pagination from '../components/Pagination';

interface Signal {
    symbol: string;
    name: string;
    signal: string;
    confidence: number;
    trade: {
        buy_price: number;
        target_price: number;
        stop_loss: number;
        risk_reward: number;
    };
    reasons: string[];
}

const sectorColors: Record<string, string> = {
    'STRONG BUY': 'bg-green-500/20 text-green-400',
    BUY: 'bg-green-500/10 text-green-400',
    HOLD: 'bg-amber-500/10 text-amber-400',
    SELL: 'bg-red-500/10 text-red-400',
    'STRONG SELL': 'bg-red-500/20 text-red-400',
};

export default function AISignalsPage() {
    const [signals, setSignals] = useState<Signal[]>([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(true);

    // Server-side params
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(25);
    const [search, setSearch] = useState('');
    const [activeFilter, setActiveFilter] = useState('All');
    const [sortKey, setSortKey] = useState('confidence');
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

    // Debounced search
    const [debouncedSearch, setDebouncedSearch] = useState('');
    useEffect(() => {
        const t = setTimeout(() => setDebouncedSearch(search), 400);
        return () => clearTimeout(t);
    }, [search]);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const res = await getLatestSignals({
                page,
                size: pageSize,
                sort: sortKey,
                order: sortDir,
                globalFilter: debouncedSearch || (activeFilter !== 'All' ? activeFilter : undefined),
            });
            const trades = res?.data || [];
            setSignals(trades);
            setTotal(res?.total ?? trades.length);
        } catch {
            setSignals([]);
        } finally {
            setLoading(false);
        }
    }, [page, pageSize, sortKey, sortDir, debouncedSearch, activeFilter]);

    useEffect(() => { fetchData(); }, [fetchData]);
    useEffect(() => { setPage(0); }, [debouncedSearch, activeFilter]);

    const fmt = (n: number) => n?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '—';
    const filters = ['All', 'BUY', 'SELL', 'HOLD'];

    const handleSort = (key: string) => {
        if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
        else { setSortKey(key); setSortDir('desc'); }
        setPage(0);
    };

    return (
        <>
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-white text-3xl md:text-4xl font-black tracking-tight flex items-center gap-3">
                        <Brain className="w-8 h-8 text-primary" /> AI Trade Signals
                    </h1>
                    <p className="text-slate-400 mt-2">ML-generated signals with confidence scores • {total.toLocaleString()} stocks analyzed</p>
                </div>
            </div>

            {/* Search + Filters */}
            <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2 bg-surface-dark border border-slate-700 rounded-xl px-4 py-3 flex-1 max-w-sm">
                    <Search className="w-4 h-4 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search by symbol or name..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        className="bg-transparent text-sm text-white placeholder-slate-400 outline-none flex-1"
                    />
                </div>
                <div className="flex gap-2">
                    {filters.map(f => (
                        <button
                            key={f}
                            onClick={() => setActiveFilter(f)}
                            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${activeFilter === f ? 'bg-primary text-white' : 'bg-surface-dark border border-slate-700 text-slate-400 hover:text-white'
                                }`}
                        >
                            {f}
                        </button>
                    ))}
                </div>
            </div>

            {/* Table */}
            <div className="w-full overflow-hidden rounded-xl border border-slate-800 bg-surface-dark/50 shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-slate-700 bg-surface-dark">
                                {[
                                    { key: 'symbol', label: 'Symbol' },
                                    { key: 'signal', label: 'Signal', center: true },
                                    { key: 'confidence', label: 'Confidence', center: true },
                                    { key: 'buy_price', label: 'Buy Price', right: true },
                                    { key: 'target_price', label: 'Target', right: true },
                                    { key: 'stop_loss', label: 'Stop Loss', right: true },
                                    { key: 'risk_reward', label: 'R:R', center: true },
                                    { key: '', label: 'Action', center: true },
                                ].map(col => (
                                    <th key={col.label}
                                        onClick={() => col.key && handleSort(col.key)}
                                        className={`p-4 text-xs font-bold uppercase tracking-wider text-slate-400 ${col.center ? 'text-center' : col.right ? 'text-right' : ''} ${col.key ? 'cursor-pointer hover:text-white transition-colors' : ''}`}
                                    >
                                        <span className="inline-flex items-center gap-1">
                                            {col.label}
                                            {sortKey === col.key && <span className="text-primary">{sortDir === 'asc' ? '↑' : '↓'}</span>}
                                        </span>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {loading ? (
                                <tr><td colSpan={8} className="p-16 text-center">
                                    <div className="flex items-center justify-center gap-3 text-slate-400">
                                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                        Loading signals...
                                    </div>
                                </td></tr>
                            ) : signals.length === 0 ? (
                                <tr><td colSpan={8} className="p-10 text-center text-slate-500">No signals found. Try adjusting your search or filters.</td></tr>
                            ) : (
                                signals.map((s) => (
                                    <tr key={s.symbol} className="hover:bg-slate-800/50 transition-colors">
                                        <td className="p-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-9 h-9 rounded-lg flex items-center justify-center text-white text-xs font-bold ${s.signal?.includes('BUY') ? 'bg-green-600' : s.signal?.includes('SELL') ? 'bg-red-600' : 'bg-amber-600'
                                                    }`}>
                                                    {s.symbol?.replace('.NS', '').slice(0, 3)}
                                                </div>
                                                <div>
                                                    <p className="text-white font-bold text-sm">{s.symbol?.replace('.NS', '')}</p>
                                                    <p className="text-slate-500 text-xs truncate max-w-[120px]">{s.name}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4 text-center">
                                            <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${sectorColors[s.signal] || 'bg-slate-700 text-slate-300'}`}>
                                                {s.signal}
                                            </span>
                                        </td>
                                        <td className="p-4 text-center">
                                            <div className="flex flex-col items-center gap-1">
                                                <span className="text-white font-bold text-sm">{s.confidence?.toFixed(0)}%</span>
                                                <div className="w-16 h-1.5 bg-slate-700 rounded-full">
                                                    <div className={`h-1.5 rounded-full ${s.confidence >= 75 ? 'bg-green-400' : s.confidence >= 50 ? 'bg-amber-400' : 'bg-red-400'}`} style={{ width: `${s.confidence}%` }} />
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4 text-right font-mono text-white text-sm">₹{fmt(s.trade?.buy_price || 0)}</td>
                                        <td className="p-4 text-right font-mono text-green-400 text-sm">₹{fmt(s.trade?.target_price || 0)}</td>
                                        <td className="p-4 text-right font-mono text-red-400 text-sm">₹{fmt(s.trade?.stop_loss || 0)}</td>
                                        <td className="p-4 text-center font-mono text-white text-sm">{s.trade?.risk_reward?.toFixed(1) || '—'}</td>
                                        <td className="p-4 text-center">
                                            <Link
                                                to={`/trade/${s.symbol?.replace('.NS', '')}`}
                                                className="inline-block px-4 py-2 rounded-lg bg-primary/20 text-primary text-xs font-bold hover:bg-primary/30 transition-colors"
                                            >
                                                Trade
                                            </Link>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
                <Pagination
                    page={page}
                    pageSize={pageSize}
                    total={total}
                    onPageChange={setPage}
                    onPageSizeChange={setPageSize}
                    loading={loading}
                />
            </div>
        </>
    );
}
