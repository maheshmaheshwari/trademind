import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Search, TrendingUp, TrendingDown, BarChart3, Filter } from 'lucide-react';
import { getAllStocks } from '../api';
import Pagination from '../components/Pagination';

interface Stock {
    symbol: string;
    name: string;
    sector: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    change: number;
    change_pct: number;
    prev_close: number;
    date: string;
}

export default function MarketPage() {
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [total, setTotal] = useState(0);
    const [sectors, setSectors] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);

    // Server-side params
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(25);
    const [search, setSearch] = useState('');
    const [selectedSector, setSelectedSector] = useState('All');
    const [sortKey, setSortKey] = useState<string>('symbol');
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

    // Debounced search
    const [debouncedSearch, setDebouncedSearch] = useState('');
    useEffect(() => {
        const t = setTimeout(() => setDebouncedSearch(search), 400);
        return () => clearTimeout(t);
    }, [search]);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const res = await getAllStocks({
                page,
                size: pageSize,
                sort: sortKey,
                order: sortDir,
                globalFilter: debouncedSearch || undefined,
                sector: selectedSector !== 'All' ? selectedSector : undefined,
            });
            setStocks(res.stocks || res.data || []);
            setTotal(res.total ?? res.stocks?.length ?? 0);
            if (res.sectors?.length) setSectors(res.sectors);
        } catch {
            setStocks([]);
        } finally {
            setLoading(false);
        }
    }, [page, pageSize, sortKey, sortDir, debouncedSearch, selectedSector]);

    useEffect(() => { fetchData(); }, [fetchData]);

    // Reset page on filter change
    useEffect(() => { setPage(0); }, [debouncedSearch, selectedSector]);

    const handleSort = (key: string) => {
        if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
        else { setSortKey(key); setSortDir('asc'); }
        setPage(0);
    };

    const fmt = (n: number) => n != null ? n.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '—';
    const volFmt = (v: number) => v >= 10000000 ? `${(v / 10000000).toFixed(1)}Cr` : v >= 100000 ? `${(v / 100000).toFixed(1)}L` : v?.toLocaleString();

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-white text-2xl font-bold flex items-center gap-2"><BarChart3 className="w-6 h-6 text-primary" /> Market</h1>
                    <p className="text-slate-400 text-sm mt-1">Nifty 500 Stocks — {total.toLocaleString()} stocks</p>
                </div>
            </div>

            {/* Search + Sector Filter */}
            <div className="flex gap-3 items-center flex-wrap">
                <div className="relative flex-1 min-w-[200px] max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                    <input
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        placeholder="Search stocks..."
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-surface-dark border border-slate-800 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-primary/50"
                    />
                </div>
                <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-slate-400" />
                    <select
                        value={selectedSector}
                        onChange={e => setSelectedSector(e.target.value)}
                        className="bg-surface-dark border border-slate-800 text-white text-sm rounded-xl px-3 py-2.5 focus:outline-none focus:border-primary/50"
                    >
                        <option value="All">All Sectors</option>
                        {sectors.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>
            </div>

            {/* Table */}
            <div className="rounded-xl border border-slate-800 bg-surface-dark overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-slate-800">
                                {[
                                    { key: 'name', label: 'Stock' },
                                    { key: 'sector', label: 'Sector' },
                                    { key: 'close', label: 'Price' },
                                    { key: 'change_pct', label: 'Change %' },
                                    { key: 'volume', label: 'Volume' },
                                    { key: 'open', label: 'Open' },
                                    { key: 'high', label: 'High' },
                                    { key: 'low', label: 'Low' },
                                ].map(col => (
                                    <th key={col.key}
                                        onClick={() => handleSort(col.key)}
                                        className="text-left text-slate-400 text-xs font-semibold uppercase tracking-wider p-4 cursor-pointer hover:text-white transition-colors select-none"
                                    >
                                        <span className="flex items-center gap-1">
                                            {col.label}
                                            {sortKey === col.key && (
                                                <span className="text-primary">{sortDir === 'asc' ? '↑' : '↓'}</span>
                                            )}
                                        </span>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {loading ? (
                                <tr><td colSpan={8} className="p-16 text-center">
                                    <div className="flex items-center justify-center gap-3 text-slate-400">
                                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                        Loading stocks...
                                    </div>
                                </td></tr>
                            ) : stocks.length === 0 ? (
                                <tr><td colSpan={8} className="p-16 text-center text-slate-500">No stocks found</td></tr>
                            ) : (
                                stocks.map(s => {
                                    const isUp = s.change_pct >= 0;
                                    return (
                                        <tr key={s.symbol} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors group">
                                            <td className="p-4">
                                                <Link to={`/market/${s.symbol.replace('.NS', '')}`} className="flex items-center gap-3 group-hover:text-primary transition-colors">
                                                    <div className={`w-9 h-9 rounded-lg flex items-center justify-center text-white text-xs font-bold ${isUp ? 'bg-green-600/80' : 'bg-red-600/80'}`}>
                                                        {s.symbol.replace('.NS', '').slice(0, 3)}
                                                    </div>
                                                    <div>
                                                        <p className="text-white font-semibold text-sm">{s.name}</p>
                                                        <p className="text-slate-500 text-xs">{s.symbol.replace('.NS', '')}</p>
                                                    </div>
                                                </Link>
                                            </td>
                                            <td className="p-4 text-slate-400 text-xs">{s.sector}</td>
                                            <td className="p-4 text-white font-semibold">₹{fmt(s.close)}</td>
                                            <td className="p-4">
                                                <div className={`flex items-center gap-1 font-semibold text-sm ${isUp ? 'text-green-400' : 'text-red-400'}`}>
                                                    {isUp ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                                                    {isUp ? '+' : ''}{s.change_pct?.toFixed(2)}%
                                                </div>
                                            </td>
                                            <td className="p-4 text-slate-300 text-sm">{volFmt(s.volume)}</td>
                                            <td className="p-4 text-slate-400">₹{fmt(s.open)}</td>
                                            <td className="p-4 text-green-400/80">₹{fmt(s.high)}</td>
                                            <td className="p-4 text-red-400/80">₹{fmt(s.low)}</td>
                                        </tr>
                                    );
                                })
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
        </div>
    );
}
