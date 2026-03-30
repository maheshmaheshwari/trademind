import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Search, ChevronRight, Loader2 } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { getPositions, getPortfolioSummary, squareOff } from '../api';
import Pagination from '../components/Pagination';

const tabs = ['Open Positions', 'Performance'];

export default function PortfolioPage() {
    const { user, refreshUser } = useAuth();
    const [activeTab, setActiveTab] = useState(0);
    const [positions, setPositions] = useState<any[]>([]);
    const [total, setTotal] = useState(0);
    const [portfolio, setPortfolio] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [squaringOff, setSquaringOff] = useState<string | null>(null);

    // Server-side params
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(25);
    const [search, setSearch] = useState('');

    // Debounced search
    const [debouncedSearch, setDebouncedSearch] = useState('');
    useEffect(() => {
        const t = setTimeout(() => setDebouncedSearch(search), 400);
        return () => clearTimeout(t);
    }, [search]);

    const fetchData = useCallback(async () => {
        if (!user) return;
        setLoading(true);
        try {
            const [pos, port] = await Promise.all([
                getPositions(user.id, {
                    page,
                    size: pageSize,
                    globalFilter: debouncedSearch || undefined,
                }),
                getPortfolioSummary(user.id),
            ]);
            setPositions(pos.positions || []);
            setTotal(pos.total ?? pos.positions?.length ?? 0);
            setPortfolio(port);
        } catch {
            setPositions([]);
        } finally {
            setLoading(false);
        }
    }, [user, page, pageSize, debouncedSearch]);

    useEffect(() => { fetchData(); }, [fetchData]);
    useEffect(() => { setPage(0); }, [debouncedSearch]);

    const handleSquareOff = async (symbol: string) => {
        if (!user) return;
        setSquaringOff(symbol);
        try {
            await squareOff(user.id, symbol);
            await fetchData();
            await refreshUser();
        } catch (err: any) {
            alert(err.message);
        } finally {
            setSquaringOff(null);
        }
    };

    const fmt = (n: number) => n?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '0';
    const pnlColor = (n: number) => (n || 0) >= 0 ? 'text-green-400' : 'text-red-400';

    return (
        <>
            <div className="flex items-center gap-2 text-sm text-slate-400">
                <Link to="/dashboard" className="hover:text-white">Dashboard</Link>
                <ChevronRight className="w-3 h-3" />
                <span className="text-white font-medium">Portfolio & Positions</span>
            </div>

            <div className="flex flex-wrap justify-between items-center gap-4">
                <h1 className="text-white text-3xl md:text-4xl font-black tracking-tight">Portfolio Overview</h1>
            </div>

            {/* Stats */}
            {portfolio && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                        <p className="text-slate-400 text-sm">Total Investment</p>
                        <p className="text-white text-2xl font-bold">₹{fmt(portfolio.invested)}</p>
                        <div className="w-8 h-1 bg-primary rounded-full mt-1" />
                    </div>
                    <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                        <p className="text-slate-400 text-sm">Total Value</p>
                        <p className="text-white text-2xl font-bold">₹{fmt(portfolio.total_value)}</p>
                    </div>
                    <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                        <p className="text-slate-400 text-sm">Unrealized P&L</p>
                        <div className="flex items-center gap-2">
                            <p className={`text-2xl font-bold ${pnlColor(portfolio.unrealized_pnl)}`}>
                                {portfolio.unrealized_pnl >= 0 ? '+' : ''}₹{fmt(portfolio.unrealized_pnl)}
                            </p>
                        </div>
                    </div>
                    <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                        <p className="text-slate-400 text-sm">Total P&L</p>
                        <div className="flex items-center gap-2">
                            <p className={`text-2xl font-bold ${pnlColor(portfolio.total_pnl)}`}>
                                {portfolio.total_pnl >= 0 ? '+' : ''}₹{fmt(portfolio.total_pnl)}
                            </p>
                        </div>
                        <p className="text-slate-500 text-xs">Win Rate: {portfolio.win_rate}%</p>
                    </div>
                </div>
            )}

            {/* Tabs + Search */}
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex gap-6">
                    {tabs.map((tab, i) => (
                        <button key={tab} onClick={() => setActiveTab(i)} className={`pb-2 text-sm font-medium transition-colors ${activeTab === i ? 'text-white border-b-2 border-primary' : 'text-slate-400 hover:text-white'}`}>
                            {tab}
                        </button>
                    ))}
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 bg-surface-dark border border-slate-700 rounded-xl px-3 py-2">
                        <Search className="w-4 h-4 text-slate-400" />
                        <input type="text" placeholder="Filter positions..." value={search} onChange={(e) => setSearch(e.target.value)} className="bg-transparent text-sm text-white placeholder-slate-400 outline-none w-40" />
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="w-full overflow-hidden rounded-xl border border-slate-800 bg-surface-dark/50 shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-slate-700 bg-surface-dark">
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400">Instrument</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-center">Qty.</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">Avg. Price</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">Current</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">Invested</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">P&L</th>
                                <th className="p-5 text-xs font-bold uppercase tracking-wider text-slate-400 text-center">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {loading ? (
                                <tr><td colSpan={7} className="p-16 text-center">
                                    <div className="flex items-center justify-center gap-3 text-slate-400">
                                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                        Loading positions...
                                    </div>
                                </td></tr>
                            ) : positions.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="p-10 text-center text-slate-500">
                                        No open positions. <Link to="/signals" className="text-primary hover:underline">Browse AI signals</Link> to start trading.
                                    </td>
                                </tr>
                            ) : (
                                positions.map((p) => (
                                    <tr key={p.symbol} className="group hover:bg-slate-800/50 transition-colors">
                                        <td className="p-5">
                                            <div className="flex items-center gap-3">
                                                <div className="h-10 w-10 rounded-lg bg-primary/20 flex items-center justify-center text-primary text-xs font-bold">
                                                    {p.symbol?.replace('.NS', '').slice(0, 3)}
                                                </div>
                                                <div>
                                                    <p className="font-bold text-white">{p.symbol?.replace('.NS', '')}</p>
                                                    <p className="text-xs text-slate-500">{p.name || p.symbol}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-5 text-center text-white font-medium">{p.quantity}</td>
                                        <td className="p-5 text-right font-mono text-slate-300">₹{fmt(p.avg_buy_price)}</td>
                                        <td className="p-5 text-right font-mono text-white font-medium">₹{fmt(p.current_price)}</td>
                                        <td className="p-5 text-right font-mono text-slate-300">₹{fmt(p.invested_amount)}</td>
                                        <td className="p-5 text-right">
                                            <p className={`font-bold ${pnlColor(p.unrealized_pnl)}`}>
                                                {(p.unrealized_pnl || 0) >= 0 ? '+' : ''}₹{fmt(p.unrealized_pnl || 0)}
                                            </p>
                                            <p className={`text-xs ${pnlColor(p.unrealized_pnl_pct)}`}>
                                                {(p.unrealized_pnl_pct || 0) >= 0 ? '+' : ''}{(p.unrealized_pnl_pct || 0).toFixed(2)}%
                                            </p>
                                        </td>
                                        <td className="p-5 text-center">
                                            <button
                                                onClick={() => handleSquareOff(p.symbol)}
                                                disabled={squaringOff === p.symbol}
                                                className="px-3 py-1.5 rounded-lg border border-red-500/30 text-red-400 text-xs font-medium hover:bg-red-500/10 transition-colors disabled:opacity-50 flex items-center gap-1 mx-auto"
                                            >
                                                {squaringOff === p.symbol ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
                                                Square Off
                                            </button>
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
