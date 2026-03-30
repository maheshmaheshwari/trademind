import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Search, Plus, ChevronRight, ChevronDown } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { getOrders } from '../api';
import Pagination from '../components/Pagination';

const statusTabs = ['All Orders', 'Executed', 'Pending', 'Cancelled'];

const statusColors: Record<string, string> = {
    EXECUTED: 'text-green-400 bg-green-400/10 border-green-400/20',
    PENDING: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20',
    CANCELLED: 'text-slate-400 bg-slate-400/10 border-slate-400/20',
};

const purposeLabels: Record<string, string> = {
    ENTRY: 'Buy Entry',
    STOP_LOSS: 'Stop Loss',
    TARGET: 'Target',
    SQUARE_OFF: 'Square Off',
};

export default function OrderHistoryPage() {
    const { user } = useAuth();
    const [orders, setOrders] = useState<any[]>([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(true);
    const [expandedBrackets, setExpandedBrackets] = useState<Set<string>>(new Set());

    // Server-side params
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(25);
    const [search, setSearch] = useState('');
    const [activeTab, setActiveTab] = useState(0);

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
            // Combine tab filter + search into globalFilter
            const tabName = statusTabs[activeTab];
            let filter = debouncedSearch || '';
            if (tabName !== 'All Orders') filter = filter ? `${filter} ${tabName.toUpperCase()}` : tabName.toUpperCase();

            const res = await getOrders(user.id, {
                page,
                size: pageSize,
                sort: 'created_at',
                order: 'desc',
                globalFilter: filter || undefined,
            });
            setOrders(res.orders || []);
            setTotal(res.total ?? res.orders?.length ?? 0);
        } catch {
            setOrders([]);
        } finally {
            setLoading(false);
        }
    }, [user, page, pageSize, debouncedSearch, activeTab]);

    useEffect(() => { fetchData(); }, [fetchData]);
    useEffect(() => { setPage(0); }, [debouncedSearch, activeTab]);

    const toggleBracket = (bracketId: string) => {
        setExpandedBrackets((prev) => {
            const next = new Set(prev);
            if (next.has(bracketId)) next.delete(bracketId);
            else next.add(bracketId);
            return next;
        });
    };

    const fmt = (n: number) => n?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '—';

    // Stats from current visible data (we don't have global stats from server, show per-page)
    const executed = orders.filter((o) => o.status === 'EXECUTED').length;
    const totalPnl = orders.reduce((a, o) => a + (o.pnl || 0), 0);

    return (
        <>
            <div className="flex items-center gap-2 text-sm text-slate-400">
                <Link to="/dashboard" className="hover:text-white">Dashboard</Link>
                <ChevronRight className="w-3 h-3" />
                <span className="text-white font-medium">Order History</span>
            </div>

            <div className="flex flex-wrap justify-between items-start gap-4">
                <div>
                    <h1 className="text-white text-3xl md:text-4xl font-black tracking-tight">Order History</h1>
                    <p className="text-slate-400 text-sm mt-2 max-w-xl">Track bracket orders, stop-loss triggers, and realized P&L.</p>
                </div>
                <Link to="/signals" className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-primary text-white text-sm font-bold hover:bg-primary/90 transition-colors">
                    <Plus className="w-4 h-4" /> New Order
                </Link>
            </div>

            {/* Search + Filters */}
            <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2 bg-surface-dark border border-slate-700 rounded-xl px-4 py-3 flex-1 max-w-sm">
                    <Search className="w-4 h-4 text-slate-400" />
                    <input type="text" placeholder="Search by symbol..." value={search} onChange={(e) => setSearch(e.target.value)} className="bg-transparent text-sm text-white placeholder-slate-400 outline-none flex-1" />
                </div>
                <div className="flex border border-slate-700 rounded-xl overflow-hidden">
                    {statusTabs.map((tab, i) => (
                        <button key={tab} onClick={() => setActiveTab(i)} className={`px-4 py-2.5 text-sm font-medium transition-colors ${activeTab === i ? 'bg-primary text-white' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>
                            {tab}
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
                                <th className="p-4 w-10"></th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400">Instrument</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400">Type</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400">Purpose</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400 text-center">Qty</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">Price</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400 text-right">P&L</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400 text-center">Status</th>
                                <th className="p-4 text-xs font-bold uppercase tracking-wider text-slate-400">Time</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {loading ? (
                                <tr><td colSpan={9} className="p-16 text-center">
                                    <div className="flex items-center justify-center gap-3 text-slate-400">
                                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                        Loading orders...
                                    </div>
                                </td></tr>
                            ) : orders.length === 0 ? (
                                <tr><td colSpan={9} className="p-10 text-center text-slate-500">No orders found. Start trading from AI signals.</td></tr>
                            ) : (
                                orders.map((o) => (
                                    <tr key={o.id} className="hover:bg-slate-800/50 transition-colors">
                                        <td className="p-4">
                                            {o.bracket_id && (
                                                <button onClick={() => toggleBracket(o.bracket_id)} className="text-slate-500 hover:text-white">
                                                    {expandedBrackets.has(o.bracket_id) ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                                </button>
                                            )}
                                        </td>
                                        <td className="p-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${o.order_type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                                                    }`}>{o.order_type === 'BUY' ? '↗' : '↘'}</div>
                                                <div>
                                                    <p className="font-bold text-white">{o.symbol?.replace('.NS', '')}</p>
                                                    <p className="text-xs text-slate-500">{o.name}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className={`text-xs font-bold ${o.order_type === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{o.order_type}</span>
                                        </td>
                                        <td className="p-4 text-sm text-slate-300">{purposeLabels[o.order_purpose] || o.order_purpose}</td>
                                        <td className="p-4 text-center text-white font-bold">{o.quantity}</td>
                                        <td className="p-4 text-right font-mono text-white">₹{fmt(o.fill_price || o.price)}</td>
                                        <td className="p-4 text-right">
                                            {o.pnl != null ? (
                                                <span className={`font-bold ${o.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {o.pnl >= 0 ? '+' : ''}₹{fmt(o.pnl)}
                                                </span>
                                            ) : <span className="text-slate-600">—</span>}
                                        </td>
                                        <td className="p-4 text-center">
                                            <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold border ${statusColors[o.status] || 'text-slate-400 bg-slate-400/10 border-slate-400/20'}`}>
                                                <span className="w-1.5 h-1.5 rounded-full bg-current" />
                                                {o.status}
                                            </span>
                                        </td>
                                        <td className="p-4 text-sm text-slate-400">{o.created_at?.split(' ')[0]}</td>
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

            {/* Bottom stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="rounded-xl p-6 bg-gradient-to-br from-primary/10 to-primary/5 border border-primary/20">
                    <p className="text-primary text-sm font-medium">Total Orders</p>
                    <p className="text-white text-3xl font-bold mt-1">{total}</p>
                </div>
                <div className="rounded-xl p-6 bg-gradient-to-br from-green-500/10 to-green-500/5 border border-green-500/20">
                    <p className="text-green-400 text-sm font-medium">Executed (this page)</p>
                    <p className="text-white text-3xl font-bold mt-1">{executed}</p>
                </div>
                <div className={`rounded-xl p-6 bg-gradient-to-br ${totalPnl >= 0 ? 'from-green-500/10 to-green-500/5 border-green-500/20' : 'from-red-500/10 to-red-500/5 border-red-500/20'} border`}>
                    <p className={`text-sm font-medium ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>Page P&L</p>
                    <p className="text-white text-3xl font-bold mt-1">{totalPnl >= 0 ? '+' : ''}₹{fmt(totalPnl)}</p>
                </div>
            </div>
        </>
    );
}
