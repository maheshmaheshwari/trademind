import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { TrendingUp, ArrowUp, Briefcase, Wallet, ChevronRight } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { getPortfolioSummary, getLatestSignals } from '../api';

interface PortfolioData {
    balance: number;
    invested: number;
    total_value: number;
    realized_pnl: number;
    unrealized_pnl: number;
    total_pnl: number;
    open_positions: number;
    wins: number;
    losses: number;
    win_rate: number;
    positions: any[];
}

interface Signal {
    symbol: string;
    name: string;
    signal: string;
    confidence: number;
    trade?: { buy_price?: number; target_price?: number };
}

export default function DashboardPage() {
    const { user } = useAuth();
    const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
    const [signals, setSignals] = useState<Signal[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!user) return;
        const load = async () => {
            try {
                const [portData, sigData] = await Promise.all([
                    getPortfolioSummary(user.id),
                    getLatestSignals(),
                ]);
                setPortfolio(portData);
                const trades = sigData?.data?.trades || sigData?.data?.actionable_trades || [];
                setSignals(trades.slice(0, 5));
            } catch {
                // Backend might be down; use fallback data from user object
                setPortfolio({
                    balance: user.virtual_balance,
                    invested: user.virtual_invested,
                    total_value: user.virtual_balance + user.virtual_invested,
                    realized_pnl: user.total_pnl,
                    unrealized_pnl: 0,
                    total_pnl: user.total_pnl,
                    open_positions: 0,
                    wins: user.win_count,
                    losses: user.loss_count,
                    win_rate: user.win_count + user.loss_count > 0 ? Math.round((user.win_count / (user.win_count + user.loss_count)) * 100) : 0,
                    positions: [],
                });
            } finally {
                setLoading(false);
            }
        };
        load();
    }, [user]);

    const fmt = (n: number) => n?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '0';
    const pnlColor = (n: number) => n >= 0 ? 'text-green-400' : 'text-red-400';

    if (loading) {
        return <div className="flex items-center justify-center py-20 text-slate-400 text-lg">Loading dashboard...</div>;
    }

    const p = portfolio!;

    return (
        <>
            {/* Welcome */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-white text-2xl font-bold">Welcome back, {user?.display_name || user?.username}!</h1>
                    <p className="text-slate-400 text-sm mt-1">Here's your portfolio overview</p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-500/20 border border-green-500/20">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-green-400 text-xs font-bold">PAPER TRADING</span>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                        <Briefcase className="w-4 h-4" />
                        Total Portfolio Value
                    </div>
                    <p className="text-white text-3xl font-bold tracking-tight">₹{fmt(p.total_value)}</p>
                    <p className={`text-sm flex items-center gap-1 ${pnlColor(p.total_pnl)}`}>
                        <TrendingUp className="w-3 h-3" /> {p.total_pnl >= 0 ? '+' : ''}₹{fmt(p.total_pnl)}
                    </p>
                </div>
                <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                        <TrendingUp className="w-4 h-4" />
                        Unrealized P&L
                    </div>
                    <p className={`text-3xl font-bold tracking-tight ${pnlColor(p.unrealized_pnl)}`}>
                        {p.unrealized_pnl >= 0 ? '+' : ''}₹{fmt(p.unrealized_pnl)}
                    </p>
                    <p className="text-slate-500 text-sm">{p.open_positions} open positions</p>
                </div>
                <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                        <Wallet className="w-4 h-4" />
                        Invested Amount
                    </div>
                    <p className="text-white text-3xl font-bold tracking-tight">₹{fmt(p.invested)}</p>
                    <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
                        <div className="bg-primary h-2 rounded-full" style={{ width: `${Math.min((p.invested / (p.invested + p.balance)) * 100, 100)}%` }} />
                    </div>
                </div>
                <div className="flex flex-col gap-2 rounded-xl p-6 bg-surface-dark border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                        <Wallet className="w-4 h-4" />
                        Virtual Balance
                    </div>
                    <p className="text-white text-3xl font-bold tracking-tight">₹{fmt(p.balance)}</p>
                    <p className="text-slate-500 text-xs mt-1">
                        Win Rate: {p.win_rate}% ({p.wins}W/{p.losses}L)
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Positions */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                    <div className="rounded-xl bg-surface-dark border border-slate-800">
                        <div className="flex items-center justify-between p-5 border-b border-slate-800">
                            <h3 className="text-white font-bold">Open Positions ({p.positions.length})</h3>
                            <Link to="/portfolio" className="text-primary text-sm font-medium hover:underline">View All</Link>
                        </div>
                        {p.positions.length === 0 ? (
                            <div className="p-8 text-center">
                                <p className="text-slate-400 text-sm">No open positions yet.</p>
                                <Link to="/signals" className="text-primary text-sm font-medium hover:underline mt-2 inline-block">
                                    Browse AI Signals to start trading →
                                </Link>
                            </div>
                        ) : (
                            p.positions.slice(0, 5).map((pos: any) => (
                                <Link to={`/trade/${pos.symbol}`} key={pos.symbol} className="flex items-center justify-between p-5 border-b border-slate-800/50 last:border-0 hover:bg-slate-800/30 transition-colors">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center text-primary text-xs font-bold">
                                            {pos.symbol?.slice(0, 3)}
                                        </div>
                                        <div>
                                            <p className="text-white font-bold text-sm">{pos.symbol}</p>
                                            <p className="text-slate-500 text-xs">Qty: {pos.quantity} • Avg: ₹{fmt(pos.avg_buy_price)}</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className={`font-bold text-sm ${pnlColor(pos.unrealized_pnl || 0)}`}>
                                            {(pos.unrealized_pnl || 0) >= 0 ? '+' : ''}₹{fmt(pos.unrealized_pnl || 0)}
                                        </p>
                                        <p className={`text-xs ${pnlColor(pos.unrealized_pnl_pct || 0)}`}>
                                            {(pos.unrealized_pnl_pct || 0) >= 0 ? '+' : ''}{(pos.unrealized_pnl_pct || 0).toFixed(2)}%
                                        </p>
                                    </div>
                                </Link>
                            ))
                        )}
                    </div>
                </div>

                {/* Right: AI Top Picks */}
                <div className="flex flex-col gap-6">
                    <div className="rounded-xl bg-surface-dark border border-slate-800">
                        <div className="flex items-center justify-between p-5 border-b border-slate-800">
                            <h3 className="text-white font-bold">AI Top Picks</h3>
                            <span className="text-xs text-primary font-bold border border-primary/30 px-2 py-0.5 rounded-full">AI</span>
                        </div>
                        {signals.length === 0 ? (
                            <div className="p-8 text-center text-slate-400 text-sm">
                                No signals available yet. Run the signal generator.
                            </div>
                        ) : (
                            signals.map((s) => (
                                <Link to={`/trade/${s.symbol?.replace('.NS', '')}`} key={s.symbol} className="flex items-center justify-between px-5 py-4 border-b border-slate-800/50 last:border-0 hover:bg-slate-800/30 transition-colors">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-white text-xs font-bold ${s.signal?.includes('BUY') ? 'bg-green-600' : s.signal?.includes('SELL') ? 'bg-red-600' : 'bg-amber-600'
                                            }`}>
                                            {s.symbol?.replace('.NS', '').slice(0, 3)}
                                        </div>
                                        <div>
                                            <p className="text-white font-bold text-sm">{s.symbol?.replace('.NS', '')}</p>
                                            <p className={`text-xs font-bold ${s.signal?.includes('BUY') ? 'text-green-400' : s.signal?.includes('SELL') ? 'text-red-400' : 'text-amber-400'
                                                }`}>{s.signal}</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-primary text-sm font-bold">{s.confidence?.toFixed(0)}%</p>
                                        {s.trade?.buy_price && <p className="text-slate-500 text-xs">₹{fmt(s.trade.buy_price)}</p>}
                                    </div>
                                </Link>
                            ))
                        )}
                        <div className="px-5 py-4">
                            <Link
                                to="/signals"
                                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-slate-700 text-white text-sm font-medium hover:bg-slate-800 transition-colors"
                            >
                                View Full Analysis <ChevronRight className="w-4 h-4" />
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
