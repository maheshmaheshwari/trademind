import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown, Brain, Shield, Loader2, CheckCircle, AlertTriangle, Zap, Activity, Target, BarChart3, Clock, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { createChart, ColorType, CandlestickSeries, HistogramSeries, AreaSeries } from 'lightweight-charts';
import { useAuth } from '../AuthContext';
import { getStockWatchlist, getLatestSignals, getStockPrices, getStockIndicators, getOrders, executeSignal } from '../api';

interface PriceData { date: string; open: number; high: number; low: number; close: number; volume: number; }

export default function TradeExecutionPage() {
    const { symbol: rawSymbol } = useParams<{ symbol: string }>();
    const navigate = useNavigate();
    const { user, refreshUser } = useAuth();
    const symbol = rawSymbol?.includes('.NS') ? rawSymbol : `${rawSymbol}.NS`;
    const displaySymbol = rawSymbol?.replace('.NS', '') || 'STOCK';

    const chartRef = useRef<HTMLDivElement>(null);
    const [stockData, setStockData] = useState<any>(null);
    const [signalData, setSignalData] = useState<any>(null);
    const [prices, setPrices] = useState<PriceData[]>([]);
    const [indicators, setIndicators] = useState<any>(null);
    const [recentOrders, setRecentOrders] = useState<any[]>([]);
    const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
    const [investment, setInvestment] = useState('50000');
    const [loading, setLoading] = useState(true);
    const [executing, setExecuting] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState('');
    const [mode, setMode] = useState<'PAPER' | 'LIVE'>('PAPER');
    const [showLiveConfirm, setShowLiveConfirm] = useState(false);
    const [chartRange, setChartRange] = useState(90);
    const [manualBuyPrice, setManualBuyPrice] = useState('');
    const [manualTarget, setManualTarget] = useState('');
    const [manualSL, setManualSL] = useState('');

    useEffect(() => {
        const load = async () => {
            try {
                const [w, s, p, ind, orders] = await Promise.all([
                    getStockWatchlist(symbol).catch(() => null),
                    getLatestSignals().catch(() => ({ data: { trades: [] } })),
                    getStockPrices(symbol, 365).catch(() => ({ data: [] })),
                    getStockIndicators(symbol).catch(() => null),
                    user ? getOrders(user.id).catch(() => ({ orders: [] })) : Promise.resolve({ orders: [] }),
                ]);
                setStockData(w);
                setPrices(p?.data || []);
                setIndicators(ind);
                const trades = s?.data?.trades || [];
                const match = trades.find((t: any) => t.symbol === symbol || t.symbol?.replace('.NS', '') === displaySymbol);
                setSignalData(match || null);
                // Auto-fill price fields from signal or stock data
                const curPrice = w?.latest_price?.close || match?.price?.current || match?.trade?.buy_price || 0;
                if (match?.trade) {
                    setManualBuyPrice(String(match.trade.buy_price || curPrice));
                    setManualTarget(String(match.trade.target_price || ''));
                    setManualSL(String(match.trade.stop_loss || ''));
                } else if (curPrice) {
                    setManualBuyPrice(String(curPrice));
                    // Auto-calculate 5% target and 3% SL
                    setManualTarget(String((curPrice * 1.05).toFixed(2)));
                    setManualSL(String((curPrice * 0.97).toFixed(2)));
                }
                // Filter orders for this symbol
                const allOrders = orders?.orders || [];
                setRecentOrders(allOrders.filter((o: any) => o.symbol === symbol).slice(0, 5));
            } catch { } finally { setLoading(false); }
        };
        load();
    }, [symbol, displaySymbol, user]);

    // Chart rendering
    useEffect(() => {
        if (!chartRef.current || prices.length === 0) return;
        const container = chartRef.current;
        container.innerHTML = '';

        // Filter by range
        const now = new Date();
        const cutoff = new Date(now.getTime() - chartRange * 24 * 60 * 60 * 1000);
        const filtered = prices.filter(p => new Date(p.date) >= cutoff);
        const deduped = new Map<string, PriceData>();
        filtered.forEach(p => deduped.set(p.date, p));
        const uniquePrices = Array.from(deduped.values()).sort((a, b) => a.date.localeCompare(b.date));

        if (uniquePrices.length === 0) return;

        const isDark = !document.documentElement.classList.contains('light');
        const chartBg = isDark ? '#0f172a' : '#ffffff';
        const chartText = isDark ? '#94a3b8' : '#64748b';
        const chartGrid = isDark ? '#1e293b' : '#f1f5f9';
        const chartBorder = isDark ? '#334155' : '#e2e8f0';

        const chart = createChart(container, {
            layout: { background: { type: ColorType.Solid, color: chartBg }, textColor: chartText, fontSize: 11 },
            grid: { vertLines: { color: chartGrid }, horzLines: { color: chartGrid } },
            width: container.clientWidth, height: 300,
            crosshair: { mode: 0 },
            timeScale: { borderColor: chartBorder, timeVisible: false },
            rightPriceScale: { borderColor: chartBorder },
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#22c55e', downColor: '#ef4444',
            borderDownColor: '#ef4444', borderUpColor: '#22c55e',
            wickDownColor: '#ef4444', wickUpColor: '#22c55e',
        });
        candleSeries.setData(uniquePrices.map(p => ({
            time: p.date as string, open: p.open, high: p.high, low: p.low, close: p.close,
        })) as any);

        // Add SL/Target lines if signal data
        if (signalData?.trade) {
            const slPrice = signalData.trade.stop_loss;
            const targetPrice = signalData.trade.target_price;
            if (slPrice) {
                candleSeries.createPriceLine({ price: slPrice, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'SL' });
            }
            if (targetPrice) {
                candleSeries.createPriceLine({ price: targetPrice, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Target' });
            }
        }

        const volumeSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: '' });
        volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
        volumeSeries.setData(uniquePrices.map(p => ({
            time: p.date as string, value: p.volume,
            color: p.close >= p.open ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)',
        })) as any);

        chart.timeScale().fitContent();
        const ro = new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth }));
        ro.observe(container);
        return () => { ro.disconnect(); chart.remove(); };
    }, [prices, chartRange, signalData]);

    const handleExecute = async () => {
        if (!user) return;
        const buyP = parseFloat(manualBuyPrice) || 0;
        const targetP = parseFloat(manualTarget) || 0;
        const slP = parseFloat(manualSL) || 0;
        if (buyP <= 0) { setError('Buy price is required'); return; }
        if (targetP <= 0) { setError('Target price is required'); return; }
        if (slP <= 0) { setError('Stop loss is required'); return; }
        if (mode === 'LIVE' && !showLiveConfirm) { setShowLiveConfirm(true); return; }
        setShowLiveConfirm(false);
        setExecuting(true); setError(''); setResult(null);
        try {
            const res = await executeSignal({
                user_id: user.id, symbol,
                name: signalData?.name || displaySymbol,
                investment_amount: parseFloat(investment),
                buy_price: buyP,
                target_price: targetP,
                stop_loss: slP,
                signal: signalData?.signal || side,
                confidence: signalData?.confidence || 0,
                horizon: signalData?.model?.horizon,
                max_safe_qty: signalData?.position?.max_safe_qty,
                mode,
            });
            setResult(res);
            await refreshUser();
        } catch (err: any) { setError(err.message); }
        finally { setExecuting(false); }
    };

    const fmt = (n?: number) => n != null ? n.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '—';
    const pctChange = (cur: number, prev: number) => prev ? ((cur - prev) / prev * 100).toFixed(2) : '0.00';

    if (loading) return (
        <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
        </div>
    );

    const price = parseFloat(manualBuyPrice) || stockData?.latest_price?.close || signalData?.price?.current || 0;
    const prevClose = stockData?.latest_price?.open || price;
    const dayChange = price - prevClose;
    const dayChangePct = pctChange(price, prevClose);
    const target = parseFloat(manualTarget) || 0;
    const sl = parseFloat(manualSL) || 0;
    const rr = (target > 0 && sl > 0 && price > 0) ? ((target - price) / (price - sl)) : 0;
    const conf = signalData?.confidence || 0;
    const canTrade = price > 0 && target > 0 && sl > 0;
    const ind = indicators?.indicators;

    return (
        <div className="flex flex-col gap-6">
            {/* Header */}
            <div className="flex items-center gap-4 flex-wrap">
                <button onClick={() => navigate(-1)} className="p-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white transition-colors"><ArrowLeft className="w-5 h-5" /></button>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                        <h1 className="text-white text-2xl font-bold">{displaySymbol}</h1>
                        {signalData && (
                            <span className={`px-3 py-1 rounded-lg text-xs font-bold ${signalData.signal?.includes('BUY') ? 'bg-green-500/20 text-green-400 border border-green-500/30' : signalData.signal?.includes('SELL') ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'}`}>
                                AI: {signalData.signal}
                            </span>
                        )}
                    </div>
                    <p className="text-slate-400 text-sm mt-0.5">NSE • {signalData?.name || displaySymbol} • {mode === 'LIVE' ? 'Live Trading' : 'Paper Trading'}</p>
                </div>
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${mode === 'LIVE' ? 'bg-red-500/20 border-red-500/20' : 'bg-green-500/20 border-green-500/20'}`}>
                    <div className={`w-2 h-2 rounded-full animate-pulse ${mode === 'LIVE' ? 'bg-red-500' : 'bg-green-500'}`} />
                    <span className={`text-xs font-bold ${mode === 'LIVE' ? 'text-red-400' : 'text-green-400'}`}>{mode}</span>
                </div>
            </div>

            {/* Price + Change row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="rounded-xl p-4 bg-surface-dark border border-slate-800">
                    <p className="text-slate-400 text-xs">Current Price</p>
                    <p className="text-white text-2xl font-bold mt-1">₹{fmt(price)}</p>
                    <div className={`flex items-center gap-1 mt-1 text-xs font-semibold ${dayChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {dayChange >= 0 ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                        {dayChange >= 0 ? '+' : ''}{fmt(dayChange)} ({dayChangePct}%)
                    </div>
                </div>
                <div className="rounded-xl p-4 bg-surface-dark border border-slate-800">
                    <p className="text-slate-400 text-xs">Day Range</p>
                    <p className="text-white font-bold mt-1">₹{fmt(stockData?.latest_price?.low)} — ₹{fmt(stockData?.latest_price?.high)}</p>
                    <p className="text-slate-500 text-xs mt-1">Vol: {fmt(stockData?.latest_price?.volume)}</p>
                </div>
                <div className="rounded-xl p-4 bg-green-500/5 border border-green-500/20">
                    <p className="text-green-400 text-xs font-bold flex items-center gap-1"><Target className="w-3 h-3" />Target</p>
                    <p className="text-white font-bold mt-1 text-lg">₹{fmt(target)}</p>
                    {target > 0 && price > 0 && <p className="text-green-400 text-xs mt-1">+{pctChange(target, price)}% upside</p>}
                </div>
                <div className="rounded-xl p-4 bg-red-500/5 border border-red-500/20">
                    <p className="text-red-400 text-xs font-bold flex items-center gap-1"><Shield className="w-3 h-3" />Stop Loss</p>
                    <p className="text-white font-bold mt-1 text-lg">₹{fmt(sl)}</p>
                    {sl > 0 && price > 0 && <p className="text-red-400 text-xs mt-1">{pctChange(sl, price)}% risk</p>}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left column: chart + indicators + orders */}
                <div className="lg:col-span-2 flex flex-col gap-6">

                    {/* Price Chart */}
                    <div className="rounded-xl bg-surface-dark border border-slate-800 overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
                            <h3 className="text-white font-bold flex items-center gap-2"><BarChart3 className="w-4 h-4 text-primary" />Price Chart</h3>
                            <div className="flex gap-1">
                                {[{ label: '1M', days: 30 }, { label: '3M', days: 90 }, { label: '6M', days: 180 }, { label: '1Y', days: 365 }].map(r => (
                                    <button key={r.days} onClick={() => setChartRange(r.days)}
                                        className={`px-3 py-1 text-xs rounded-lg font-bold transition-all ${chartRange === r.days ? 'bg-primary text-white' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>{r.label}</button>
                                ))}
                            </div>
                        </div>
                        <div ref={chartRef} className="w-full" style={{ minHeight: 300 }} />
                    </div>

                    {/* AI Analysis + Technical Indicators */}
                    {(signalData || ind) && (
                        <div className="rounded-xl p-5 bg-gradient-to-br from-primary/5 to-purple-500/5 border border-primary/20">
                            <div className="flex items-center gap-2 mb-4">
                                <Brain className="w-5 h-5 text-primary" />
                                <h3 className="text-white font-bold">AI Analysis & Indicators</h3>
                                {conf > 0 && <span className="text-primary text-xs font-bold ml-auto">{conf.toFixed(0)}% Confidence</span>}
                            </div>
                            <div className="grid grid-cols-3 md:grid-cols-6 gap-3 text-sm">
                                {signalData?.trade?.buy_price && <div className="p-2.5 rounded-lg bg-slate-900/50"><p className="text-slate-400 text-xs">Buy Price</p><p className="text-white font-bold mt-0.5">₹{fmt(signalData.trade.buy_price)}</p></div>}
                                {rr > 0 && <div className="p-2.5 rounded-lg bg-slate-900/50"><p className="text-slate-400 text-xs">Risk:Reward</p><p className="text-primary font-bold mt-0.5">1:{rr.toFixed(1)}</p></div>}
                                {ind?.rsi_14 != null && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/50">
                                        <p className="text-slate-400 text-xs">RSI (14)</p>
                                        <p className={`font-bold mt-0.5 ${ind.rsi_14 > 70 ? 'text-red-400' : ind.rsi_14 < 30 ? 'text-green-400' : 'text-white'}`}>{ind.rsi_14.toFixed(1)}</p>
                                    </div>
                                )}
                                {ind?.macd != null && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/50">
                                        <p className="text-slate-400 text-xs">MACD</p>
                                        <p className={`font-bold mt-0.5 ${ind.macd > 0 ? 'text-green-400' : 'text-red-400'}`}>{ind.macd.toFixed(2)}</p>
                                    </div>
                                )}
                                {ind?.adx_14 != null && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/50">
                                        <p className="text-slate-400 text-xs">ADX</p>
                                        <p className={`font-bold mt-0.5 ${ind.adx_14 > 25 ? 'text-green-400' : 'text-slate-300'}`}>{ind.adx_14.toFixed(1)}</p>
                                    </div>
                                )}
                                {ind?.atr_14 != null && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/50">
                                        <p className="text-slate-400 text-xs">ATR (14)</p>
                                        <p className="text-white font-bold mt-0.5">₹{ind.atr_14.toFixed(2)}</p>
                                    </div>
                                )}
                            </div>
                            {signalData?.model && (
                                <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
                                    <span>Model: {signalData.model.name}</span>
                                    <span>Horizon: {signalData.model.horizon}</span>
                                    {indicators?.date && <span>Updated: {indicators.date}</span>}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Recent Orders for this stock */}
                    <div className="rounded-xl bg-surface-dark border border-slate-800 overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
                            <h3 className="text-white font-bold flex items-center gap-2"><Clock className="w-4 h-4 text-slate-400" />Recent Orders — {displaySymbol}</h3>
                            <Link to="/orders" className="text-primary text-xs hover:underline">View All</Link>
                        </div>
                        {recentOrders.length > 0 ? (
                            <div className="divide-y divide-slate-800">
                                {recentOrders.map((o, i) => (
                                    <div key={i} className="flex items-center justify-between px-4 py-3 text-sm">
                                        <div className="flex items-center gap-3">
                                            <span className={`px-2 py-0.5 rounded text-xs font-bold ${o.order_type === 'BUY' ? 'bg-green-500/20 text-green-400' : o.order_type === 'SELL' ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-300'}`}>{o.order_type}</span>
                                            <span className="text-slate-300">{o.order_purpose}</span>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <span className="text-white font-bold">₹{fmt(o.price)} × {o.quantity}</span>
                                            <span className={`px-2 py-0.5 rounded text-xs ${o.status === 'EXECUTED' ? 'bg-green-500/20 text-green-400' : o.status === 'PENDING' ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700 text-slate-400'}`}>{o.status}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="p-8 text-center text-slate-500 text-sm">No previous orders for {displaySymbol}</div>
                        )}
                    </div>

                    {/* Execution Result */}
                    {result?.status === 'executed' && (
                        <div className="rounded-xl p-6 bg-green-500/10 border border-green-500/20">
                            <div className="flex items-center gap-3 mb-4"><CheckCircle className="w-6 h-6 text-green-400" /><h3 className="text-green-400 font-bold text-lg">Order Executed! {mode === 'LIVE' && <span className="text-red-400 text-sm">(LIVE)</span>}</h3></div>
                            <div className="grid grid-cols-3 gap-4 text-sm">
                                <div><p className="text-slate-400">Qty</p><p className="text-white font-bold mt-1">{result.position?.quantity}</p></div>
                                <div><p className="text-slate-400">Invested</p><p className="text-white font-bold mt-1">₹{fmt(result.position?.invested)}</p></div>
                                <div><p className="text-slate-400">Balance</p><p className="text-white font-bold mt-1">₹{fmt(result.account?.balance_after)}</p></div>
                            </div>
                            {result.gtt && (
                                <div className="mt-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-sm">
                                    <p className="text-amber-400 font-bold flex items-center gap-1"><Zap className="w-4 h-4" /> GTT Orders on Angel One</p>
                                    <div className="flex gap-4 mt-2 text-slate-300"><span>SL: #{result.gtt.sl_rule_id || '—'}</span><span>Target: #{result.gtt.target_rule_id || '—'}</span></div>
                                </div>
                            )}
                            <div className="flex gap-3 mt-4">
                                <Link to="/portfolio" className="px-4 py-2 rounded-xl bg-primary text-white text-sm font-bold">View Portfolio</Link>
                                <Link to="/orders" className="px-4 py-2 rounded-xl border border-slate-700 text-white text-sm">Orders</Link>
                            </div>
                        </div>
                    )}
                    {result?.status === 'rejected' && (
                        <div className="rounded-xl p-6 bg-red-500/10 border border-red-500/20">
                            <div className="flex items-center gap-3"><AlertTriangle className="w-6 h-6 text-red-400" /><h3 className="text-red-400 font-bold">Rejected</h3></div>
                            <p className="text-red-300 text-sm mt-2">{result.reason}</p>
                        </div>
                    )}
                </div>

                {/* Right column: Order Form */}
                <div className="flex flex-col gap-6">
                    <div className="rounded-xl bg-surface-dark border border-slate-800 p-6 sticky top-6">
                        <h3 className="text-white font-bold mb-4">Place Order</h3>

                        {/* Paper / Live Toggle */}
                        <div className="mb-4">
                            <label className="text-xs text-slate-500 mb-2 block">Trading Mode</label>
                            <div className="grid grid-cols-2 gap-2">
                                <button onClick={() => { setMode('PAPER'); setShowLiveConfirm(false); }}
                                    className={`py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-1.5
                                    ${mode === 'PAPER' ? 'bg-green-500/20 text-green-400 border border-green-500/30 shadow-lg shadow-green-500/10' : 'bg-slate-800 text-slate-400 border border-transparent'}`}>
                                    <Shield className="w-4 h-4" />Paper
                                </button>
                                <button onClick={() => setMode('LIVE')}
                                    className={`py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-1.5
                                    ${mode === 'LIVE' ? 'bg-red-500/20 text-red-400 border border-red-500/30 shadow-lg shadow-red-500/10' : 'bg-slate-800 text-slate-400 border border-transparent'}`}>
                                    <Zap className="w-4 h-4" />Live
                                </button>
                            </div>
                        </div>

                        {mode === 'LIVE' && (
                            <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs">
                                <p className="text-red-400 font-bold flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5" /> Real Money</p>
                                <p className="text-red-300/80 mt-1">Orders on Angel One. SL & Target as GTT (365 days).</p>
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-2 mb-4">
                            <button onClick={() => setSide('BUY')} className={`py-3 rounded-xl text-sm font-bold transition-all ${side === 'BUY' ? 'bg-green-500 text-white shadow-lg shadow-green-500/25' : 'bg-slate-800 text-slate-400'}`}>BUY</button>
                            <button onClick={() => setSide('SELL')} className={`py-3 rounded-xl text-sm font-bold transition-all ${side === 'SELL' ? 'bg-red-500 text-white shadow-lg shadow-red-500/25' : 'bg-slate-800 text-slate-400'}`}>SELL</button>
                        </div>

                        <div className="mb-4">
                            <label className="text-sm text-slate-400 mb-2 block">Investment (₹)</label>
                            <input type="number" value={investment} onChange={(e) => setInvestment(e.target.value)} className="w-full bg-slate-950/50 border border-slate-700 text-white rounded-xl p-3.5 outline-none focus:ring-2 focus:ring-primary/50" />
                            <div className="flex gap-2 mt-2">
                                {['25000', '50000', '100000', '200000'].map(v => (
                                    <button key={v} onClick={() => setInvestment(v)} className={`flex-1 py-1.5 text-xs rounded-lg border ${investment === v ? 'border-primary text-primary' : 'border-slate-700 text-slate-400'}`}>₹{(+v / 1000)}K</button>
                                ))}
                            </div>
                        </div>

                        {/* Editable Price Fields */}
                        <div className="mb-4 space-y-3">
                            <div>
                                <label className="text-xs text-slate-400 mb-1 block">Buy Price (₹)</label>
                                <input type="number" step="0.01" value={manualBuyPrice} onChange={(e) => setManualBuyPrice(e.target.value)} className="w-full bg-slate-950/50 border border-slate-700 text-white rounded-xl p-3 text-sm outline-none focus:ring-2 focus:ring-primary/50" />
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className="text-xs text-green-400 mb-1 block">Target (₹)</label>
                                    <input type="number" step="0.01" value={manualTarget} onChange={(e) => setManualTarget(e.target.value)} className="w-full bg-slate-950/50 border border-green-500/30 text-white rounded-xl p-3 text-sm outline-none focus:ring-2 focus:ring-green-500/50" />
                                </div>
                                <div>
                                    <label className="text-xs text-red-400 mb-1 block">Stop Loss (₹)</label>
                                    <input type="number" step="0.01" value={manualSL} onChange={(e) => setManualSL(e.target.value)} className="w-full bg-slate-950/50 border border-red-500/30 text-white rounded-xl p-3 text-sm outline-none focus:ring-2 focus:ring-red-500/50" />
                                </div>
                            </div>
                        </div>

                        {/* Order Summary */}
                        <div className="mb-4 p-3 rounded-xl bg-slate-900 border border-slate-800 text-sm space-y-1.5">
                            {price > 0 && <div className="flex justify-between text-slate-400"><span>Est. Qty</span><span className="text-white font-bold">{Math.floor(+investment / price)}</span></div>}
                            {rr > 0 && <div className="flex justify-between text-slate-400"><span>R:R Ratio</span><span className="text-primary font-bold">1:{rr.toFixed(1)}</span></div>}
                            {target > 0 && price > 0 && <div className="flex justify-between text-slate-400"><span>Upside</span><span className="text-green-400 font-bold">+{((target - price) / price * 100).toFixed(1)}%</span></div>}
                            {sl > 0 && price > 0 && <div className="flex justify-between text-slate-400"><span>Downside</span><span className="text-red-400 font-bold">{((sl - price) / price * 100).toFixed(1)}%</span></div>}
                            <div className="flex justify-between text-slate-400"><span>Balance</span><span className="text-white font-bold">₹{fmt(user?.virtual_balance || 0)}</span></div>
                        </div>

                        {error && <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">{error}</div>}

                        {showLiveConfirm && (
                            <div className="mb-4 p-4 rounded-xl bg-red-500/10 border-2 border-red-500/40">
                                <p className="text-red-400 font-bold text-sm flex items-center gap-2"><AlertTriangle className="w-4 h-4" /> Confirm LIVE Trade</p>
                                <p className="text-red-300/80 text-xs mt-1">REAL order on Angel One. Click Execute again.</p>
                                <button onClick={() => setShowLiveConfirm(false)} className="mt-2 text-xs text-slate-400 underline">Cancel</button>
                            </div>
                        )}

                        <button onClick={handleExecute} disabled={executing || !canTrade || result?.status === 'executed'} className={`w-full py-4 rounded-xl font-bold text-white shadow-lg transition-all disabled:opacity-50 flex items-center justify-center gap-2 ${showLiveConfirm ? 'bg-red-600 hover:bg-red-700 shadow-red-500/25' :
                            mode === 'LIVE' ? 'bg-red-500 hover:bg-red-600 shadow-red-500/25' :
                                side === 'BUY' ? 'bg-green-500 hover:bg-green-600 shadow-green-500/25' : 'bg-red-500 hover:bg-red-600 shadow-red-500/25'}`}>
                            {executing ? <><Loader2 className="w-5 h-5 animate-spin" />Executing...</> :
                                result?.status === 'executed' ? <><CheckCircle className="w-5 h-5" />Executed</> :
                                    showLiveConfirm ? <><AlertTriangle className="w-5 h-5" />Confirm LIVE Order</> :
                                        mode === 'LIVE' ? <><Zap className="w-5 h-5" />Execute LIVE {side}</> :
                                            <><TrendingUp className="w-5 h-5" />Execute {side}</>}
                        </button>

                        {!signalData && <p className="text-center text-amber-500 text-xs mt-3">Manual mode — no AI signal. Prices auto-set from market data.</p>}
                        <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-500">
                            {mode === 'LIVE' ? <><Zap className="w-4 h-4 text-amber-500" /><span>GTT orders (365 days)</span></> : <><Shield className="w-4 h-4 text-green-500" /><span>Auto bracket order</span></>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
