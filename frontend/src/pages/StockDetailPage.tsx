import { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown, BarChart3, Activity, Target, Shield, CandlestickChart, LineChart, AreaChart, BarChart2 } from 'lucide-react';
import { createChart, ColorType, CandlestickSeries, HistogramSeries, LineSeries, AreaSeries, BarSeries } from 'lightweight-charts';
import { getStockPrices, getStockIndicators, getAllStocks } from '../api';

interface PriceData {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface Indicators {
    rsi_14?: number;
    macd?: number;
    macd_signal?: number;
    macd_hist?: number;
    bb_upper?: number;
    bb_middle?: number;
    bb_lower?: number;
    sma_20?: number;
    sma_50?: number;
    sma_200?: number;
    ema_9?: number;
    ema_21?: number;
    atr_14?: number;
    adx_14?: number;
    stoch_k?: number;
    stoch_d?: number;
    obv?: number;
}

interface SupportResistance {
    support_1?: number;
    support_2?: number;
    support_3?: number;
    resistance_1?: number;
    resistance_2?: number;
    resistance_3?: number;
}

export default function StockDetailPage() {
    const { symbol } = useParams<{ symbol: string }>();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [prices, setPrices] = useState<PriceData[]>([]);
    const [indicators, setIndicators] = useState<Indicators | null>(null);
    const [sr, setSR] = useState<SupportResistance | null>(null);
    const [signal, setSignal] = useState('');
    const [signalStrength, setSignalStrength] = useState(0);
    const [indicatorDate, setIndicatorDate] = useState('');
    const [loading, setLoading] = useState(true);
    const [chartLoading, setChartLoading] = useState(false);
    const [timeRange, setTimeRange] = useState(365);
    const [stockName, setStockName] = useState('');
    const [chartType, setChartType] = useState<'candle' | 'line' | 'area' | 'bar'>('candle');

    const fullSymbol = symbol ? `${symbol}.NS` : '';

    // Initial load: indicators, stock name (only when symbol changes)
    useEffect(() => {
        if (!fullSymbol) return;
        setLoading(true);
        Promise.all([
            getStockPrices(fullSymbol, 365).catch(() => ({ data: [] })),
            getStockIndicators(fullSymbol).catch(() => null),
            getAllStocks(symbol).catch(() => ({ stocks: [] })),
        ]).then(([priceRes, indRes, stocksRes]) => {
            setPrices(priceRes?.data || []);
            const stockInfo = (stocksRes?.stocks || []).find((s: any) => s.symbol === fullSymbol);
            if (stockInfo) setStockName(stockInfo.name);
            if (indRes) {
                setIndicators(indRes.indicators || null);
                setSR(indRes.support_resistance || null);
                setSignal(indRes.signal || '');
                setSignalStrength(indRes.signal_strength || 0);
                setIndicatorDate(indRes.date || '');
            }
        }).finally(() => setLoading(false));
    }, [fullSymbol]);

    // Chart data reload: only when time range changes (not on initial load)
    const initialLoadDone = useRef(false);
    useEffect(() => {
        if (!fullSymbol) return;
        if (!initialLoadDone.current) {
            initialLoadDone.current = true;
            return; // skip first render, handled by above effect
        }
        setChartLoading(true);
        getStockPrices(fullSymbol, timeRange)
            .then(res => setPrices(res?.data || []))
            .catch(() => setPrices([]))
            .finally(() => setChartLoading(false));
    }, [fullSymbol, timeRange]);

    // Chart rendering
    useEffect(() => {
        if (!chartContainerRef.current || prices.length === 0) return;

        const container = chartContainerRef.current;
        container.innerHTML = '';

        const isDark = !document.documentElement.classList.contains('light');
        const chartBg = isDark ? '#0f172a' : '#ffffff';
        const chartText = isDark ? '#94a3b8' : '#64748b';
        const chartGrid = isDark ? '#1e293b' : '#f1f5f9';
        const chartBorder = isDark ? '#334155' : '#e2e8f0';

        const chart = createChart(container, {
            layout: {
                background: { type: ColorType.Solid, color: chartBg },
                textColor: chartText,
                fontSize: 12,
            },
            grid: {
                vertLines: { color: chartGrid },
                horzLines: { color: chartGrid },
            },
            width: container.clientWidth,
            height: 420,
            crosshair: {
                mode: 0,
            },
            timeScale: {
                borderColor: chartBorder,
                timeVisible: false,
            },
            rightPriceScale: {
                borderColor: chartBorder,
            },
        });

        // Deduplicate by date (keep last occurrence) and sort ascending
        const deduped = new Map<string, PriceData>();
        prices.forEach(p => deduped.set(p.date, p));
        const uniquePrices = Array.from(deduped.values()).sort((a, b) => a.date.localeCompare(b.date));

        // Add main price series based on chart type
        if (chartType === 'candle') {
            const series = chart.addSeries(CandlestickSeries, {
                upColor: '#22c55e',
                downColor: '#ef4444',
                borderDownColor: '#ef4444',
                borderUpColor: '#22c55e',
                wickDownColor: '#ef4444',
                wickUpColor: '#22c55e',
            });
            series.setData(uniquePrices.map(p => ({
                time: p.date as string,
                open: p.open, high: p.high, low: p.low, close: p.close,
            })) as any);
        } else if (chartType === 'line') {
            const series = chart.addSeries(LineSeries, {
                color: '#6366f1',
                lineWidth: 2,
                crosshairMarkerVisible: true,
                crosshairMarkerRadius: 4,
            });
            series.setData(uniquePrices.map(p => ({
                time: p.date as string,
                value: p.close,
            })) as any);
        } else if (chartType === 'area') {
            const series = chart.addSeries(AreaSeries, {
                lineColor: '#6366f1',
                topColor: 'rgba(99, 102, 241, 0.4)',
                bottomColor: 'rgba(99, 102, 241, 0.02)',
                lineWidth: 2,
                crosshairMarkerVisible: true,
            });
            series.setData(uniquePrices.map(p => ({
                time: p.date as string,
                value: p.close,
            })) as any);
        } else if (chartType === 'bar') {
            const series = chart.addSeries(BarSeries, {
                upColor: '#22c55e',
                downColor: '#ef4444',
            });
            series.setData(uniquePrices.map(p => ({
                time: p.date as string,
                open: p.open, high: p.high, low: p.low, close: p.close,
            })) as any);
        }

        // Volume histogram (always shown)
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' },
            priceScaleId: '',
        });
        volumeSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.85, bottom: 0 },
        });
        volumeSeries.setData(uniquePrices.map(p => ({
            time: p.date as string,
            value: p.volume,
            color: p.close >= p.open ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
        })) as any);

        chart.timeScale().fitContent();

        const resizeObserver = new ResizeObserver(() => {
            chart.applyOptions({ width: container.clientWidth });
        });
        resizeObserver.observe(container);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
        };
    }, [prices, chartType]);

    const fmt = (n?: number) => n != null ? n.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '—';

    const latest = prices.length > 0 ? prices[prices.length - 1] : null;
    const prev = prices.length > 1 ? prices[prices.length - 2] : null;
    const change = latest && prev ? latest.close - prev.close : 0;
    const changePct = prev ? (change / prev.close * 100) : 0;
    const isUp = change >= 0;

    const signalColor = signal?.includes('BUY') ? 'text-green-400 bg-green-400/10 border-green-400/20' :
        signal?.includes('SELL') ? 'text-red-400 bg-red-400/10 border-red-400/20' :
            'text-amber-400 bg-amber-400/10 border-amber-400/20';

    const timeRanges = [
        { label: '1M', days: 30 },
        { label: '3M', days: 90 },
        { label: '6M', days: 180 },
        { label: '1Y', days: 365 },
        { label: '3Y', days: 1095 },
        { label: '5Y', days: 1825 },
    ];

    if (loading) return <div className="flex items-center justify-center py-20 text-slate-400">Loading stock data...</div>;

    return (
        <>
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link to="/market" className="p-2 rounded-lg bg-surface-dark border border-slate-700 hover:border-primary transition-colors">
                    <ArrowLeft className="w-5 h-5 text-slate-400" />
                </Link>
                <div className="flex-1">
                    <div className="flex items-center gap-4">
                        <h1 className="text-white text-3xl md:text-4xl font-black tracking-tight">{stockName || symbol}</h1>
                        {signal && (
                            <span className={`px-3 py-1 rounded-full text-xs font-bold border ${signalColor}`}>
                                {signal}
                            </span>
                        )}
                    </div>
                    <p className="text-slate-400 mt-1">{symbol} • {fullSymbol} • NSE</p>
                </div>
                <div className="text-right">
                    <p className="text-white text-3xl font-black">₹{fmt(latest?.close)}</p>
                    <p className={`text-lg font-bold ${isUp ? 'text-green-400' : 'text-red-400'}`}>
                        {isUp ? '+' : ''}{change?.toFixed(2)} ({isUp ? '+' : ''}{changePct?.toFixed(2)}%)
                    </p>
                </div>
            </div>

            {/* Price Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[
                    { label: 'Open', value: latest?.open, icon: BarChart3 },
                    { label: 'High', value: latest?.high, icon: TrendingUp, color: 'text-green-400' },
                    { label: 'Low', value: latest?.low, icon: TrendingDown, color: 'text-red-400' },
                    { label: 'Volume', value: latest?.volume, icon: Activity, isVol: true },
                    { label: 'Prev Close', value: prev?.close, icon: Target },
                ].map(({ label, value, icon: Icon, color, isVol }) => (
                    <div key={label} className="rounded-xl p-4 bg-surface-dark border border-slate-800">
                        <div className="flex items-center gap-2 text-slate-400 text-xs mb-1"><Icon className="w-3.5 h-3.5" />{label}</div>
                        <p className={`font-bold text-lg ${color || 'text-white'}`}>
                            {isVol ? (value as number)?.toLocaleString('en-IN') : `₹${fmt(value as number)}`}
                        </p>
                    </div>
                ))}
            </div>

            {/* Chart */}
            <div className="rounded-xl border border-slate-800 bg-surface-dark overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-slate-700">
                    <div className="flex items-center gap-2">
                        <h2 className="text-white font-bold">Price Chart</h2>
                        <div className="flex ml-4 bg-slate-800 rounded-lg p-0.5">
                            {[
                                { type: 'candle' as const, icon: CandlestickChart, label: 'Candle' },
                                { type: 'line' as const, icon: LineChart, label: 'Line' },
                                { type: 'area' as const, icon: AreaChart, label: 'Area' },
                                { type: 'bar' as const, icon: BarChart2, label: 'OHLC' },
                            ].map(ct => (
                                <button
                                    key={ct.type}
                                    onClick={() => setChartType(ct.type)}
                                    title={ct.label}
                                    className={`p-1.5 rounded-md transition-all ${chartType === ct.type
                                        ? 'bg-primary text-white shadow-lg'
                                        : 'text-slate-400 hover:text-white'}`}
                                >
                                    <ct.icon className="w-4 h-4" />
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="flex gap-1">
                        {timeRanges.map(t => (
                            <button
                                key={t.days}
                                onClick={() => setTimeRange(t.days)}
                                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${timeRange === t.days ? 'bg-primary text-white' : 'bg-slate-800 text-slate-400 hover:text-white'}`}
                            >{t.label}</button>
                        ))}
                    </div>
                </div>
                <div className="relative">
                    {chartLoading && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#0f172a]/80 backdrop-blur-sm">
                            <div className="flex items-center gap-3">
                                <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                <span className="text-slate-400 text-sm">Loading chart...</span>
                            </div>
                        </div>
                    )}
                    <div ref={chartContainerRef} className="w-full" style={{ minHeight: 420 }} />
                </div>
            </div>

            {/* Indicators Grid */}
            {indicators && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Technical Indicators */}
                    <div className="rounded-xl border border-slate-800 bg-surface-dark p-5">
                        <h3 className="text-white font-bold mb-4 flex items-center gap-2"><Activity className="w-4 h-4 text-primary" />Technical Indicators</h3>
                        <div className="space-y-3">
                            {/* RSI */}
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">RSI (14)</span>
                                <div className="flex items-center gap-2">
                                    <div className="w-24 h-1.5 bg-slate-700 rounded-full">
                                        <div className={`h-1.5 rounded-full ${(indicators.rsi_14 || 50) > 70 ? 'bg-red-400' : (indicators.rsi_14 || 50) < 30 ? 'bg-green-400' : 'bg-amber-400'}`}
                                            style={{ width: `${indicators.rsi_14 || 50}%` }} />
                                    </div>
                                    <span className={`text-sm font-bold ${(indicators.rsi_14 || 50) > 70 ? 'text-red-400' : (indicators.rsi_14 || 50) < 30 ? 'text-green-400' : 'text-white'}`}>
                                        {indicators.rsi_14?.toFixed(1)}
                                    </span>
                                </div>
                            </div>
                            {/* MACD */}
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">MACD</span>
                                <span className={`text-sm font-bold ${(indicators.macd || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {indicators.macd?.toFixed(2)}
                                </span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">MACD Signal</span>
                                <span className="text-white text-sm font-bold">{indicators.macd_signal?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">MACD Histogram</span>
                                <span className={`text-sm font-bold ${(indicators.macd_hist || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {indicators.macd_hist?.toFixed(2)}
                                </span>
                            </div>
                            {/* ADX */}
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">ADX (14)</span>
                                <span className={`text-sm font-bold ${(indicators.adx_14 || 0) > 25 ? 'text-green-400' : 'text-slate-400'}`}>
                                    {indicators.adx_14?.toFixed(1)}
                                </span>
                            </div>
                            {/* ATR */}
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">ATR (14)</span>
                                <span className="text-white text-sm font-bold">{indicators.atr_14?.toFixed(2)}</span>
                            </div>
                            {/* Stochastic */}
                            <div className="flex justify-between items-center">
                                <span className="text-slate-400 text-sm">Stochastic %K / %D</span>
                                <span className="text-white text-sm font-bold">
                                    {indicators.stoch_k?.toFixed(1)} / {indicators.stoch_d?.toFixed(1)}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Moving Averages + Bollinger */}
                    <div className="rounded-xl border border-slate-800 bg-surface-dark p-5">
                        <h3 className="text-white font-bold mb-4 flex items-center gap-2"><TrendingUp className="w-4 h-4 text-primary" />Moving Averages</h3>
                        <div className="space-y-3">
                            {[
                                { label: 'SMA 20', value: indicators.sma_20 },
                                { label: 'SMA 50', value: indicators.sma_50 },
                                { label: 'SMA 200', value: indicators.sma_200 },
                                { label: 'EMA 9', value: indicators.ema_9 },
                                { label: 'EMA 21', value: indicators.ema_21 },
                            ].map(({ label, value }) => (
                                <div key={label} className="flex justify-between items-center">
                                    <span className="text-slate-400 text-sm">{label}</span>
                                    <div className="flex items-center gap-2">
                                        <span className="text-white text-sm font-bold">₹{fmt(value)}</span>
                                        {latest && value && (
                                            <span className={`text-[10px] ${latest.close > value ? 'text-green-400' : 'text-red-400'}`}>
                                                {latest.close > value ? '▲ Above' : '▼ Below'}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                            <div className="border-t border-slate-700 pt-3 mt-3">
                                <p className="text-slate-400 text-xs mb-2">Bollinger Bands</p>
                                <div className="flex justify-between">
                                    <span className="text-red-400 text-xs">Lower: ₹{fmt(indicators.bb_lower)}</span>
                                    <span className="text-white text-xs">Mid: ₹{fmt(indicators.bb_middle)}</span>
                                    <span className="text-green-400 text-xs">Upper: ₹{fmt(indicators.bb_upper)}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Support/Resistance */}
                    {sr && (
                        <div className="rounded-xl border border-slate-800 bg-surface-dark p-5">
                            <h3 className="text-white font-bold mb-4 flex items-center gap-2"><Shield className="w-4 h-4 text-primary" />Support & Resistance</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <p className="text-green-400 text-xs font-bold uppercase">Resistance</p>
                                    {[sr.resistance_3, sr.resistance_2, sr.resistance_1].filter(Boolean).map((v, i) => (
                                        <div key={i} className="flex justify-between">
                                            <span className="text-slate-400 text-sm">R{3 - i}</span>
                                            <span className="text-green-400 text-sm font-bold">₹{fmt(v)}</span>
                                        </div>
                                    ))}
                                </div>
                                <div className="space-y-2">
                                    <p className="text-red-400 text-xs font-bold uppercase">Support</p>
                                    {[sr.support_1, sr.support_2, sr.support_3].filter(Boolean).map((v, i) => (
                                        <div key={i} className="flex justify-between">
                                            <span className="text-slate-400 text-sm">S{i + 1}</span>
                                            <span className="text-red-400 text-sm font-bold">₹{fmt(v)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Signal Summary */}
                    {signal && (
                        <div className="rounded-xl border border-slate-800 bg-surface-dark p-5">
                            <h3 className="text-white font-bold mb-4 flex items-center gap-2"><Target className="w-4 h-4 text-primary" />AI Signal</h3>
                            <div className="flex items-center gap-4 mb-3">
                                <span className={`px-4 py-2 rounded-xl text-lg font-black border ${signalColor}`}>{signal}</span>
                                <div>
                                    <p className="text-slate-400 text-xs">Strength</p>
                                    <p className="text-white font-bold">{signalStrength?.toFixed(1)}%</p>
                                </div>
                            </div>
                            <p className="text-slate-400 text-xs">Analysis date: {indicatorDate}</p>
                            <Link
                                to={`/trade/${symbol}`}
                                className="mt-4 inline-block w-full text-center px-6 py-3 rounded-xl bg-primary text-white font-bold hover:bg-primary/90 transition-colors"
                            >
                                Trade {symbol}
                            </Link>
                        </div>
                    )}
                </div>
            )}
        </>
    );
}
