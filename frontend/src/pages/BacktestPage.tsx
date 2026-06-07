import ReactApexChart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';
import { useTheme } from '../ThemeContext';
import { useGetBacktestSummaryQuery } from '../services/tradeMindApiService';
import {
  Brain, TrendingUp, Target, Zap, CheckCircle, AlertCircle,
} from 'lucide-react';

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number, dec = 1) {
  return n.toFixed(dec);
}

function chartBase(isDark: boolean): ApexOptions {
  const text2 = isDark ? '#AEB9CE' : '#4A5670';
  const grid  = isDark ? 'rgba(255,255,255,.05)' : 'rgba(15,23,42,.06)';
  return {
    chart: { background: 'transparent', toolbar: { show: false }, zoom: { enabled: false }, animations: { enabled: true, speed: 450 } },
    theme: { mode: isDark ? 'dark' : 'light' },
    tooltip: { theme: isDark ? 'dark' : 'light', style: { fontFamily: 'inherit', fontSize: '12px' } },
    grid: { borderColor: grid, strokeDashArray: 0, xaxis: { lines: { show: false } }, yaxis: { lines: { show: true } } },
    xaxis: { labels: { style: { colors: text2, fontFamily: 'inherit', fontSize: '11px' } }, axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: { labels: { style: { colors: text2, fontFamily: 'inherit', fontSize: '11px' } } },
    dataLabels: { enabled: false },
    legend: { show: false },
  };
}

// ── sub-components ────────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = '#3B82F6' }: {
  icon: React.ElementType; label: string; value: string; sub?: string; color?: string;
}) {
  return (
    <div className="rounded-[14px] border border-line bg-surface p-5 flex gap-4 items-start">
      <span className="w-10 h-10 rounded-[10px] grid place-items-center flex-shrink-0"
        style={{ background: color + '1A', color }}>
        <Icon size={20} />
      </span>
      <div className="flex flex-col gap-0.5 min-w-0">
        <span className="text-[12px] font-semibold uppercase tracking-[.07em] text-ink-3">{label}</span>
        <span className="text-[26px] font-bold text-ink leading-none">{value}</span>
        {sub && <span className="text-[12px] text-ink-2 mt-1">{sub}</span>}
      </div>
    </div>
  );
}

function SectionCard({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-[14px] border border-line bg-surface p-5 flex flex-col gap-4">
      <div>
        <h3 className="text-[15px] font-bold text-ink">{title}</h3>
        {sub && <p className="text-[12.5px] text-ink-3 mt-0.5">{sub}</p>}
      </div>
      {children}
    </div>
  );
}

const SIGNAL_COLORS: Record<string, string> = {
  'STRONG BUY':  '#10B981',
  'BUY':         '#34D399',
  'HOLD':        '#F59E0B',
  'SELL':        '#F97316',
  'STRONG SELL': '#EF4444',
};

// ── main page ─────────────────────────────────────────────────────────────────

export default function BacktestPage() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const { data, isLoading, isError } = useGetBacktestSummaryQuery();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-ink-3 text-[14px]">
        Loading model performance data…
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex items-center justify-center h-64 text-loss text-[14px]">
        Failed to load performance data. Make sure the backend is running.
      </div>
    );
  }

  const ms = data.model_stats;
  const ss = data.signal_stats;
  const history: any[] = data.history ?? [];

  const buyCount  = (ss.distribution?.STRONG_BUY ?? 0) + (ss.distribution?.BUY ?? 0);
  const sellCount = (ss.distribution?.SELL ?? 0)       + (ss.distribution?.STRONG_SELL ?? 0);

  // ── Horizon chart (grouped bar) ───────────────────────────────────────
  const horizonLabels = ms.by_horizon.map((h: any) => h.horizon.replace(' ', '\n'));
  const horizonAcc   = ms.by_horizon.map((h: any) => h.avg_accuracy);
  const horizonPrec  = ms.by_horizon.map((h: any) => h.avg_precision);

  const horizonOptions: ApexOptions = {
    ...chartBase(isDark),
    chart: { ...(chartBase(isDark).chart), type: 'bar' },
    plotOptions: { bar: { borderRadius: 5, columnWidth: '55%', grouped: true } },
    colors: ['#3B82F6', '#10B981'],
    xaxis: { ...(chartBase(isDark).xaxis), categories: horizonLabels },
    yaxis: { ...chartBase(isDark).yaxis, min: 60, max: 100, labels: { ...(chartBase(isDark).yaxis as any).labels, formatter: (v: number) => v + '%' } },
    tooltip: {
      ...chartBase(isDark).tooltip,
      y: { formatter: (v: number) => v.toFixed(1) + '%' },
    },
    legend: { show: true, position: 'top', labels: { colors: isDark ? '#AEB9CE' : '#4A5670' } },
  };
  const horizonSeries = [
    { name: 'Accuracy',  data: horizonAcc  },
    { name: 'Precision', data: horizonPrec },
  ];

  // ── Model type donut ──────────────────────────────────────────────────
  const modelLabels  = ms.by_model_type.map((m: any) => m.model);
  const modelCounts  = ms.by_model_type.map((m: any) => m.count);
  const donutColors  = ['#3B82F6','#10B981','#F59E0B','#8B5CF6','#EF4444','#06B6D4','#EC4899'];

  const donutOptions: ApexOptions = {
    ...chartBase(isDark),
    chart: { ...(chartBase(isDark).chart), type: 'donut' },
    labels: modelLabels,
    colors: donutColors.slice(0, modelLabels.length),
    plotOptions: { pie: { donut: { size: '65%', labels: { show: true, total: { show: true, label: 'Models', color: isDark ? '#AEB9CE' : '#4A5670', fontSize: '13px', fontWeight: 600 } } } } },
    legend: { show: true, position: 'bottom', labels: { colors: isDark ? '#AEB9CE' : '#4A5670' }, fontSize: '12px' },
    dataLabels: { enabled: false },
    tooltip: { ...chartBase(isDark).tooltip, y: { formatter: (v: number) => v + ' models' } },
  };

  // ── Signal distribution bar ───────────────────────────────────────────
  const sigLabels  = ss.by_signal_type.map((s: any) => s.signal);
  const sigCounts  = ss.by_signal_type.map((s: any) => s.count);
  const sigColors  = ss.by_signal_type.map((s: any) => SIGNAL_COLORS[s.signal] ?? '#3B82F6');

  const sigOptions: ApexOptions = {
    ...chartBase(isDark),
    chart: { ...(chartBase(isDark).chart), type: 'bar' },
    plotOptions: { bar: { borderRadius: 5, horizontal: true, barHeight: '55%' } },
    colors: sigColors,
    xaxis: { ...chartBase(isDark).xaxis, categories: sigLabels },
    yaxis: { labels: { style: { colors: isDark ? '#AEB9CE' : '#4A5670', fontFamily: 'inherit', fontSize: '12px' }, maxWidth: 110 } },
    tooltip: { ...chartBase(isDark).tooltip, y: { formatter: (v: number) => v + ' stocks' } },
  };
  const sigSeries = [{ name: 'Stocks', data: sigCounts }];

  // ── History timeline (area) ───────────────────────────────────────────
  const histDates = history.map(h => h.date);
  const histBuy   = history.map(h => h.buy_signals);
  const histSell  = history.map(h => h.sell_signals);

  const histOptions: ApexOptions = {
    ...chartBase(isDark),
    chart: { ...(chartBase(isDark).chart), type: 'area', stacked: false },
    stroke: { curve: 'smooth', width: 2 },
    fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.3, opacityTo: 0.05 } },
    colors: ['#10B981', '#EF4444'],
    xaxis: { ...chartBase(isDark).xaxis, categories: histDates },
    yaxis: { ...chartBase(isDark).yaxis, labels: { ...(chartBase(isDark).yaxis as any).labels, formatter: (v: number) => Math.round(v).toString() } },
    legend: { show: true, position: 'top', labels: { colors: isDark ? '#AEB9CE' : '#4A5670' } },
    tooltip: { ...chartBase(isDark).tooltip, x: { show: true } },
  };
  const histSeries = [
    { name: 'Buy Signals',  data: histBuy  },
    { name: 'Sell Signals', data: histSell },
  ];

  return (
    <div className="flex flex-col gap-6">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-[22px] font-bold text-ink tracking-tight">AI Performance Center</h1>
          <p className="text-[13px] text-ink-3 mt-1">
            Model accuracy, signal quality, and portfolio simulation across {ms.total_models} trained models
          </p>
        </div>
        <div className="flex items-center gap-2 h-8 px-3 rounded-full border border-line bg-surface-2 text-[12px] text-ink-3">
          <span className="w-2 h-2 rounded-full bg-gain" />
          Last updated: {ss.generated_at ? new Date(ss.generated_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'}
        </div>
      </div>

      {/* ── Stat cards ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Brain}
          label="Avg Model Accuracy"
          value={fmt(ms.avg_accuracy) + '%'}
          sub={`${ms.successful_models} of ${ms.total_models} models trained`}
          color="#3B82F6"
        />
        <StatCard
          icon={Target}
          label="Avg Precision"
          value={fmt(ms.avg_precision) + '%'}
          sub={`${ms.high_quality_models} models ≥ 70% on both`}
          color="#10B981"
        />
        <StatCard
          icon={TrendingUp}
          label="Actionable Signals"
          value={buyCount.toString()}
          sub={`${sellCount} sell signals · ${ss.total_signals} total`}
          color="#8B5CF6"
        />
        <StatCard
          icon={Zap}
          label="Avg Confidence"
          value={fmt(ss.avg_confidence) + '%'}
          sub={`Avg expected return ${fmt(ss.avg_expected_return)}% per signal`}
          color="#F59E0B"
        />
      </div>

      {/* ── Charts row 1 ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <SectionCard
          title="Accuracy & Precision by Horizon"
          sub="How well models predict across 6 time horizons"
        >
          <ReactApexChart
            type="bar"
            series={horizonSeries}
            options={horizonOptions}
            height={240}
          />
          <p className="text-[11.5px] text-ink-3">
            Precision = of all BUY signals, what % actually outperformed the threshold. Higher precision = fewer false positives.
          </p>
        </SectionCard>

        <SectionCard
          title="Best Model Distribution"
          sub={`${ms.by_model_type.length} model types compete per stock — winner is saved`}
        >
          <ReactApexChart
            type="donut"
            series={modelCounts}
            options={donutOptions}
            height={240}
          />
          <p className="text-[11.5px] text-ink-3">
            Each stock's best model is selected by harmonic mean of accuracy + precision. Ensemble & stacking variants dominate.
          </p>
        </SectionCard>

      </div>

      {/* ── Charts row 2 ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <SectionCard
          title="Current Signal Distribution"
          sub="Breakdown of all 377 signals generated today"
        >
          <ReactApexChart
            type="bar"
            series={sigSeries}
            options={sigOptions}
            height={220}
          />
          <div className="grid grid-cols-3 gap-3 mt-1">
            {ss.by_signal_type.filter((s: any) => ['STRONG BUY','BUY','STRONG SELL'].includes(s.signal)).map((s: any) => (
              <div key={s.signal} className="rounded-[9px] p-2.5 border border-line bg-surface-2 text-center">
                <div className="text-[11px] font-semibold" style={{ color: SIGNAL_COLORS[s.signal] }}>{s.signal}</div>
                <div className="text-[18px] font-bold text-ink mt-0.5">{s.count}</div>
                <div className="text-[11px] text-ink-3">{fmt(s.avg_confidence)}% conf</div>
              </div>
            ))}
          </div>
        </SectionCard>

        {history.length > 1 ? (
          <SectionCard
            title="Buy vs Sell Signals Over Time"
            sub="Signal evolution across model run history"
          >
            <ReactApexChart
              type="area"
              series={histSeries}
              options={histOptions}
              height={220}
            />
            <p className="text-[11.5px] text-ink-3">
              Each data point = one complete signal generation run across all Nifty 500 models.
            </p>
          </SectionCard>
        ) : (
          <SectionCard
            title="Expected Return by Signal Strength"
            sub="Model-predicted returns, weighted by confidence"
          >
            <div className="flex flex-col gap-3 mt-1">
              {ss.by_signal_type.filter((s: any) => s.count > 0).map((s: any) => (
                <div key={s.signal} className="flex items-center gap-3">
                  <span className="w-[90px] text-[12px] font-semibold flex-shrink-0" style={{ color: SIGNAL_COLORS[s.signal] }}>
                    {s.signal}
                  </span>
                  <div className="flex-1 h-5 rounded-full bg-surface-2 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.min(100, Math.abs(s.avg_expected_return) * 7)}%`,
                        background: SIGNAL_COLORS[s.signal],
                        opacity: 0.8,
                      }}
                    />
                  </div>
                  <span className="text-[12px] font-mono text-ink-2 w-14 text-right flex-shrink-0">
                    {s.avg_expected_return > 0 ? '+' : ''}{fmt(s.avg_expected_return)}%
                  </span>
                  <span className="text-[11px] text-ink-3 w-16 text-right flex-shrink-0">
                    {s.count} stocks
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[11.5px] text-ink-3 mt-2">
              Expected return = model's predicted upside to target price, based on ATR and horizon thresholds.
            </p>
          </SectionCard>
        )}

      </div>

      {/* ── Top signals table ────────────────────────────────────────────── */}
      <SectionCard
        title="Top 10 High-Confidence Signals"
        sub="Current STRONG BUY picks ranked by model confidence"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-line">
                {['Stock', 'Signal', 'Confidence', 'Expected Return', 'Model', 'Accuracy', 'Horizon'].map(h => (
                  <th key={h} className="text-left text-[11px] font-semibold uppercase tracking-[.07em] text-ink-3 pb-2.5 pr-4 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ss.top_signals.map((t: any, i: number) => {
                const sigColor = SIGNAL_COLORS[t.signal] ?? '#3B82F6';
                const expRet = t.trade?.expected_return_pct ?? 0;
                return (
                  <tr key={t.symbol + i} className="border-b border-line/50 hover:bg-surface-hover/50 transition-colors">
                    <td className="py-2.5 pr-4">
                      <div className="font-semibold text-ink">{t.symbol?.replace('.NS', '')}</div>
                      {t.name && <div className="text-[11px] text-ink-3 truncate max-w-[120px]">{t.name}</div>}
                    </td>
                    <td className="py-2.5 pr-4">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold" style={{ background: sigColor + '1A', color: sigColor }}>
                        {t.signal === 'STRONG BUY' ? <CheckCircle size={11} /> : t.signal === 'STRONG SELL' ? <AlertCircle size={11} /> : null}
                        {t.signal}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-surface-2 overflow-hidden">
                          <div className="h-full rounded-full bg-accent" style={{ width: `${t.confidence}%` }} />
                        </div>
                        <span className="font-mono text-[12px] text-ink-2">{fmt(t.confidence)}%</span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-4">
                      <span className={`font-mono font-semibold text-[13px] ${expRet >= 0 ? 'text-gain' : 'text-loss'}`}>
                        {expRet >= 0 ? '+' : ''}{fmt(expRet)}%
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-ink-2">{t.model?.name ?? '—'}</td>
                    <td className="py-2.5 pr-4">
                      <span className="font-mono text-[12px] text-ink-2">{fmt(t.model?.accuracy ?? 0)}%</span>
                    </td>
                    <td className="py-2.5 text-ink-3">{t.model?.horizon ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* ── Investor callout ─────────────────────────────────────────────── */}
      <div className="rounded-[14px] border border-accent/30 bg-accent/5 p-5 flex gap-4 items-start">
        <span className="w-10 h-10 rounded-[10px] grid place-items-center flex-shrink-0 bg-accent/10 text-accent">
          <Brain size={20} />
        </span>
        <div>
          <h4 className="text-[14px] font-bold text-ink">Model Methodology</h4>
          <p className="text-[12.5px] text-ink-2 mt-1 leading-relaxed">
            TradeMind trains <strong className="text-ink">{ms.successful_models} individual ML models</strong> (XGBoost, LightGBM, RandomForest, Ensemble) per stock across 6 prediction horizons.
            Models are selected by harmonic mean of accuracy and precision on a held-out time-series test set with purged cross-validation to prevent data leakage.
            Only models meeting <strong className="text-ink">≥ 70% accuracy and ≥ 70% precision</strong> are deployed — {ms.high_quality_models} models currently meet this bar.
            Signals are probability-calibrated using isotonic regression so a 85% confidence score reflects an 85% historical win rate.
          </p>
          <p className="text-[11.5px] text-ink-3 mt-2 italic">
            Disclaimer: Past model performance does not guarantee future returns. This is a research tool, not financial advice.
          </p>
        </div>
      </div>

    </div>
  );
}
