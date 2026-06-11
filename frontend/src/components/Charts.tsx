import ReactApexChart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';
import { useTheme } from '../ThemeContext';

// ─── shared theme helpers ───────────────────────────────────────────────────

function baseOptions(isDark: boolean): ApexOptions {
  const text2 = isDark ? '#AEB9CE' : '#4A5670';
  const grid  = isDark ? 'rgba(255,255,255,.05)' : 'rgba(15,23,42,.06)';
  return {
    chart: {
      background: 'transparent',
      toolbar: { show: false },
      zoom: { enabled: false },
      animations: { enabled: true, speed: 400 },
    },
    theme: { mode: isDark ? 'dark' : 'light' },
    tooltip: {
      theme: isDark ? 'dark' : 'light',
      style: { fontFamily: 'DM Sans, sans-serif', fontSize: '12px' },
      x: { show: false },
    },
    grid: {
      borderColor: grid,
      strokeDashArray: 0,
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
      padding: { left: 4, right: 4 },
    },
    xaxis: {
      labels: { style: { colors: text2, fontFamily: 'JetBrains Mono, monospace', fontSize: '11px' } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      labels: { style: { colors: text2, fontFamily: 'JetBrains Mono, monospace', fontSize: '11px' } },
    },
    dataLabels: { enabled: false },
    legend: { show: false },
    stroke: { curve: 'smooth' },
  };
}

// ─── Sparkline ──────────────────────────────────────────────────────────────

interface SparklineProps {
  data: number[];
  color?: string;
  w?: number;
  h?: number;
}

export function Sparkline({ data, color = '#3B82F6', w = 120, h = 36 }: SparklineProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const options: ApexOptions = {
    chart: {
      type: 'line',
      sparkline: { enabled: true },
      background: 'transparent',
      animations: { enabled: false },
    },
    stroke: { curve: 'smooth', width: 2, colors: [color] },
    fill: {
      type: 'gradient',
      gradient: {
        shade: 'dark',
        type: 'vertical',
        opacityFrom: 0.3,
        opacityTo: 0,
        stops: [0, 100],
        colorStops: [
          { offset: 0, color, opacity: 0.28 },
          { offset: 100, color, opacity: 0 },
        ],
      },
    },
    tooltip: { enabled: false },
    theme: { mode: isDark ? 'dark' : 'light' },
  };

  return (
    <ReactApexChart
      type="area"
      series={[{ data }]}
      options={options}
      width={w}
      height={h}
    />
  );
}

// ─── AreaChart ──────────────────────────────────────────────────────────────

interface AreaChartProps {
  data: number[];
  color?: string;
  h?: number;
  labels?: string[];
  currency?: boolean; // true → ₹ prefix + compact formatting; false (default) → plain number
}

export function AreaChart({ data, color = '#3B82F6', h = 230, labels, currency = false }: AreaChartProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const base = baseOptions(isDark);

  function fmtAxis(v: number) {
    if (!currency) return Math.round(v).toLocaleString('en-IN');
    if (Math.abs(v) >= 1e7) return '₹' + (v / 1e7).toFixed(1) + 'Cr';
    if (Math.abs(v) >= 1e5) return '₹' + (v / 1e5).toFixed(1) + 'L';
    return '₹' + Math.round(v).toLocaleString('en-IN');
  }

  function fmtTip(v: number) {
    if (!currency) return v.toLocaleString('en-IN');
    if (Math.abs(v) >= 1e7) return '₹' + (v / 1e7).toFixed(2) + ' Cr';
    if (Math.abs(v) >= 1e5) return '₹' + (v / 1e5).toFixed(2) + ' L';
    return '₹' + v.toLocaleString('en-IN');
  }

  const options: ApexOptions = {
    ...base,
    chart: { ...base.chart, type: 'area' },
    stroke: { curve: 'smooth', width: 2.4, colors: [color] },
    fill: {
      type: 'gradient',
      gradient: {
        type: 'vertical',
        colorStops: [
          { offset: 0, color, opacity: 0.22 },
          { offset: 100, color, opacity: 0 },
        ],
      },
    },
    xaxis: {
      ...base.xaxis,
      categories: labels ?? data.map((_, i) => i.toString()),
      tickAmount: labels ? labels.length - 1 : undefined,
      tickPlacement: 'on',
      labels: {
        ...base.xaxis?.labels,
        show: !!labels,
        rotate: 0,
        hideOverlappingLabels: false,
        showDuplicates: false,
      },
    },
    yaxis: {
      ...(base.yaxis as object),
      labels: {
        ...(Array.isArray(base.yaxis) ? {} : base.yaxis?.labels),
        formatter: fmtAxis,
      },
    },
    markers: { size: 0, hover: { size: 5, sizeOffset: 2 } },
    colors: [color],
    tooltip: {
      ...base.tooltip,
      y: { formatter: fmtTip },
    },
  };

  return (
    <ReactApexChart
      type="area"
      series={[{ name: 'Value', data }]}
      options={options}
      width="100%"
      height={h}
    />
  );
}

// ─── Donut ──────────────────────────────────────────────────────────────────

export interface DonutSlice {
  sector: string;
  val: number;
  color: string;
}

interface DonutProps {
  data: DonutSlice[];
  size?: number;
  centerTop?: string;
  centerBottom?: string;
}

export function Donut({ data, size = 280, centerTop, centerBottom }: DonutProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const text2 = isDark ? '#AEB9CE' : '#4A5670';
  const inkColor = isDark ? '#EEF2F9' : '#0E1726';

  const options: ApexOptions = {
    chart: {
      type: 'donut',
      background: 'transparent',
      toolbar: { show: false },
      animations: { enabled: true, speed: 400 },
    },
    theme: { mode: isDark ? 'dark' : 'light' },
    colors: data.map(d => d.color),
    labels: data.map(d => d.sector),
    dataLabels: { enabled: false },
    legend: {
      show: true,
      position: 'right',
      fontFamily: 'DM Sans, sans-serif',
      fontSize: '12.5px',
      labels: { colors: text2 },
      markers: { size: 5 },
      itemMargin: { vertical: 4 },
    },
    plotOptions: {
      pie: {
        donut: {
          size: '70%',
          labels: {
            show: true,
            name: {
              show: true,
              fontSize: '13px',
              fontFamily: 'DM Sans, sans-serif',
              color: text2,
              offsetY: -4,
            },
            value: {
              show: true,
              fontSize: '18px',
              fontFamily: 'JetBrains Mono, monospace',
              fontWeight: 700,
              color: inkColor,
              offsetY: 4,
              formatter: (v: string) => Math.round(+v / data.reduce((a, d) => a + d.val, 0) * 100) + '%',
            },
            total: {
              show: true,
              label: centerTop ?? 'Total',
              fontSize: '13px',
              fontFamily: 'DM Sans, sans-serif',
              color: text2,
              formatter: () => centerBottom ?? '',
            },
          },
        },
      },
    },
    stroke: { width: 2, colors: [isDark ? '#111827' : '#FFFFFF'] },
    tooltip: {
      theme: isDark ? 'dark' : 'light',
      style: { fontFamily: 'DM Sans, sans-serif' },
      y: {
        formatter: (v: number) => {
          const total = data.reduce((a, d) => a + d.val, 0);
          return Math.round(v / total * 100) + '%';
        },
      },
    },
  };

  return (
    <ReactApexChart
      type="donut"
      series={data.map(d => d.val)}
      options={options}
      width="100%"
      height={size}
    />
  );
}

// ─── Gauge (sentiment) ──────────────────────────────────────────────────────

interface GaugeProps {
  value: number; // 0–100
  size?: number;
}

export function Gauge({ value, size = 280 }: GaugeProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const label = value >= 60 ? 'BULLISH' : value >= 40 ? 'NEUTRAL' : 'BEARISH';
  const color = value >= 60 ? '#10B981' : value >= 40 ? '#F59E0B' : '#EF4444';
  const text3 = isDark ? '#6B7890' : '#8A95AD';

  const options: ApexOptions = {
    chart: {
      type: 'radialBar',
      background: 'transparent',
      toolbar: { show: false },
      animations: { enabled: true, speed: 600 },
    },
    theme: { mode: isDark ? 'dark' : 'light' },
    plotOptions: {
      radialBar: {
        startAngle: -135,
        endAngle: 135,
        hollow: { size: '62%', background: 'transparent' },
        track: {
          background: isDark ? '#1C2740' : '#E9EEF7',
          strokeWidth: '100%',
          margin: 0,
        },
        dataLabels: {
          name: {
            show: true,
            fontSize: '13px',
            fontFamily: 'DM Sans, sans-serif',
            fontWeight: 600,
            color: text3,
            offsetY: 30,
          },
          value: {
            show: true,
            fontSize: '26px',
            fontFamily: 'JetBrains Mono, monospace',
            fontWeight: 700,
            color,
            offsetY: -14,
            formatter: () => String(value),
          },
        },
      },
    },
    colors: [color],
    labels: [label],
    stroke: { lineCap: 'round' },
    tooltip: { enabled: false },
  };

  return (
    <ReactApexChart
      type="radialBar"
      series={[value]}
      options={options}
      width="100%"
      height={size}
    />
  );
}

// ─── FlowBars (FII / DII) ───────────────────────────────────────────────────

export interface FlowBar {
  day: string;
  fii: number;
  dii: number;
}

interface FlowBarsProps {
  data: FlowBar[];
  h?: number;
}

export function FlowBars({ data, h = 210 }: FlowBarsProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const base = baseOptions(isDark);

  const options: ApexOptions = {
    ...base,
    chart: { ...base.chart, type: 'bar' },
    plotOptions: {
      bar: {
        columnWidth: '55%',
        borderRadius: 2,
        dataLabels: { position: 'top' },
      },
    },
    colors: ['#3B82F6', '#F59E0B'],
    xaxis: {
      ...base.xaxis,
      categories: data.map(d => d.day),
    },
    yaxis: {
      ...(base.yaxis as object),
      labels: {
        ...(Array.isArray(base.yaxis) ? {} : base.yaxis?.labels),
        formatter: (v: number) => (v >= 0 ? '+' : '') + Math.round(v).toLocaleString('en-IN'),
      },
    },
    tooltip: {
      ...base.tooltip,
      shared: true,
      intersect: false,
      y: {
        formatter: (v: number) => (v >= 0 ? '+' : '') + Math.round(v).toLocaleString('en-IN') + ' Cr',
      },
    },
    legend: {
      show: true,
      position: 'top',
      horizontalAlign: 'right',
      fontFamily: 'DM Sans, sans-serif',
      fontSize: '12px',
      labels: { colors: isDark ? '#AEB9CE' : '#4A5670' },
      markers: { size: 5 },
    },
  };

  return (
    <ReactApexChart
      type="bar"
      series={[
        { name: 'FII Net', data: data.map(d => d.fii) },
        { name: 'DII Net', data: data.map(d => d.dii) },
      ]}
      options={options}
      width="100%"
      height={h}
    />
  );
}
