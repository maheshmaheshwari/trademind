# Frontend Redesign — Implementation Plan

**Source:** Claude Design export (`/tmp/design_files/trademind/`)  
**Target:** `frontend/src/` (React + TypeScript + MUI + TailwindCSS)  
**Design spec:** DM Sans + JetBrains Mono, dark/light theme, deep navy palette  
**Styling rule:** Tailwind classes first. CSS variables only where Tailwind cannot do it.  
**Icons:** `lucide-react` (already installed). Google brand icon kept as inline SVG.  
**Charts:** `react-apexcharts` + `apexcharts` — replaces hand-built SVG charts.

---

## Tailwind vs CSS Variables — Decision Rule

| Use **Tailwind classes** for | Use **CSS variables** only for |
|---|---|
| All colors (`bg-`, `text-`, `border-`) | Density multiplier `--u` (dynamic `calc()` scaling) |
| All spacing (`p-`, `m-`, `gap-`) | Sidebar width transition (`--sidebar-w`) |
| Typography (`font-`, `text-`, `tracking-`) | ApexCharts color props (pass hex strings, not Tailwind classes) |
| Borders, radius, shadows | Shimmer/skeleton gradient animation |
| Flexbox, grid, layout | Navbar height `--navbar-h` (used in sidebar calc) |
| Hover, focus, active states (`hover:`, `focus:`) | |
| Dark/light theme (`dark:`) | |
| Transitions (`transition-`, `duration-`, `ease-`) | |
| Animations (pulse, spin — extend if needed) | |

---

## Step 0 — Update `tailwind.config.ts` (do before anything else)

Replace the current minimal config with the full design palette as named Tailwind colors.  
This lets every component use `bg-surface`, `text-muted`, `border-line` etc. without arbitrary values.

```ts
// tailwind.config.ts
import plugin from 'tailwindcss/plugin';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],

  // Use data-theme attribute for dark mode (matches the design's [data-theme="dark"])
  darkMode: ['selector', '[data-theme="dark"]'],

  theme: {
    extend: {
      colors: {
        // ── Dark theme palette (defaults) ──
        bg:           { DEFAULT: '#0A0E1A' },
        surface:      { DEFAULT: '#111827', 2: '#161F33', 3: '#1C2740', hover: '#1A2438' },
        line:         { DEFAULT: 'rgba(255,255,255,0.07)', strong: 'rgba(255,255,255,0.13)' },
        ink:          { DEFAULT: '#EEF2F9', 2: '#AEB9CE', 3: '#6B7890' },
        accent:       { DEFAULT: '#3B82F6', soft: 'rgba(59,130,246,0.14)', 2: '#60A5FA' },
        gain:         { DEFAULT: '#10B981', soft: 'rgba(16,185,129,0.14)' },
        loss:         { DEFAULT: '#EF4444', soft: 'rgba(239,68,68,0.14)' },
        gold:         { DEFAULT: '#F59E0B', soft: 'rgba(245,158,11,0.16)' },
      },
      fontFamily: {
        sans:  ['DM Sans', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'ui-monospace', 'SF Mono', 'monospace'],
      },
      borderRadius: {
        card:  '14px',
        'card-sm': '9px',
        'card-lg': '18px',
      },
      boxShadow: {
        sm:  '0 1px 2px rgba(0,0,0,.18)',
        md:  '0 6px 24px rgba(0,0,0,.28)',
        lg:  '0 18px 50px rgba(0,0,0,.40)',
      },
      keyframes: {
        shimmer: { '0%': { backgroundPosition: '100% 0' }, '100%': { backgroundPosition: '-100% 0' } },
        pulse:   { '0%,100%': { boxShadow: '0 0 0 0 rgba(16,185,129,0)' }, '70%': { boxShadow: '0 0 0 7px rgba(16,185,129,0)' } },
        slideIn: { from: { transform: 'translateX(34px)' }, to: { transform: 'translateX(0)' } },
        pop:     { from: { transform: 'translateY(10px)' }, to: { transform: 'translateY(0)' } },
        toastIn: { from: { transform: 'translateY(12px)', opacity: '0' }, to: { transform: 'translateY(0)', opacity: '1' } },
        pageIn:  { from: { transform: 'translateY(8px)' }, to: { transform: 'none' } },
      },
      animation: {
        shimmer:  'shimmer 1.3s ease-in-out infinite',
        'pulse-dot': 'pulse 2s infinite',
        'slide-in': 'slideIn .3s cubic-bezier(.4,0,.2,1) both',
        pop:       'pop .22s cubic-bezier(.4,0,.2,1) both',
        'toast-in': 'toastIn .3s cubic-bezier(.4,0,.2,1) both',
        'page-in': 'pageIn .32s cubic-bezier(.4,0,.2,1) both',
      },
    },
  },
  plugins: [
    plugin(({ addVariant }) => {
      addVariant('light', '[data-theme="light"] &');
    }),
  ],
};
```

**Light theme overrides** — add to `src/index.css` (minimal, only what CSS vars still need):

```css
/* index.css — only what Tailwind can't do */
:root {
  --sidebar-w: 240px;
  --sidebar-w-collapsed: 72px;
  --navbar-h: 60px;
  --u: 1;   /* density multiplier */
}
[data-density="compact"] { --u: .82; }
[data-density="comfy"]   { --u: 1.2; }

/* Light theme color overrides for Tailwind's named colors */
[data-theme="light"] {
  --tw-bg: #EEF1F8;
  /* Override Tailwind colors at runtime via CSS vars for light mode */
}

/* SVG chart grid lines and skeleton — cannot use Tailwind */
.chart-grid  { stroke: rgba(255,255,255,.05); }
[data-theme="light"] .chart-grid { stroke: rgba(15,23,42,.06); }
.skel-bg     { background: linear-gradient(90deg,rgba(255,255,255,.04) 25%,rgba(255,255,255,.09) 37%,rgba(255,255,255,.04) 63%); background-size: 400% 100%; }
[data-theme="light"] .skel-bg { background: linear-gradient(90deg,rgba(15,23,42,.04) 25%,rgba(15,23,42,.08) 37%,rgba(15,23,42,.04) 63%); background-size: 400% 100%; }

/* Density spacing — calc() with --u cannot be replaced by Tailwind */
.density-pad  { padding: calc(18px * var(--u)); }
.density-gap  { gap: calc(16px * var(--u)); }
.density-head { padding: calc(15px * var(--u)) calc(18px * var(--u)); }
```

**Light mode strategy:** Use Tailwind's `dark:` prefix for the dark variant, plus the `light:` custom variant for light-only overrides. Example: `bg-bg dark:bg-bg light:bg-[#EEF1F8]`.

---

## File Map — Design → Implementation

| Design file | Target file(s) | Action |
|---|---|---|
| `styles.css` | `tailwind.config.ts` + `src/index.css` | Tailwind config + minimal CSS vars only |
| `icons.jsx` | `src/components/Icons.tsx` | New file — SVG icon set |
| `charts.jsx` | `src/components/Charts.tsx` | New file — SVG charts (keep CSS vars for stroke/fill) |
| `ui.jsx` | `src/components/ui/` (multiple) | Split into shared primitives, Tailwind classes |
| `shell.jsx` | `src/components/Layout.tsx` + `Navbar.tsx` | Rewrite both with Tailwind |
| `page-auth.jsx` | `src/pages/AuthPage.tsx` | Rewrite |
| `page-dashboard.jsx` | `src/pages/DashboardPage.tsx` | Rewrite |
| `page-signals.jsx` | `src/pages/AISignalsPage.tsx` | Rewrite + add StockDrawer |
| `page-market.jsx` | `src/pages/MarketPage.tsx` | Rewrite |
| `page-portfolio.jsx` | `src/pages/PortfolioPage.tsx` | Rewrite + add modal |
| `page-trades.jsx` | `src/pages/TradesPage.tsx` | Rewrite |
| `app.jsx` | `src/App.tsx` | Update routing + theme/density state |
| `data.jsx` | `src/hooks/useTradeData.ts` | API wiring (replace mock data) |

---

## Implementation Order

### Phase 1 — Design System Foundation

**Step 1.1 — `tailwind.config.ts` + `src/index.css`**
- Apply the full config from Step 0 above
- `index.css` keeps only: CSS var density scale, sidebar/navbar dimensions, SVG chart grid, skeleton gradient
- Add Google Fonts to `frontend/index.html`:
  ```html
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  ```
- Update `ThemeContext.tsx`: toggle sets `document.documentElement.setAttribute('data-theme', theme)` and adds/removes `dark` class for Tailwind's dark variant

**Step 1.2 — Icon component** (`src/components/Icons.tsx`) ✅ DONE
- Uses `lucide-react` — maps 30+ semantic names to Lucide components
- `<Icon name="dashboard" size={20} className="text-ink-3" />` — Tailwind color classes work via `currentColor`
- Google brand icon kept as inline SVG (not in lucide)
- Full map: `dashboard→LayoutDashboard`, `signals/trendUp→TrendingUp`, `market→LineChart`, `portfolio/pie→PieChart`, `trades→ArrowLeftRight`, `flow→BarChart2`, `news→Newspaper`, `sparkle→Sparkles`, etc.

**Step 1.3 — Chart components** (`src/components/Charts.tsx`) ✅ DONE
- Uses `react-apexcharts` + `apexcharts` (installed)
- All charts read `useTheme()` and pass `mode: 'dark' | 'light'` to ApexCharts — auto theme-aware
- `Sparkline` → ApexCharts area sparkline (sparkline: enabled)
- `AreaChart` → ApexCharts area chart with INR formatter on Y axis and tooltip
- `Donut` → ApexCharts donut with center labels (total/sector)
- `Gauge` → ApexCharts radialBar (startAngle: -135, endAngle: 135) — BULLISH/NEUTRAL/BEARISH
- `FlowBars` → ApexCharts grouped bar chart, FII (blue) + DII (gold), shared tooltip

**Step 1.4 — Shared UI primitives** (`src/components/ui/`)

| Component | Tailwind classes used | Notes |
|---|---|---|
| `SignalBadge` | `inline-flex items-center gap-1 h-[23px] px-2 rounded-[7px] text-[11.5px] font-bold` + `bg-gain-soft text-gain` / `bg-loss-soft text-loss` / `bg-gold-soft text-gold` | BUY/SELL/HOLD |
| `SymbolCell` | `flex items-center gap-3` logo: `w-8 h-8 rounded-[9px] grid place-items-center font-bold text-xs` | Symbol + sector |
| `Conf` | `flex items-center gap-2` track: `flex-1 h-1.5 rounded-full bg-surface-3` fill: `h-full rounded-full` | Confidence bar |
| `Delta` | `inline-flex items-center gap-0.5 font-semibold` + `text-gain` / `text-loss` | +/- with icon |
| `Card` | `bg-surface border border-line rounded-card relative` header: `flex items-center justify-between border-b border-line` | Surface card |
| `Th` + `useSort` | `text-left text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 cursor-pointer` | Sortable header |
| `Pager` | `flex items-center justify-between gap-3 p-3 border-t border-line flex-wrap` btn: `min-w-8 h-8 px-2 rounded-lg border border-line bg-transparent` | Pagination |
| `Skeleton` | `rounded-[7px] bg-surface-3 skel-bg animate-shimmer` | Shimmer uses CSS var gradient |
| `ToastProvider` | `fixed bottom-5 left-1/2 -translate-x-1/2 z-[200] flex flex-col gap-2.5 items-center` toast: `min-w-[300px] flex items-center gap-3 p-3 rounded-[13px] bg-surface-2 border border-line-strong shadow-lg animate-toast-in` | Bottom-center |

**Step 1.5 — Theme + density context** (`src/ThemeContext.tsx`)
- Add `density` state: `'compact' | 'balanced' | 'comfy'`
- On theme change: toggle `dark` class + `data-theme` attribute on `<html>`
- On density change: set `data-density` attribute on `<html>` (CSS vars `--u` pick it up)
- Persist both to `localStorage`

---

### Phase 2 — Shell (Sidebar + Navbar)

**Step 2.1 — Sidebar** (`src/components/Layout.tsx`)

Key Tailwind classes:
```
sidebar wrapper:   flex-shrink-0 bg-surface border-r border-line flex flex-col z-[5]
                   transition-[width] duration-[260ms] ease-[cubic-bezier(.4,0,.2,1)]
expanded:          w-60    (240px)
collapsed:         w-[72px]
brand:             flex items-center gap-2.5 h-[60px] px-4.5 border-b border-line overflow-hidden
logo:              w-[34px] h-[34px] rounded-[10px] bg-gradient-to-br from-accent to-blue-800
nav item:          flex items-center gap-3 h-[42px] px-3 rounded-[9px] text-ink-2 font-medium
                   hover:bg-surface-hover hover:text-ink transition-colors duration-150
active item:       bg-accent-soft text-accent-2 (+ before pseudo for left border via CSS)
collapsed labels:  opacity-0 pointer-events-none (when .collapsed)
footer:            border-t border-line p-3
```

**Step 2.2 — Navbar** (`src/components/Navbar.tsx`)

Key Tailwind classes:
```
navbar:            h-[60px] flex-shrink-0 flex items-center gap-3.5
                   border-b border-line backdrop-blur-[14px] bg-surface/70 z-[4]
search input:      w-full h-10 pl-10 pr-14 rounded-[11px] border border-line bg-surface-2
                   text-ink text-[13.5px] focus:border-accent focus:bg-surface outline-none
market status:     flex items-center gap-2 h-9 px-3 rounded-full text-[12.5px] font-semibold border
open:              text-gain bg-gain-soft border-line
closed:            text-loss bg-loss-soft border-line
icon btn:          w-[38px] h-[38px] rounded-[10px] border border-line bg-transparent
                   text-ink-2 grid place-items-center hover:bg-surface-hover hover:text-ink
avatar:            w-[38px] h-[38px] rounded-[11px] bg-gradient-to-br from-indigo-500 to-accent
                   grid place-items-center font-bold text-sm text-white
dropdown:          absolute right-0 top-[46px] w-[210px] bg-surface border border-line-strong
                   rounded-[13px] shadow-lg z-10 overflow-hidden p-1.5
```

---

### Phase 3 — Pages

**Step 3.1 — Auth Page** (`src/pages/AuthPage.tsx`)

Layout: `grid grid-cols-[1.1fr_1fr] min-h-screen bg-bg`

Left panel (showcase):
- `relative overflow-hidden border-r border-line p-12 flex flex-col justify-between`
- `bg-gradient-to-br from-surface to-bg`
- Signal preview cards: `flex items-center justify-between p-3 rounded-xl border border-line backdrop-blur-sm`

Right panel (form):
- `flex flex-col p-7 px-8 relative`
- Form inputs: `h-11 pl-10 rounded-[11px] border border-line bg-surface-2 text-ink text-sm w-full focus:border-accent outline-none`
- Primary button: `w-full h-[46px] rounded-[11px] bg-accent text-white font-semibold shadow-[0_4px_14px_rgba(59,130,246,.32)] hover:bg-accent-2`
- Wire login/register to existing `AuthContext`

**Step 3.2 — Dashboard** (`src/pages/DashboardPage.tsx`)

Stat cards grid: `grid grid-cols-4 gap-4 max-lg:grid-cols-2`  
Stat card: `bg-surface border border-line rounded-card p-[calc(17px*var(--u))] relative overflow-hidden`  
Stat value: `text-[calc(27px*var(--u))] font-bold tracking-tight mt-2.5 mb-1 font-mono`

Index + Sentiment row: `grid gap-4 mb-4` style `gridTemplateColumns: '1.7fr 1fr'`

Signal cards:
- Rich variant: `bg-surface-2 border border-line rounded-card p-[calc(15px*var(--u))] hover:border-line-strong hover:-translate-y-0.5 hover:bg-surface-hover transition-all duration-100`
- Compact variant: `grid grid-cols-[1.5fr_auto_1fr_90px] items-center gap-3.5 p-2.5 px-[15px]`
- Bold variant: same as rich but with `border-l-[3px]` in gain/loss/gold color

Recent trades table: standard `.tbl` pattern — see table classes in Step 1.4

**Step 3.3 — AI Signals Page** (`src/pages/AISignalsPage.tsx`)

Filter bar: `flex items-center gap-3 flex-wrap`  
Search: same as navbar search, `max-w-[280px]`  
Segmented control: `inline-flex bg-surface-2 border border-line rounded-[10px] p-[3px] gap-0.5`  
Segment btn: `border-none bg-transparent text-ink-2 text-[12.5px] font-semibold px-3 py-1.5 rounded-[7px]`  
Active segment: `bg-accent text-white`  
Range slider: custom styled via `appearance-none h-[5px] rounded-full bg-surface-3`

Table: `w-full border-collapse text-[13px]`  
`th`: `text-left text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 px-3.5 py-[calc(11px*var(--u))] border-b border-line sticky top-0 bg-surface z-[1]`  
`td`: `px-3.5 py-[calc(12px*var(--u))] border-b border-line whitespace-nowrap`  
`tr:hover td`: `bg-surface-2`

**StockDrawer** (`src/components/StockDrawer.tsx`):  
`fixed top-0 right-0 bottom-0 w-[520px] max-w-[92vw] bg-surface border-l border-line z-[95] flex flex-col shadow-lg animate-slide-in`  
Scrim: `fixed inset-0 bg-[rgba(3,6,15,.55)] backdrop-blur-sm z-[90]`

**Step 3.4 — Market Overview** (`src/pages/MarketPage.tsx`)

Index cards: `grid grid-cols-4 gap-4 max-lg:grid-cols-2`  
Card: `bg-surface border border-line rounded-card p-[calc(18px*var(--u))] relative overflow-hidden`

FII/DII + Breadth: `grid gap-4` with `gridTemplateColumns: '1.7fr 1fr'`

Heatmap grid: `grid grid-cols-4 gap-2 max-md:grid-cols-2`  
Heat cell: `rounded-[9px] p-[13px] flex flex-col gap-1 min-h-[78px] justify-between border border-white/[.06] hover:-translate-y-0.5 transition-transform duration-100`  
Colors: inline style for background rgba (intensity based on % change — must be computed, not Tailwind)

Gainers/Losers: `grid grid-cols-2 gap-4`

**Step 3.5 — Portfolio Page** (`src/pages/PortfolioPage.tsx`)

Summary grid: `grid grid-cols-3 gap-4`  
P&L card gradient: inline `style={{ background: pnl >= 0 ? 'linear-gradient(135deg,...)' : '...' }}` (dynamic, can't use Tailwind)

Chart + Donut: same `1.7fr 1fr` grid as Dashboard

Holdings table: same table pattern as Signals page

**AddPositionModal** (`src/components/AddPositionModal.tsx`):  
`fixed inset-0 z-[100] grid place-items-center p-5`  
Modal card: `w-[480px] max-w-full bg-surface border border-line-strong rounded-card-lg shadow-lg relative z-[101] animate-pop max-h-[90vh] flex flex-col`  
Form inputs: `h-11 px-3.5 rounded-[11px] border border-line bg-surface-2 text-ink font-sans text-sm focus:border-accent outline-none`

**Step 3.6 — Trades & Orders** (`src/pages/TradesPage.tsx`)

Tabs: `flex gap-1 border-b border-line mb-[calc(18px*var(--u))]`  
Tab: `border-none bg-transparent text-ink-2 font-sans text-sm font-semibold px-4 py-3 cursor-pointer relative whitespace-nowrap`  
Active tab: `text-accent-2` + `after:absolute after:left-3 after:right-3 after:-bottom-px after:h-[2.5px] after:rounded-t after:bg-accent`  
Count badge: `ml-1.5 text-[11px] bg-surface-3 text-ink-2 px-1.5 py-px rounded-full`

Close button: `h-8 px-3 rounded-[9px] text-[12.5px] font-semibold bg-loss-soft text-loss border-transparent hover:bg-loss hover:text-white transition-colors`

---

### Phase 4 — API Wiring

Replace mock data with real API calls:

| Data | Endpoint | Hook |
|---|---|---|
| Stocks + signals | `GET /api/signals` | `useGetSignalsQuery` |
| Market indices | `GET /api/market/overview` | `useGetMarketOverviewQuery` |
| Portfolio holdings | `GET /api/portfolio` | `useGetPortfolioQuery` |
| Open positions | `GET /api/trades?status=open` | `useGetTradesQuery` |
| Trade history | `GET /api/trades/history` | `useGetTradeHistoryQuery` |
| GTT orders | `GET /api/orders/gtt` | `useGetGTTOrdersQuery` |
| FII/DII data | `GET /api/market/fii-dii` | `useGetFiiDiiQuery` |
| Sector heatmap | `GET /api/market/sectors` | `useGetSectorsQuery` |
| Market breadth | `GET /api/market/breadth` | `useGetBreadthQuery` |

Use existing `src/services/tradeMindApiService.ts` + RTK Query. Skeleton components handle loading states.

---

### Phase 5 — Cleanup

- `StockDetailPage.tsx` → replaced by `StockDrawer`, remove route or keep for deep-link
- `TradeExecutionPage.tsx` → replaced by `AddPositionModal`, remove
- `OrderHistoryPage.tsx` → merged into `TradesPage` history tab, remove
- `RiskSettingsPage.tsx` → defer to settings modal, keep for now
- Update `App.tsx` routes

---

## Backend Changes Needed

| UI action | Required endpoint |
|---|---|
| Refresh Signals button | `POST /api/signals/refresh` — async trigger |
| Close position | `POST /api/trades/{id}/close` |
| Export CSV | Handle client-side (blob download) from RTK Query data |
| Market status pill | `GET /api/health` already has `market_open` — already done |
| GTT Sync | `POST /api/orders/gtt/sync` |

---

## Dependencies

```bash
# Already installed
lucide-react        # icons

# Newly installed (Day 1)
apexcharts
react-apexcharts
```

Add to `frontend/index.html`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
```

---

## Recommended Sequence

```
Day 1
├── Step 0    — tailwind.config.ts + index.css
├── Step 1.1  — ThemeContext (dark class + data-theme + density)
├── Step 1.2  — Icons component
└── Step 1.3  — Charts component

Day 2
├── Step 1.4  — All 9 UI primitives (SignalBadge, Card, Table, Pager, Toast, etc.)
└── Phase 2   — Sidebar + Navbar

Day 3
├── Step 3.1  — Auth page
├── Step 3.2  — Dashboard
└── Step 3.3  — Signals + StockDrawer

Day 4
├── Step 3.4  — Market Overview
├── Step 3.5  — Portfolio + AddPositionModal
└── Step 3.6  — Trades & Orders

Day 5
├── Phase 4   — Wire all pages to real API
└── Phase 5   — Cleanup + route updates
```

---

## Key Notes

1. **Dark mode** — configure Tailwind with `['selector', '[data-theme="dark"]']` so `dark:` variants activate on the design's `data-theme` attribute, not a separate `dark` class.

2. **Color naming** — use semantic names (`bg-surface`, `text-ink-2`, `border-line`) not raw hex. Extend `tailwind.config.ts` once, use everywhere.

3. **Density spacing** — anything that scales with density must use `calc(Xpx * var(--u))` as an inline style or the `.density-pad` utility class. Do not use fixed Tailwind spacing for paddings that should respond to density.

4. **SVG charts** — pass colors as prop strings (resolved hex or named CSS var string). The chart components themselves stay theme-aware by accepting color props from callers.

5. **Font mono on numbers** — use `font-mono` Tailwind class on any element showing prices, P&L, percentages, or counts.

6. **Heatmap cell colors** — background opacity is computed dynamically from % change magnitude. Use inline `style={{ background: \`rgba(..., ${intensity})\` }}` — Tailwind cannot generate these at runtime.

7. **TypeScript types** — create `src/types/trade.ts` with: `Stock`, `Signal`, `Trade`, `IndexData`, `HeatmapSector`, `Holding`, `GTTOrder`, `NewsItem`.
