# TradeMind AI — Design → Production Implementation Plan

**Source design**: `trademind-design/project/src/`  
**Target**: `frontend/src/` + `backend/`  
**Date**: 2026-06-04

---

## 1. Gap Analysis

### 🔴 MISSING — build from scratch

| # | What | Files |
|---|------|-------|
| 1 | **AutopilotPage** — "AI Authorized Trades" page entirely absent | New `frontend/src/pages/AutopilotPage.tsx` |
| 2 | **`/autopilot` route** | `frontend/src/App.tsx` |
| 3 | **"AI Authorized" sidebar nav item** | `frontend/src/components/Layout.tsx` |
| 4 | **`authorized_trades` DB table** | `backend/database/schema_pg.py` |
| 5 | **`autopilot_settings` DB table** | `backend/database/schema_pg.py` |
| 6 | **Autopilot API routes** | New `backend/api/routes/autopilot.py` |
| 7 | **Register autopilot router** | `backend/api/server.py` |
| 8 | **Autopilot types + RTK Query endpoints** | `frontend/src/services/tradeMindApiService.ts` |
| 9 | **Notification popover CSS** — `.notif-pop`, `.notif-head`, `.notif-list`, `.notif-item`, `.notif-item.unread`, `.notif-ic`, `.notif-foot` used in Navbar but not defined | `frontend/src/index.css` |

---

### 🟡 PARTIALLY DIFFERENT — needs update

| # | What | Design | Current | Fix |
|---|------|--------|---------|-----|
| 10 | **Signal card variants** | 3 styles: `rich` / `compact` / `bold`, user-selectable | Only `rich` exists | Add `compact` + `bold` variants to Dashboard; wire style selector in Settings → Appearance |
| 11 | **Navbar avatar menu links** | Opens Settings with specific tab (Profile / Preferences / Notifications / Security) | Routes to `/settings/risk` (stale path) | Fix to navigate to `/settings` with correct `tab` state |
| 12 | **Market status live clock** | Shows `NSE · 15:24` with a live IST clock | Shows `NSE` only | Add a 1-second `setInterval` clock in Navbar |
| 13 | **Settings → Appearance tab** | Theme + Accent colour picker (4 swatches) + Density + Signal card style | Has Theme + Density; missing Accent and Signal style | Add accent colour state to ThemeContext; add two new controls in Appearance tab |

---

### ✅ ALREADY CORRECT — no changes needed

- Dashboard (stat cards, NIFTY chart, sentiment gauge, top signals, recent trades)
- AI Signals page (filters, table, confidence slider, pagination)
- Market page (index cards, FII/DII chart, breadth, heatmap, gainers/losers)
- Portfolio page (chart, donut, holdings table, Add Position modal)
- Trades page (open positions, history, GTT tabs)
- Watchlist page (grid/table toggle, stat cards, remove)
- Settings page (all 5 tabs: Profile, Brokers, Notifications, Appearance, Security)
- StockDrawer (price chart, metrics, horizon breakdown, news)
- Toast system, dark/light theme, density scaling, collapsible sidebar

---

## 2. Detailed Spec per Item

---

### Item 1–3 + 6–8: AutopilotPage + route + sidebar + backend

#### Page layout (from `page-autopilot.jsx`)

```
┌─────────────────────────────────────────────────────┐
│ Page header                                         │
│   "AI Authorized Trades"                            │
│   subtitle: "Trades you've authorized the AI to     │
│              place & manage via Angel One"           │
│                                            [Autopilot pill ▶] │
├──────────┬──────────┬──────────┬──────────┤
│ Capital  │ Active   │ Realized │ Projected│  ← 4 stat cards
│ Under AI │ Mandates │ P&L      │ Profit   │
├─────────────────────────────────────────────────────┤
│ AI banner: "AI is managing N trades across M sectors"│
├─────────────────────────────────────────────────────┤
│ Filter: [All][Running][Pending][Target hit][Stopped] │
├─────────────────────────────────────────────────────┤
│ Table:                                              │
│ Stock | Signal | Auth ₹ | Qty | Entry→Target |      │
│ Exp.Profit | Max Loss | Live P&L | Status | Action  │
└─────────────────────────────────────────────────────┘
```

#### Backend DB schema additions (`schema_pg.py`)

```sql
-- authorized_trades
CREATE TABLE IF NOT EXISTS authorized_trades (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol      TEXT NOT NULL,
    name        TEXT,
    sector      TEXT,
    signal      TEXT NOT NULL DEFAULT 'BUY',
    mode        TEXT DEFAULT 'PAPER',
    qty         INTEGER NOT NULL DEFAULT 0,
    amount      DOUBLE PRECISION NOT NULL DEFAULT 0,
    entry       DOUBLE PRECISION,
    target      DOUBLE PRECISION,
    sl          DOUBLE PRECISION,
    exp_profit  DOUBLE PRECISION DEFAULT 0,
    max_loss    DOUBLE PRECISION DEFAULT 0,
    cmp         DOUBLE PRECISION,
    actual_pnl  DOUBLE PRECISION,
    status      TEXT DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING','EXECUTED','COMPLETED','STOPPED')),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- autopilot_settings (one row per user)
CREATE TABLE IF NOT EXISTS autopilot_settings (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    enabled    BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Backend API endpoints (`routes/autopilot.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/autopilot/status` | `?user_id=` → `{ enabled, capital, active, realized_pnl, projected_profit }` |
| `POST` | `/api/autopilot/toggle` | `{ user_id }` → flip `enabled`; return new state |
| `GET` | `/api/autopilot/trades` | `?user_id=&status=` → list of authorized trades |
| `POST` | `/api/autopilot/trades` | Authorize a trade (body: symbol, user_id, amount, qty, entry, target, sl, signal, mode) |
| `DELETE` | `/api/autopilot/trades/{trade_id}` | Revoke a trade (set status = STOPPED) |

#### Frontend types (add to `tradeMindApiService.ts`)

```ts
export interface AuthorizedTrade {
  id: number;
  user_id: number;
  symbol: string;
  name: string;
  sector: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  mode: 'PAPER' | 'LIVE';
  qty: number;
  amount: number;
  entry: number;
  target: number;
  sl: number;
  exp_profit: number;
  max_loss: number;
  cmp: number | null;
  actual_pnl: number | null;
  status: 'PENDING' | 'EXECUTED' | 'COMPLETED' | 'STOPPED';
  created_at: string;
}

export interface AutopilotStatus {
  enabled: boolean;
  capital: number;
  active: number;
  realized_pnl: number;
  projected_profit: number;
}
```

#### Frontend RTK Query endpoints (add to `tradeMindApiService.ts`)

```ts
getAutopilotStatus: builder.query<AutopilotStatus, number>({
  query: (userId) => ({ url: '/api/autopilot/status', params: { user_id: userId } }),
  providesTags: ['Autopilot'],
}),
toggleAutopilot: builder.mutation<{ enabled: boolean }, number>({
  query: (userId) => ({ url: '/api/autopilot/toggle', method: 'POST', data: { user_id: userId } }),
  invalidatesTags: ['Autopilot'],
}),
getAuthorizedTrades: builder.query<{ data: AuthorizedTrade[]; total: number }, { userId: number; status?: string }>({
  query: ({ userId, status }) => ({ url: '/api/autopilot/trades', params: { user_id: userId, status } }),
  providesTags: ['Autopilot'],
}),
authorizeTradeAuto: builder.mutation<{ status: string; data: AuthorizedTrade }, Partial<AuthorizedTrade>>({
  query: (data) => ({ url: '/api/autopilot/trades', method: 'POST', data }),
  invalidatesTags: ['Autopilot'],
}),
revokeAuthorizedTrade: builder.mutation<{ status: string }, number>({
  query: (id) => ({ url: `/api/autopilot/trades/${id}`, method: 'DELETE' }),
  invalidatesTags: ['Autopilot'],
}),
```

#### Sidebar change (`Layout.tsx`)

Add between "AI Signals" and "Market Overview":

```ts
{ id: 'autopilot', path: '/autopilot', label: 'AI Authorized', Icon: BrainCircuit }
```

Use `BrainCircuit` from lucide-react (closest to design's `brain` icon).

#### Route change (`App.tsx`)

```tsx
<Route path="/autopilot" element={<AutopilotPage />} />
```

---

### Item 9: Notification CSS (`index.css`)

Add at the bottom of `index.css` — these classes are already referenced in `Navbar.tsx` but never defined:

```css
/* ── Notification popover ── */
.notif-pop { position: absolute; right: 0; top: 46px; width: 360px; ... }
.notif-head { display: flex; align-items: center; justify-content: space-between; ... }
.notif-list { max-height: 380px; overflow-y: auto; }
.notif-item { display: flex; align-items: flex-start; gap: 11px; padding: 12px 16px; ... }
.notif-item.unread { background: color-mix(in srgb, var(--accent) 5%, transparent); }
.notif-ic { width: 32px; height: 32px; border-radius: 9px; display: grid; ... }
.notif-foot { width: 100%; display: flex; align-items: center; ... }
```

(Full CSS values taken verbatim from design's `styles.css` lines 426–437.)

---

### Item 10: Signal card variants (`DashboardPage.tsx`)

The design's `SignalCard` has three render paths based on `variant` prop:

| Variant | Layout | When used |
|---------|--------|-----------|
| `rich` | Full card with sparkline, horizon, exp. return, conf bar | Default |
| `compact` | Single row: symbol · badge · price · conf bar | Condensed view |
| `bold` | Left border accent, large return %, conf bar | Bold/dramatic |

- Add `variant` prop to `SignalCard` in `DashboardPage.tsx`
- Store preference in `ThemeContext` (new `signalStyle` field alongside `density`)
- Wire the selector in **Settings → Appearance** tab

---

### Item 11: Navbar avatar menu links

Current code navigates to `/settings/risk` — that route redirects but doesn't open the right tab.  
Fix: navigate to `/settings` and pass `{ state: { tab: 'profile' | 'appearance' | 'notifications' | 'security' } }`.  
`SettingsPage.tsx` already reads `useLocation().state?.tab` on mount — just fix the `navigate()` calls.

---

### Item 12: Live market clock in Navbar

```tsx
const [clock, setClock] = useState('');
useEffect(() => {
  const tick = () => {
    const t = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' });
    setClock(t);
  };
  tick();
  const id = setInterval(tick, 1000);
  return () => clearInterval(id);
}, []);
```

Then in the market status pill: `NSE · {clock}`.

---

### Item 13: Accent colour + signal style in Settings → Appearance

**ThemeContext changes:**
- Add `accent: string` (default `#3B82F6`) and `signalStyle: 'rich'|'compact'|'bold'` to context state
- On `accent` change: `document.documentElement.style.setProperty('--accent', accent)` and also set `--accent-soft` to `accent + '22'`

**Settings → Appearance tab changes:**
- Add accent colour row: 4 circular swatches (`#3B82F6`, `#8B5CF6`, `#14B8A6`, `#F59E0B`)
- Add signal style row: segmented control `rich | compact | bold`

---

## 3. Implementation Order

```
Step 1  backend/database/schema_pg.py     — add 2 SQL blocks + register in init_timescale()
Step 2  backend/api/routes/autopilot.py   — new file, 5 endpoints
Step 3  backend/api/server.py             — include autopilot router
Step 4  frontend/src/index.css            — add notif-pop CSS block
Step 5  frontend/src/ThemeContext.tsx     — add accent + signalStyle
Step 6  frontend/src/components/Layout.tsx — add AI Authorized nav item
Step 7  frontend/src/App.tsx              — add /autopilot route + import
Step 8  frontend/src/services/tradeMindApiService.ts — add Autopilot types + 5 endpoints
Step 9  frontend/src/pages/AutopilotPage.tsx — new file
Step 10 frontend/src/components/Navbar.tsx — fix avatar menu links + live clock
Step 11 frontend/src/pages/DashboardPage.tsx — add compact + bold signal card variants
Step 12 frontend/src/pages/SettingsPage.tsx — add accent + signalStyle controls
```

---

## 4. File Change Summary

| File | Change type |
|------|-------------|
| `backend/database/schema_pg.py` | Edit — add 2 SQL tables + register |
| `backend/api/routes/autopilot.py` | **New file** |
| `backend/api/server.py` | Edit — import + include_router |
| `frontend/src/index.css` | Edit — append notif CSS block |
| `frontend/src/ThemeContext.tsx` | Edit — add accent + signalStyle |
| `frontend/src/components/Layout.tsx` | Edit — add AI Authorized nav item |
| `frontend/src/App.tsx` | Edit — add /autopilot route |
| `frontend/src/services/tradeMindApiService.ts` | Edit — add types + 5 endpoints |
| `frontend/src/pages/AutopilotPage.tsx` | **New file** |
| `frontend/src/components/Navbar.tsx` | Edit — fix links + live clock |
| `frontend/src/pages/DashboardPage.tsx` | Edit — add card variants |
| `frontend/src/pages/SettingsPage.tsx` | Edit — add accent + signal style |

**Total: 10 edits + 2 new files**
