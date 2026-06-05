# Day 6 — Implementation Plan

**Source:** New design bundle — `page-watchlist.jsx`, `page-settings.jsx`, updated `shell.jsx`  
**Scope:** 2 new pages + notifications system + full RTK Query migration + backend planning

---

## RTK Query Migration Strategy

All API calls will be migrated from the current `api.ts` (raw `fetch`) to RTK Query hooks,
following the **exact same pattern** used in the medical-main project at
`/Users/maheshmaheshwari/Documents/medical-main/medistock/frontend/src/api/`.

### Pattern reference

```
medical-main pattern               →  TradeMind equivalent
─────────────────────────────────────────────────────────
src/api/axiosBaseQuery.ts          →  src/services/tradeMindBaseQuery.ts (rewrite)
src/api/axiosInterceptor.ts        →  src/services/tradeMindInterceptor.ts (new)
src/api/apiSlice.ts                →  src/services/tradeMindApiService.ts (expand)
src/store.ts                       →  src/app/store.ts (already exists, keep)
Component: useGetDashboardStatsQuery →  Component: useGetPortfolioSummaryQuery
```

### Key differences from current setup

| Current (`api.ts`) | Target (RTK Query) |
|---|---|
| Raw `fetch` calls | Axios via `axiosBaseQuery` |
| Manual `localStorage.getItem('trademind_token')` auth | Axios interceptor handles auth header automatically |
| No caching — every render re-fetches | RTK Query cache with `keepUnusedDataFor` |
| No loading/error states built in | `isLoading`, `isFetching`, `isError` from hooks |
| No cache invalidation | `providesTags` / `invalidatesTags` |
| Pages call functions imperatively | Pages use `useGet*Query` / `use*Mutation` hooks |
| No optimistic updates | Mutations can do optimistic updates via `onQueryStarted` |

---

## Phase A — Infrastructure: RTK Query Foundation

### A1. Rewrite `src/services/tradeMindBaseQuery.ts`

Replace the current wrapper with a proper `axiosBaseQuery` matching the medical-main pattern:

```ts
// src/services/tradeMindBaseQuery.ts
import type { BaseQueryFn } from '@reduxjs/toolkit/query';
import axios from 'axios';
import type { AxiosError, AxiosRequestConfig } from 'axios';
import tradeMindInterceptor from './tradeMindInterceptor';

// Create a single shared axios instance with interceptors attached
const axiosInstance = tradeMindInterceptor(axios.create());

const tradeMindBaseQuery = (baseURL: string): BaseQueryFn<{
  url: string;
  method?: AxiosRequestConfig['method'];
  data?: AxiosRequestConfig['data'];
  params?: AxiosRequestConfig['params'];
  headers?: AxiosRequestConfig['headers'];
}, unknown, unknown> =>
  async ({ url, method = 'GET', data, params, headers }) => {
    try {
      const response = await axiosInstance({ url: baseURL + url, method, data, params, headers });
      return { data: response.data };
    } catch (axiosError) {
      const err = axiosError as AxiosError;
      return { error: { status: err?.response?.status, data: err?.response?.data || err?.message } };
    }
  };

export default tradeMindBaseQuery;
```

### A2. Create `src/services/tradeMindInterceptor.ts` (new)

Handles auth token injection and 401 → logout, exactly like `axiosInterceptor.ts` in medical-main:

```ts
// src/services/tradeMindInterceptor.ts
import axios, { AxiosError, AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

const TOKEN_KEY = 'trademind_token';

const onRequest = (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token && !config.headers['Authorization']) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  if (!(config.data instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json; charset=UTF-8';
  }
  return config;
};

const onErrorResponse = (error: AxiosError | Error): Promise<AxiosError> => {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    if (status === 401) {
      // Clear token and dispatch custom event so AuthContext can react
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem('trademind_user');
      window.dispatchEvent(new Event('trademind:unauthorized'));
    }
  }
  return Promise.reject(error);
};

const tradeMindInterceptor = (instance: AxiosInstance): AxiosInstance => {
  instance.interceptors.request.use(onRequest, onErrorResponse);
  instance.interceptors.response.use(res => res, onErrorResponse);
  return instance;
};

export default tradeMindInterceptor;
```

### A3. Expand `src/services/tradeMindApiService.ts` — ALL endpoints

Replace the current partial service with a complete `createApi` covering every endpoint:

```ts
export const tradeMindApiService = createApi({
  reducerPath: 'tradeMindApi',
  baseQuery: tradeMindBaseQuery(import.meta.env.VITE_BASE_URL || 'http://localhost:8000'),
  tagTypes: [
    'User', 'Portfolio', 'Positions', 'Orders', 'Signals',
    'Market', 'Watchlist', 'Notifications', 'Settings', 'GTT',
  ],
  endpoints: (builder) => ({
    // ── Auth ──────────────────────────────────────────────────────────────
    login: builder.mutation<{ token: string; user: User }, { username: string; password: string }>({
      query: (data) => ({ url: '/api/trading/login', method: 'POST', data }),
    }),
    register: builder.mutation<{ token: string; user: User }, { username: string; password: string; display_name?: string }>({
      query: (data) => ({ url: '/api/trading/register', method: 'POST', data }),
    }),
    getMe: builder.query<User, void>({
      query: () => ({ url: '/api/trading/me' }),
      providesTags: ['User'],
    }),

    // ── Portfolio / Dashboard ─────────────────────────────────────────────
    getPortfolioSummary: builder.query<PortfolioSummary, number>({
      query: (userId) => ({ url: `/api/trading/portfolio/${userId}` }),
      providesTags: ['Portfolio'],
    }),
    getTodayPnl: builder.query<{ today_pnl: number; today_pnl_pct: number }, number>({
      query: (userId) => ({ url: `/api/trading/pnl/today/${userId}` }),
      keepUnusedDataFor: 60,
    }),

    // ── Positions & Orders ────────────────────────────────────────────────
    getPositions: builder.query<{ data: OpenPosition[]; total: number }, { userId: number; size?: number }>({
      query: ({ userId, size = 100 }) => ({ url: `/api/trading/positions/${userId}`, params: { size } }),
      providesTags: ['Positions'],
    }),
    getOrders: builder.query<{ data: Trade[]; total: number }, { userId: number; size?: number }>({
      query: ({ userId, size = 50 }) => ({ url: `/api/trading/orders/${userId}`, params: { size } }),
      providesTags: ['Orders'],
    }),
    squareOff: builder.mutation<{ status: string; realized_pnl: number }, { userId: number; symbol: string }>({
      query: ({ userId, symbol }) => ({ url: `/api/trading/square-off/${userId}/${encodeURIComponent(symbol)}`, method: 'POST', data: {} }),
      invalidatesTags: ['Positions', 'Portfolio'],
    }),
    executeSignal: builder.mutation<any, ExecuteSignalParams>({
      query: (data) => ({ url: '/api/trading/execute-signal', method: 'POST', data }),
      invalidatesTags: ['Positions', 'Portfolio'],
    }),

    // ── Signals ───────────────────────────────────────────────────────────
    getLatestSignals: builder.query<SignalsResponse, void>({
      query: () => ({ url: '/api/signals/latest' }),
      providesTags: ['Signals'],
      keepUnusedDataFor: 300,
    }),
    getActionableSignals: builder.query<{ data: Stock[] }, void>({
      query: () => ({ url: '/api/signals/actionable' }),
      providesTags: ['Signals'],
      keepUnusedDataFor: 300,
    }),
    refreshSignals: builder.mutation<{ status: string; message: string }, void>({
      query: () => ({ url: '/api/signals/refresh', method: 'POST', data: {} }),
      invalidatesTags: ['Signals'],
    }),

    // ── Stocks ────────────────────────────────────────────────────────────
    getStocks: builder.query<{ data: Stock[]; total: number }, StockQueryParams>({
      query: (params) => ({ url: '/api/stocks', params }),
      keepUnusedDataFor: 120,
    }),
    getStockDetail: builder.query<{ data: StockDetail }, string>({
      query: (symbol) => ({ url: `/api/watchlist/${encodeURIComponent(symbol)}` }),
      keepUnusedDataFor: 60,
    }),

    // ── Market ────────────────────────────────────────────────────────────
    getMarketOverview: builder.query<MarketOverviewResponse, void>({
      query: () => ({ url: '/api/market/overview' }),
      providesTags: ['Market'],
      keepUnusedDataFor: 120,
    }),

    // ── GTT ───────────────────────────────────────────────────────────────
    getGTTOrders: builder.query<{ data: GTTOrder[] }, void>({
      query: () => ({ url: '/api/orders/gtt' }),
      providesTags: ['GTT'],
    }),
    syncGTT: builder.mutation<{ status: string; count: number }, void>({
      query: () => ({ url: '/api/orders/gtt/sync', method: 'POST', data: {} }),
      invalidatesTags: ['GTT'],
    }),

    // ── Watchlist ─────────────────────────────────────────────────────────
    getWatchlist: builder.query<{ data: WatchlistItem[]; total: number }, number>({
      query: (userId) => ({ url: `/api/users/${userId}/watchlist` }),
      providesTags: ['Watchlist'],
    }),
    addToWatchlist: builder.mutation<{ status: string; data: WatchlistItem }, { userId: number; symbol: string }>({
      query: ({ userId, symbol }) => ({ url: `/api/users/${userId}/watchlist/${encodeURIComponent(symbol)}`, method: 'POST', data: {} }),
      invalidatesTags: ['Watchlist'],
    }),
    removeFromWatchlist: builder.mutation<{ status: string }, { userId: number; symbol: string }>({
      query: ({ userId, symbol }) => ({ url: `/api/users/${userId}/watchlist/${encodeURIComponent(symbol)}`, method: 'DELETE' }),
      invalidatesTags: ['Watchlist'],
    }),
    updateWatchlistAlerts: builder.mutation<{ status: string }, { userId: number; symbol: string; alert_above?: number; alert_below?: number }>({
      query: ({ userId, symbol, ...data }) => ({ url: `/api/users/${userId}/watchlist/${encodeURIComponent(symbol)}/alerts`, method: 'PUT', data }),
      invalidatesTags: ['Watchlist'],
    }),

    // ── Notifications ─────────────────────────────────────────────────────
    getNotifications: builder.query<{ data: Notification[]; unread: number }, void>({
      query: () => ({ url: '/api/notifications' }),
      providesTags: ['Notifications'],
      keepUnusedDataFor: 30,
    }),
    markNotificationsRead: builder.mutation<{ status: string }, void>({
      query: () => ({ url: '/api/notifications/mark-read', method: 'POST', data: {} }),
      invalidatesTags: ['Notifications'],
    }),
    deleteNotification: builder.mutation<{ status: string }, number>({
      query: (id) => ({ url: `/api/notifications/${id}`, method: 'DELETE' }),
      invalidatesTags: ['Notifications'],
    }),

    // ── Risk Settings ─────────────────────────────────────────────────────
    getRiskSettings: builder.query<RiskSettings, number>({
      query: (userId) => ({ url: `/api/trading/risk-settings/${userId}` }),
      providesTags: ['Settings'],
    }),
    updateRiskSettings: builder.mutation<{ status: string }, { userId: number; settings: Partial<RiskSettings> }>({
      query: ({ userId, settings }) => ({ url: `/api/trading/risk-settings/${userId}`, method: 'PUT', data: settings }),
      invalidatesTags: ['Settings'],
    }),

    // ── Health ────────────────────────────────────────────────────────────
    getHealth: builder.query<{ status: string; market_open: boolean }, void>({
      query: () => ({ url: '/api/health' }),
      keepUnusedDataFor: 60,
    }),
  }),
});

// Export all auto-generated hooks
export const {
  useLoginMutation, useRegisterMutation, useGetMeQuery,
  useGetPortfolioSummaryQuery, useGetTodayPnlQuery,
  useGetPositionsQuery, useGetOrdersQuery,
  useSquareOffMutation, useExecuteSignalMutation,
  useGetLatestSignalsQuery, useGetActionableSignalsQuery, useRefreshSignalsMutation,
  useGetStocksQuery, useLazyGetStocksQuery, useGetStockDetailQuery, useLazyGetStockDetailQuery,
  useGetMarketOverviewQuery,
  useGetGTTOrdersQuery, useSyncGTTMutation,
  useGetWatchlistQuery, useAddToWatchlistMutation, useRemoveFromWatchlistMutation, useUpdateWatchlistAlertsMutation,
  useGetNotificationsQuery, useMarkNotificationsReadMutation, useDeleteNotificationMutation,
  useGetRiskSettingsQuery, useUpdateRiskSettingsMutation,
  useGetHealthQuery,
} = tradeMindApiService;
```

### A4. `src/app/store.ts` — update reducerPath

Update `reducerPath` from `'tradeMindApiService'` to `'tradeMindApi'` to match the new service.

### A5. Delete `src/api.ts`

Once migration is complete, `api.ts` is removed entirely. All pages import hooks from
`'../services/tradeMindApiService'` — same as how medical-main imports from `'../../api/apiSlice'`.

### A6. Update `src/AuthContext.tsx`

- Listen for the `trademind:unauthorized` custom event dispatched by the interceptor
- Call `logout()` when received (clears local state)
- Remove manual token management from within auth functions — the interceptor handles it

---

## Phase B — Page Migration: Replace `api.ts` calls with RTK Query hooks

Each page currently calls raw `api.ts` functions inside `useEffect`. Replace with RTK Query hooks.

### Migration example (DashboardPage)

**Before (current):**
```ts
const [portData, setPortData] = useState(null);
const [loading, setLoading] = useState(true);

useEffect(() => {
  getPortfolioSummary(user.id).then(d => setPortData(d)).finally(() => setLoading(false));
}, [user]);
```

**After (RTK Query):**
```ts
const { data: portData, isLoading } = useGetPortfolioSummaryQuery(user!.id);
const { data: actionableData } = useGetActionableSignalsQuery();
const { data: marketData } = useGetMarketOverviewQuery();
const [refreshSignals, { isLoading: refreshing }] = useRefreshSignalsMutation();
```

### All pages to migrate

| Page | Current api.ts calls | New RTK Query hooks |
|---|---|---|
| `DashboardPage` | `getPortfolioSummary`, `getTodayPnl`, `getActionableSignals`, `getOrders`, `getPositions`, `getMarketOverview` | `useGetPortfolioSummaryQuery`, `useGetTodayPnlQuery`, `useGetActionableSignalsQuery`, `useGetOrdersQuery`, `useGetPositionsQuery`, `useGetMarketOverviewQuery` |
| `AISignalsPage` | `getAllStocks` | `useGetStocksQuery` (with `skip` for lazy), `useLazyGetStocksQuery` for search |
| `MarketPage` | `getMarketOverview` | `useGetMarketOverviewQuery` |
| `PortfolioPage` | `getPortfolioSummary` | `useGetPortfolioSummaryQuery` |
| `TradesPage` | `getPositions`, `getOrders`, `squareOff`, `getGTTOrders`, `syncGTTOrders` | `useGetPositionsQuery`, `useGetOrdersQuery`, `useSquareOffMutation`, `useGetGTTOrdersQuery`, `useSyncGTTMutation` |
| `SettingsPage` | `getRiskSettings`, `updateRiskSettings` | `useGetRiskSettingsQuery`, `useUpdateRiskSettingsMutation` |
| `WatchlistPage` *(new)* | — | `useGetWatchlistQuery`, `useAddToWatchlistMutation`, `useRemoveFromWatchlistMutation` |
| `StockDrawer` | `getStockDetail`, `getPositions`, `executeSignal` | `useLazyGetStockDetailQuery`, `useGetPositionsQuery`, `useExecuteSignalMutation` |
| `AddPositionModal` | `getAllStocks` | `useLazyGetStocksQuery` |
| `Navbar` | — | `useGetNotificationsQuery`, `useMarkNotificationsReadMutation` |
| `AuthPage` | `registerUser`, `loginUser` | `useLoginMutation`, `useRegisterMutation` |

### Loading/error state pattern

```ts
// Query
const { data, isLoading, isFetching, isError, refetch } = useGetPortfolioSummaryQuery(userId);

// Mutation
const [squareOff, { isLoading: closingTrade }] = useSquareOffMutation();
await squareOff({ userId, symbol }).unwrap(); // .unwrap() throws on error

// Conditional (skip when no user)
const { data } = useGetPortfolioSummaryQuery(user?.id ?? 0, { skip: !user });
```

---

## Phase C — New Pages

### C1. `src/pages/WatchlistPage.tsx` (new)

```
Data source: useGetWatchlistQuery(user.id)
Mutations:   useAddToWatchlistMutation, useRemoveFromWatchlistMutation
Layout:
  Header: "Watchlist" + "N stocks · X BUY · Y SELL" + Grid/Table toggle + "Add Stocks" btn
  4 stat cards: Watchlist Items | Avg Confidence | Price Alerts | In Profit Zone
  Grid view: watch-card per stock (SymbolCell, price, sparkline, SignalBadge+horizon pill, expReturn, alert thresholds, ×)
  Table view: sortable (Stock, LTP, Change, Signal, Confidence, Alerts, Remove)
  Empty state: bookmark icon + "Browse Signals" CTA
  Card CSS classes from design: .watch-card, .watch-x
```

### C2. `src/pages/SettingsPage.tsx` (full redesign of RiskSettingsPage)

```
Data source: useGetRiskSettingsQuery(user.id)
Mutations:   useUpdateRiskSettingsMutation
Layout: 220px sticky sidebar (.settings-nav) + content panel
Tabs (matching design exactly):
  profile       — Avatar, form (Full name, Email, Phone, PAN), Trading Preferences card
  brokers       — Angel One / Zerodha / Upstox / Groww connect cards (.broker-row)
  notifications — Toggle rows (.pref-row) + delivery channel chips (.chan-btn)
  appearance    — Theme swatches (.theme-swatch) + accent dots + density seg + signal-style seg
  security      — Password form + 2FA card + Active Sessions
```

---

## Phase D — Navbar Notifications Panel

### D1. Update `src/components/Navbar.tsx`

```
Data source: useGetNotificationsQuery() — polls/refetches every 60s (pollingInterval: 60000)
Mutations:   useMarkNotificationsReadMutation, useDeleteNotificationMutation
New elements:
  Bell button: shows badge dot with unread count from query data
  Notification panel (.notif-pop): 360px dropdown (same CSS as design)
    Header: "Notifications" + "N new" pill + "Mark all read" button
    List (.notif-list): up to 6 items — icon, title (nowrap), msg, time, unread tint
    Footer (.notif-foot): "Notification settings" → navigate('/settings', { state: { tab: 'notifications' } })
  Avatar dropdown:
    Profile → /settings?tab=profile
    Preferences → /settings?tab=appearance
    Notifications → /settings?tab=notifications
    Security → /settings?tab=security
```

---

## Phase E — Route + Layout Updates

### E1. `src/components/Layout.tsx`

- Watchlist nav: `path: '/watchlist'` (currently points to `/portfolio`)
- Settings nav: `path: '/settings'` (currently `/settings/risk`)

### E2. `src/App.tsx`

```ts
import WatchlistPage from './pages/WatchlistPage';
import SettingsPage  from './pages/SettingsPage';
// Remove: RiskSettingsPage

<Route path="/watchlist"    element={<WatchlistPage />} />
<Route path="/settings"     element={<SettingsPage />} />
<Route path="/settings/risk" element={<Navigate to="/settings" replace />} />  // legacy redirect
```

### E3. Delete `src/pages/RiskSettingsPage.tsx`

Replaced entirely by `SettingsPage.tsx`.

---

## Phase F — Index.css additions (design CSS)

Add new CSS classes from the design's `styles.css` (lines 390–442) to `index.css`:

```css
/* Watchlist */
.watch-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: calc(15px * var(--u)); cursor: pointer; transition: border .15s, transform .1s; }
.watch-card:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.watch-x { width: 28px; height: 28px; border-radius: 8px; border: 1px solid transparent; background: transparent; color: var(--text-3); display: grid; place-items: center; cursor: pointer; }
.watch-x:hover { background: var(--red-soft); color: var(--red); }

/* Settings layout */
.settings-layout { display: grid; grid-template-columns: 220px 1fr; gap: calc(20px * var(--u)); align-items: start; }
.settings-nav { position: sticky; top: 0; display: flex; flex-direction: column; gap: 3px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; }
.settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }
.broker-row, .pref-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.broker-row { padding: 13px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); }
.broker-logo { width: 42px; height: 42px; border-radius: 11px; display: grid; place-items: center; flex-shrink: 0; }
.pref-row { padding: 14px 0; }
.tgl { width: 44px; height: 25px; border-radius: 999px; background: var(--surface-3); border: 1px solid var(--border); padding: 2px; cursor: pointer; transition: background .18s; flex-shrink: 0; }
.tgl.on { background: var(--accent); border-color: var(--accent); }
.tgl-knob { display: block; width: 19px; height: 19px; border-radius: 50%; background: #fff; transition: transform .18s; }
.tgl.on .tgl-knob { transform: translateX(19px); }
.chan-btn { display: inline-flex; align-items: center; gap: 6px; height: 36px; padding: 0 15px; border-radius: 10px; border: 1px solid var(--border); background: var(--surface-2); color: var(--text-2); font-size: 13px; font-weight: 600; cursor: pointer; transition: all .15s; }
.chan-btn.on { background: var(--accent-soft); color: var(--accent-2); }
.theme-swatch { display: flex; align-items: center; gap: 10px; padding: 10px 14px 10px 10px; border-radius: var(--radius-sm); border: 2px solid var(--border); background: var(--surface-2); cursor: pointer; }
.theme-swatch.on { border-color: var(--accent); }
.accent-dot { width: 38px; height: 38px; border-radius: 11px; border: 2px solid transparent; cursor: pointer; display: grid; place-items: center; color: #fff; }
.accent-dot.on { box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px currentColor; }

/* Notifications popover */
.notif-pop { position: absolute; right: 0; top: 46px; width: 360px; max-width: 90vw; background: var(--surface); border: 1px solid var(--border-strong); border-radius: 15px; box-shadow: var(--shadow-lg); z-index: 10; overflow: hidden; }
.notif-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--border); }
.notif-list { max-height: 380px; overflow-y: auto; }
.notif-item { display: flex; align-items: flex-start; gap: 11px; padding: 12px 16px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background .12s; }
.notif-item:hover { background: var(--surface-2); }
.notif-item.unread { background: color-mix(in srgb, var(--accent) 5%, transparent); }
.notif-ic { width: 32px; height: 32px; border-radius: 9px; display: grid; place-items: center; flex-shrink: 0; }
.notif-foot { width: 100%; display: flex; align-items: center; justify-content: center; gap: 8px; padding: 13px; border: none; background: var(--surface-2); color: var(--text-2); font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }
.notif-foot:hover { background: var(--surface-hover); color: var(--text); }

@media (max-width: 980px) { .settings-layout { grid-template-columns: 1fr; } .settings-nav { flex-direction: row; flex-wrap: wrap; position: relative; } .settings-grid { grid-template-columns: 1fr; } }
```

---

## Backend Implementation Plan

### Current State Audit

**Already implemented** (no work needed):

| Endpoint | File | Status |
|---|---|---|
| `POST /api/trading/register` | `routes/trading.py:97` | ✅ Done |
| `POST /api/trading/login` | `routes/trading.py:112` | ✅ Done |
| `GET /api/trading/me` | `routes/trading.py:125` | ✅ Done |
| `GET /api/trading/portfolio/{user_id}` | `routes/trading.py:331` | ✅ Done |
| `GET /api/trading/pnl/today/{user_id}` | `routes/trading.py:360` | ✅ Done |
| `GET /api/trading/positions/{user_id}` | `routes/trading.py:223` | ✅ Done |
| `GET /api/trading/orders/{user_id}` | `routes/trading.py:266` | ✅ Done |
| `POST /api/trading/square-off/{user_id}/{symbol}` | `routes/trading.py:309` | ✅ Done |
| `POST /api/trading/execute-signal` | `routes/trading.py:168` | ✅ Done |
| `GET /api/trading/risk-settings/{user_id}` | `routes/trading.py:343` | ✅ Done |
| `PUT /api/trading/risk-settings/{user_id}` | `routes/trading.py:350` | ✅ Done |
| `GET /api/signals/latest` | `routes/signals.py:76` | ✅ Done |
| `GET /api/signals/actionable` | `routes/signals.py:124` | ✅ Done |
| `GET /api/market/overview` | `server.py:179` | ✅ Done |
| `GET /api/health` | `server.py:148` | ✅ Done |
| `GET /api/watchlist/{symbol}` | `server.py:205` | ✅ Done (stock detail) |
| `GET /api/heatmap/sectors` | `server.py:245` | ✅ Done |

**Missing** — must be built for Day 6:

| Endpoint | Purpose | Priority |
|---|---|---|
| `POST /api/signals/refresh` | Trigger async signal regeneration | High |
| `GET /api/users/{userId}/watchlist` | User's saved watchlist | High |
| `POST /api/users/{userId}/watchlist/{symbol}` | Add stock | High |
| `DELETE /api/users/{userId}/watchlist/{symbol}` | Remove stock | High |
| `PUT /api/users/{userId}/watchlist/{symbol}/alerts` | Price alert thresholds | Medium |
| `GET /api/notifications` | List user notifications | High |
| `POST /api/notifications/mark-read` | Mark all read | High |
| `DELETE /api/notifications/{id}` | Delete single | Medium |
| `GET /api/orders/gtt` | List GTT orders from `orders` table | Medium |
| `POST /api/orders/gtt/sync` | Trigger Angel One GTT sync | Medium |

---

### Backend Change 1 — `POST /api/signals/refresh`

**File:** `backend/api/routes/signals.py`

Add after the existing `GET /signals/avoid` handler:

```python
@router.post("/refresh")
async def refresh_signals(background_tasks: BackgroundTasks):
    """Kick off async regeneration of all trade signals from stored ML models."""
    from database.db import get_all_symbols

    def _run_refresh():
        import subprocess, sys, os
        script = os.path.join(os.path.dirname(__file__), '../../generate_trades.py')
        subprocess.run([sys.executable, script], check=False)

    background_tasks.add_task(_run_refresh)
    return {"status": "ok", "message": "Signal refresh started in background"}
```

Add `BackgroundTasks` import at top of file: `from fastapi import APIRouter, Query, BackgroundTasks`.

**Register in `server.py`:** No change needed — signals router is already registered.

---

### Backend Change 2 — Watchlist Table + Routes

#### 2a. Schema (`backend/database/schema_pg.py`)

Add after `SQL_POSITIONS` (around line 192):

```sql
SQL_WATCHLIST = """
CREATE TABLE IF NOT EXISTS watchlist (
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol      TEXT   NOT NULL,
    alert_above DOUBLE PRECISION,
    alert_below DOUBLE PRECISION,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, symbol)
);
"""
SQL_IDX_WATCHLIST = "CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id);"
```

Add both to the `SCHEMA_STATEMENTS` list in `init_timescale()`.

**Note:** The `users` table (not `trading_users`) is the correct FK target — confirmed from `schema_pg.py:19`.

#### 2b. DB helpers (`backend/database/db.py`)

Add at end of file:

```python
def get_watchlist(user_id: int) -> List[Dict]:
    conn = get_connection()
    rows = _query_rows(conn, "SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
    conn.close()
    return rows

def add_to_watchlist(user_id: int, symbol: str) -> bool:
    conn = get_connection()
    try:
        _execute(conn, "INSERT INTO watchlist (user_id, symbol) VALUES (?, ?) ON CONFLICT DO NOTHING", (user_id, symbol))
        conn.commit()
        return True
    finally:
        conn.close()

def remove_from_watchlist(user_id: int, symbol: str) -> bool:
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        conn.commit()
        return True
    finally:
        conn.close()

def update_watchlist_alerts(user_id: int, symbol: str, alert_above: float = None, alert_below: float = None) -> bool:
    conn = get_connection()
    try:
        _execute(conn,
            "UPDATE watchlist SET alert_above = ?, alert_below = ? WHERE user_id = ? AND symbol = ?",
            (alert_above, alert_below, user_id, symbol))
        conn.commit()
        return True
    finally:
        conn.close()
```

`_query_rows` is an internal helper identical to `_rows_to_dicts` — use the existing pattern from `db.py`.

#### 2c. New route file (`backend/api/routes/watchlist.py`)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from database.db import get_watchlist, add_to_watchlist, remove_from_watchlist, update_watchlist_alerts

router = APIRouter(prefix="/api/users", tags=["Watchlist"])

class AlertRequest(BaseModel):
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None

@router.get("/{user_id}/watchlist")
async def list_watchlist(user_id: int):
    items = get_watchlist(user_id)
    return {"data": items, "total": len(items)}

@router.post("/{user_id}/watchlist/{symbol}", status_code=201)
async def add_watchlist(user_id: int, symbol: str):
    add_to_watchlist(user_id, symbol)
    return {"status": "ok", "data": {"user_id": user_id, "symbol": symbol}}

@router.delete("/{user_id}/watchlist/{symbol}")
async def remove_watchlist(user_id: int, symbol: str):
    remove_from_watchlist(user_id, symbol)
    return {"status": "ok"}

@router.put("/{user_id}/watchlist/{symbol}/alerts")
async def set_alerts(user_id: int, symbol: str, req: AlertRequest):
    update_watchlist_alerts(user_id, symbol, req.alert_above, req.alert_below)
    return {"status": "ok"}
```

#### 2d. Register in `server.py`

```python
from api.routes.watchlist import router as watchlist_router
app.include_router(watchlist_router)
```

**URL design note:** Using `/api/users/{userId}/watchlist` instead of `/api/watchlist/{userId}` avoids collision with the existing `/api/watchlist/{symbol}` stock-detail endpoint — FastAPI would route them identically since both path params are strings.

---

### Backend Change 3 — Notifications Table + Routes

#### 3a. Schema (`backend/database/schema_pg.py`)

Add after `SQL_WATCHLIST`:

```sql
SQL_NOTIFICATIONS = """
CREATE TABLE IF NOT EXISTS notifications (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type       TEXT NOT NULL CHECK (type IN ('trade','signal','price','news','system')),
    title      TEXT NOT NULL,
    message    TEXT,
    icon       TEXT,
    color      TEXT,
    is_read    BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""
SQL_IDX_NOTIF = "CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;"
```

Add both to `SCHEMA_STATEMENTS`.

#### 3b. DB helpers (`backend/database/db.py`)

```python
def get_notifications(user_id: int, limit: int = 50) -> List[Dict]:
    conn = get_connection()
    rows = _query_rows(conn,
        "SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit))
    unread = sum(1 for r in rows if not r.get("is_read"))
    conn.close()
    return rows, unread

def mark_notifications_read(user_id: int) -> None:
    conn = get_connection()
    try:
        _execute(conn, "UPDATE notifications SET is_read = TRUE WHERE user_id = ? AND is_read = FALSE", (user_id,))
        conn.commit()
    finally:
        conn.close()

def delete_notification(notif_id: int, user_id: int) -> bool:
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM notifications WHERE id = ? AND user_id = ?", (notif_id, user_id))
        conn.commit()
        return True
    finally:
        conn.close()

def insert_notification(user_id: int, type: str, title: str, message: str = None, icon: str = None, color: str = None) -> None:
    conn = get_connection()
    try:
        _execute(conn,
            "INSERT INTO notifications (user_id, type, title, message, icon, color) VALUES (?,?,?,?,?,?)",
            (user_id, type, title, message, icon, color))
        conn.commit()
    finally:
        conn.close()
```

#### 3c. New route file (`backend/api/routes/notifications.py`)

```python
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional
from database.db import get_notifications, mark_notifications_read, delete_notification

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])

async def _get_user_id(authorization: Optional[str] = Header(None)) -> int:
    """Extract user_id from JWT — reuse the same logic as trading.py:get_current_user."""
    from api.routes.trading import get_current_user
    user = await get_current_user(authorization)
    return user["id"]

@router.get("")
async def list_notifications(user_id: int = Depends(_get_user_id)):
    rows, unread = get_notifications(user_id)
    return {"data": rows, "unread": unread}

@router.post("/mark-read")
async def mark_read(user_id: int = Depends(_get_user_id)):
    mark_notifications_read(user_id)
    return {"status": "ok"}

@router.delete("/{notif_id}")
async def remove_notification(notif_id: int, user_id: int = Depends(_get_user_id)):
    delete_notification(notif_id, user_id)
    return {"status": "ok"}
```

#### 3d. Register in `server.py`

```python
from api.routes.notifications import router as notifications_router
app.include_router(notifications_router)
```

---

### Backend Change 4 — GTT Routes

GTT data already lives in the `orders` table (`gtt_rule_id`, `gtt_status` columns). A scheduler job (`jobs.py:229`) already calls `sync_gtt_statuses()` from `trading/gtt_manager.py`.

All that's needed is two HTTP endpoints to expose this.

**File:** `backend/api/routes/trades.py` — add at bottom:

```python
@router.get("/orders/gtt", tags=["GTT"])
async def get_gtt_orders():
    """Return all orders that have a gtt_rule_id (Angel One GTT orders)."""
    from database.db import get_connection, _rows_to_dicts
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE gtt_rule_id IS NOT NULL ORDER BY created_at DESC")
    rows = _rows_to_dicts(cur)
    conn.close()
    return {"data": rows, "total": len(rows)}

@router.post("/orders/gtt/sync", tags=["GTT"])
async def sync_gtt(background_tasks: BackgroundTasks):
    """Manually trigger an Angel One GTT status sync."""
    from scheduler.jobs import sync_gtt_status_job
    background_tasks.add_task(sync_gtt_status_job)
    return {"status": "ok", "message": "GTT sync started"}
```

Trades router is already registered in `server.py:78` — no additional registration needed.

---

### Backend Change 5 — Scheduler: Notification Hooks

**File:** `backend/scheduler/jobs.py`

#### 5a. Signal-change notifier (add to EOD job)

After the nightly signal refresh completes, compare new vs previous signals for all users with watchlist entries and fire notifications:

```python
def notify_signal_changes_job():
    """After EOD refresh, create notifications for watchlist signal changes."""
    try:
        from database.db import get_connection, _rows_to_dicts, insert_notification
        conn = get_connection()
        # Get all watchlist entries with their current and previous signals
        cur = conn.cursor()
        cur.execute("""
            SELECT w.user_id, w.symbol,
                   s.signal AS new_signal, s.confidence,
                   LAG(s.signal) OVER (PARTITION BY s.symbol ORDER BY s.generated_at) AS prev_signal
            FROM watchlist w
            JOIN trade_signals s ON s.symbol = w.symbol
            WHERE s.generated_at >= NOW() - INTERVAL '2 days'
        """)
        rows = _rows_to_dicts(cur)
        conn.close()
        for row in rows:
            if row["new_signal"] != row["prev_signal"] and row["prev_signal"] is not None:
                insert_notification(
                    user_id=row["user_id"],
                    type="signal",
                    title=f"{row['symbol']} signal changed",
                    message=f"{row['prev_signal']} → {row['new_signal']} ({row['confidence']:.0%} confidence)",
                    icon="TrendingUp",
                    color="#3B82F6"
                )
    except Exception as e:
        logger.error(f"Signal change notifier failed: {e}")
```

Wire into the existing EOD job at `jobs.py` — call `notify_signal_changes_job()` at the end of `eod_update_job()`.

#### 5b. Price alert checker (add to hourly job)

```python
def price_alert_job():
    """Check watchlist price alerts and fire notifications when thresholds are crossed."""
    try:
        from database.db import get_connection, _rows_to_dicts, insert_notification, get_latest_indicators
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, symbol, alert_above, alert_below FROM watchlist WHERE alert_above IS NOT NULL OR alert_below IS NOT NULL")
        alerts = _rows_to_dicts(cur)
        conn.close()
        for alert in alerts:
            ind = get_latest_indicators(alert["symbol"])
            if not ind:
                continue
            price = ind.get("close") or ind.get("ltp")
            if alert["alert_above"] and price and price >= alert["alert_above"]:
                insert_notification(
                    user_id=alert["user_id"], type="price",
                    title=f"{alert['symbol']} above ₹{alert['alert_above']:,.2f}",
                    message=f"Current price ₹{price:,.2f}", icon="ArrowUp", color="#10B981"
                )
            elif alert["alert_below"] and price and price <= alert["alert_below"]:
                insert_notification(
                    user_id=alert["user_id"], type="price",
                    title=f"{alert['symbol']} below ₹{alert['alert_below']:,.2f}",
                    message=f"Current price ₹{price:,.2f}", icon="ArrowDown", color="#EF4444"
                )
    except Exception as e:
        logger.error(f"Price alert job failed: {e}")
```

Register in `init_scheduler()`:

```python
scheduler.add_job(
    price_alert_job,
    CronTrigger(hour="9-15", minute="0", day_of_week="mon-fri", timezone="Asia/Kolkata"),
    id="price_alerts", name="Price Alert Checker", misfire_grace_time=300, replace_existing=True
)
```

---

### Schema Migration

After adding new SQL constants in `schema_pg.py`, run the idempotent init to create tables:

```bash
cd backend && source venv/bin/activate
python -c "from database.db import init_database; init_database()"
```

Both `CREATE TABLE IF NOT EXISTS` statements are safe to run against an existing DB.

---

### New File Summary

| File | Action | Purpose |
|---|---|---|
| `backend/api/routes/watchlist.py` | Create | User watchlist CRUD |
| `backend/api/routes/notifications.py` | Create | Notification CRUD |
| `backend/database/schema_pg.py` | Edit | Add `watchlist` + `notifications` tables |
| `backend/database/db.py` | Edit | Add 7 helper functions |
| `backend/api/routes/signals.py` | Edit | Add `POST /signals/refresh` |
| `backend/api/routes/trades.py` | Edit | Add `GET/POST /orders/gtt*` |
| `backend/api/server.py` | Edit | Register 2 new routers |
| `backend/scheduler/jobs.py` | Edit | Add 2 notification jobs |

---

## Execution Order

```
Day 6
├── Backend (do first — unblocks frontend wiring)
│   ├── B1  Schema: add watchlist + notifications tables to schema_pg.py
│   ├── B2  DB: add helper functions to db.py
│   ├── B3  New routes: watchlist.py + notifications.py
│   ├── B4  Edit: signals.py (refresh) + trades.py (GTT) + server.py (register)
│   ├── B5  Run init_database() to create tables
│   └── B6  Scheduler: add notification jobs to jobs.py
│
├── Phase A  — RTK Query infrastructure (baseQuery, interceptor, full service, store update)
├── Phase B  — Migrate all pages from api.ts → hooks
├── Phase C  — WatchlistPage + SettingsPage (new/redesigned)
├── Phase D  — Navbar notifications panel
├── Phase E  — Routes + layout + delete RiskSettingsPage
└── Phase F  — index.css additions
```
