import { createApi } from '@reduxjs/toolkit/query/react';
import tradeMindBaseQuery from './tradeMindBaseQuery';
import type {
  Stock, OpenPosition, Trade, GTTOrder,
  WatchlistItem, Notification,
} from '../types';

// ---------------------------------------------------------------------------
// Local types (not in global types/index.ts)
// ---------------------------------------------------------------------------

export interface User {
  id: number;
  username: string;
  display_name: string;
  virtual_balance: number;
  virtual_invested: number;
  total_pnl: number;
  win_count: number;
  loss_count: number;
  mode: string;
  email?: string;
  phone?: string;
  avatar_url?: string;
  totp_enabled?: boolean;
  default_account?: string;
  currency?: string;
}

export interface PortfolioSummary {
  user: { id: number; username: string; display_name: string };
  balance: number;          // available cash
  invested: number;         // total capital deployed
  total_value: number;      // balance + invested + unrealized
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  open_positions: number;
  wins: number;
  losses: number;
  win_rate: number;
  positions: import('../types').OpenPosition[];
}

export interface RiskSettings {
  id?: number;
  user_id?: number;
  max_daily_loss: number;
  max_daily_trades: number;
  max_position_pct: number;
  auto_stop_loss: boolean;
  auto_target: boolean;
  stop_loss_pct: number;
  target_pct: number;
  max_position_size: number;
  mode: 'PAPER' | 'LIVE';
  updated_at?: string;
}

export interface TradeSignal {
  symbol: string;
  signal: string;
  confidence: number;
  trade: {
    type: string;
    buy_price: number | null;
    target_price: number | null;
    stop_loss: number | null;
    risk_reward: number | null;
    expected_return_pct: number;
  };
  price: { current: number; atr_14: number; atr_pct: number };
  model: { name: string; horizon: string; accuracy: number; precision: number };
  sentiment: Record<string, number>;
  top_drivers: Array<{ feature: string; importance: number }>;
  generated_at: string;
}

export interface SignalsResponse {
  generated_at: string;
  total_models: number;
  total_signals: number;
  summary: { STRONG_BUY: number; BUY: number; HOLD: number; SELL: number; STRONG_SELL: number };
  actionable_trades: TradeSignal[];
  avoid_list: TradeSignal[];
  hold_list: TradeSignal[];
}

export interface StockQueryParams {
  page?: number;
  size?: number;
  sort?: string;
  order?: 'asc' | 'desc';
  search?: string;
  sector?: string;
  globalFilter?: string;
}

export interface ExecuteSignalParams {
  user_id: number;
  symbol: string;
  name?: string;
  investment_amount: number;
  buy_price: number;
  target_price: number;
  stop_loss: number;
  signal?: string;
  confidence?: number;
  horizon?: string;
  max_safe_qty?: number;
  mode?: 'PAPER' | 'LIVE';
}

export interface MarketOverviewResponse {
  indices:  Array<{ name: string; value: number; change: number; pct: number; spark: number[] }>;
  breadth:  { advances: number; declines: number; unchanged: number };
  fii_dii:  Array<{ day: string; fii: number; dii: number }>;
  vix:      number;
  gainers:  Array<{ symbol: string; name: string; price: number; change: number; signal: string; confidence: number }>;
  losers:   Array<{ symbol: string; name: string; price: number; change: number; signal: string; confidence: number }>;
  heatmap:  SectorPerformance[];
  sentiment_score: number | null;
  fear_greed:      string | null;
}

export interface SectorPerformance {
  sector:      string;
  change:      number;
  avg_conf:    number;
  stock_count: number;
  buy_count:   number;
  sell_count:  number;
  hold_count:  number;
  stocks:      Array<{ symbol: string; name: string; price: number; change: number; signal: string; confidence: number }>;
}

// ---------------------------------------------------------------------------
// Autopilot types
// ---------------------------------------------------------------------------

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
  updated_at: string;
}

export interface AutopilotStatus {
  enabled: boolean;
  capital: number;
  active: number;
  realized_pnl: number;
  projected_profit: number;
}

// ---------------------------------------------------------------------------
// API Service
// ---------------------------------------------------------------------------

const API_BASE_URL = import.meta.env.VITE_BASE_URL || 'http://localhost:8000';

export const tradeMindApiService = createApi({
  reducerPath: 'tradeMindApi',
  baseQuery: tradeMindBaseQuery(API_BASE_URL),
  tagTypes: [
    'User', 'Portfolio', 'Positions', 'Orders', 'Signals',
    'Market', 'Watchlist', 'Notifications', 'Settings', 'GTT', 'Sectors',
    'Autopilot', 'Brokers', 'Sessions', 'Preferences', 'NotifPreferences',
  ],
  endpoints: (builder) => ({

    // ── Auth ──────────────────────────────────────────────────────────────
    login: builder.mutation<
      { token?: string; user?: User; mfa_required?: boolean; mfa_token?: string; status: string },
      { username: string; password: string }
    >({
      query: (data) => ({ url: '/api/trading/login', method: 'POST', data }),
    }),
    loginMfa: builder.mutation<{ token: string; user: User }, { mfa_token: string; totp_code: string }>({
      query: (data) => ({ url: '/auth/login/mfa', method: 'POST', data }),
    }),
    register: builder.mutation<{ token: string; user: User }, { username: string; password: string; display_name?: string }>({
      query: (data) => ({ url: '/api/trading/register', method: 'POST', data }),
    }),
    getMe: builder.query<User, void>({
      query: () => ({ url: '/auth/me' }),
      providesTags: ['User'],
    }),
    updateMe: builder.mutation<User, { display_name?: string; email?: string; phone?: string }>({
      query: (data) => ({ url: '/auth/me', method: 'PATCH', data }),
      invalidatesTags: ['User'],
    }),
    changePassword: builder.mutation<{ status: string }, { current_password: string; new_password: string }>({
      query: (data) => ({ url: '/auth/password/change', method: 'POST', data }),
    }),
    requestPasswordReset: builder.mutation<{ status: string }, { email: string }>({
      query: (data) => ({ url: '/auth/password/reset-request', method: 'POST', data }),
    }),
    confirmPasswordReset: builder.mutation<{ status: string }, { email: string; otp: string; new_password: string }>({
      query: (data) => ({ url: '/auth/password/reset-confirm', method: 'POST', data }),
    }),
    getPreferences: builder.query<{ default_account?: string; currency?: string }, void>({
      query: () => ({ url: '/auth/preferences' }),
      providesTags: ['Preferences'],
    }),
    updatePreferences: builder.mutation<{ status: string }, { default_account?: string; currency?: string }>({
      query: (data) => ({ url: '/auth/preferences', method: 'PUT', data }),
      invalidatesTags: ['Preferences'],
    }),
    getNotifPreferences: builder.query<Record<string, boolean>, void>({
      query: () => ({ url: '/api/notifications/preferences' }),
      providesTags: ['NotifPreferences'],
    }),
    updateNotifPreferences: builder.mutation<{ status: string }, Record<string, boolean>>({
      query: (data) => ({ url: '/api/notifications/preferences', method: 'PUT', data }),
      invalidatesTags: ['NotifPreferences'],
    }),
    getBrokers: builder.query<{ data: Array<{ name: string; broker: string; connected: boolean; desc?: string }> }, void>({
      query: () => ({ url: '/api/brokers' }),
      providesTags: ['Brokers'],
    }),
    connectBrokerAngelOne: builder.mutation<{ status: string }, { client_id: string; password: string; totp: string }>({
      query: (data) => ({ url: '/api/brokers/angel-one/connect', method: 'POST', data }),
      invalidatesTags: ['Brokers'],
    }),
    disconnectBroker: builder.mutation<{ status: string }, string>({
      query: (broker) => ({ url: `/api/brokers/${broker}/disconnect`, method: 'DELETE' }),
      invalidatesTags: ['Brokers'],
    }),
    totpSetup: builder.mutation<{ qr_uri: string; secret: string }, void>({
      query: () => ({ url: '/auth/totp/setup', method: 'POST', data: {} }),
    }),
    totpConfirm: builder.mutation<{ status: string }, { code: string }>({
      query: (data) => ({ url: '/auth/totp/confirm', method: 'POST', data }),
      invalidatesTags: ['User'],
    }),
    totpDisable: builder.mutation<{ status: string }, { code: string }>({
      query: (data) => ({ url: '/auth/totp/disable', method: 'POST', data }),
      invalidatesTags: ['User'],
    }),
    getSessions: builder.query<{ data: Array<{ id: string; device: string; location?: string; last_active: string; current: boolean }> }, void>({
      query: () => ({ url: '/auth/sessions' }),
      providesTags: ['Sessions'],
    }),
    revokeSession: builder.mutation<{ status: string }, string>({
      query: (session_id) => ({ url: `/auth/sessions/${session_id}`, method: 'DELETE' }),
      invalidatesTags: ['Sessions'],
    }),
    revokeAllSessions: builder.mutation<{ status: string }, void>({
      query: () => ({ url: '/auth/sessions', method: 'DELETE' }),
      invalidatesTags: ['Sessions'],
    }),
    getMarketStatus: builder.query<{ open: boolean; next_open?: string; next_close?: string }, void>({
      query: () => ({ url: '/api/market/status' }),
      keepUnusedDataFor: 60,
      // pollingInterval set at call-site via refetchOnFocus / pollingInterval option
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
    addPosition: builder.mutation<{ status: string; data: OpenPosition }, { symbol: string; quantity: number; buy_price: number; account_type?: string; notes?: string }>({
      query: (data) => ({ url: '/api/portfolio/positions', method: 'POST', data }),
      invalidatesTags: ['Positions', 'Portfolio'],
    }),
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
    squareOffAll: builder.mutation<{ status: string }, number>({
      query: (userId) => ({ url: `/api/trading/square-off-all/${userId}`, method: 'POST', data: {} }),
      invalidatesTags: ['Positions', 'Portfolio'],
    }),
    executeSignal: builder.mutation<{ status: string }, ExecuteSignalParams>({
      query: (data) => ({ url: '/api/trading/execute-signal', method: 'POST', data }),
      invalidatesTags: ['Positions', 'Portfolio'],
    }),

    // ── Signals ───────────────────────────────────────────────────────────
    getLatestSignals: builder.query<{ data: SignalsResponse }, void>({
      query: () => ({ url: '/api/signals/latest' }),
      providesTags: ['Signals'],
      keepUnusedDataFor: 300,
    }),
    getActionableSignals: builder.query<{ data: TradeSignal[] }, void>({
      query: () => ({ url: '/api/signals/actionable' }),
      providesTags: ['Signals'],
      keepUnusedDataFor: 300,
    }),
    getSignalHistory: builder.query<{ data: SignalsResponse[] }, void>({
      query: () => ({ url: '/api/signals/history' }),
      keepUnusedDataFor: 0,
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
    getStockDetail: builder.query<{ data: Stock }, string>({
      query: (symbol) => ({ url: `/api/stocks/${encodeURIComponent(symbol)}` }),
      keepUnusedDataFor: 60,
    }),
    getStockHistory: builder.query<{ prices: number[]; labels: string[]; change_pct: number }, { symbol: string; range: string }>({
      query: ({ symbol, range }) => ({ url: `/api/stocks/${encodeURIComponent(symbol)}/history`, params: { range } }),
      keepUnusedDataFor: 60,
    }),
    getStockPrices: builder.query<{ data: number[] }, { symbol: string; days?: number }>({
      query: ({ symbol, days = 365 }) => ({ url: `/api/prices/${encodeURIComponent(symbol)}`, params: { days, interval: '1d' } }),
      keepUnusedDataFor: 120,
    }),
    getStockIndicators: builder.query<{ data: Record<string, number> }, string>({
      query: (symbol) => ({ url: `/api/indicators/${encodeURIComponent(symbol)}` }),
      keepUnusedDataFor: 300,
    }),

    // ── Market ────────────────────────────────────────────────────────────
    getMarketOverview: builder.query<MarketOverviewResponse, void>({
      query: () => ({ url: '/api/market/overview' }),
      providesTags: ['Market'],
      keepUnusedDataFor: 120,
    }),
    getMarketSectors: builder.query<SectorPerformance[], void>({
      query: () => ({ url: '/api/market/sectors' }),
      providesTags: ['Sectors'],
      keepUnusedDataFor: 300,
    }),

    // ── GTT ───────────────────────────────────────────────────────────────
    getGTTOrders: builder.query<{ data: GTTOrder[]; total: number }, number>({
      query: (userId) => ({ url: `/api/orders/gtt/${userId}` }),
      providesTags: ['GTT'],
    }),
    syncGTT: builder.mutation<{ status: string; count: number }, void>({
      query: () => ({ url: '/api/orders/gtt/sync', method: 'POST', data: {} }),
      invalidatesTags: ['GTT'],
    }),

    // ── User Analytics ────────────────────────────────────────────────────
    getUserAnalytics: builder.query<any, number>({
      query: (userId) => ({ url: `/api/trading/analytics/${userId}` }),
      keepUnusedDataFor: 300,
    }),
    getUserSignalVolume: builder.query<{ data: any[]; total: number }, number>({
      query: (userId) => ({ url: `/api/trading/analytics/${userId}/volume` }),
      keepUnusedDataFor: 120,
    }),

    // ── Watchlist ─────────────────────────────────────────────────────────
    // Note: /api/users/{userId}/watchlist avoids collision with /api/watchlist/{symbol} (stock detail)
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

    // ── User-wise News ────────────────────────────────────────────────────
    getUserWatchlistNews: builder.query<{ data: any[]; total: number }, { userId: number; limit?: number }>({
      query: ({ userId, limit = 50 }) => ({ url: `/api/news/watchlist/${userId}`, params: { limit } }),
      providesTags: ['Watchlist'],
      keepUnusedDataFor: 120,
    }),
    getUserWatchlistSentiment: builder.query<{ per_stock: any[]; overall: any }, number>({
      query: (userId) => ({ url: `/api/news/watchlist/${userId}/summary` }),
      providesTags: ['Watchlist'],
      keepUnusedDataFor: 300,
    }),
    getStockNews: builder.query<{ data: any[]; article_count: number; avg_sentiment: number }, { symbol: string; limit?: number }>({
      query: ({ symbol, limit = 20 }) => ({ url: `/api/news/stock/${encodeURIComponent(symbol)}`, params: { limit } }),
      keepUnusedDataFor: 120,
    }),
    getMarketNews: builder.query<{ data: any[]; total: number }, number | void>({
      query: (limit = 30) => ({ url: '/api/news/market', params: { limit } }),
      keepUnusedDataFor: 120,
    }),
    getUserSignalHistory: builder.query<{ data: any[]; total: number }, { userId: number; limit?: number }>({
      query: ({ userId, limit = 50 }) => ({ url: `/api/signals/history/${userId}`, params: { limit } }),
      keepUnusedDataFor: 60,
    }),

    // ── Autopilot ─────────────────────────────────────────────────────────
    getAutopilotStatus: builder.query<AutopilotStatus, number>({
      query: (userId) => ({ url: '/api/autopilot/status', params: { user_id: userId } }),
      providesTags: ['Autopilot'],
      keepUnusedDataFor: 30,
    }),
    toggleAutopilot: builder.mutation<{ enabled: boolean }, number>({
      query: (userId) => ({ url: '/api/autopilot/toggle', method: 'POST', data: { user_id: userId } }),
      invalidatesTags: ['Autopilot'],
    }),
    getAuthorizedTrades: builder.query<{ data: AuthorizedTrade[]; total: number }, { userId: number; status?: string }>({
      query: ({ userId, status }) => ({ url: '/api/autopilot/trades', params: { user_id: userId, ...(status && status !== 'All' ? { status } : {}) } }),
      providesTags: ['Autopilot'],
    }),
    authorizeTradeAuto: builder.mutation<{ status: string; data: AuthorizedTrade }, Partial<AuthorizedTrade> & { user_id: number }>({
      query: (data) => ({ url: '/api/autopilot/trades', method: 'POST', data }),
      invalidatesTags: ['Autopilot'],
    }),
    revokeAuthorizedTrade: builder.mutation<{ status: string }, number>({
      query: (id) => ({ url: `/api/autopilot/trades/${id}`, method: 'DELETE' }),
      invalidatesTags: ['Autopilot'],
    }),

    // ── Health ────────────────────────────────────────────────────────────
    getHealth: builder.query<{ status: string; market_open: boolean }, void>({
      query: () => ({ url: '/api/health' }),
      keepUnusedDataFor: 60,
    }),

    // ── Backtest / Model Performance ──────────────────────────────────────
    getBacktestSummary: builder.query<any, void>({
      query: () => ({ url: '/api/backtest/summary' }),
      keepUnusedDataFor: 300,
    }),
  }),
});

export const {
  // Auth
  useLoginMutation, useLoginMfaMutation, useRegisterMutation, useGetMeQuery,
  useUpdateMeMutation, useChangePasswordMutation,
  useRequestPasswordResetMutation, useConfirmPasswordResetMutation,
  useGetPreferencesQuery, useUpdatePreferencesMutation,
  useGetNotifPreferencesQuery, useUpdateNotifPreferencesMutation,
  useGetBrokersQuery, useConnectBrokerAngelOneMutation,
  useDisconnectBrokerMutation,
  useTotpSetupMutation, useTotpConfirmMutation, useTotpDisableMutation,
  useGetSessionsQuery, useRevokeSessionMutation, useRevokeAllSessionsMutation,
  useGetMarketStatusQuery,
  // Portfolio
  useGetPortfolioSummaryQuery, useGetTodayPnlQuery,
  // Positions & Orders
  useAddPositionMutation,
  useGetPositionsQuery, useGetOrdersQuery,
  useSquareOffMutation, useSquareOffAllMutation, useExecuteSignalMutation,
  // Signals
  useGetLatestSignalsQuery, useGetActionableSignalsQuery,
  useGetSignalHistoryQuery, useRefreshSignalsMutation,
  // Stocks
  useGetStocksQuery, useLazyGetStocksQuery,
  useGetStockDetailQuery, useLazyGetStockDetailQuery,
  useGetStockHistoryQuery,
  useGetStockPricesQuery, useGetStockIndicatorsQuery,
  // Market
  useGetMarketOverviewQuery, useGetMarketSectorsQuery,
  // GTT
  useGetGTTOrdersQuery, useSyncGTTMutation,
  // Analytics
  useGetUserAnalyticsQuery, useGetUserSignalVolumeQuery,
  // Watchlist
  useGetWatchlistQuery, useAddToWatchlistMutation,
  useRemoveFromWatchlistMutation, useUpdateWatchlistAlertsMutation,
  // Notifications
  useGetNotificationsQuery, useMarkNotificationsReadMutation, useDeleteNotificationMutation,
  // Settings
  useGetRiskSettingsQuery, useUpdateRiskSettingsMutation,
  // User-wise News
  useGetUserWatchlistNewsQuery, useGetUserWatchlistSentimentQuery,
  useGetStockNewsQuery, useGetMarketNewsQuery,
  useGetUserSignalHistoryQuery,
  // Autopilot
  useGetAutopilotStatusQuery, useToggleAutopilotMutation,
  useGetAuthorizedTradesQuery, useAuthorizeTradeAutoMutation,
  useRevokeAuthorizedTradeMutation,
  // Health
  useGetHealthQuery,
  // Backtest
  useGetBacktestSummaryQuery,
} = tradeMindApiService;
