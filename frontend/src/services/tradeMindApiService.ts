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
}

export interface PortfolioSummary {
  id: number;
  total_invested: number;
  current_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  open_positions: number;
  win_rate: number;
  virtual_balance: number;
}

export interface RiskSettings {
  max_position_size: number;
  max_portfolio_risk: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  max_open_positions: number;
  mode: 'PAPER' | 'LIVE';
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
// API Service
// ---------------------------------------------------------------------------

const API_BASE_URL = import.meta.env.VITE_BASE_URL || 'http://localhost:8000';

export const tradeMindApiService = createApi({
  reducerPath: 'tradeMindApi',
  baseQuery: tradeMindBaseQuery(API_BASE_URL),
  tagTypes: [
    'User', 'Portfolio', 'Positions', 'Orders', 'Signals',
    'Market', 'Watchlist', 'Notifications', 'Settings', 'GTT', 'Sectors',
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

    // ── Health ────────────────────────────────────────────────────────────
    getHealth: builder.query<{ status: string; market_open: boolean }, void>({
      query: () => ({ url: '/api/health' }),
      keepUnusedDataFor: 60,
    }),
  }),
});

export const {
  // Auth
  useLoginMutation, useRegisterMutation, useGetMeQuery,
  // Portfolio
  useGetPortfolioSummaryQuery, useGetTodayPnlQuery,
  // Positions & Orders
  useGetPositionsQuery, useGetOrdersQuery,
  useSquareOffMutation, useSquareOffAllMutation, useExecuteSignalMutation,
  // Signals
  useGetLatestSignalsQuery, useGetActionableSignalsQuery,
  useGetSignalHistoryQuery, useRefreshSignalsMutation,
  // Stocks
  useGetStocksQuery, useLazyGetStocksQuery,
  useGetStockDetailQuery, useLazyGetStockDetailQuery,
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
  // Health
  useGetHealthQuery,
} = tradeMindApiService;
