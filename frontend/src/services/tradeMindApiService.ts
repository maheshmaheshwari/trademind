/**
 * TradeMind AI — API Service
 * 
 * All API endpoints defined using RTK Query + Axios (same pattern as TAQA).
 * Uses injectEndpoints pattern for code splitting.
 */
import { tradeMindApiClient } from './tradeMindApiClient';

// ==========================================
// Interfaces
// ==========================================

interface TradeSignal {
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
  price: {
    current: number;
    atr_14: number;
    atr_pct: number;
  };
  model: {
    name: string;
    horizon: string;
    accuracy: number;
    precision: number;
  };
  sentiment: Record<string, number>;
  top_drivers: Array<{ feature: string; importance: number }>;
  generated_at: string;
}

interface SignalsResponse {
  data: {
    generated_at: string;
    total_models: number;
    total_signals: number;
    summary: {
      STRONG_BUY: number;
      BUY: number;
      HOLD: number;
      SELL: number;
      STRONG_SELL: number;
    };
    actionable_trades: TradeSignal[];
    avoid_list: TradeSignal[];
    hold_list: TradeSignal[];
  };
}

interface Portfolio {
  id: number;
  name: string;
  investment_amount: number;
  time_horizon: string;
  risk_profile: string;
  created_at: string;
  updated_at: string;
  sectors: Array<{
    sector: string;
    allocation_pct: number;
    ai_suggested_pct: number;
    num_stocks: number;
  }>;
  stocks: Array<{
    symbol: string;
    sector: string;
    signal: string;
    confidence: number;
    buy_price: number;
    target_price: number;
    stop_loss: number;
    allocated_amount: number;
    quantity: number;
    status: string;
  }>;
  total_stocks: number;
}

interface SectorInfo {
  sector: string;
  total_stocks: number;
  signals: Record<string, number>;
}

interface CreatePortfolioRequest {
  name: string;
  investment_amount: number;
  time_horizon: string;
  risk_profile: string;
}

interface UpdateSectorsRequest {
  id: number;
  sectors: Array<{ sector: string; allocation_pct: number }>;
}

// ==========================================
// API Service with all endpoints
// ==========================================

const API_BASE_URL = import.meta.env.VITE_BASE_URL || 'http://localhost:8000';

export const tradeMindApiService = tradeMindApiClient(
  'tradeMindApiService',
  API_BASE_URL,
).injectEndpoints({
  endpoints: (builder) => ({

    // ==========================================
    // Trade Signals
    // ==========================================
    
    /** Get latest trade signals with buy/target/stop-loss */
    getLatestSignals: builder.query<SignalsResponse, void>({
      query: () => ({
        url: '/api/signals/latest',
        method: 'GET',
      }),
      keepUnusedDataFor: 60,
      providesTags: ['Signals'],
    }),

    /** Get signal run history */
    getSignalHistory: builder.query<any, void>({
      query: () => ({
        url: '/api/signals/history',
        method: 'GET',
      }),
      keepUnusedDataFor: 0,
    }),

    /** Get only actionable (BUY/STRONG BUY) signals */
    getActionableSignals: builder.query<any, void>({
      query: () => ({
        url: '/api/signals/actionable',
        method: 'GET',
      }),
      keepUnusedDataFor: 60,
    }),

    /** Get avoid list (SELL/STRONG SELL) */
    getAvoidSignals: builder.query<any, void>({
      query: () => ({
        url: '/api/signals/avoid',
        method: 'GET',
      }),
      keepUnusedDataFor: 60,
    }),

    // ==========================================
    // Portfolio Management
    // ==========================================

    /** List all portfolios */
    getAllPortfolios: builder.query<any, void>({
      query: () => ({
        url: '/api/portfolio',
        method: 'GET',
      }),
      keepUnusedDataFor: 0,
      providesTags: ['Portfolio'],
    }),

    /** Get single portfolio by ID */
    getPortfolioById: builder.query<{ data: Portfolio }, number>({
      query: (id) => ({
        url: `/api/portfolio/${id}`,
        method: 'GET',
      }),
      keepUnusedDataFor: 0,
      providesTags: ['Portfolio'],
    }),

    /** Create portfolio with AI allocation */
    createPortfolio: builder.mutation<any, CreatePortfolioRequest>({
      query: (data) => ({
        url: '/api/portfolio/create',
        method: 'POST',
        data,
      }),
      invalidatesTags: ['Portfolio'],
    }),

    /** Update sector allocations */
    updatePortfolioSectors: builder.mutation<any, UpdateSectorsRequest>({
      query: ({ id, sectors }) => ({
        url: `/api/portfolio/${id}/sectors`,
        method: 'PUT',
        data: { sectors },
      }),
      invalidatesTags: ['Portfolio'],
    }),

    /** Rebalance portfolio with current signals */
    rebalancePortfolio: builder.mutation<any, number>({
      query: (id) => ({
        url: `/api/portfolio/${id}/rebalance`,
        method: 'POST',
      }),
      invalidatesTags: ['Portfolio'],
    }),

    /** Delete portfolio */
    deletePortfolio: builder.mutation<any, number>({
      query: (id) => ({
        url: `/api/portfolio/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Portfolio'],
    }),

    // ==========================================
    // Sectors
    // ==========================================

    /** Get all sectors with stock counts and signal summary */
    getAllSectors: builder.query<{ data: SectorInfo[] }, void>({
      query: () => ({
        url: '/api/portfolio/sectors',
        method: 'GET',
      }),
      keepUnusedDataFor: 300,
      providesTags: ['Sectors'],
    }),

    // ==========================================
    // Stock Data
    // ==========================================

    /** Get stock price history */
    getStockPrices: builder.query<any, { symbol: string; limit?: number }>({
      query: ({ symbol, limit }) => ({
        url: `/prices/${symbol}`,
        method: 'GET',
        params: { limit },
      }),
      keepUnusedDataFor: 60,
    }),

    /** Get stock technical indicators */
    getStockIndicators: builder.query<any, { symbol: string }>({
      query: ({ symbol }) => ({
        url: `/indicators/${symbol}`,
        method: 'GET',
      }),
      keepUnusedDataFor: 300,
    }),

    /** Get stock sentiment data */
    getStockSentiment: builder.query<any, { symbol: string }>({
      query: ({ symbol }) => ({
        url: `/sentiment/${symbol}`,
        method: 'GET',
      }),
      keepUnusedDataFor: 600,
    }),

    /** Get watchlist (combined stock data) */
    getWatchlist: builder.query<any, string>({
      query: (symbol) => ({
        url: `/watchlist/${symbol}`,
        method: 'GET',
      }),
      keepUnusedDataFor: 60,
    }),

    // ==========================================
    // Market Overview
    // ==========================================

    /** Get market overview (Nifty, Sensex, VIX, FII/DII) */
    getMarketOverview: builder.query<any, void>({
      query: () => ({
        url: '/market/overview',
        method: 'GET',
      }),
      keepUnusedDataFor: 120,
    }),

    /** Get sector heatmap data */
    getSectorHeatmap: builder.query<any, void>({
      query: () => ({
        url: '/heatmap/sectors',
        method: 'GET',
      }),
      keepUnusedDataFor: 300,
    }),

    /** Health check */
    getHealthCheck: builder.query<any, void>({
      query: () => ({
        url: '/health',
        method: 'GET',
      }),
      keepUnusedDataFor: 0,
    }),
  }),
});

// ==========================================
// Export hooks (auto-generated by RTK Query)
// ==========================================
export const {
  // Signals
  useGetLatestSignalsQuery,
  useGetSignalHistoryQuery,
  useGetActionableSignalsQuery,
  useGetAvoidSignalsQuery,

  // Portfolio
  useGetAllPortfoliosQuery,
  useGetPortfolioByIdQuery,
  useCreatePortfolioMutation,
  useUpdatePortfolioSectorsMutation,
  useRebalancePortfolioMutation,
  useDeletePortfolioMutation,

  // Sectors
  useGetAllSectorsQuery,

  // Stock Data
  useGetStockPricesQuery,
  useGetStockIndicatorsQuery,
  useGetStockSentimentQuery,
  useGetWatchlistQuery,

  // Market
  useGetMarketOverviewQuery,
  useGetSectorHeatmapQuery,
  useGetHealthCheckQuery,
} = tradeMindApiService;
