import { http, HttpResponse } from 'msw';
import {
  STOCKS, TOP_SIGNALS, HOLDINGS, OPEN_POS, TRADE_HISTORY, GTT,
  INDICES, FII_DII, HEATMAP, GAINERS, LOSERS, BREADTH, SENTIMENT_SCORE,
  MARKET_NEWS, stockNews, horizonBreakdown,
  PF_INVESTED, PF_CURRENT, PF_PNL, ALLOC, PNL_HISTORY,
  SECTORS, SECTOR_COLORS, WATCHLIST, NOTIFS,
} from './data';

// mutable in-memory watchlist + notifs for this session
let _watchlist = [...WATCHLIST];
let _notifs    = [...NOTIFS];

const B = 'http://localhost:8000';

const MOCK_USER = {
  id: 1,
  username: 'demo',
  display_name: 'Demo User',
  virtual_balance: 1000000,
  virtual_invested: PF_INVESTED,
  total_pnl: PF_PNL,
  win_count: 12,
  loss_count: 4,
  mode: 'paper',
};

export const handlers = [

  // ── Auth (/api/trading/*) ─────────────────────────────────────────────────
  http.post(`${B}/api/trading/register`, () =>
    HttpResponse.json({ token: 'mock-jwt-token', user: MOCK_USER })
  ),

  http.post(`${B}/api/trading/login`, () =>
    HttpResponse.json({ token: 'mock-jwt-token', user: MOCK_USER })
  ),

  http.get(`${B}/api/trading/me`, () =>
    HttpResponse.json(MOCK_USER)
  ),

  http.get(`${B}/api/trading/user/:userId`, () =>
    HttpResponse.json(MOCK_USER)
  ),

  // ── Portfolio / Dashboard ─────────────────────────────────────────────────
  http.get(`${B}/api/trading/portfolio/:userId`, () =>
    HttpResponse.json({
      total_invested: PF_INVESTED,
      current_value: PF_CURRENT,
      total_pnl: PF_PNL,
      pnl_pct: ((PF_PNL / PF_INVESTED) * 100).toFixed(2),
      holdings: HOLDINGS,
      allocation: ALLOC,
      pnl_history: PNL_HISTORY,
    })
  ),

  http.get(`${B}/api/trading/pnl/today/:userId`, () =>
    HttpResponse.json({
      today_pnl: Math.round(PF_PNL * 0.06),
      today_pnl_pct: 1.21,
    })
  ),

  // ── Signals ───────────────────────────────────────────────────────────────
  http.get(`${B}/api/signals/latest`, () =>
    HttpResponse.json({
      data: {
        generated_at: new Date().toISOString(),
        total_signals: STOCKS.length,
        summary: {
          STRONG_BUY: 0,
          BUY:  STOCKS.filter(s => s.signal === 'BUY').length,
          HOLD: STOCKS.filter(s => s.signal === 'HOLD').length,
          SELL: STOCKS.filter(s => s.signal === 'SELL').length,
          STRONG_SELL: 0,
        },
        actionable_trades: TOP_SIGNALS.map(s => ({
          symbol: s.symbol,
          signal: s.signal,
          confidence: s.confidence,
          trade: {
            type: s.signal,
            buy_price: s.price,
            target_price: +(s.price * (1 + s.expReturn / 100)).toFixed(2),
            stop_loss: +(s.price * 0.93).toFixed(2),
            risk_reward: 2.5,
            expected_return_pct: s.expReturn,
          },
          price: { current: s.price, atr_14: s.price * 0.015, atr_pct: 1.5 },
          model: { name: 'Ensemble', horizon: s.horizon, accuracy: 0.72, precision: s.confidence / 100 },
          sentiment: { score: s.sentiment },
          top_drivers: [],
          generated_at: new Date().toISOString(),
        })),
        avoid_list: STOCKS.filter(s => s.signal === 'SELL').slice(0, 5),
        hold_list: STOCKS.filter(s => s.signal === 'HOLD').slice(0, 10),
      },
    })
  ),

  http.get(`${B}/api/signals/actionable`, () =>
    HttpResponse.json({ data: TOP_SIGNALS })
  ),

  http.get(`${B}/api/signals/avoid`, () =>
    HttpResponse.json({ data: STOCKS.filter(s => s.signal === 'SELL').slice(0, 10) })
  ),

  http.get(`${B}/api/signals/stock/:symbol`, ({ params }) => {
    const stock = STOCKS.find(s => s.symbol === params.symbol);
    return stock
      ? HttpResponse.json({ data: stock })
      : HttpResponse.json({ error: 'not found' }, { status: 404 });
  }),

  http.post(`${B}/api/signals/refresh`, () =>
    HttpResponse.json({ status: 'ok', message: '498 stocks re-scored · 14 new BUY signals' })
  ),

  // ── Trade execution ───────────────────────────────────────────────────────
  http.post(`${B}/api/trading/execute-signal`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>;
    return HttpResponse.json({
      status: 'executed',
      trade: { id: Math.floor(Math.random() * 10000), ...body, executed_at: new Date().toISOString() },
    });
  }),

  // ── Positions & Orders ────────────────────────────────────────────────────
  http.get(`${B}/api/trading/positions/:userId`, () =>
    HttpResponse.json({ data: OPEN_POS, total: OPEN_POS.length, page: 0, size: 50 })
  ),

  http.get(`${B}/api/trading/orders/:userId`, () =>
    HttpResponse.json({ data: TRADE_HISTORY, total: TRADE_HISTORY.length, page: 0, size: 50 })
  ),

  http.post(`${B}/api/trading/square-off/:userId/:symbol`, () =>
    HttpResponse.json({ status: 'closed', realized_pnl: Math.round(Math.random() * 5000 - 2000) })
  ),

  http.post(`${B}/api/trading/square-off-all/:userId`, () =>
    HttpResponse.json({ status: 'all_closed', count: OPEN_POS.length })
  ),

  // ── Risk Settings ─────────────────────────────────────────────────────────
  http.get(`${B}/api/trading/risk-settings/:userId`, () =>
    HttpResponse.json({
      max_position_size: 50000,
      max_daily_loss: 10000,
      stop_loss_pct: 7,
      target_pct: 15,
      mode: 'paper',
    })
  ),

  http.put(`${B}/api/trading/risk-settings/:userId`, async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({ status: 'updated', data: body });
  }),

  // ── GTT Orders ────────────────────────────────────────────────────────────
  http.get(`${B}/api/orders/gtt`, () =>
    HttpResponse.json({ data: GTT })
  ),

  http.post(`${B}/api/orders/gtt/sync`, () =>
    HttpResponse.json({ status: 'synced', count: GTT.length })
  ),

  // ── Market Data ───────────────────────────────────────────────────────────
  http.get(`${B}/api/market/overview`, () =>
    HttpResponse.json({
      indices: INDICES,
      fii_dii: FII_DII,
      heatmap: HEATMAP,
      gainers: GAINERS,
      losers: LOSERS,
      breadth: BREADTH,
      sentiment: SENTIMENT_SCORE,
      news: MARKET_NEWS,
    })
  ),

  http.get(`${B}/api/market/fii-dii`, () =>
    HttpResponse.json({ data: FII_DII })
  ),

  http.get(`${B}/api/market/sectors`, () =>
    HttpResponse.json({ data: HEATMAP })
  ),

  http.get(`${B}/api/market/breadth`, () =>
    HttpResponse.json({ data: BREADTH })
  ),

  // ── Stocks ────────────────────────────────────────────────────────────────
  http.get(`${B}/api/stocks`, ({ request }) => {
    const url = new URL(request.url);
    const search = url.searchParams.get('search') ?? '';
    const sector = url.searchParams.get('sector') ?? '';
    const page   = Number(url.searchParams.get('page') ?? 0);
    const size   = Number(url.searchParams.get('size') ?? 50);

    let filtered = STOCKS;
    if (search) filtered = filtered.filter(s => s.symbol.toLowerCase().includes(search.toLowerCase()) || s.name.toLowerCase().includes(search.toLowerCase()));
    if (sector) filtered = filtered.filter(s => s.sector === sector);

    const total = filtered.length;
    const data  = filtered.slice(page * size, (page + 1) * size);
    return HttpResponse.json({ data, total, page, size, sectors: SECTORS, sector_colors: SECTOR_COLORS });
  }),

  http.get(`${B}/api/prices/:symbol`, ({ params, request }) => {
    const stock = STOCKS.find(s => s.symbol === decodeURIComponent(params.symbol as string));
    const url   = new URL(request.url);
    const days  = Number(url.searchParams.get('days') ?? 30);
    const pts   = stock?.spark ?? [];
    return HttpResponse.json({
      symbol: params.symbol,
      data: pts.map((close, i) => ({
        date: new Date(Date.now() - (pts.length - i) * 86400000).toISOString().slice(0, 10),
        open: close * 0.998, high: close * 1.012, low: close * 0.985, close, volume: 1000000,
      })).slice(-days),
    });
  }),

  http.get(`${B}/api/indicators/:symbol`, ({ params }) => {
    const stock = STOCKS.find(s => s.symbol === decodeURIComponent(params.symbol as string));
    return HttpResponse.json({ symbol: params.symbol, rsi: 55, macd: 0.12, adx: 22, price: stock?.price ?? 0 });
  }),

  http.get(`${B}/api/watchlist/:symbol`, ({ params }) => {
    const stock = STOCKS.find(s => s.symbol === decodeURIComponent(params.symbol as string));
    return stock
      ? HttpResponse.json({ data: { ...stock, news: stockNews(stock.symbol), horizons: horizonBreakdown(stock) } })
      : HttpResponse.json({ error: 'not found' }, { status: 404 });
  }),

  // ── Health ────────────────────────────────────────────────────────────────
  http.get(`${B}/api/health`, () =>
    HttpResponse.json({ status: 'ok', market_open: true, version: '1.0.0' })
  ),

  http.get(`${B}/health`, () =>
    HttpResponse.json({ status: 'ok', market_open: true, version: '1.0.0' })
  ),

  // ── Portfolio API (RTK Query service endpoints) ───────────────────────────
  http.get(`${B}/api/portfolio`, () =>
    HttpResponse.json({ data: [{ id: 1, name: 'My Portfolio', total_stocks: HOLDINGS.length }] })
  ),

  http.get(`${B}/api/portfolio/sectors`, () =>
    HttpResponse.json({
      data: SECTORS.map(sector => ({
        sector,
        total_stocks: STOCKS.filter(s => s.sector === sector).length,
        signals: {
          BUY:  STOCKS.filter(s => s.sector === sector && s.signal === 'BUY').length,
          SELL: STOCKS.filter(s => s.sector === sector && s.signal === 'SELL').length,
          HOLD: STOCKS.filter(s => s.sector === sector && s.signal === 'HOLD').length,
        },
        color: SECTOR_COLORS[sector],
      })),
    })
  ),

  // ── Watchlist ─────────────────────────────────────────────────────────────
  http.get(`${B}/api/watchlist/:userId`, () =>
    HttpResponse.json({ data: _watchlist, total: _watchlist.length })
  ),

  http.post(`${B}/api/watchlist/:userId/:symbol`, ({ params }) => {
    const symbol = decodeURIComponent(params.symbol as string);
    if (_watchlist.find(w => w.symbol === symbol)) {
      return HttpResponse.json({ status: 'already_added' });
    }
    const s = STOCKS.find(x => x.symbol === symbol);
    if (!s) return HttpResponse.json({ error: 'not found' }, { status: 404 });
    const item = { ...s, alertAbove: +(s.price * 1.06).toFixed(0), alertBelow: +(s.price * 0.93).toFixed(0), addedAt: new Date().toISOString() };
    _watchlist = [item, ..._watchlist];
    return HttpResponse.json({ status: 'added', data: item });
  }),

  http.delete(`${B}/api/watchlist/:userId/:symbol`, ({ params }) => {
    const symbol = decodeURIComponent(params.symbol as string);
    _watchlist = _watchlist.filter(w => w.symbol !== symbol);
    return HttpResponse.json({ status: 'removed' });
  }),

  http.put(`${B}/api/watchlist/:userId/:symbol/alerts`, async ({ params, request }) => {
    const symbol = decodeURIComponent(params.symbol as string);
    const body = await request.json() as { alert_above?: number; alert_below?: number };
    _watchlist = _watchlist.map(w => w.symbol === symbol ? { ...w, alertAbove: body.alert_above ?? w.alertAbove, alertBelow: body.alert_below ?? w.alertBelow } : w);
    return HttpResponse.json({ status: 'updated' });
  }),

  // ── Notifications ─────────────────────────────────────────────────────────
  http.get(`${B}/api/notifications`, () =>
    HttpResponse.json({ data: _notifs, unread: _notifs.filter(n => !n.is_read).length })
  ),

  http.post(`${B}/api/notifications/mark-read`, () => {
    _notifs = _notifs.map(n => ({ ...n, is_read: true }));
    return HttpResponse.json({ status: 'ok' });
  }),

  http.delete(`${B}/api/notifications/:id`, ({ params }) => {
    _notifs = _notifs.filter(n => n.id !== +params.id!);
    return HttpResponse.json({ status: 'deleted' });
  }),

  // ── Catch-all: return 503 instead of passthrough to dead backend ──────────
  // Must be LAST — catches any endpoint not handled above
  http.all(`${B}/*`, ({ request }) => {
    console.warn(`[MSW] Unhandled: ${request.method} ${request.url}`);
    return HttpResponse.json(
      { error: 'Not mocked', detail: `${request.method} ${new URL(request.url).pathname} is not in mock handlers` },
      { status: 503 }
    );
  }),
];
