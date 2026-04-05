const API_BASE = 'http://localhost:8000';

// ==========================================
// Server-Side Table Params
// ==========================================

export interface TableParams {
  page?: number;
  size?: number;
  sort?: string;
  order?: 'asc' | 'desc';
  globalFilter?: string;
}

function buildQuery(params?: TableParams): string {
  if (!params) return '';
  const q = new URLSearchParams();
  if (params.page != null) q.set('page', String(params.page));
  if (params.size != null) q.set('size', String(params.size));
  if (params.sort) q.set('sort', params.sort);
  if (params.order) q.set('order', params.order);
  if (params.globalFilter) q.set('globalFilter', params.globalFilter);
  return q.toString();
}

// ==========================================
// Token Management
// ==========================================

function getToken(): string | null {
  return localStorage.getItem('trademind_token');
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token
    ? { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
    : { 'Content-Type': 'application/json' };
}

// ==========================================
// Auth API
// ==========================================

export async function registerUser(username: string, password: string, displayName?: string) {
  const res = await fetch(`${API_BASE}/api/trading/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, display_name: displayName || username }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Registration failed');
  }
  const data = await res.json();
  // Store token
  localStorage.setItem('trademind_token', data.token);
  return data;
}

export async function loginUser(username: string, password: string) {
  const res = await fetch(`${API_BASE}/api/trading/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Invalid username or password');
  }
  const data = await res.json();
  localStorage.setItem('trademind_token', data.token);
  return data;
}

export async function getMe() {
  const res = await fetch(`${API_BASE}/api/trading/me`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Not authenticated');
  return res.json();
}

export async function getUser(userId: number) {
  const res = await fetch(`${API_BASE}/api/trading/user/${userId}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('User not found');
  return res.json();
}

// ==========================================
// Portfolio & Dashboard
// ==========================================

export async function getPortfolioSummary(userId: number) {
  const res = await fetch(`${API_BASE}/api/trading/portfolio/${userId}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load portfolio');
  return res.json();
}

export async function getTodayPnl(userId: number) {
  const res = await fetch(`${API_BASE}/api/trading/pnl/today/${userId}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load PnL');
  return res.json();
}

// ==========================================
// Trade Signals
// ==========================================

export async function getLatestSignals(params?: TableParams) {
  const q = buildQuery(params);
  const url = q ? `${API_BASE}/api/signals/latest?${q}` : `${API_BASE}/api/signals/latest`;
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load signals');
  return res.json();
}

export async function getSignalForStock(symbol: string) {
  const res = await fetch(`${API_BASE}/api/signals/stock/${symbol}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load signal');
  return res.json();
}

export async function getActionableSignals() {
  const res = await fetch(`${API_BASE}/api/signals/actionable`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load signals');
  return res.json();
}

// ==========================================
// Trade Execution
// ==========================================

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

export async function executeSignal(params: ExecuteSignalParams) {
  const res = await fetch(`${API_BASE}/api/trading/execute-signal`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Trade execution failed');
  }
  return res.json();
}

// ==========================================
// Positions & Orders
// ==========================================

export async function getPositions(userId: number, params?: TableParams) {
  const q = buildQuery(params);
  const res = await fetch(`${API_BASE}/api/trading/positions/${userId}?${q}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load positions');
  return res.json();
}

export async function getOrders(userId: number, params?: TableParams) {
  const q = buildQuery(params);
  const res = await fetch(`${API_BASE}/api/trading/orders/${userId}?${q}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load orders');
  return res.json();
}

export async function squareOff(userId: number, symbol: string, sellPrice?: number) {
  const res = await fetch(`${API_BASE}/api/trading/square-off/${userId}/${encodeURIComponent(symbol)}`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ sell_price: sellPrice || null }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Square off failed');
  }
  return res.json();
}

export async function squareOffAll(userId: number) {
  const res = await fetch(`${API_BASE}/api/trading/square-off-all/${userId}`, { method: 'POST', headers: authHeaders() });
  if (!res.ok) throw new Error('Square off all failed');
  return res.json();
}

// ==========================================
// Risk Settings
// ==========================================

export async function getRiskSettings(userId: number) {
  const res = await fetch(`${API_BASE}/api/trading/risk-settings/${userId}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load risk settings');
  return res.json();
}

export async function updateRiskSettings(userId: number, settings: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/api/trading/risk-settings/${userId}`, {
    method: 'PUT',
    headers: authHeaders(),
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error('Failed to update risk settings');
  return res.json();
}

// ==========================================
// Market Data
// ==========================================

export async function getMarketOverview() {
  const res = await fetch(`${API_BASE}/api/market/overview`);
  if (!res.ok) throw new Error('Failed to load market overview');
  return res.json();
}

export async function getStockWatchlist(symbol: string) {
  const res = await fetch(`${API_BASE}/api/watchlist/${encodeURIComponent(symbol)}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load watchlist');
  return res.json();
}

// ==========================================
// Stocks List & Detail
// ==========================================

export async function getAllStocks(params?: TableParams & { search?: string; sector?: string }) {
  const q = new URLSearchParams();
  if (params?.search) q.set('search', params.search);
  if (params?.sector) q.set('sector', params.sector);
  if (params?.page != null) q.set('page', String(params.page));
  if (params?.size != null) q.set('size', String(params.size));
  if (params?.sort) q.set('sort', params.sort);
  if (params?.order) q.set('order', params.order);
  if (params?.globalFilter) q.set('globalFilter', params.globalFilter);
  const res = await fetch(`${API_BASE}/api/stocks?${q}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load stocks');
  return res.json();
}

export async function getStockPrices(symbol: string, days = 365) {
  const res = await fetch(`${API_BASE}/api/prices/${encodeURIComponent(symbol)}?days=${days}&interval=1d`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load price data');
  return res.json();
}

export async function getStockIndicators(symbol: string) {
  const res = await fetch(`${API_BASE}/api/indicators/${encodeURIComponent(symbol)}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load indicators');
  return res.json();
}

export function clearToken() {
  localStorage.removeItem('trademind_token');
}
