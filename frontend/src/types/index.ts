export type SignalType = 'BUY' | 'SELL' | 'HOLD';
export type Horizon = '1W' | '2W' | '1M' | '2M' | '3M' | '6M';
export type SentLabel = 'pos' | 'neg' | 'neu';

export interface Stock {
  id: number;
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;            // % change today
  signal: SignalType;
  confidence: number;        // 0–100
  horizon: Horizon;
  expReturn: number;         // expected return %
  sentiment: number;         // -1 to +1
  updatedMin: number;        // minutes ago
  spark: number[];           // sparkline data points
  mcap: number;              // market cap in Cr
  volume: number;            // volume in M
  high52: number;
  low52: number;
  pe: number;
  target_price?: number;     // from trade signal
  stop_loss?: number;        // from trade signal
}

export interface Holding extends Stock {
  quantity: number;
  avg_buy_price: number;   // avg entry price
  current_price: number;   // current market price
  invested_amount: number;
  current_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface OpenPosition extends Stock {
  avg_buy_price: number;
  stop_loss: number;
  target_price: number;
  quantity: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  created_at?: string;
}

export interface Trade {
  id: number;
  symbol: string;
  name: string;
  sector: string;
  order_type: 'BUY' | 'SELL' | 'LIMIT' | 'MARKET';
  order_purpose?: string;
  quantity: number;
  price: number;
  value: number | null;
  created_at: string;
  pnl: number | null;
  status: string;
}

export interface GTTOrder {
  id: number;
  symbol: string;
  name: string;
  type: string;
  side: 'BUY' | 'SELL';
  trigger: number;
  ltp: number;
  qty: number;
  status: 'ACTIVE' | 'TRIGGERED' | 'EXPIRED';
  created: string;
}

export interface IndexData {
  name: string;
  value: number;
  change: number;
  pct: number;
  spark: number[];
}

export interface FIIDIIBar {
  day: string;
  fii: number;
  dii: number;
}

export interface HeatmapSector {
  sector:      string;
  change:      number;
  stock_count?: number;
  buy_count?:  number;
  sell_count?: number;
  hold_count?: number;
  avg_conf?:   number;
}

export interface Breadth {
  advances: number;
  declines: number;
  unchanged: number;
}

export interface AllocSlice {
  sector: string;
  val: number;
  color: string;
}

export interface NewsItem {
  src: string;
  time: string;
  sent: SentLabel;
  title: string;
}

export interface HorizonBreakdown {
  h: Horizon;
  sig: SignalType;
  conf: number;
}

export interface WatchlistItem extends Stock {
  alertAbove: number;
  alertBelow: number;
  addedAt: string;
}

export type NotifType = 'trade' | 'signal' | 'price' | 'news';

export interface Notification {
  id:         number;
  user_id?:   number;
  type:       NotifType;
  icon:       string;
  color:      string;
  title:      string;
  message:    string;
  created_at: string;
  is_read:    boolean;
}
