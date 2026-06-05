/**
 * Mock market data — deterministic (seeded RNG, ported from design data.jsx)
 * Used when VITE_USE_MOCK=true  (npm run dev:mock)
 */
import type {
  Stock, Holding, OpenPosition, Trade, GTTOrder,
  IndexData, FIIDIIBar, HeatmapSector, Breadth, AllocSlice, NewsItem, SignalType, Horizon,
  WatchlistItem, Notification,
} from '../types';

// ─── seeded RNG ────────────────────────────────────────────────────────────

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng   = mulberry32(20260601);
const rand  = (a: number, b: number) => a + (b - a) * rng();
const randI = (a: number, b: number) => Math.floor(rand(a, b + 1));
const pick  = <T>(arr: T[]): T => arr[Math.floor(rng() * arr.length)];

// ─── constants ─────────────────────────────────────────────────────────────

export const SECTORS = ['IT','Banking','Financials','Energy','Auto','FMCG','Pharma','Metals','Cement','Infra','Telecom','Power'];

export const SECTOR_COLORS: Record<string, string> = {
  IT:'#3B82F6', Banking:'#8B5CF6', Financials:'#6366F1', Energy:'#F59E0B', Auto:'#EC4899',
  FMCG:'#10B981', Pharma:'#14B8A6', Metals:'#F97316', Cement:'#A78BFA', Infra:'#0EA5E9',
  Telecom:'#EF4444', Power:'#EAB308',
};

export const HORIZONS: Horizon[] = ['1W','2W','1M','2M','3M','6M'];

// ─── helpers ───────────────────────────────────────────────────────────────

function sparkline(n: number, trend: number): number[] {
  const pts: number[] = []; let v = 50 + rand(-8, 8);
  for (let i = 0; i < n; i++) { v += trend * rand(.2, 1.4) + rand(-3.2, 3.2); v = Math.max(8, Math.min(92, v)); pts.push(v); }
  return pts;
}

export function fmtAgo(m: number): string {
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

// ─── base stock list (real Nifty 500 constituents) ─────────────────────────

const BASE: [string, string, string, number][] = [
  ['RELIANCE','Reliance Industries','Energy',2912],['TCS','Tata Consultancy','IT',4087],
  ['HDFCBANK','HDFC Bank','Banking',1684],['INFY','Infosys','IT',1832],
  ['ICICIBANK','ICICI Bank','Banking',1247],['HINDUNILVR','Hindustan Unilever','FMCG',2398],
  ['ITC','ITC Ltd','FMCG',436],['SBIN','State Bank of India','Banking',842],
  ['BHARTIARTL','Bharti Airtel','Telecom',1589],['KOTAKBANK','Kotak Mahindra Bank','Banking',1763],
  ['LT','Larsen & Toubro','Infra',3621],['BAJFINANCE','Bajaj Finance','Financials',7218],
  ['HCLTECH','HCL Technologies','IT',1894],['ASIANPAINT','Asian Paints','FMCG',2287],
  ['AXISBANK','Axis Bank','Banking',1142],['MARUTI','Maruti Suzuki','Auto',12640],
  ['SUNPHARMA','Sun Pharma','Pharma',1812],['TITAN','Titan Company','FMCG',3398],
  ['ULTRACEMCO','UltraTech Cement','Cement',11240],['WIPRO','Wipro','IT',548],
  ['NESTLEIND','Nestle India','FMCG',2456],['ONGC','ONGC','Energy',268],
  ['NTPC','NTPC','Power',412],['POWERGRID','Power Grid Corp','Power',324],
  ['TATAMOTORS','Tata Motors','Auto',978],['TATASTEEL','Tata Steel','Metals',164],
  ['JSWSTEEL','JSW Steel','Metals',942],['ADANIENT','Adani Enterprises','Infra',2876],
  ['ADANIPORTS','Adani Ports','Infra',1421],['COALINDIA','Coal India','Energy',478],
  ['GRASIM','Grasim Industries','Cement',2614],['BAJAJFINSV','Bajaj Finserv','Financials',1718],
  ['TECHM','Tech Mahindra','IT',1672],['HDFCLIFE','HDFC Life Insurance','Financials',682],
  ['DRREDDY',"Dr Reddy's Labs",'Pharma',1289],['CIPLA','Cipla','Pharma',1564],
  ['BRITANNIA','Britannia Industries','FMCG',4912],['EICHERMOT','Eicher Motors','Auto',4847],
  ['DIVISLAB',"Divi's Laboratories",'Pharma',5928],['HEROMOTOCO','Hero MotoCorp','Auto',4621],
  ['BPCL','Bharat Petroleum','Energy',312],['INDUSINDBK','IndusInd Bank','Banking',984],
  ['SBILIFE','SBI Life Insurance','Financials',1542],['APOLLOHOSP','Apollo Hospitals','Pharma',6824],
  ['BAJAJAUTO','Bajaj Auto','Auto',9218],['TATACONSUM','Tata Consumer','FMCG',1098],
  ['HINDALCO','Hindalco Industries','Metals',642],['MM','Mahindra & Mahindra','Auto',2891],
  ['SHREECEM','Shree Cement','Cement',26840],['DLF','DLF Ltd','Infra',812],
  ['VEDL','Vedanta','Metals',448],['GAIL','GAIL India','Energy',214],
  ['DABUR','Dabur India','FMCG',512],['GODREJCP','Godrej Consumer','FMCG',1284],
  ['PIDILITIND','Pidilite Industries','FMCG',2962],['AMBUJACEM','Ambuja Cements','Cement',614],
  ['SIEMENS','Siemens','Infra',6842],['BANKBARODA','Bank of Baroda','Banking',248],
  ['PNB','Punjab National Bank','Banking',104],['IOC','Indian Oil Corp','Energy',168],
  ['LICI','LIC of India','Financials',982],['TVSMOTOR','TVS Motor','Auto',2418],
  ['ZOMATO','Zomato','IT',264],['IRCTC','IRCTC','Infra',812],
  ['LTIM','LTIMindtree','IT',5912],['PERSISTENT','Persistent Systems','IT',5824],
  ['COFORGE','Coforge','IT',6248],['MPHASIS','Mphasis','IT',2814],
  ['TORNTPHARM','Torrent Pharma','Pharma',3142],['LUPIN','Lupin','Pharma',2089],
  ['AUROPHARMA','Aurobindo Pharma','Pharma',1342],['BIOCON','Biocon','Pharma',348],
  ['JINDALSTEL','Jindal Steel','Metals',924],['SAIL','SAIL','Metals',128],
  ['NMDC','NMDC','Metals',214],['ACC','ACC Ltd','Cement',2184],
  ['DMART','Avenue Supermarts','FMCG',4218],['TRENT','Trent Ltd','FMCG',6842],
  ['NAUKRI','Info Edge','IT',7842],['INDIGO','InterGlobe Aviation','Infra',4621],
  ['CANBK','Canara Bank','Banking',102],['FEDERALBNK','Federal Bank','Banking',184],
  ['IDFCFIRSTB','IDFC First Bank','Banking',68],['RECLTD','REC Ltd','Financials',512],
  ['PFC','Power Finance Corp','Financials',468],['IRFC','Indian Railway Finance','Financials',148],
];

// ─── STOCKS ────────────────────────────────────────────────────────────────

export const STOCKS: Stock[] = BASE.map(([symbol, name, sector, price], i) => {
  const bias = pick([1, 1, -1, 1, -1, 0]);
  const signal: SignalType = bias > 0 ? 'BUY' : bias < 0 ? 'SELL' : 'HOLD';
  const trend = signal === 'BUY' ? 1 : signal === 'SELL' ? -1 : rand(-.4, .4);
  const change = +(trend * rand(.3, 3.8) + rand(-.6, .6)).toFixed(2);
  const confidence = randI(signal === 'HOLD' ? 52 : 64, 96);
  const horizon = pick(HORIZONS);
  const expReturn = +(signal === 'BUY' ? rand(3, 28) : signal === 'SELL' ? -rand(3, 22) : rand(-3, 4)).toFixed(1);
  const sentiment = +(trend * rand(.1, .7) + rand(-.2, .2)).toFixed(2);
  return {
    id: i, symbol, name, sector, price: +price.toFixed(2), change,
    signal, confidence, horizon, expReturn, sentiment,
    updatedMin: randI(2, 180), spark: sparkline(28, trend),
    mcap: +(price * randI(40, 900) / 100).toFixed(0),
    volume: randI(2, 180),
    high52: +(price * rand(1.05, 1.4)).toFixed(0),
    low52: +(price * rand(.6, .92)).toFixed(0),
    pe: +rand(12, 68).toFixed(1),
  };
});

export const TOP_SIGNALS = STOCKS
  .filter(s => s.signal !== 'HOLD')
  .sort((a, b) => b.confidence - a.confidence)
  .slice(0, 5);

// ─── PORTFOLIO ─────────────────────────────────────────────────────────────

const HOLD_SYMS = ['RELIANCE','TCS','HDFCBANK','INFY','TATAMOTORS','BHARTIARTL','SUNPHARMA','JSWSTEEL','TITAN','LTIM'];

export const HOLDINGS: Holding[] = HOLD_SYMS.map(sym => {
  const s = STOCKS.find(x => x.symbol === sym)!;
  const qty = randI(5, 140);
  const avg = +(s.price * rand(.72, 1.08)).toFixed(2);
  const invested = qty * avg;
  const current  = qty * s.price;
  return { ...s, qty, avg, cmp: s.price, invested: +invested.toFixed(0), current: +current.toFixed(0),
    pnl: +(current - invested).toFixed(0), pnlPct: +(((s.price - avg) / avg) * 100).toFixed(2) };
});

export const PF_INVESTED = HOLDINGS.reduce((a, h) => a + h.invested, 0);
export const PF_CURRENT  = HOLDINGS.reduce((a, h) => a + h.current, 0);
export const PF_PNL      = PF_CURRENT - PF_INVESTED;

export const ALLOC: AllocSlice[] = (() => {
  const m: Record<string, number> = {};
  HOLDINGS.forEach(h => { m[h.sector] = (m[h.sector] ?? 0) + h.current; });
  return Object.entries(m)
    .map(([sector, val]) => ({ sector, val, color: SECTOR_COLORS[sector] }))
    .sort((a, b) => b.val - a.val);
})();

function pnlSeries(n: number, end: number): number[] {
  const pts: number[] = []; let v = end * rand(.55, .7);
  const step = (end - v) / n;
  for (let i = 0; i < n; i++) { v += step + rand(-end * .018, end * .022); pts.push(Math.max(0, v)); }
  pts[pts.length - 1] = end;
  return pts;
}
export const PNL_HISTORY: Record<string, number[]> = {
  '30D': pnlSeries(30, PF_CURRENT),
  '90D': pnlSeries(90, PF_CURRENT),
  '1Y':  pnlSeries(52, PF_CURRENT),
};

// ─── OPEN POSITIONS ────────────────────────────────────────────────────────

export const OPEN_POS: OpenPosition[] = ['RELIANCE','TATAMOTORS','JSWSTEEL','BHARTIARTL','SUNPHARMA','LTIM'].map(sym => {
  const s = STOCKS.find(x => x.symbol === sym)!;
  const entry  = +(s.price * rand(.88, .98)).toFixed(2);
  const sl     = +(entry * rand(.9, .96)).toFixed(2);
  const target = +(entry * rand(1.06, 1.18)).toFixed(2);
  const qty    = randI(10, 120);
  return { ...s, entry, sl, target, qty,
    pnl: +((s.price - entry) * qty).toFixed(0),
    pnlPct: +(((s.price - entry) / entry) * 100).toFixed(2),
    days: randI(1, 42),
  };
});

// ─── TRADE HISTORY ─────────────────────────────────────────────────────────

export const TRADE_HISTORY: Trade[] = (() => {
  const out: Trade[] = [];
  const today = new Date(2026, 5, 1);
  for (let i = 0; i < 46; i++) {
    const s = pick(STOCKS);
    const side = pick(['BUY', 'SELL'] as const);
    const d = new Date(today); d.setDate(d.getDate() - randI(1, 140));
    const qty  = randI(5, 150);
    const pr   = +(s.price * rand(.8, 1.15)).toFixed(2);
    const realized = +((rng() > .42 ? 1 : -1) * s.price * qty * rand(.005, .09)).toFixed(0);
    out.push({ id: i, symbol: s.symbol, name: s.name, sector: s.sector, side, qty, price: pr,
      value: +(qty * pr).toFixed(0), date: d, realized, status: 'EXECUTED' });
  }
  return out.sort((a, b) => b.date.getTime() - a.date.getTime());
})();

// ─── GTT ORDERS ────────────────────────────────────────────────────────────

export const GTT: GTTOrder[] = ['INFY','HDFCBANK','TITAN','COALINDIA','DRREDDY'].map((sym, i) => {
  const s = STOCKS.find(x => x.symbol === sym)!;
  return {
    id: i, symbol: s.symbol, name: s.name,
    type: pick(['Single', 'OCO']),
    side: pick(['BUY', 'SELL'] as const),
    trigger: +(s.price * rand(.92, 1.08)).toFixed(2),
    ltp: s.price, qty: randI(10, 80),
    status: pick(['ACTIVE', 'ACTIVE', 'TRIGGERED', 'EXPIRED'] as const),
    created: fmtAgo(randI(120, 4000)),
  };
});

// ─── MARKET ────────────────────────────────────────────────────────────────

export const INDICES: IndexData[] = [
  { name:'NIFTY 50',  value:24862.30, change:142.65, pct:0.58,  spark: sparkline(40, 1) },
  { name:'NIFTY 500', value:22918.75, change:118.40, pct:0.52,  spark: sparkline(40, 1) },
  { name:'SENSEX',    value:81472.18, change:421.30, pct:0.52,  spark: sparkline(40, 1) },
  { name:'INDIA VIX', value:13.42,    change:-0.86,  pct:-6.02, spark: sparkline(40,-1) },
];

export const FII_DII: FIIDIIBar[] = (() => {
  const labels = ['Mon','Tue','Wed','Thu','Fri','Mon','Tue','Wed','Thu','Fri'];
  return labels.map(day => ({ day, fii: +rand(-4200, 5400).toFixed(0), dii: +rand(-2600, 4800).toFixed(0) }));
})();

export const HEATMAP: HeatmapSector[] = SECTORS.map(sector => ({
  sector, change: +rand(-3.4, 3.8).toFixed(2), mcap: randI(2, 28),
}));

export const GAINERS = [...STOCKS].sort((a, b) => b.change - a.change).slice(0, 5);
export const LOSERS  = [...STOCKS].sort((a, b) => a.change - b.change).slice(0, 5);

export const BREADTH: Breadth = { advances: 1842, declines: 1264, unchanged: 122 };
export const SENTIMENT_SCORE  = 68;

// ─── NEWS ──────────────────────────────────────────────────────────────────

export const MARKET_NEWS: NewsItem[] = [
  { src:'Economic Times', time:'32m', sent:'pos', title:'RBI holds repo rate at 6.25%, signals dovish stance for H2 FY26' },
  { src:'Mint',           time:'1h',  sent:'pos', title:'FII inflows top ₹12,400 Cr this week as global funds rotate into India' },
  { src:'Moneycontrol',   time:'2h',  sent:'neu', title:'Q4 earnings season: IT majors guide for muted but stable growth' },
  { src:'Business Standard', time:'3h', sent:'neg', title:'Metal stocks slip as China demand outlook weakens for Q3' },
  { src:'Bloomberg Quint', time:'4h', sent:'pos', title:'Auto sales hit record high in May, two-wheelers lead recovery' },
];

export function stockNews(symbol: string): NewsItem[] {
  const templates: [string, string][] = [
    ['pos','Brokerage upgrades to BUY, raises target on strong order book'],
    ['pos','Q4 net profit beats estimates, margins expand 180 bps YoY'],
    ['neu','Management reiterates FY26 guidance in analyst call'],
    ['neg','Promoter pledge rises marginally; analysts flag caution'],
    ['pos','New capacity expansion announced, capex of ₹4,200 Cr'],
  ];
  const srcs = ['ET','Mint','MC','BQ'] as const;
  // use symbol to seed variation per stock
  const offset = [...symbol].reduce((a, c) => a + c.charCodeAt(0), 0) % templates.length;
  return templates.map((t, i) => ({
    sent: t[0] as 'pos' | 'neg' | 'neu',
    title: t[1],
    src: srcs[(i + offset) % srcs.length],
    time: fmtAgo(randI(20, 600)),
  }));
}

// ─── horizon breakdown per stock ───────────────────────────────────────────

export function horizonBreakdown(stock: Stock) {
  return HORIZONS.map((h, i) => {
    const seed = (stock.id * 7 + i * 13) % 100;
    const sig: SignalType = seed > 62 ? 'BUY' : seed < 22 ? 'SELL' : seed < 40 ? 'HOLD' : 'BUY';
    const conf = 52 + ((stock.confidence + seed) % 44);
    return { h, sig, conf };
  });
}

// ─── WATCHLIST ─────────────────────────────────────────────────────────────

const WATCH_SYMS = ['INFY','ZOMATO','TRENT','TITAN','LTIM','ADANIENT','DMART','TATAMOTORS','HDFCBANK','RELIANCE','BHARTIARTL','ACC'];

export const WATCHLIST: WatchlistItem[] = WATCH_SYMS.map((sym) => {
  const s = STOCKS.find(x => x.symbol === sym)!;
  return {
    ...s,
    alertAbove: +(s.price * rand(1.04, 1.08)).toFixed(0),
    alertBelow: +(s.price * rand(0.91, 0.95)).toFixed(0),
    addedAt: new Date(Date.now() - randI(1, 30) * 86400000).toISOString(),
  };
});

// ─── NOTIFICATIONS ─────────────────────────────────────────────────────────

export const NOTIFS: Notification[] = [
  { id:1, type:'trade',  icon:'checkCircle', color:'var(--green)',   title:'Order executed',    message:'BUY 40 × RELIANCE filled @ ₹2,912.00', created_at: new Date(Date.now()-2*60000).toISOString(),  is_read:false },
  { id:2, type:'signal', icon:'sparkle',     color:'var(--gold)',    title:'New AI signal',     message:'INFY upgraded to BUY · 88% confidence', created_at: new Date(Date.now()-18*60000).toISOString(), is_read:false },
  { id:3, type:'price',  icon:'bell',        color:'var(--accent-2)',title:'Price alert',       message:'TATAMOTORS crossed ₹980 (your target)', created_at: new Date(Date.now()-41*60000).toISOString(), is_read:false },
  { id:4, type:'trade',  icon:'trendDown',   color:'var(--red)',     title:'Stop-loss hit',     message:'JSWSTEEL closed at SL · −₹1,240 realized', created_at: new Date(Date.now()-60*60000).toISOString(),  is_read:true  },
  { id:5, type:'signal', icon:'brain',       color:'#8B5CF6',        title:'Signals refreshed', message:'498 stocks re-scored · 14 new BUY calls', created_at: new Date(Date.now()-2*3600000).toISOString(), is_read:true  },
  { id:6, type:'news',   icon:'news',        color:'var(--text-2)',  title:'Market news',       message:'RBI holds repo rate at 6.25%', created_at: new Date(Date.now()-3*3600000).toISOString(), is_read:true  },
];
