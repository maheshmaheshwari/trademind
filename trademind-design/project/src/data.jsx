/* ===== Mock market data — deterministic (seeded) ===== */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const rng = mulberry32(20260601);
const rand=(a,b)=>a+(b-a)*rng();
const randInt=(a,b)=>Math.floor(rand(a,b+1));
const pick=arr=>arr[Math.floor(rng()*arr.length)];

const SECTORS = ["IT","Banking","Financials","Energy","Auto","FMCG","Pharma","Metals","Cement","Infra","Telecom","Power"];
const SECTOR_COLORS = {
  IT:"#3B82F6", Banking:"#8B5CF6", Financials:"#6366F1", Energy:"#F59E0B", Auto:"#EC4899",
  FMCG:"#10B981", Pharma:"#14B8A6", Metals:"#F97316", Cement:"#A78BFA", Infra:"#0EA5E9",
  Telecom:"#EF4444", Power:"#EAB308"
};

// symbol → brand-ish color for the square avatar
function symColor(sym){ const h=[...sym].reduce((a,c)=>a+c.charCodeAt(0),0); const palette=["#3B82F6","#8B5CF6","#10B981","#F59E0B","#EC4899","#14B8A6","#6366F1","#F97316","#0EA5E9","#EF4444","#A78BFA","#22C55E"]; return palette[h%palette.length]; }

// curated real-ish Nifty constituents (name, sector)
const BASE = [
  ["RELIANCE","Reliance Industries","Energy",2912],["TCS","Tata Consultancy","IT",4087],
  ["HDFCBANK","HDFC Bank","Banking",1684],["INFY","Infosys","IT",1832],
  ["ICICIBANK","ICICI Bank","Banking",1247],["HINDUNILVR","Hindustan Unilever","FMCG",2398],
  ["ITC","ITC Ltd","FMCG",436],["SBIN","State Bank of India","Banking",842],
  ["BHARTIARTL","Bharti Airtel","Telecom",1589],["KOTAKBANK","Kotak Mahindra Bank","Banking",1763],
  ["LT","Larsen & Toubro","Infra",3621],["BAJFINANCE","Bajaj Finance","Financials",7218],
  ["HCLTECH","HCL Technologies","IT",1894],["ASIANPAINT","Asian Paints","FMCG",2287],
  ["AXISBANK","Axis Bank","Banking",1142],["MARUTI","Maruti Suzuki","Auto",12640],
  ["SUNPHARMA","Sun Pharma","Pharma",1812],["TITAN","Titan Company","FMCG",3398],
  ["ULTRACEMCO","UltraTech Cement","Cement",11240],["WIPRO","Wipro","IT",548],
  ["NESTLEIND","Nestle India","FMCG",2456],["ONGC","ONGC","Energy",268],
  ["NTPC","NTPC","Power",412],["POWERGRID","Power Grid Corp","Power",324],
  ["TATAMOTORS","Tata Motors","Auto",978],["TATASTEEL","Tata Steel","Metals",164],
  ["JSWSTEEL","JSW Steel","Metals",942],["ADANIENT","Adani Enterprises","Infra",2876],
  ["ADANIPORTS","Adani Ports","Infra",1421],["COALINDIA","Coal India","Energy",478],
  ["GRASIM","Grasim Industries","Cement",2614],["BAJAJFINSV","Bajaj Finserv","Financials",1718],
  ["TECHM","Tech Mahindra","IT",1672],["HDFCLIFE","HDFC Life Insurance","Financials",682],
  ["DRREDDY","Dr Reddy's Labs","Pharma",1289],["CIPLA","Cipla","Pharma",1564],
  ["BRITANNIA","Britannia Industries","FMCG",4912],["EICHERMOT","Eicher Motors","Auto",4847],
  ["DIVISLAB","Divi's Laboratories","Pharma",5928],["HEROMOTOCO","Hero MotoCorp","Auto",4621],
  ["BPCL","Bharat Petroleum","Energy",312],["INDUSINDBK","IndusInd Bank","Banking",984],
  ["SBILIFE","SBI Life Insurance","Financials",1542],["APOLLOHOSP","Apollo Hospitals","Pharma",6824],
  ["BAJAJ-AUTO","Bajaj Auto","Auto",9218],["TATACONSUM","Tata Consumer","FMCG",1098],
  ["HINDALCO","Hindalco Industries","Metals",642],["MM","Mahindra & Mahindra","Auto",2891],
  ["SHREECEM","Shree Cement","Cement",26840],["DLF","DLF Ltd","Infra",812],
  ["VEDL","Vedanta","Metals",448],["GAIL","GAIL India","Energy",214],
  ["DABUR","Dabur India","FMCG",512],["GODREJCP","Godrej Consumer","FMCG",1284],
  ["PIDILITIND","Pidilite Industries","FMCG",2962],["AMBUJACEM","Ambuja Cements","Cement",614],
  ["SIEMENS","Siemens","Infra",6842],["BANKBARODA","Bank of Baroda","Banking",248],
  ["PNB","Punjab National Bank","Banking",104],["IOC","Indian Oil Corp","Energy",168],
  ["MOTHERSON","Samvardhana Motherson","Auto",182],["LICI","LIC of India","Financials",982],
  ["VARROC","Varroc Engineering","Auto",542],["TVSMOTOR","TVS Motor","Auto",2418],
  ["ZOMATO","Zomato","IT",264],["PAYTM","One97 Communications","IT",648],
  ["NYKAA","FSN E-Commerce","FMCG",182],["IRCTC","IRCTC","Infra",812],
  ["LTIM","LTIMindtree","IT",5912],["PERSISTENT","Persistent Systems","IT",5824],
  ["COFORGE","Coforge","IT",6248],["MPHASIS","Mphasis","IT",2814],
  ["TORNTPHARM","Torrent Pharma","Pharma",3142],["LUPIN","Lupin","Pharma",2089],
  ["AUROPHARMA","Aurobindo Pharma","Pharma",1342],["BIOCON","Biocon","Pharma",348],
  ["JINDALSTEL","Jindal Steel","Metals",924],["SAIL","SAIL","Metals",128],
  ["NMDC","NMDC","Metals",214],["ACC","ACC Ltd","Cement",2184],
  ["DMART","Avenue Supermarts","FMCG",4218],["TRENT","Trent Ltd","FMCG",6842],
  ["NAUKRI","Info Edge","IT",7842],["INDIGO","InterGlobe Aviation","Infra",4621],
  ["CANBK","Canara Bank","Banking",102],["FEDERALBNK","Federal Bank","Banking",184],
  ["IDFCFIRSTB","IDFC First Bank","Banking",68],["RECLTD","REC Ltd","Financials",512],
  ["PFC","Power Finance Corp","Financials",468],["IRFC","Indian Railway Finance","Financials",148],
];

const HORIZONS = ["1W","2W","1M","2M","3M","6M"];
const SIGNAL_TYPES = ["BUY","SELL","HOLD"];

function sparkline(n, trend){
  const pts=[]; let v=50+rand(-8,8);
  for(let i=0;i<n;i++){ v+=trend*rand(.2,1.4)+rand(-3.2,3.2); v=Math.max(8,Math.min(92,v)); pts.push(v); }
  return pts;
}

const STOCKS = BASE.map((b,i)=>{
  const [symbol,name,sector,price]=b;
  const bias = pick([1,1,-1,1,-1,0]); // more buys
  const signal = bias>0?"BUY":bias<0?"SELL":"HOLD";
  const trend = signal==="BUY"?1:signal==="SELL"?-1:rand(-.4,.4);
  const change = +(trend*rand(.3,3.8)+rand(-.6,.6)).toFixed(2);
  const confidence = randInt(signal==="HOLD"?52:64, 96);
  const horizon = pick(HORIZONS);
  const expReturn = +(signal==="BUY"?rand(3,28):signal==="SELL"?-rand(3,22):rand(-3,4)).toFixed(1);
  const sentiment = +(trend*rand(.1,.7)+rand(-.2,.2)).toFixed(2);
  const mins = randInt(2,180);
  return { id:i, symbol, name, sector, price:+price.toFixed(2), change,
    signal, confidence, horizon, expReturn, sentiment,
    updatedMin: mins, spark: sparkline(28, trend),
    mcap: +(price*randInt(40,900)/100).toFixed(0),
    volume: randInt(2,180), high52:+(price*rand(1.05,1.4)).toFixed(0), low52:+(price*rand(.6,.92)).toFixed(0),
    pe: +rand(12,68).toFixed(1) };
});

function fmtAgo(m){ if(m<60) return m+"m ago"; const h=Math.floor(m/60); return h+"h ago"; }
function inr(n,dec=2){ return "₹"+Number(n).toLocaleString("en-IN",{minimumFractionDigits:dec,maximumFractionDigits:dec}); }
function inrCompact(n){ const a=Math.abs(n); if(a>=1e7) return "₹"+(n/1e7).toFixed(2)+" Cr"; if(a>=1e5) return "₹"+(n/1e5).toFixed(2)+" L"; return "₹"+n.toLocaleString("en-IN"); }
function pct(n){ return (n>0?"+":"")+n.toFixed(2)+"%"; }
function signed(n,dec=2){ return (n>0?"+":"")+Number(n).toLocaleString("en-IN",{minimumFractionDigits:dec,maximumFractionDigits:dec}); }

const TOP_SIGNALS = STOCKS.filter(s=>s.signal!=="HOLD").sort((a,b)=>b.confidence-a.confidence).slice(0,5);

// ---- Watchlist ----
const WATCH_SYMS = ["RELIANCE","INFY","TATAMOTORS","HDFCBANK","ZOMATO","DMART","TRENT","BHARTIARTL","SUNPHARMA","LTIM","ADANIENT","TITAN"];
const WATCHLIST = WATCH_SYMS.map(sym=>{ const s=STOCKS.find(x=>x.symbol===sym);
  return { ...s, alertAbove:+(s.price*rand(1.03,1.09)).toFixed(0), alertBelow:+(s.price*rand(.9,.96)).toFixed(0), addedAgo:fmtAgo(randInt(120,9000)) };
});

// ---- Portfolio ----
const HOLD_SYMS = ["RELIANCE","TCS","HDFCBANK","INFY","TATAMOTORS","BHARTIARTL","SUNPHARMA","JSWSTEEL","TITAN","LTIM"];
const HOLDINGS = HOLD_SYMS.map(sym=>{
  const s=STOCKS.find(x=>x.symbol===sym);
  const qty = randInt(5,140);
  const avg = +(s.price*rand(.72,1.08)).toFixed(2);
  const cmp = s.price;
  const invested = qty*avg, current = qty*cmp;
  return { ...s, qty, avg, cmp, invested:+invested.toFixed(0), current:+current.toFixed(0),
    pnl:+(current-invested).toFixed(0), pnlPct:+(((cmp-avg)/avg)*100).toFixed(2) };
});
const PF_INVESTED = HOLDINGS.reduce((a,h)=>a+h.invested,0);
const PF_CURRENT = HOLDINGS.reduce((a,h)=>a+h.current,0);
const PF_PNL = PF_CURRENT-PF_INVESTED;

// allocation by sector
const ALLOC = (()=>{ const m={}; HOLDINGS.forEach(h=>{m[h.sector]=(m[h.sector]||0)+h.current;}); return Object.entries(m).map(([sector,val])=>({sector,val,color:SECTOR_COLORS[sector]})).sort((a,b)=>b.val-a.val); })();

// P&L history line (30d / 90d / 1Y)
function pnlSeries(n, end){ const pts=[]; let v=end*rand(.55,.7); const step=(end-v)/n; for(let i=0;i<n;i++){ v+=step+rand(-end*.018,end*.022); pts.push(Math.max(0,v)); } pts[pts.length-1]=end; return pts; }
const PNL_HISTORY = { "30D":pnlSeries(30,PF_CURRENT), "90D":pnlSeries(90,PF_CURRENT), "1Y":pnlSeries(52,PF_CURRENT) };

// ---- Open positions ----
const OPEN_POS = ["RELIANCE","TATAMOTORS","JSWSTEEL","BHARTIARTL","SUNPHARMA","LTIM"].map(sym=>{
  const s=STOCKS.find(x=>x.symbol===sym);
  const entry=+(s.price*rand(.88,.98)).toFixed(2);
  const sl=+(entry*rand(.9,.96)).toFixed(2);
  const target=+(entry*rand(1.06,1.18)).toFixed(2);
  const qty=randInt(10,120);
  return { ...s, entry, sl, target, qty, pnl:+((s.price-entry)*qty).toFixed(0), pnlPct:+(((s.price-entry)/entry)*100).toFixed(2), days:randInt(1,42) };
});

// ---- Trade history ----
const TRADE_HISTORY = (()=>{
  const out=[]; const today=new Date(2026,5,1);
  for(let i=0;i<46;i++){
    const s=pick(STOCKS); const side=pick(["BUY","SELL"]);
    const d=new Date(today); d.setDate(d.getDate()-randInt(1,140));
    const qty=randInt(5,150); const pr=+(s.price*rand(.8,1.15)).toFixed(2);
    const realized=+((rng()>.42?1:-1)*s.price*qty*rand(.005,.09)).toFixed(0);
    out.push({ id:i, symbol:s.symbol, name:s.name, sector:s.sector, side, qty, price:pr,
      value:+(qty*pr).toFixed(0), date:d, realized, status:"EXECUTED" });
  }
  return out.sort((a,b)=>b.date-a.date);
})();

// ---- GTT orders ----
const GTT = ["INFY","HDFCBANK","TITAN","COALINDIA","DRREDDY"].map((sym,i)=>{
  const s=STOCKS.find(x=>x.symbol===sym);
  const type=pick(["Single","OCO"]);
  const trigger=+(s.price*rand(.92,1.08)).toFixed(2);
  return { id:i, symbol:s.symbol, name:s.name, type, side:pick(["BUY","SELL"]),
    trigger, ltp:s.price, qty:randInt(10,80), status:pick(["ACTIVE","ACTIVE","TRIGGERED","EXPIRED"]),
    created:fmtAgo(randInt(120,4000)) };
});

// ---- AI Authorized Trades (autopilot mandates executed via Angel One) ----
// every trade the user has authorized the AI engine to place automatically
const AUTH_SYMS = ["RELIANCE","TCS","INFY","HDFCBANK","TATAMOTORS","BHARTIARTL","SUNPHARMA","JSWSTEEL","TITAN","LTIM","ICICIBANK","MARUTI","BAJFINANCE","DRREDDY"];
const AUTH_TRADES = AUTH_SYMS.map((sym,i)=>{
  const s=STOCKS.find(x=>x.symbol===sym);
  const amount = pick([10000,15000,25000,30000,50000,75000,100000]);
  const entry = +(s.price*rand(.9,1.0)).toFixed(2);
  const qty = Math.max(1, Math.floor(amount/entry));
  const target = +(entry*(1+Math.abs(s.expReturn)/100)).toFixed(2);
  const stop = +(entry*rand(.92,.97)).toFixed(2);
  // status mix: executed (running), pending (awaiting trigger), completed (target hit), stopped
  const roll = (s.id*13+i*7)%100;
  const status = roll<46?"EXECUTED":roll<70?"PENDING":roll<88?"COMPLETED":"STOPPED";
  const expProfit = +((target-entry)*qty).toFixed(0);
  const maxLoss = +((entry-stop)*qty).toFixed(0);
  let actualPnl=null, exitPrice=null;
  if(status==="EXECUTED") actualPnl=+((s.price-entry)*qty).toFixed(0);
  else if(status==="COMPLETED"){ exitPrice=target; actualPnl=expProfit; }
  else if(status==="STOPPED"){ exitPrice=stop; actualPnl=-maxLoss; }
  return { id:i, symbol:s.symbol, name:s.name, sector:s.sector, signal:s.signal, confidence:s.confidence,
    horizon:s.horizon, amount, entry, qty, target, stop, cmp:s.price, status,
    expProfit, maxLoss, actualPnl, exitPrice, expReturn:s.expReturn,
    mode: i%4===0?"Paper":"Live", broker:"Angel One",
    authAgo: fmtAgo(randInt(60,7000)), spark:s.spark };
});
const AUTH_CAPITAL = AUTH_TRADES.reduce((a,t)=>a+t.amount,0);
const AUTH_ACTIVE = AUTH_TRADES.filter(t=>t.status==="EXECUTED"||t.status==="PENDING").length;
const AUTH_REALIZED = AUTH_TRADES.filter(t=>t.actualPnl!=null).reduce((a,t)=>a+t.actualPnl,0);
const AUTH_EXP_PROFIT = AUTH_TRADES.filter(t=>t.status==="EXECUTED"||t.status==="PENDING").reduce((a,t)=>a+t.expProfit,0);

// ---- Market overview ----
const INDICES = [
  { name:"NIFTY 50", value:24862.30, change:142.65, pct:0.58, spark:sparkline(40,1) },
  { name:"NIFTY 500", value:22918.75, change:118.40, pct:0.52, spark:sparkline(40,1) },
  { name:"SENSEX", value:81472.18, change:421.30, pct:0.52, spark:sparkline(40,1) },
  { name:"INDIA VIX", value:13.42, change:-0.86, pct:-6.02, spark:sparkline(40,-1) },
];

const FII_DII = (()=>{ const out=[]; const today=new Date(2026,5,1); const labels=["Mon","Tue","Wed","Thu","Fri","Mon","Tue","Wed","Thu","Fri"];
  for(let i=9;i>=0;i--){ out.push({ day:labels[9-i], fii:+rand(-4200,5400).toFixed(0), dii:+rand(-2600,4800).toFixed(0) }); } return out; })();

const HEATMAP = SECTORS.map(s=>{ const c=+(rand(-3.4,3.8)).toFixed(2); return { sector:s, change:c, mcap:randInt(2,28) }; });

const GAINERS = [...STOCKS].sort((a,b)=>b.change-a.change).slice(0,5);
const LOSERS = [...STOCKS].sort((a,b)=>a.change-b.change).slice(0,5);

const BREADTH = { advances: 1842, declines: 1264, unchanged: 122 };
const SENTIMENT_SCORE = 68; // 0-100 bullish

// ---- Notifications (navbar bell) ----
const NOTIFS = [
  { id:1, type:"trade", icon:"checkCircle", color:"var(--green)", title:"Order executed", msg:"BUY 40 × RELIANCE filled @ ₹2,912.00", time:"2m", unread:true },
  { id:2, type:"signal", icon:"sparkle", color:"var(--gold)", title:"New AI signal", msg:"INFY upgraded to BUY · 88% confidence", time:"18m", unread:true },
  { id:3, type:"price", icon:"bell", color:"var(--accent-2)", title:"Price alert", msg:"TATAMOTORS crossed ₹980 (your target)", time:"41m", unread:true },
  { id:4, type:"trade", icon:"trendDown", color:"var(--red)", title:"Stop-loss hit", msg:"JSWSTEEL closed at SL · −₹1,240 realized", time:"1h", unread:false },
  { id:5, type:"signal", icon:"brain", color:"#8B5CF6", title:"Signals refreshed", msg:"498 stocks re-scored · 14 new BUY calls", time:"2h", unread:false },
  { id:6, type:"news", icon:"news", color:"var(--text-2)", title:"Market news", msg:"RBI holds repo rate at 6.25%", time:"3h", unread:false },
];

const NEWS = [
  { src:"Economic Times", time:"32m", sent:"pos", title:"RBI holds repo rate at 6.25%, signals dovish stance for H2 FY26" },
  { src:"Mint", time:"1h", sent:"pos", title:"FII inflows top ₹12,400 Cr this week as global funds rotate into India" },
  { src:"Moneycontrol", time:"2h", sent:"neu", title:"Q4 earnings season: IT majors guide for muted but stable growth" },
  { src:"Business Standard", time:"3h", sent:"neg", title:"Metal stocks slip as China demand outlook weakens for Q3" },
  { src:"Bloomberg Quint", time:"4h", sent:"pos", title:"Auto sales hit record high in May, two-wheelers lead recovery" },
];

function stockNews(sym){
  const tmpl=[
    ["pos","Brokerage upgrades to BUY, raises target on strong order book"],
    ["pos","Q4 net profit beats estimates, margins expand 180 bps YoY"],
    ["neu","Management reiterates FY26 guidance in analyst call"],
    ["neg","Promoter pledge rises marginally; analysts flag caution"],
    ["pos","New capacity expansion announced, capex of ₹4,200 Cr"],
  ];
  return tmpl.map((t,i)=>({ sent:t[0], title:t[1], src:pick(["ET","Mint","MC","BQ"]), time:fmtAgo(randInt(20,600)) }));
}

window.DATA = {
  SECTORS, SECTOR_COLORS, symColor, STOCKS, HORIZONS, SIGNAL_TYPES, TOP_SIGNALS,
  HOLDINGS, PF_INVESTED, PF_CURRENT, PF_PNL, ALLOC, PNL_HISTORY, WATCHLIST,
  OPEN_POS, TRADE_HISTORY, GTT, AUTH_TRADES, AUTH_CAPITAL, AUTH_ACTIVE, AUTH_REALIZED, AUTH_EXP_PROFIT, INDICES, FII_DII, HEATMAP, GAINERS, LOSERS,
  BREADTH, SENTIMENT_SCORE, NEWS, NOTIFS, stockNews,
  fmt: { fmtAgo, inr, inrCompact, pct, signed }
};
