/* ===== Stock detail drawer (reused across pages) ===== */
function StockDrawer({ stock, onClose, toast }){
  const D=DATA, F=D.fmt;
  if(!stock) return null;
  const s = D.STOCKS.find(x=>x.symbol===stock.symbol) || stock;
  const col = s.signal==="BUY"?"var(--green)":s.signal==="SELL"?"var(--red)":"var(--gold)";
  const lineCol = s.change>=0?"var(--green)":"var(--red)";
  const news = D.stockNews(s.symbol);
  // per-horizon breakdown
  const breakdown = D.HORIZONS.map((h,i)=>{
    const seed=(s.id*7+i*13)%100;
    const sig = seed>62?"BUY":seed<22?"SELL":seed<40?"HOLD":"BUY";
    const conf = 52+((s.confidence+seed)%44);
    return { h, sig, conf };
  });
  useEffectU(()=>{ const onKey=e=>{ if(e.key==="Escape") onClose(); }; window.addEventListener("keydown",onKey); return ()=>window.removeEventListener("keydown",onKey); },[]);
  return (
    <>
      <div className="scrim" onClick={onClose}/>
      <aside className="drawer">
        <div className="drawer-head">
          <div className="row gap-sm">
            <span className="sym-logo" style={{width:42,height:42,fontSize:15,background:D.symColor(s.symbol)+"22",color:D.symColor(s.symbol)}}>{s.symbol.slice(0,2)}</span>
            <div className="col" style={{gap:1}}>
              <div className="row gap-sm"><span style={{fontWeight:700,fontSize:17}}>{s.symbol}</span><span className="pill">{s.sector}</span></div>
              <span style={{fontSize:12.5,color:"var(--text-2)"}}>{s.name}</span>
            </div>
          </div>
          <button className="drawer-x" onClick={onClose}><Icon name="x" size={18}/></button>
        </div>
        <div className="drawer-body">
          {/* price header */}
          <div className="row between" style={{alignItems:"flex-end"}}>
            <div className="col" style={{gap:2}}>
              <span className="mono" style={{fontSize:30,fontWeight:700,letterSpacing:"-.02em"}}>{F.inr(s.price)}</span>
              <Delta value={s.change} size={14}/>
            </div>
            <div className="col" style={{alignItems:"flex-end",gap:6}}>
              <SignalBadge signal={s.signal}/>
              <span style={{fontSize:12,color:"var(--text-3)"}}>Updated {F.fmtAgo(s.updatedMin)}</span>
            </div>
          </div>

          {/* price chart */}
          <div>
            <div className="row between" style={{marginBottom:8}}>
              <span className="card-title" style={{fontSize:13.5}}>Price · 1M</span>
              <div className="seg subtle" style={{transform:"scale(.92)",transformOrigin:"right"}}>{["1D","1W","1M","3M","1Y"].map((p,i)=><button key={p} className={i===2?"on":""}>{p}</button>)}</div>
            </div>
            <AreaChart data={s.spark} color={lineCol} h={170}/>
          </div>

          {/* key metrics */}
          <div className="metric-grid">
            <div><div className="ml">Mkt Cap</div><div className="mv">₹{s.mcap.toLocaleString("en-IN")} Cr</div></div>
            <div><div className="ml">P/E</div><div className="mv">{s.pe}</div></div>
            <div><div className="ml">Volume</div><div className="mv">{s.volume}M</div></div>
            <div><div className="ml">52W High</div><div className="mv">{F.inr(s.high52,0)}</div></div>
            <div><div className="ml">52W Low</div><div className="mv">{F.inr(s.low52,0)}</div></div>
            <div><div className="ml">Sentiment</div><div className="mv" style={{color:s.sentiment>=0?"var(--green)":"var(--red)"}}>{F.signed(s.sentiment)}</div></div>
          </div>

          {/* signal breakdown per horizon */}
          <div>
            <h4 className="card-title" style={{fontSize:13.5,marginBottom:12}}><span className="ic"><Icon name="layers" size={16}/></span>Signal breakdown by horizon</h4>
            <div className="col" style={{gap:9}}>
              {breakdown.map(b=>{ const bc=b.sig==="BUY"?"var(--green)":b.sig==="SELL"?"var(--red)":"var(--gold)";
                return (
                  <div key={b.h} className="hbar-row">
                    <span className="mono" style={{fontWeight:700,fontSize:12.5}}>{b.h}</span>
                    <div className="hbar-track"><div className="hbar-fill" style={{width:b.conf+"%",background:bc}}>{b.sig}</div></div>
                    <span className="mono" style={{fontSize:12,fontWeight:600,minWidth:34,textAlign:"right"}}>{b.conf}%</span>
                  </div>
                ); })}
            </div>
          </div>

          {/* news */}
          <div>
            <h4 className="card-title" style={{fontSize:13.5,marginBottom:6}}><span className="ic"><Icon name="news" size={16}/></span>News & sentiment</h4>
            {news.map((n,i)=>{ const nc=n.sent==="pos"?"var(--green)":n.sent==="neg"?"var(--red)":"var(--text-3)";
              return (
                <div key={i} className="news-item">
                  <span className="news-dot" style={{background:nc}}/>
                  <div className="col" style={{gap:3}}>
                    <span style={{fontSize:13,fontWeight:500,lineHeight:1.4}}>{n.title}</span>
                    <div className="news-meta"><span>{n.src}</span><span>·</span><span>{n.time}</span><span className={"tag-sent "+(n.sent==="neu"?"neu":n.sent==="pos"?"pos":"neg")}>{n.sent==="pos"?"Bullish":n.sent==="neg"?"Bearish":"Neutral"}</span></div>
                  </div>
                </div>
              ); })}
          </div>
        </div>
        <div style={{padding:"16px 22px",borderTop:"1px solid var(--border)",display:"flex",gap:10}}>
          <button className="btn btn-ghost" style={{flex:1}} onClick={()=>toast({type:"info",title:s.symbol+" added to watchlist"})}><Icon name="bookmark" size={17}/>Watchlist</button>
          <button className="btn btn-primary" style={{flex:2}} onClick={()=>{ toast({type:"success",title:"Order placed",msg:`${s.signal} ${s.symbol} @ ${F.inr(s.price)} · paper trade`}); onClose(); }}>
            <Icon name={s.signal==="SELL"?"arrowDown":"arrowUp"} size={17}/>{s.signal==="HOLD"?"Trade":s.signal} {s.symbol}
          </button>
        </div>
      </aside>
    </>
  );
}

/* ===== Signals Page ===== */
function SignalsPage({ openStock }){
  const D=DATA, F=D.fmt;
  const [loading,setLoading]=useStateU(true);
  const [horizon,setHorizon]=useStateU("All");
  const [sigType,setSigType]=useStateU("All");
  const [conf,setConf]=useStateU(50);
  const [sector,setSector]=useStateU("All");
  const [amount,setAmount]=useStateU(25000);
  const [q,setQ]=useStateU("");
  const [page,setPage]=useStateU(1);
  const { sort, toggle, apply } = useSort("confidence","desc");
  const perPage=12;
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),600); return ()=>clearTimeout(t); },[]);
  useEffectU(()=>{ setPage(1); },[horizon,sigType,conf,sector,q,sort]);
  const AMOUNTS=[10000,25000,50000,100000,200000];
  // per-row computed: qty affordable + expected ₹ P&L from expected return
  const calc=(s)=>{ const qty=Math.max(0,Math.floor(amount/s.price)); const deployed=qty*s.price; const expPnl=Math.round(deployed*s.expReturn/100); return { qty, deployed, expPnl }; };

  const filtered = apply(D.STOCKS.filter(s=>
    (horizon==="All"||s.horizon===horizon) &&
    (sigType==="All"||s.signal===sigType) &&
    s.confidence>=conf &&
    (sector==="All"||s.sector===sector) &&
    (q===""||s.symbol.toLowerCase().includes(q.toLowerCase())||s.name.toLowerCase().includes(q.toLowerCase()))
  ));
  const pages=Math.ceil(filtered.length/perPage)||1;
  const rows=filtered.slice((page-1)*perPage,page*perPage);

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">AI Signals</h1>
          <p className="page-sub">Machine-learning signals across all <b className="num">498</b> Nifty 500 constituents · refreshed every 15 min</p>
        </div>
        <div className="row gap-sm">
          <span className="badge buy"><Icon name="arrowUp" size={12}/>{D.STOCKS.filter(s=>s.signal==="BUY").length} BUY</span>
          <span className="badge sell"><Icon name="arrowDown" size={12}/>{D.STOCKS.filter(s=>s.signal==="SELL").length} SELL</span>
          <span className="badge hold"><Icon name="target" size={12}/>{D.STOCKS.filter(s=>s.signal==="HOLD").length} HOLD</span>
        </div>
      </div>

      {/* filters */}
      <Card className="" style={{marginBottom:"calc(16px * var(--u))"}}>
        <div className="card-pad" style={{display:"flex",flexDirection:"column",gap:14}}>
          <div className="filter-bar">
            <div className="nav-search" style={{maxWidth:280,flex:"1 1 220px"}}>
              <Icon name="search" size={17}/>
              <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search symbol or name…" style={{background:"var(--surface-2)"}}/>
            </div>
            <div className="field">
              <span className="field-label">Sector</span>
              <select className="input" value={sector} onChange={e=>setSector(e.target.value)}>
                <option>All</option>{D.SECTORS.map(s=><option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="field">
              <span className="field-label">Signal</span>
              <div className="seg">{["All","BUY","SELL","HOLD"].map(t=><button key={t} className={sigType===t?"on":""} onClick={()=>setSigType(t)}>{t}</button>)}</div>
            </div>
          </div>
          <div className="filter-bar">
            <div className="field">
              <span className="field-label">Horizon</span>
              <div className="seg">{["All",...D.HORIZONS].map(h=><button key={h} className={horizon===h?"on":""} onClick={()=>setHorizon(h)}>{h}</button>)}</div>
            </div>
            <div className="field" style={{flex:"1 1 220px",maxWidth:320}}>
              <span className="field-label">Min Confidence · {conf}%</span>
              <div className="range-wrap">
                <input type="range" className="rng" min="50" max="95" value={conf} onChange={e=>setConf(+e.target.value)} style={{flex:1}}/>
                <span className="mono" style={{fontWeight:700,minWidth:38}}>{conf}%</span>
              </div>
            </div>
            <div className="field">
              <span className="field-label">Investment / trade</span>
              <div className="row gap-sm">
                <div className="amount-input">
                  <span className="ai-prefix">₹</span>
                  <input className="input" type="number" value={amount} min="1000" step="1000" onChange={e=>setAmount(Math.max(0,+e.target.value))} style={{paddingLeft:24,width:120,minWidth:0}}/>
                </div>
                <div className="seg subtle">{AMOUNTS.map(a=><button key={a} className={amount===a?"on":""} onClick={()=>setAmount(a)}>{a>=100000?(a/100000)+"L":(a/1000)+"k"}</button>)}</div>
              </div>
            </div>
            <div className="field" style={{justifyContent:"flex-end"}}>
              <span className="field-label">&nbsp;</span>
              <span style={{fontSize:13,color:"var(--text-2)"}}><b className="num">{filtered.length}</b> stocks match</span>
            </div>
          </div>
        </div>
      </Card>

      {/* table */}
      <Card pad={false}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr>
              <Th label="Stock" k="symbol" sort={sort} toggle={toggle}/>
              <Th label="Sector" k="sector" sort={sort} toggle={toggle}/>
              <th>Signal</th>
              <Th label="Confidence" k="confidence" sort={sort} toggle={toggle}/>
              <th>Horizon</th>
              <Th label="Exp. Return" k="expReturn" sort={sort} toggle={toggle} align="right"/>
              <th style={{textAlign:"right"}}>Qty @ ₹{(amount/1000)}k</th>
              <th style={{textAlign:"right"}}>Exp. P&L (₹)</th>
              <Th label="Updated" k="updatedMin" sort={sort} toggle={toggle} align="right"/>
            </tr></thead>
            <tbody>
              {loading ? <SkeletonRows cols={9} rows={10}/> : rows.length===0 ? (
                <tr><td colSpan={9}><div className="empty">No signals match your filters. Try lowering the confidence threshold.</div></td></tr>
              ) : rows.map(s=>{ const c=calc(s);
                return (
                <tr key={s.id} className="clickable" onClick={()=>openStock(s)}>
                  <td><SymbolCell s={s} showSector={false}/></td>
                  <td><span className="pill" style={{color:D.SECTOR_COLORS[s.sector]}}>{s.sector}</span></td>
                  <td><SignalBadge signal={s.signal}/></td>
                  <td style={{minWidth:130}}><Conf value={s.confidence}/></td>
                  <td><span className="pill">{s.horizon}</span></td>
                  <td className="num-cell" style={{textAlign:"right",color:s.expReturn>=0?"var(--green)":"var(--red)",fontWeight:600}}>{F.pct(s.expReturn)}</td>
                  <td className="num-cell" style={{textAlign:"right"}}><span>{c.qty}</span><span className="dim" style={{fontSize:11,display:"block"}}>{F.inr(c.deployed,0)}</span></td>
                  <td className="num-cell" style={{textAlign:"right",fontWeight:700,color:c.expPnl>=0?"var(--green)":"var(--red)"}}>{c.qty===0?<span className="dim" style={{fontWeight:400}}>—</span>:F.signed(c.expPnl,0)}</td>
                  <td className="num-cell dim" style={{textAlign:"right",fontSize:12}}>{F.fmtAgo(s.updatedMin)}</td>
                </tr>
              ); })}
            </tbody>
          </table>
        </div>
        <Pager page={page} pages={pages} total={filtered.length} perPage={perPage} onPage={setPage} label="signals"/>
      </Card>
    </div>
  );
}

Object.assign(window, { StockDrawer, SignalsPage });
