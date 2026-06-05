/* ===== Signal Card (3 style variants, driven by tweak) ===== */
function SignalCard({ s, variant="rich", onClick }){
  const col = s.signal==="BUY"?"var(--green)":s.signal==="SELL"?"var(--red)":"var(--gold)";
  const line = s.signal==="SELL"?"var(--red)":s.signal==="HOLD"?"var(--gold)":"var(--green)";

  if(variant==="compact"){
    return (
      <button className="sigc compact" onClick={onClick}>
        <SymbolCell s={s}/>
        <div className="row gap-sm"><SignalBadge signal={s.signal}/><span className="pill">{s.horizon}</span></div>
        <div className="col" style={{alignItems:"flex-end",gap:2}}>
          <span className="mono" style={{fontWeight:700,fontSize:13.5}}>{DATA.fmt.inr(s.price)}</span>
          <Delta value={s.change} size={11.5}/>
        </div>
        <div style={{width:64}}><Conf value={s.confidence}/></div>
      </button>
    );
  }
  if(variant==="bold"){
    return (
      <button className="sigc bold" onClick={onClick} style={{borderLeft:`3px solid ${col}`}}>
        <div className="row between">
          <SymbolCell s={s}/>
          <SignalBadge signal={s.signal}/>
        </div>
        <div className="row between" style={{alignItems:"flex-end",marginTop:14}}>
          <div className="col" style={{gap:1}}>
            <span style={{fontSize:11,color:"var(--text-3)",fontWeight:600,textTransform:"uppercase",letterSpacing:".04em"}}>Expected · {s.horizon}</span>
            <span className="mono" style={{fontSize:24,fontWeight:700,color:col,letterSpacing:"-.02em"}}>{DATA.fmt.pct(s.expReturn)}</span>
          </div>
          <Sparkline data={s.spark} color={line} w={92} h={42} fill sw={2.2}/>
        </div>
        <div className="row between" style={{marginTop:12}}>
          <span style={{fontSize:11.5,color:"var(--text-2)"}}>Confidence</span>
          <span className="mono" style={{fontWeight:700,color:col}}>{s.confidence}%</span>
        </div>
        <div className="conf-track" style={{marginTop:5}}><div className="conf-fill" style={{width:s.confidence+"%",background:col}}/></div>
      </button>
    );
  }
  // rich (default)
  return (
    <button className="sigc rich" onClick={onClick}>
      <div className="row between">
        <SymbolCell s={s}/>
        <SignalBadge signal={s.signal}/>
      </div>
      <div className="row between" style={{margin:"13px 0 11px",alignItems:"center"}}>
        <div className="col" style={{gap:1}}>
          <span className="mono" style={{fontSize:18,fontWeight:700}}>{DATA.fmt.inr(s.price)}</span>
          <Delta value={s.change} size={12}/>
        </div>
        <Sparkline data={s.spark} color={line} w={88} h={38} fill/>
      </div>
      <div className="divider"/>
      <div className="row between" style={{marginTop:11}}>
        <div className="col" style={{gap:2}}>
          <span className="dim" style={{fontSize:11}}>Horizon</span>
          <span className="pill">{s.horizon}</span>
        </div>
        <div className="col" style={{gap:2,alignItems:"flex-end"}}>
          <span className="dim" style={{fontSize:11}}>Exp. Return</span>
          <span className="mono" style={{fontWeight:700,color:col,fontSize:13.5}}>{DATA.fmt.pct(s.expReturn)}</span>
        </div>
      </div>
      <div className="row between" style={{marginTop:11,gap:10}}>
        <span className="dim" style={{fontSize:11,whiteSpace:"nowrap"}}>Confidence</span>
        <div style={{flex:1}}><Conf value={s.confidence}/></div>
      </div>
    </button>
  );
}

/* ===== Dashboard ===== */
function StatCard({ label, value, delta, deltaSuffix="%", icon, color, spark, sparkColor }){
  return (
    <div className="stat">
      <div className="stat-top">
        <span className="stat-label">{label}</span>
        <span className="stat-ic" style={{background:color+"1f",color}}><Icon name={icon} size={18}/></span>
      </div>
      <div className="stat-val">{value}</div>
      {delta!=null && <span className="stat-delta" style={{color:delta>=0?"var(--green)":"var(--red)"}}>
        <Icon name={delta>=0?"arrowUpRight":"arrowDown"} size={13}/>{(delta>=0?"+":"")+delta+deltaSuffix}<span className="dim" style={{fontWeight:500,marginLeft:2}}>today</span>
      </span>}
      {spark && <div className="spark-bg"><Sparkline data={spark} color={sparkColor||color} w={300} h={38} fill sw={1.6}/></div>}
    </div>
  );
}

function Dashboard({ openStock, toast, tweaks }){
  const [loading,setLoading]=useStateU(true);
  const [refreshing,setRefreshing]=useStateU(false);
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),650); return ()=>clearTimeout(t); },[]);
  const D=DATA; const F=D.fmt;
  const nifty=D.INDICES[0];
  const recent=D.TRADE_HISTORY.slice(0,6);
  const sigVar = tweaks.signalStyle;

  function refresh(){ setRefreshing(true); setLoading(true);
    setTimeout(()=>{ setRefreshing(false); setLoading(false); toast({type:"success",title:"Signals refreshed",msg:"498 stocks re-scored · 14 new BUY signals"}); },900); }

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Good afternoon, Arjun 👋</h1>
          <p className="page-sub">Your AI engine scanned <b className="num">498</b> Nifty 500 stocks · last run {F.fmtAgo(8)}</p>
        </div>
        <div className="row gap-sm wrap">
          <button className="btn btn-ghost" onClick={()=>toast({type:"info",title:"Added to Watchlist",msg:"Top 5 signals saved to your watchlist"})}><Icon name="bookmark" size={17}/>Add to Watchlist</button>
          <button className="btn btn-gold" onClick={refresh}><Icon name="refresh" size={17} style={{animation:refreshing?"spin 1s linear infinite":"none"}}/>Refresh Signals</button>
        </div>
      </div>

      {/* stats */}
      <div className="grid stat-grid" style={{marginBottom:"calc(16px * var(--u))"}}>
        <StatCard label="Portfolio Value" value={F.inrCompact(D.PF_CURRENT)} delta={1.84} icon="wallet" color="var(--accent)" spark={nifty.spark} sparkColor="var(--accent)"/>
        <StatCard label="Today's P&L" value={F.signed(Math.round(D.PF_PNL*0.06))} delta={1.21} icon="trendUp" color="var(--green)" spark={D.INDICES[1].spark} sparkColor="var(--green)"/>
        <StatCard label="Active Positions" value={D.OPEN_POS.length} icon="layers" color="var(--gold)"/>
        <StatCard label="Win Rate" value="68.4%" delta={2.3} icon="target" color="#8B5CF6"/>
      </div>

      {/* index + sentiment */}
      <div className="grid" style={{gridTemplateColumns:"1.7fr 1fr",marginBottom:"calc(16px * var(--u))"}}>
        <Card title="NIFTY 50" sub="NSE · Intraday" icon="market" pad={false}
          right={<div className="row gap-lg"><div className="col" style={{alignItems:"flex-end",gap:0}}>
            <span className="mono" style={{fontSize:20,fontWeight:700}}>{nifty.value.toLocaleString("en-IN")}</span>
            <span style={{fontSize:12.5,fontWeight:600,color:"var(--green)"}}>+{nifty.change} ({F.pct(nifty.pct)})</span>
          </div></div>}>
          <div className="card-pad" style={{paddingTop:8}}>
            {loading ? <Skeleton h={210}/> : <AreaChart data={nifty.spark} color="var(--green)" h={210} labels={["9:15","11:00","12:45","14:30","15:30"]}/>}
            <div className="row gap-sm wrap" style={{marginTop:12}}>
              {D.INDICES.map(ix=>(
                <div key={ix.name} className="row gap-sm" style={{padding:"7px 11px",borderRadius:9,background:"var(--surface-2)",fontSize:12}}>
                  <span style={{color:"var(--text-2)",fontWeight:600}}>{ix.name}</span>
                  <span className="mono" style={{fontWeight:700}}>{ix.value.toLocaleString("en-IN")}</span>
                  <span style={{color:ix.pct>=0?"var(--green)":"var(--red)",fontWeight:600}} className="num">{F.pct(ix.pct)}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>
        <Card title="Market Sentiment" sub="Driven by news + flows" icon="brain">
          <div className="card-pad" style={{display:"flex",flexDirection:"column",alignItems:"center",gap:14}}>
            {loading ? <Skeleton w={190} h={140} r={12}/> : <Gauge value={D.SENTIMENT_SCORE}/>}
            <div className="row" style={{width:"100%",justifyContent:"space-around",borderTop:"1px solid var(--border)",paddingTop:13}}>
              <div className="col" style={{alignItems:"center",gap:1}}><span className="mono" style={{fontWeight:700,color:"var(--green)"}}>{D.BREADTH.advances}</span><span className="dim" style={{fontSize:11}}>Advances</span></div>
              <div className="col" style={{alignItems:"center",gap:1}}><span className="mono" style={{fontWeight:700,color:"var(--red)"}}>{D.BREADTH.declines}</span><span className="dim" style={{fontSize:11}}>Declines</span></div>
              <div className="col" style={{alignItems:"center",gap:1}}><span className="mono" style={{fontWeight:700,color:"var(--gold)"}}>{(D.BREADTH.advances/D.BREADTH.declines).toFixed(2)}</span><span className="dim" style={{fontSize:11}}>A/D Ratio</span></div>
            </div>
          </div>
        </Card>
      </div>

      {/* top signals */}
      <Card title="Top AI Signals Today" sub="Highest-confidence calls across Nifty 500" icon="sparkle"
        right={<button className="btn btn-ghost btn-sm" onClick={()=>window.__setPage("signals")}>View all 498<Icon name="chevRight" size={15}/></button>}>
        <div className="card-pad">
          {loading ? (
            <div className="sig-grid"><div className="row gap-sm" style={{flexWrap:"wrap"}}>{Array.from({length:5}).map((_,i)=><Skeleton key={i} w="220px" h={sigVar==="compact"?60:190} r={12}/>)}</div></div>
          ) : (
            <div className={sigVar==="compact"?"sig-list":"sig-grid"}>
              {D.TOP_SIGNALS.map(s=><SignalCard key={s.id} s={s} variant={sigVar} onClick={()=>openStock(s)}/>)}
            </div>
          )}
        </div>
      </Card>

      {/* recent trades */}
      <div style={{marginTop:"calc(16px * var(--u))"}}>
        <Card title="Recent Trades" sub="Latest executions across your accounts" icon="trades" pad={false}
          right={<button className="btn btn-ghost btn-sm" onClick={()=>window.__setPage("trades")}>All trades<Icon name="chevRight" size={15}/></button>}>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Side</th><th style={{textAlign:"right"}}>Entry Price</th><th style={{textAlign:"right"}}>Current</th><th style={{textAlign:"right"}}>P&L</th><th style={{textAlign:"right"}}>P&L %</th><th>AI Signal</th></tr></thead>
              <tbody>
                {loading ? <SkeletonRows cols={7}/> : recent.map(t=>{ const s=D.STOCKS.find(x=>x.symbol===t.symbol); const cur=s.price; const pnlpct=((cur-t.price)/t.price)*100;
                  return (
                    <tr key={t.id} className="clickable" onClick={()=>openStock(s)}>
                      <td><SymbolCell s={t}/></td>
                      <td><span className={"pill"} style={{color:t.side==="BUY"?"var(--green)":"var(--red)"}}>{t.side}</span></td>
                      <td className="num-cell" style={{textAlign:"right"}}>{F.inr(t.price)}</td>
                      <td className="num-cell" style={{textAlign:"right"}}>{F.inr(cur)}</td>
                      <td className="num-cell" style={{textAlign:"right",color:pnlpct>=0?"var(--green)":"var(--red)"}}>{F.signed(Math.round((cur-t.price)*t.qty))}</td>
                      <td style={{textAlign:"right"}}><Delta value={pnlpct} size={12.5} icon={false}/></td>
                      <td><SignalBadge signal={s.signal}/></td>
                    </tr>
                  ); })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

Object.assign(window, { Dashboard, SignalCard, StatCard });
