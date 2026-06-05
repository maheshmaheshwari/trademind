/* ===== Market Overview ===== */
function IndexCard({ ix, F }){
  const pos=ix.pct>=0; const col=pos?"var(--green)":"var(--red)";
  return (
    <div className="card card-pad" style={{position:"relative",overflow:"hidden"}}>
      <div className="row between">
        <span style={{fontWeight:700,fontSize:13.5,letterSpacing:".01em",whiteSpace:"nowrap"}}>{ix.name}</span>
        <span className="pill" style={{color:col,background:pos?"var(--green-soft)":"var(--red-soft)",border:"none"}}>{F.pct(ix.pct)}</span>
      </div>
      <div className="mono" style={{fontSize:23,fontWeight:700,margin:"9px 0 2px"}}>{ix.value.toLocaleString("en-IN")}</div>
      <div style={{fontSize:12.5,fontWeight:600,color:col}} className="num">{(pos?"+":"")+ix.change}</div>
      <div style={{marginTop:10,marginLeft:-2,marginRight:-2}}><Sparkline data={ix.spark} color={col} w={260} h={40} fill sw={2}/></div>
    </div>
  );
}

function MarketPage({ openStock }){
  const D=DATA, F=D.fmt;
  const [loading,setLoading]=useStateU(true);
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),600); return ()=>clearTimeout(t); },[]);
  const heat=[...D.HEATMAP].sort((a,b)=>b.change-a.change);
  function heatColor(c){ const a=Math.min(Math.abs(c)/3.5,1); return c>=0?`rgba(16,185,129,${.12+a*.5})`:`rgba(239,68,68,${.12+a*.5})`; }

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Market Overview</h1>
          <p className="page-sub">Live indices, institutional flows & sector rotation · NSE</p>
        </div>
        <div className="mkt-status open"><span className="dot"/>MARKET OPEN<span style={{color:"var(--text-3)",fontWeight:500,fontFamily:"var(--font-mono)",fontSize:11.5}}>15:24:08</span></div>
      </div>

      {/* index cards */}
      <div className="grid stat-grid" style={{marginBottom:"calc(16px * var(--u))"}}>
        {loading ? Array.from({length:4}).map((_,i)=><Skeleton key={i} h={150} r={14}/>) : D.INDICES.map(ix=><IndexCard key={ix.name} ix={ix} F={F}/>)}
      </div>

      {/* flows + breadth */}
      <div className="grid" style={{gridTemplateColumns:"1.7fr 1fr",marginBottom:"calc(16px * var(--u))"}}>
        <Card title="FII / DII Activity" sub="Net buy / sell · last 10 sessions (₹ Cr)" icon="flow">
          <div className="card-pad">{loading?<Skeleton h={210}/>:<FlowBars data={D.FII_DII}/>}</div>
        </Card>
        <Card title="Market Breadth" sub="Advance / Decline" icon="trendUp">
          <div className="card-pad" style={{display:"flex",flexDirection:"column",gap:16}}>
            <div className="row" style={{height:14,borderRadius:999,overflow:"hidden",gap:2}}>
              <div style={{flex:D.BREADTH.advances,background:"var(--green)",height:"100%"}}/>
              <div style={{flex:D.BREADTH.unchanged,background:"var(--text-3)",height:"100%"}}/>
              <div style={{flex:D.BREADTH.declines,background:"var(--red)",height:"100%"}}/>
            </div>
            <div className="row between">
              <div className="col" style={{gap:1}}><span className="mono" style={{fontSize:22,fontWeight:700,color:"var(--green)"}}>{D.BREADTH.advances}</span><span className="dim" style={{fontSize:11.5}}>Advancing</span></div>
              <div className="col" style={{gap:1,alignItems:"center"}}><span className="mono" style={{fontSize:22,fontWeight:700,color:"var(--text-2)"}}>{D.BREADTH.unchanged}</span><span className="dim" style={{fontSize:11.5}}>Unchanged</span></div>
              <div className="col" style={{gap:1,alignItems:"flex-end"}}><span className="mono" style={{fontSize:22,fontWeight:700,color:"var(--red)"}}>{D.BREADTH.declines}</span><span className="dim" style={{fontSize:11.5}}>Declining</span></div>
            </div>
            <div className="divider"/>
            <div className="row between">
              <span style={{fontSize:13,color:"var(--text-2)"}}>Advance / Decline Ratio</span>
              <span className="mono" style={{fontWeight:700,fontSize:16,color:"var(--green)"}}>{(D.BREADTH.advances/D.BREADTH.declines).toFixed(2)}</span>
            </div>
            <div className="row between">
              <span style={{fontSize:13,color:"var(--text-2)"}}>India VIX</span>
              <span className="mono" style={{fontWeight:700,fontSize:16,color:"var(--green)"}}>13.42 <span style={{fontSize:12}}>▼6.0%</span></span>
            </div>
          </div>
        </Card>
      </div>

      {/* sector heatmap */}
      <Card title="Sector Heatmap" sub="12 sectors · % change today" icon="layers" pad={false} style={{marginBottom:"calc(16px * var(--u))"}}>
        <div className="card-pad">
          {loading ? <Skeleton h={170}/> : (
            <div className="heat-grid">
              {heat.map(h=>(
                <div key={h.sector} className="heat-cell" style={{background:heatColor(h.change)}}>
                  <span className="hc-name">{h.sector}</span>
                  <div className="row between" style={{alignItems:"flex-end"}}>
                    <span className="hc-val" style={{color:h.change>=0?"var(--green)":"var(--red)"}}>{F.pct(h.change)}</span>
                    <span className="dim" style={{fontSize:10.5}}>₹{h.mcap}L Cr</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* gainers / losers */}
      <div className="grid" style={{gridTemplateColumns:"1fr 1fr"}}>
        {[["Top Gainers",D.GAINERS,"var(--green)","trendUp"],["Top Losers",D.LOSERS,"var(--red)","trendDown"]].map(([title,list,col,ic])=>(
          <Card key={title} title={title} icon={ic} pad={false}>
            <div className="tbl-wrap">
              <table className="tbl">
                <thead><tr><th>Stock</th><th style={{textAlign:"right"}}>LTP</th><th style={{textAlign:"right"}}>Change</th><th>Signal</th></tr></thead>
                <tbody>
                  {loading ? <SkeletonRows cols={4} rows={5}/> : list.map(s=>(
                    <tr key={s.id} className="clickable" onClick={()=>openStock(s)}>
                      <td><SymbolCell s={s}/></td>
                      <td className="num-cell" style={{textAlign:"right"}}>{F.inr(s.price)}</td>
                      <td style={{textAlign:"right"}}><Delta value={s.change} size={13} icon={false}/></td>
                      <td><SignalBadge signal={s.signal}/></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { MarketPage });
