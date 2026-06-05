/* ===== Watchlist Page ===== */
function WatchlistPage({ openStock, toast }){
  const D=DATA, F=D.fmt;
  const [loading,setLoading]=useStateU(true);
  const [view,setView]=useStateU("grid");
  const [removed,setRemoved]=useStateU([]);
  const { sort, toggle, apply } = useSort("confidence","desc");
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),600); return ()=>clearTimeout(t); },[]);

  const list = D.WATCHLIST.filter(w=>!removed.includes(w.symbol));
  const sorted = apply(list);
  function remove(sym,e){ e.stopPropagation(); setRemoved(r=>[...r,sym]); toast({type:"info",title:`${sym} removed from watchlist`}); }
  const buys=list.filter(s=>s.signal==="BUY").length, sells=list.filter(s=>s.signal==="SELL").length;

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Watchlist</h1>
          <p className="page-sub">Tracking <b className="num">{list.length}</b> stocks · <span className="pos">{buys} BUY</span> · <span className="neg">{sells} SELL</span> signals active</p>
        </div>
        <div className="row gap-sm">
          <div className="seg subtle">
            <button className={view==="grid"?"on":""} onClick={()=>setView("grid")}>Grid</button>
            <button className={view==="table"?"on":""} onClick={()=>setView("table")}>Table</button>
          </div>
          <button className="btn btn-primary" onClick={()=>{window.__setPage("signals");}}><Icon name="plus" size={17}/>Add Stocks</button>
        </div>
      </div>

      {/* alert summary strip */}
      <div className="grid stat-grid" style={{marginBottom:"calc(16px * var(--u))"}}>
        <StatCard label="Watchlist Items" value={list.length} icon="bookmark" color="var(--accent)"/>
        <StatCard label="Avg Confidence" value={list.length?Math.round(list.reduce((a,s)=>a+s.confidence,0)/list.length)+"%":"—"} icon="brain" color="#8B5CF6"/>
        <StatCard label="Price Alerts" value={list.length*2} icon="bell" color="var(--gold)"/>
        <StatCard label="In Profit Zone" value={list.filter(s=>s.change>=0).length+"/"+list.length} icon="trendUp" color="var(--green)"/>
      </div>

      {list.length===0 ? (
        <Card><div className="empty"><Icon name="bookmark" size={34}/><h3 style={{color:"var(--text)",margin:"12px 0 4px"}}>Your watchlist is empty</h3><p>Add stocks from the Signals page to track them here.</p><button className="btn btn-primary" style={{marginTop:14}} onClick={()=>window.__setPage("signals")}><Icon name="plus" size={17}/>Browse Signals</button></div></Card>
      ) : view==="grid" ? (
        <div className="sig-grid" style={{gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))"}}>
          {loading ? Array.from({length:8}).map((_,i)=><Skeleton key={i} h={172} r={14}/>) : sorted.map(s=>{
            const col=s.signal==="BUY"?"var(--green)":s.signal==="SELL"?"var(--red)":"var(--gold)";
            return (
              <div key={s.symbol} className="watch-card" onClick={()=>openStock(s)}>
                <div className="row between">
                  <SymbolCell s={s}/>
                  <button className="watch-x" onClick={(e)=>remove(s.symbol,e)} title="Remove"><Icon name="x" size={15}/></button>
                </div>
                <div className="row between" style={{alignItems:"flex-end",margin:"12px 0"}}>
                  <div className="col" style={{gap:1}}><span className="mono" style={{fontSize:19,fontWeight:700}}>{F.inr(s.price)}</span><Delta value={s.change} size={12}/></div>
                  <Sparkline data={s.spark} color={s.change>=0?"var(--green)":"var(--red)"} w={90} h={40} fill/>
                </div>
                <div className="divider"/>
                <div className="row between" style={{marginTop:11}}>
                  <div className="row gap-sm"><SignalBadge signal={s.signal}/><span className="pill">{s.horizon}</span></div>
                  <span className="mono" style={{fontWeight:700,color:col,fontSize:13.5}}>{F.pct(s.expReturn)}</span>
                </div>
                <div className="row gap-sm" style={{marginTop:11,fontSize:11.5,color:"var(--text-3)"}}>
                  <span className="row" style={{gap:4}}><Icon name="bell" size={13}/>↑{F.inr(s.alertAbove,0)}</span>
                  <span className="row" style={{gap:4}}>↓{F.inr(s.alertBelow,0)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <Card pad={false}>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr>
                <Th label="Stock" k="symbol" sort={sort} toggle={toggle}/>
                <Th label="LTP" k="price" sort={sort} toggle={toggle} align="right"/>
                <Th label="Change" k="change" sort={sort} toggle={toggle} align="right"/>
                <th>Signal</th>
                <Th label="Confidence" k="confidence" sort={sort} toggle={toggle}/>
                <th>Alerts</th>
                <th style={{textAlign:"right"}}></th>
              </tr></thead>
              <tbody>
                {loading ? <SkeletonRows cols={7} rows={8}/> : sorted.map(s=>(
                  <tr key={s.symbol} className="clickable" onClick={()=>openStock(s)}>
                    <td><SymbolCell s={s}/></td>
                    <td className="num-cell" style={{textAlign:"right"}}>{F.inr(s.price)}</td>
                    <td style={{textAlign:"right"}}><Delta value={s.change} size={12.5} icon={false}/></td>
                    <td><SignalBadge signal={s.signal}/></td>
                    <td style={{minWidth:130}}><Conf value={s.confidence}/></td>
                    <td><span className="pill" style={{color:"var(--gold)"}}><Icon name="bell" size={12}/>2 set</span></td>
                    <td style={{textAlign:"right"}}><button className="btn btn-danger btn-sm" onClick={(e)=>remove(s.symbol,e)}>Remove</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

Object.assign(window, { WatchlistPage });
