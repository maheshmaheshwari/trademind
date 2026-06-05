/* ===== Add Position Modal ===== */
function AddPositionModal({ onClose, toast }){
  const D=DATA, F=D.fmt;
  const [q,setQ]=useStateU("");
  const [sel,setSel]=useStateU(null);
  const [qty,setQty]=useStateU("");
  const [price,setPrice]=useStateU("");
  const [mode,setMode]=useStateU("paper");
  const matches = q.length? D.STOCKS.filter(s=>s.symbol.toLowerCase().includes(q.toLowerCase())||s.name.toLowerCase().includes(q.toLowerCase())).slice(0,5):[];
  function choose(s){ setSel(s); setQ(s.symbol); setPrice(String(s.price)); }
  const valid = sel && +qty>0 && +price>0;
  function submit(){ if(!valid) return; toast({type:"success",title:"Position added",msg:`${qty} × ${sel.symbol} @ ${F.inr(+price)} · ${mode==="paper"?"Paper":"Live"} account`}); onClose(); }
  return (
    <div className="modal">
      <div className="scrim" onClick={onClose}/>
      <div className="modal-card">
        <div className="modal-head">
          <div className="col" style={{gap:2}}><h3 style={{margin:0,fontSize:18,fontWeight:700}}>Add Position</h3><span style={{fontSize:13,color:"var(--text-2)"}}>Record a holding or place a paper trade</span></div>
          <button className="drawer-x" onClick={onClose}><Icon name="x" size={18}/></button>
        </div>
        <div className="modal-body">
          <div className="form-row" style={{position:"relative"}}>
            <label>Symbol</label>
            <input value={q} onChange={e=>{setQ(e.target.value);setSel(null);}} placeholder="Search e.g. RELIANCE, Infosys…" autoFocus/>
            {matches.length>0 && !sel && (
              <div style={{position:"absolute",top:74,left:0,right:0,background:"var(--surface)",border:"1px solid var(--border-strong)",borderRadius:11,boxShadow:"var(--shadow-lg)",zIndex:5,overflow:"hidden"}}>
                {matches.map(s=>(
                  <button key={s.id} className="sb-item" style={{height:46,borderRadius:0}} onClick={()=>choose(s)}>
                    <span className="sym-logo" style={{width:28,height:28,fontSize:11,background:D.symColor(s.symbol)+"22",color:D.symColor(s.symbol)}}>{s.symbol.slice(0,2)}</span>
                    <div className="col" style={{gap:0,flex:1}}><span style={{fontWeight:600,fontSize:13}}>{s.symbol}</span><span style={{fontSize:11,color:"var(--text-3)"}}>{s.name}</span></div>
                    <span className="mono" style={{fontSize:12.5,color:"var(--text-2)"}}>{F.inr(s.price)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          {sel && <div className="row gap-sm" style={{padding:"10px 12px",background:"var(--surface-2)",borderRadius:11,marginBottom:15}}>
            <span className="sym-logo" style={{width:32,height:32,fontSize:12,background:D.symColor(sel.symbol)+"22",color:D.symColor(sel.symbol)}}>{sel.symbol.slice(0,2)}</span>
            <div className="col" style={{gap:0,flex:1}}><span style={{fontWeight:600,fontSize:13.5}}>{sel.symbol}</span><span style={{fontSize:11.5,color:"var(--text-3)"}}>{sel.name}</span></div>
            <SignalBadge signal={sel.signal}/>
          </div>}
          <div className="row gap-sm" style={{gap:13}}>
            <div className="form-row" style={{flex:1}}><label>Quantity</label><input type="number" value={qty} onChange={e=>setQty(e.target.value)} placeholder="0"/></div>
            <div className="form-row" style={{flex:1}}><label>Price (₹)</label><input type="number" value={price} onChange={e=>setPrice(e.target.value)} placeholder="0.00"/></div>
          </div>
          {valid && <div className="row between" style={{padding:"11px 13px",background:"var(--accent-soft)",borderRadius:11,marginBottom:6}}>
            <span style={{fontSize:13,color:"var(--text-2)",fontWeight:500}}>Total investment</span>
            <span className="mono" style={{fontWeight:700,fontSize:15,color:"var(--accent-2)"}}>{F.inr(+qty*+price)}</span>
          </div>}
          <div className="form-row">
            <label>Account</label>
            <div className="seg" style={{alignSelf:"flex-start"}}>
              <button className={mode==="paper"?"on":""} onClick={()=>setMode("paper")}>📄 Paper</button>
              <button className={mode==="live"?"on":""} onClick={()=>setMode("live")}>⚡ Live</button>
            </div>
          </div>
        </div>
        <div className="modal-foot">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" disabled={!valid} style={{opacity:valid?1:.5,cursor:valid?"pointer":"not-allowed"}} onClick={submit}><Icon name="plus" size={17}/>Add Position</button>
        </div>
      </div>
    </div>
  );
}

/* ===== Portfolio Page ===== */
function PortfolioPage({ openStock, toast }){
  const D=DATA, F=D.fmt;
  const [loading,setLoading]=useStateU(true);
  const [modal,setModal]=useStateU(false);
  const [range,setRange]=useStateU("90D");
  const { sort, toggle, apply } = useSort("current","desc");
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),600); return ()=>clearTimeout(t); },[]);
  const pnlPct=(D.PF_PNL/D.PF_INVESTED)*100;
  const series=D.PNL_HISTORY[range];
  const holdings=apply(D.HOLDINGS);
  const labels = range==="30D"?["30d ago","20d","10d","Today"]:range==="90D"?["90d ago","60d","30d","Today"]:["1Y ago","9mo","6mo","3mo","Now"];

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Portfolio</h1>
          <p className="page-sub"><b className="num">{D.HOLDINGS.length}</b> holdings · diversified across <b className="num">{D.ALLOC.length}</b> sectors</p>
        </div>
        <button className="btn btn-primary" onClick={()=>setModal(true)}><Icon name="plus" size={17}/>Add Position</button>
      </div>

      {/* summary */}
      <div className="grid" style={{gridTemplateColumns:"repeat(3,1fr)",marginBottom:"calc(16px * var(--u))"}}>
        <div className="stat"><div className="stat-top"><span className="stat-label">Total Invested</span><span className="stat-ic" style={{background:"var(--accent-soft)",color:"var(--accent)"}}><Icon name="wallet" size={18}/></span></div><div className="stat-val">{F.inrCompact(D.PF_INVESTED)}</div><span className="dim" style={{fontSize:12}}>across {D.HOLDINGS.length} stocks</span></div>
        <div className="stat"><div className="stat-top"><span className="stat-label">Current Value</span><span className="stat-ic" style={{background:"#8B5CF61f",color:"#8B5CF6"}}><Icon name="pie" size={18}/></span></div><div className="stat-val">{F.inrCompact(D.PF_CURRENT)}</div><span className="stat-delta" style={{color:"var(--green)"}}><Icon name="arrowUpRight" size={13}/>+1.84% today</span></div>
        <div className="stat" style={{background:pnlPct>=0?"linear-gradient(135deg,var(--green-soft),transparent)":"linear-gradient(135deg,var(--red-soft),transparent)"}}><div className="stat-top"><span className="stat-label">Total P&L</span><span className="stat-ic" style={{background:pnlPct>=0?"var(--green-soft)":"var(--red-soft)",color:pnlPct>=0?"var(--green)":"var(--red)"}}><Icon name="trendUp" size={18}/></span></div><div className="stat-val" style={{color:pnlPct>=0?"var(--green)":"var(--red)"}}>{F.signed(D.PF_PNL,0).replace("+","+₹").replace("-","-₹")}</div><span style={{fontWeight:700,color:pnlPct>=0?"var(--green)":"var(--red)"}} className="num">{F.pct(pnlPct)} overall</span></div>
      </div>

      {/* chart + donut */}
      <div className="grid" style={{gridTemplateColumns:"1.7fr 1fr",marginBottom:"calc(16px * var(--u))"}}>
        <Card title="Portfolio Value" sub="Growth over time" icon="trendUp"
          right={<div className="seg subtle">{["30D","90D","1Y"].map(r=><button key={r} className={range===r?"on":""} onClick={()=>setRange(r)}>{r}</button>)}</div>}>
          <div className="card-pad" style={{paddingTop:10}}>{loading?<Skeleton h={230}/>:<AreaChart data={series} color="var(--accent)" h={230} labels={labels}/>}</div>
        </Card>
        <Card title="Allocation" sub="By sector" icon="pie">
          <div className="card-pad">{loading?<Skeleton h={180}/>:<Donut data={D.ALLOC} centerTop="Total" centerBottom={F.inrCompact(D.PF_CURRENT).replace("₹","₹")}/>}</div>
        </Card>
      </div>

      {/* holdings */}
      <Card title="Holdings" sub={`${D.HOLDINGS.length} positions`} icon="layers" pad={false}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr>
              <Th label="Symbol" k="symbol" sort={sort} toggle={toggle}/>
              <Th label="Qty" k="qty" sort={sort} toggle={toggle} align="right"/>
              <Th label="Avg Buy" k="avg" sort={sort} toggle={toggle} align="right"/>
              <Th label="CMP" k="cmp" sort={sort} toggle={toggle} align="right"/>
              <Th label="Invested" k="invested" sort={sort} toggle={toggle} align="right"/>
              <Th label="P&L" k="pnl" sort={sort} toggle={toggle} align="right"/>
              <Th label="P&L %" k="pnlPct" sort={sort} toggle={toggle} align="right"/>
              <th>AI Signal</th>
            </tr></thead>
            <tbody>
              {loading ? <SkeletonRows cols={8} rows={8}/> : holdings.map(h=>(
                <tr key={h.symbol} className="clickable" onClick={()=>openStock(h)}>
                  <td><SymbolCell s={h}/></td>
                  <td className="num-cell" style={{textAlign:"right"}}>{h.qty}</td>
                  <td className="num-cell" style={{textAlign:"right"}}>{F.inr(h.avg)}</td>
                  <td className="num-cell" style={{textAlign:"right"}}>{F.inr(h.cmp)}</td>
                  <td className="num-cell" style={{textAlign:"right",color:"var(--text-2)"}}>{F.inrCompact(h.invested)}</td>
                  <td className="num-cell" style={{textAlign:"right",fontWeight:600,color:h.pnl>=0?"var(--green)":"var(--red)"}}>{F.signed(h.pnl,0)}</td>
                  <td style={{textAlign:"right"}}><Delta value={h.pnlPct} size={12.5} icon={false}/></td>
                  <td><SignalBadge signal={h.signal}/></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {modal && <AddPositionModal onClose={()=>setModal(false)} toast={toast}/>}
    </div>
  );
}

Object.assign(window, { PortfolioPage, AddPositionModal });
