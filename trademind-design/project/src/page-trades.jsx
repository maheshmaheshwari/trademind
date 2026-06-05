/* ===== Trades & Orders ===== */
function TradesPage({ openStock, toast }){
  const D=DATA, F=D.fmt;
  const [tab,setTab]=useStateU("open");
  const [loading,setLoading]=useStateU(true);
  const [closed,setClosed]=useStateU([]);
  const [dateRange,setDateRange]=useStateU("All");
  const [sideFilter,setSideFilter]=useStateU("All");
  const { sort, toggle, apply } = useSort("date","desc");
  useEffectU(()=>{ setLoading(true); const t=setTimeout(()=>setLoading(false),500); return ()=>clearTimeout(t); },[tab]);

  const openPos = D.OPEN_POS.filter(p=>!closed.includes(p.symbol));
  function closePos(p){ setClosed(c=>[...c,p.symbol]); toast({type:p.pnl>=0?"success":"info",title:`Closed ${p.symbol}`,msg:`Realized ${F.signed(p.pnl,0)} (${F.pct(p.pnlPct)})`}); }

  // history filters
  const today=new Date(2026,5,1);
  const histFiltered = apply(D.TRADE_HISTORY.filter(t=>{
    const days=(today-t.date)/(864e5);
    const rangeOk = dateRange==="All"||(dateRange==="7D"&&days<=7)||(dateRange==="30D"&&days<=30)||(dateRange==="90D"&&days<=90);
    return rangeOk && (sideFilter==="All"||t.side===sideFilter);
  }));

  function exportCSV(){
    const head="Date,Symbol,Side,Qty,Price,Value,Realized P&L\n";
    const body=histFiltered.map(t=>`${t.date.toISOString().slice(0,10)},${t.symbol},${t.side},${t.qty},${t.price},${t.value},${t.realized}`).join("\n");
    const blob=new Blob([head+body],{type:"text/csv"}); const url=URL.createObjectURL(blob);
    const a=document.createElement("a"); a.href=url; a.download="trademind-trades.csv"; a.click(); URL.revokeObjectURL(url);
    toast({type:"success",title:"Export complete",msg:`${histFiltered.length} trades downloaded as CSV`});
  }

  const fmtDate=d=>d.toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"});
  const counts={ open:openPos.length, history:D.TRADE_HISTORY.length, gtt:D.GTT.length };

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Trades & Orders</h1>
          <p className="page-sub">Manage open positions, review history & Angel One GTT rules</p>
        </div>
        {tab==="history" && <button className="btn btn-ghost" onClick={exportCSV}><Icon name="download" size={17}/>Export CSV</button>}
      </div>

      <div className="tabs" style={{marginBottom:"calc(18px * var(--u))"}}>
        {[["open","Open Positions"],["history","Trade History"],["gtt","GTT Orders"]].map(([id,label])=>(
          <button key={id} className={"tab "+(tab===id?"on":"")} onClick={()=>setTab(id)}>{label}<span className="cnt">{counts[id]}</span></button>
        ))}
      </div>

      {/* OPEN POSITIONS */}
      {tab==="open" && (
        <Card pad={false}>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th style={{textAlign:"right"}}>Entry</th><th style={{textAlign:"right"}}>SL</th><th style={{textAlign:"right"}}>Target</th><th style={{textAlign:"right"}}>CMP</th><th style={{textAlign:"right"}}>P&L</th><th style={{textAlign:"right"}}>Days</th><th style={{textAlign:"right"}}>Action</th></tr></thead>
              <tbody>
                {loading ? <SkeletonRows cols={8} rows={6}/> : openPos.length===0 ? (
                  <tr><td colSpan={8}><div className="empty">No open positions. All trades closed 🎉</div></td></tr>
                ) : openPos.map(p=>(
                  <tr key={p.symbol}>
                    <td onClick={()=>openStock(p)} style={{cursor:"pointer"}}><SymbolCell s={p}/></td>
                    <td className="num-cell" style={{textAlign:"right"}}>{F.inr(p.entry)}</td>
                    <td className="num-cell" style={{textAlign:"right",color:"var(--red)"}}>{F.inr(p.sl)}</td>
                    <td className="num-cell" style={{textAlign:"right",color:"var(--green)"}}>{F.inr(p.target)}</td>
                    <td className="num-cell" style={{textAlign:"right"}}>{F.inr(p.price)}</td>
                    <td style={{textAlign:"right"}}>
                      <div className="col" style={{alignItems:"flex-end",gap:0}}>
                        <span className="num-cell" style={{fontWeight:600,color:p.pnl>=0?"var(--green)":"var(--red)"}}>{F.signed(p.pnl,0)}</span>
                        <span className="num" style={{fontSize:11.5,color:p.pnl>=0?"var(--green)":"var(--red)"}}>{F.pct(p.pnlPct)}</span>
                      </div>
                    </td>
                    <td className="num-cell dim" style={{textAlign:"right"}}>{p.days}d</td>
                    <td style={{textAlign:"right"}}><button className="btn btn-danger btn-sm" onClick={()=>closePos(p)}>Close</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* TRADE HISTORY */}
      {tab==="history" && (
        <Card pad={false}>
          <div className="card-head">
            <div className="filter-bar">
              <div className="field"><span className="field-label">Date Range</span>
                <div className="seg">{["All","7D","30D","90D"].map(r=><button key={r} className={dateRange===r?"on":""} onClick={()=>setDateRange(r)}>{r}</button>)}</div>
              </div>
              <div className="field"><span className="field-label">Side</span>
                <div className="seg">{["All","BUY","SELL"].map(r=><button key={r} className={sideFilter===r?"on":""} onClick={()=>setSideFilter(r)}>{r}</button>)}</div>
              </div>
            </div>
            <span style={{fontSize:12.5,color:"var(--text-2)"}}><b className="num">{histFiltered.length}</b> trades</span>
          </div>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr>
                <Th label="Date" k="date" sort={sort} toggle={toggle}/>
                <th>Symbol</th>
                <th>Side</th>
                <Th label="Qty" k="qty" sort={sort} toggle={toggle} align="right"/>
                <Th label="Price" k="price" sort={sort} toggle={toggle} align="right"/>
                <Th label="Value" k="value" sort={sort} toggle={toggle} align="right"/>
                <Th label="Realized P&L" k="realized" sort={sort} toggle={toggle} align="right"/>
                <th style={{textAlign:"right"}}>Status</th>
              </tr></thead>
              <tbody>
                {loading ? <SkeletonRows cols={8} rows={9}/> : histFiltered.slice(0,18).map(t=>(
                  <tr key={t.id}>
                    <td className="num-cell dim" style={{fontSize:12.5}}>{fmtDate(t.date)}</td>
                    <td><SymbolCell s={t}/></td>
                    <td><span className="pill" style={{color:t.side==="BUY"?"var(--green)":"var(--red)"}}>{t.side}</span></td>
                    <td className="num-cell" style={{textAlign:"right"}}>{t.qty}</td>
                    <td className="num-cell" style={{textAlign:"right"}}>{F.inr(t.price)}</td>
                    <td className="num-cell" style={{textAlign:"right",color:"var(--text-2)"}}>{F.inrCompact(t.value)}</td>
                    <td className="num-cell" style={{textAlign:"right",fontWeight:600,color:t.realized>=0?"var(--green)":"var(--red)"}}>{F.signed(t.realized,0)}</td>
                    <td style={{textAlign:"right"}}><span className="pill" style={{color:"var(--green)",background:"var(--green-soft)",border:"none"}}>Executed</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pager page={1} pages={1} total={Math.min(18,histFiltered.length)} perPage={18} onPage={()=>{}} label="of recent trades shown"/>
        </Card>
      )}

      {/* GTT ORDERS */}
      {tab==="gtt" && (
        <Card pad={false}>
          <div className="card-head">
            <div className="col" style={{gap:0}}><h3 className="card-title"><span className="ic"><Icon name="shield" size={17}/></span>Angel One GTT Rules</h3><span className="card-sub">Good-Till-Triggered orders synced from your broker</span></div>
            <button className="btn btn-ghost btn-sm" onClick={()=>toast({type:"info",title:"Synced with Angel One",msg:"GTT rules up to date"})}><Icon name="refresh" size={15}/>Sync</button>
          </div>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Type</th><th>Side</th><th style={{textAlign:"right"}}>Trigger</th><th style={{textAlign:"right"}}>LTP</th><th style={{textAlign:"right"}}>Qty</th><th>Created</th><th style={{textAlign:"right"}}>Status</th></tr></thead>
              <tbody>
                {loading ? <SkeletonRows cols={8} rows={5}/> : D.GTT.map(g=>{
                  const sc=g.status==="ACTIVE"?"var(--accent-2)":g.status==="TRIGGERED"?"var(--green)":"var(--text-3)";
                  const sbg=g.status==="ACTIVE"?"var(--accent-soft)":g.status==="TRIGGERED"?"var(--green-soft)":"var(--surface-3)";
                  return (
                    <tr key={g.id}>
                      <td><SymbolCell s={g}/></td>
                      <td><span className="pill">{g.type}</span></td>
                      <td><span className="pill" style={{color:g.side==="BUY"?"var(--green)":"var(--red)"}}>{g.side}</span></td>
                      <td className="num-cell" style={{textAlign:"right",fontWeight:600}}>{F.inr(g.trigger)}</td>
                      <td className="num-cell" style={{textAlign:"right",color:"var(--text-2)"}}>{F.inr(g.ltp)}</td>
                      <td className="num-cell" style={{textAlign:"right"}}>{g.qty}</td>
                      <td className="dim" style={{fontSize:12.5}}>{g.created}</td>
                      <td style={{textAlign:"right"}}><span className="pill" style={{color:sc,background:sbg,border:"none"}}>{g.status}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

Object.assign(window, { TradesPage });
