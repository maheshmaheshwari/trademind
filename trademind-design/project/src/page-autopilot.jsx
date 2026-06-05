/* ===== AI Authorized Trades (Autopilot) ===== */
function autoStatusMeta(st){
  switch(st){
    case "EXECUTED": return { label:"Running", color:"var(--accent-2)", bg:"var(--accent-soft)", ic:"flow" };
    case "PENDING": return { label:"Pending", color:"var(--gold)", bg:"var(--gold-soft)", ic:"clock" };
    case "COMPLETED": return { label:"Target hit", color:"var(--green)", bg:"var(--green-soft)", ic:"checkCircle" };
    case "STOPPED": return { label:"Stopped", color:"var(--red)", bg:"var(--red-soft)", ic:"alert" };
    default: return { label:st, color:"var(--text-2)", bg:"var(--surface-3)", ic:"target" };
  }
}

function AutopilotPage({ openStock, toast }){
  const D=DATA, F=D.fmt;
  const [loading,setLoading]=useStateU(true);
  const [autopilot,setAutopilot]=useStateU(true);
  const [filter,setFilter]=useStateU("All");
  const [revoked,setRevoked]=useStateU([]);
  const { sort, toggle, apply } = useSort("amount","desc");
  useEffectU(()=>{ const t=setTimeout(()=>setLoading(false),600); return ()=>clearTimeout(t); },[]);

  const all = D.AUTH_TRADES.filter(t=>!revoked.includes(t.symbol));
  const FILTERS=["All","Running","Pending","Target hit","Stopped"];
  const fmap={ "Running":"EXECUTED","Pending":"PENDING","Target hit":"COMPLETED","Stopped":"STOPPED" };
  const filtered = apply(all.filter(t=>filter==="All"||t.status===fmap[filter]));
  function revoke(sym,e){ e.stopPropagation(); setRevoked(r=>[...r,sym]); toast({type:"info",title:`Authorization revoked`,msg:`${sym} removed from AI autopilot`}); }

  const capital = all.reduce((a,t)=>a+t.amount,0);
  const active = all.filter(t=>t.status==="EXECUTED"||t.status==="PENDING").length;
  const realized = all.filter(t=>t.actualPnl!=null && (t.status==="COMPLETED"||t.status==="STOPPED")).reduce((a,t)=>a+t.actualPnl,0);
  const liveExp = all.filter(t=>t.status==="EXECUTED"||t.status==="PENDING").reduce((a,t)=>a+t.expProfit,0);
  const counts = { running:all.filter(t=>t.status==="EXECUTED").length, pending:all.filter(t=>t.status==="PENDING").length };

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">AI Authorized Trades</h1>
          <p className="page-sub">Trades you've authorized the AI to place & manage automatically via <b>Angel One</b></p>
        </div>
        <div className="row gap-sm wrap">
          <div className={"autopilot-pill "+(autopilot?"on":"")} onClick={()=>{setAutopilot(a=>!a);toast({type:autopilot?"info":"success",title:autopilot?"Autopilot paused":"Autopilot active",msg:autopilot?"AI will not place new orders":"AI will auto-execute authorized signals"});}}>
            <span className="ap-dot"/>
            <div className="col" style={{gap:0,lineHeight:1.2}}>
              <span style={{fontSize:13,fontWeight:700}}>AI Autopilot</span>
              <span style={{fontSize:10.5,opacity:.85}}>{autopilot?"Active · auto-executing":"Paused"}</span>
            </div>
            <span className={"tgl "+(autopilot?"on":"")} style={{pointerEvents:"none"}}><span className="tgl-knob"/></span>
          </div>
        </div>
      </div>

      {/* summary */}
      <div className="grid stat-grid" style={{marginBottom:"calc(16px * var(--u))"}}>
        <StatCard label="Capital Under AI" value={F.inrCompact(capital)} icon="brain" color="var(--accent)"/>
        <StatCard label="Active Mandates" value={active} icon="flow" color="var(--gold)"/>
        <div className="stat"><div className="stat-top"><span className="stat-label">Realized P&L</span><span className="stat-ic" style={{background:realized>=0?"var(--green-soft)":"var(--red-soft)",color:realized>=0?"var(--green)":"var(--red)"}}><Icon name="trendUp" size={18}/></span></div><div className="stat-val" style={{color:realized>=0?"var(--green)":"var(--red)"}}>{F.signed(realized,0)}</div><span className="dim" style={{fontSize:12}}>from closed mandates</span></div>
        <div className="stat"><div className="stat-top"><span className="stat-label">Projected Profit</span><span className="stat-ic" style={{background:"#8B5CF61f",color:"#8B5CF6"}}><Icon name="target" size={18}/></span></div><div className="stat-val" style={{color:"var(--green)"}}>+{F.inrCompact(liveExp).replace("₹","₹")}</div><span className="dim" style={{fontSize:12}}>if targets hit</span></div>
      </div>

      {/* banner */}
      <div className="ai-banner" style={{marginBottom:"calc(16px * var(--u))"}}>
        <span className="ai-banner-ic"><Icon name="sparkle" size={20}/></span>
        <div className="col" style={{gap:1,flex:1}}>
          <span style={{fontWeight:600,fontSize:13.5}}>AI is managing {active} active trades across {new Set(all.map(t=>t.sector)).size} sectors</span>
          <span style={{fontSize:12,color:"var(--text-2)"}}>Each order respects your per-trade stop-loss & target. {counts.pending} pending trigger conditions.</span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={()=>window.__openSettings&&window.__openSettings("profile")}><Icon name="settings" size={15}/>Mandate rules</button>
      </div>

      {/* filters */}
      <div className="row between wrap" style={{marginBottom:"calc(14px * var(--u))",gap:12}}>
        <div className="seg">{FILTERS.map(f=><button key={f} className={filter===f?"on":""} onClick={()=>setFilter(f)}>{f}</button>)}</div>
        <span style={{fontSize:12.5,color:"var(--text-2)"}}><b className="num">{filtered.length}</b> of {all.length} mandates</span>
      </div>

      {/* table */}
      <Card pad={false}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr>
              <Th label="Stock" k="symbol" sort={sort} toggle={toggle}/>
              <th>Signal</th>
              <Th label="Authorized ₹" k="amount" sort={sort} toggle={toggle} align="right"/>
              <Th label="Qty" k="qty" sort={sort} toggle={toggle} align="right"/>
              <th style={{textAlign:"right"}}>Entry → Target</th>
              <Th label="Exp. Profit" k="expProfit" sort={sort} toggle={toggle} align="right"/>
              <th style={{textAlign:"right"}}>Max Loss</th>
              <th style={{textAlign:"right"}}>Live / Realized P&L</th>
              <th>Status</th>
              <th style={{textAlign:"right"}}>Action</th>
            </tr></thead>
            <tbody>
              {loading ? <SkeletonRows cols={10} rows={9}/> : filtered.length===0 ? (
                <tr><td colSpan={10}><div className="empty">No mandates in this state.</div></td></tr>
              ) : filtered.map(t=>{
                const m=autoStatusMeta(t.status);
                const live = t.status==="EXECUTED" ? +((t.cmp-t.entry)*t.qty).toFixed(0) : t.actualPnl;
                return (
                  <tr key={t.symbol} className="clickable" onClick={()=>openStock(t)}>
                    <td><SymbolCell s={t}/></td>
                    <td><div className="row gap-sm"><SignalBadge signal={t.signal}/><span className="pill" style={{fontSize:10}}>{t.mode}</span></div></td>
                    <td className="num-cell" style={{textAlign:"right",fontWeight:600}}>{F.inr(t.amount,0)}</td>
                    <td className="num-cell" style={{textAlign:"right"}}>{t.qty}</td>
                    <td className="num-cell" style={{textAlign:"right",fontSize:12}}><span style={{color:"var(--text-2)"}}>{F.inr(t.entry,0)}</span><span className="dim"> → </span><span style={{color:"var(--green)"}}>{F.inr(t.target,0)}</span></td>
                    <td className="num-cell" style={{textAlign:"right",color:"var(--green)",fontWeight:600}}>+{F.inr(t.expProfit,0)}</td>
                    <td className="num-cell" style={{textAlign:"right",color:"var(--red)"}}>−{F.inr(t.maxLoss,0)}</td>
                    <td style={{textAlign:"right"}}>{live!=null ? <span className="num-cell" style={{fontWeight:600,color:live>=0?"var(--green)":"var(--red)"}}>{F.signed(live,0)}</span> : <span className="dim">—</span>}</td>
                    <td><span className="pill" style={{color:m.color,background:m.bg,border:"none"}}><Icon name={m.ic} size={12}/>{m.label}</span></td>
                    <td style={{textAlign:"right"}}>
                      {(t.status==="EXECUTED"||t.status==="PENDING")
                        ? <button className="btn btn-danger btn-sm" onClick={(e)=>revoke(t.symbol,e)}>Revoke</button>
                        : <span className="dim" style={{fontSize:12}}>Closed {t.authAgo}</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Pager page={1} pages={1} total={filtered.length} perPage={filtered.length||1} onPage={()=>{}} label="authorized trades"/>
      </Card>
    </div>
  );
}

Object.assign(window, { AutopilotPage });
