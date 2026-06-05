/* ===== Shared UI primitives ===== */
const { useState: useStateU, useEffect: useEffectU, useContext, createContext, useRef: useRefU, useCallback } = React;

function Icon({ name, size=20, ...p }){ const C=Icons[name]; return C?<C size={size} {...p}/>:null; }

// signal badge
function SignalBadge({ signal }){
  const cls = signal==="BUY"?"buy":signal==="SELL"?"sell":"hold";
  const ic = signal==="BUY"?"arrowUp":signal==="SELL"?"arrowDown":"target";
  return <span className={"badge "+cls}><Icon name={ic} size={12}/>{signal}</span>;
}

// stock symbol cell
function SymbolCell({ s, showSector=true }){
  return (
    <div className="sym-cell">
      <span className="sym-logo" style={{background:DATA.symColor(s.symbol)+"22",color:DATA.symColor(s.symbol)}}>{s.symbol.slice(0,2)}</span>
      <div className="col">
        <span className="sym-name">{s.symbol}</span>
        {showSector && <span className="sym-sub">{s.sector}</span>}
      </div>
    </div>
  );
}

function Conf({ value }){
  const col = value>=80?"var(--green)":value>=65?"var(--accent)":"var(--gold)";
  return (
    <div className="conf">
      <div className="conf-track"><div className="conf-fill" style={{width:value+"%",background:col}}/></div>
      <span className="conf-val">{value}%</span>
    </div>
  );
}

function Delta({ value, suffix="%", size, icon=true, abs=false }){
  const pos=value>=0;
  return <span style={{color:pos?"var(--green)":"var(--red)",fontWeight:600,fontSize:size,display:"inline-flex",alignItems:"center",gap:3}} className="num">
    {icon && <Icon name={pos?"trendUp":"trendDown"} size={14}/>}
    {(pos?"+":"")+(abs?Math.abs(value):value).toFixed(2)}{suffix}
  </span>;
}

function Card({ title, sub, icon, right, children, pad=true, className="", ...p }){
  return (
    <div className={"card "+className} {...p}>
      {(title||right) && (
        <div className="card-head">
          <div className="col" style={{gap:0}}>
            <h3 className="card-title">{icon && <span className="ic"><Icon name={icon} size={17}/></span>}{title}</h3>
            {sub && <span className="card-sub">{sub}</span>}
          </div>
          {right}
        </div>
      )}
      {pad && !title ? <div className="card-pad">{children}</div> : <div className={title?"":""}>{title?children:children}</div>}
    </div>
  );
}

// table header sorting
function useSort(initialKey, dir="desc"){
  const [sort,setSort]=useStateU({key:initialKey,dir});
  const toggle=(key)=>setSort(s=>s.key===key?{key,dir:s.dir==="asc"?"desc":"asc"}:{key,dir:"desc"});
  const apply=(arr)=>{ const{key,dir}=sort; if(!key) return arr; const sorted=[...arr].sort((a,b)=>{ const x=a[key],y=b[key]; if(typeof x==="string") return x.localeCompare(y); return x-y; }); return dir==="asc"?sorted:sorted.reverse(); };
  return { sort, toggle, apply };
}
function Th({ label, k, sort, toggle, align="left", w }){
  const active=sort.key===k;
  return <th className={"sortable "+(active?"sorted":"")} onClick={()=>toggle(k)} style={{textAlign:align,width:w}}>
    {label}<span className="sort-ic"><Icon name={active?(sort.dir==="asc"?"chevUp":"chevDown"):"chevsUpDown"} size={13}/></span>
  </th>;
}

// pagination
function Pager({ page, pages, total, perPage, onPage, label="rows" }){
  if(pages<=1 && total<=perPage) return (
    <div className="pager"><span className="pager-info">{total} {label}</span></div>
  );
  const nums=[]; const start=Math.max(1,Math.min(page-1,pages-2)); 
  for(let i=start;i<=Math.min(start+2,pages);i++) nums.push(i);
  const from=(page-1)*perPage+1, to=Math.min(page*perPage,total);
  return (
    <div className="pager">
      <span className="pager-info">Showing <b className="num">{from}–{to}</b> of <b className="num">{total}</b> {label}</span>
      <div className="pager-btns">
        <button className="pg-btn" disabled={page===1} onClick={()=>onPage(page-1)}><Icon name="chevLeft" size={15}/></button>
        {start>1 && <><button className="pg-btn" onClick={()=>onPage(1)}>1</button>{start>2 && <span className="pg-btn" style={{border:"none",cursor:"default"}}>…</span>}</>}
        {nums.map(n=><button key={n} className={"pg-btn "+(n===page?"active":"")} onClick={()=>onPage(n)}>{n}</button>)}
        {start+2<pages && <><span className="pg-btn" style={{border:"none",cursor:"default"}}>…</span><button className="pg-btn" onClick={()=>onPage(pages)}>{pages}</button></>}
        <button className="pg-btn" disabled={page===pages} onClick={()=>onPage(page+1)}><Icon name="chevRight" size={15}/></button>
      </div>
    </div>
  );
}

function Skeleton({ w="100%", h=14, r=7, style }){ return <span className="skel" style={{display:"block",width:w,height:h,borderRadius:r,...style}}/>; }
function SkeletonRows({ cols, rows=6 }){
  return <>{Array.from({length:rows}).map((_,i)=>(
    <tr key={i}>{Array.from({length:cols}).map((_,j)=><td key={j}><Skeleton w={j===0?"70%":"55%"} h={j===0?28:13}/></td>)}</tr>
  ))}</>;
}

// ---- Toast system ----
const ToastCtx = createContext(null);
const useToast = () => useContext(ToastCtx);
function ToastProvider({ children }){
  const [toasts,setToasts]=useStateU([]);
  const push=useCallback((t)=>{ const id=Math.random().toString(36).slice(2); setToasts(x=>[...x,{...t,id}]);
    setTimeout(()=>setToasts(x=>x.map(o=>o.id===id?{...o,out:true}:o)),t.duration||3400);
    setTimeout(()=>setToasts(x=>x.filter(o=>o.id!==id)),(t.duration||3400)+300); },[]);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toast-host">
        {toasts.map(t=>(
          <div key={t.id} className={"toast "+(t.type||"info")+(t.out?" out":"")}>
            <span className="t-ic"><Icon name={t.type==="success"?"checkCircle":t.type==="error"?"alert":"sparkle"} size={18}/></span>
            <div className="col" style={{gap:0}}>
              <span className="t-title">{t.title}</span>
              {t.msg && <span className="t-msg">{t.msg}</span>}
            </div>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

Object.assign(window, { Icon, SignalBadge, SymbolCell, Conf, Delta, Card, useSort, Th, Pager, Skeleton, SkeletonRows, ToastProvider, useToast,
  useStateU, useEffectU, useRefU, useCallback, useContext, createContext });
