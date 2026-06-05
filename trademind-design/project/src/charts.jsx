/* ===== Hand-built charts (SVG) ===== */
const { useState: useStateC, useRef: useRefC, useEffect: useEffectC, useId } = React;

// normalize points into 0..1 of height
function norm(data){ const min=Math.min(...data), max=Math.max(...data); const r=(max-min)||1; return data.map(d=>(d-min)/r); }

function Sparkline({ data, color="var(--accent)", w=120, h=36, fill=false, sw=2 }){
  const n=norm(data); const step=w/(n.length-1);
  const pts=n.map((v,i)=>[i*step, h-2-v*(h-4)]);
  const d=pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  const gid=useId().replace(/:/g,"");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{display:"block"}}>
      {fill && <>
        <defs><linearGradient id={"sg"+gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity=".28"/><stop offset="1" stopColor={color} stopOpacity="0"/>
        </linearGradient></defs>
        <path d={`${d} L ${w} ${h} L 0 ${h} Z`} fill={`url(#sg${gid})`}/>
      </>}
      <path d={d} fill="none" stroke={color} strokeWidth={sw} strokeLinejoin="round" strokeLinecap="round"/>
    </svg>
  );
}

// Area chart with axis grid + hover crosshair
function AreaChart({ data, color="var(--accent)", h=230, labels }){
  const wrapRef=useRefC(null); const [w,setW]=useStateC(640); const [hi,setHi]=useStateC(null);
  useEffectC(()=>{ const el=wrapRef.current; if(!el) return; const ro=new ResizeObserver(()=>setW(el.clientWidth)); ro.observe(el); setW(el.clientWidth); return ()=>ro.disconnect(); },[]);
  const padL=4, padR=4, padB=22, padT=10;
  const innerW=w-padL-padR, innerH=h-padB-padT;
  const min=Math.min(...data), max=Math.max(...data), r=(max-min)||1;
  const x=i=>padL+i*(innerW/(data.length-1));
  const y=v=>padT+innerH-((v-min)/r)*innerH;
  const line=data.map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1)).join(" ");
  const gid=useId().replace(/:/g,"");
  const grid=[0,.25,.5,.75,1];
  function move(e){ const rect=e.currentTarget.getBoundingClientRect(); const px=e.clientX-rect.left; const i=Math.round(((px-padL)/innerW)*(data.length-1)); setHi(Math.max(0,Math.min(data.length-1,i))); }
  return (
    <div ref={wrapRef} style={{position:"relative"}}>
      <svg width={w} height={h} onMouseMove={move} onMouseLeave={()=>setHi(null)} style={{display:"block",cursor:"crosshair"}}>
        <defs><linearGradient id={"ac"+gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity=".22"/><stop offset="1" stopColor={color} stopOpacity="0"/>
        </linearGradient></defs>
        {grid.map((g,i)=><line key={i} x1={padL} x2={w-padR} y1={padT+g*innerH} y2={padT+g*innerH} stroke="var(--grid)" strokeWidth="1"/>)}
        <path d={`${line} L ${x(data.length-1)} ${padT+innerH} L ${padL} ${padT+innerH} Z`} fill={`url(#ac${gid})`}/>
        <path d={line} fill="none" stroke={color} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round"/>
        {hi!=null && <>
          <line x1={x(hi)} x2={x(hi)} y1={padT} y2={padT+innerH} stroke="var(--border-strong)" strokeWidth="1" strokeDasharray="3 3"/>
          <circle cx={x(hi)} cy={y(data[hi])} r="4.5" fill={color} stroke="var(--surface)" strokeWidth="2"/>
        </>}
      </svg>
      {labels && <div style={{display:"flex",justifyContent:"space-between",fontSize:11,color:"var(--text-3)",fontFamily:"var(--font-mono)",marginTop:2}}>
        {labels.map((l,i)=><span key={i}>{l}</span>)}
      </div>}
      {hi!=null && <div style={{position:"absolute",top:6,left:Math.min(Math.max(x(hi)-46,0),w-100),background:"var(--surface-3)",border:"1px solid var(--border-strong)",borderRadius:8,padding:"5px 9px",fontSize:11.5,pointerEvents:"none",boxShadow:"var(--shadow-md)",whiteSpace:"nowrap"}}>
        <span className="mono" style={{fontWeight:700}}>{DATA.fmt.inrCompact(Math.round(data[hi]))}</span>
      </div>}
    </div>
  );
}

// Donut chart with center label
function Donut({ data, size=170, thickness=26, centerTop, centerBottom }){
  const total=data.reduce((a,d)=>a+d.val,0);
  const R=(size-thickness)/2, C=2*Math.PI*R; let acc=0;
  const [hov,setHov]=useStateC(null);
  return (
    <div style={{display:"flex",alignItems:"center",gap:18,flexWrap:"wrap"}}>
      <svg width={size} height={size} style={{flexShrink:0}}>
        <g transform={`rotate(-90 ${size/2} ${size/2})`}>
          {data.map((d,i)=>{
            const frac=d.val/total; const len=frac*C; const off=acc*C; acc+=frac;
            return <circle key={i} cx={size/2} cy={size/2} r={R} fill="none" stroke={d.color}
              strokeWidth={hov===i?thickness+4:thickness} strokeDasharray={`${len} ${C-len}`} strokeDashoffset={-off}
              style={{transition:"stroke-width .15s",cursor:"pointer",opacity:hov==null||hov===i?1:.45}}
              onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}/>;
          })}
        </g>
        <text x="50%" y="46%" textAnchor="middle" style={{fontSize:13,fill:"var(--text-3)",fontWeight:600}}>{hov!=null?data[hov].sector:centerTop}</text>
        <text x="50%" y="60%" textAnchor="middle" className="mono" style={{fontSize:18,fill:"var(--text)",fontWeight:700}}>{hov!=null?Math.round(data[hov].val/total*100)+"%":centerBottom}</text>
      </svg>
      <div style={{display:"flex",flexDirection:"column",gap:8,flex:1,minWidth:140}}>
        {data.map((d,i)=>(
          <div key={i} className="row between" style={{opacity:hov==null||hov===i?1:.5,transition:"opacity .15s",cursor:"pointer"}} onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}>
            <div className="row gap-sm"><span style={{width:10,height:10,borderRadius:3,background:d.color}}/><span style={{fontSize:12.5,fontWeight:500}}>{d.sector}</span></div>
            <span className="mono" style={{fontSize:12.5,color:"var(--text-2)"}}>{Math.round(d.val/total*100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Sentiment gauge (semicircle)
function Gauge({ value, size=190 }){
  const w=size, h=size*0.62; const cx=w/2, cy=h-6, R=w/2-16;
  const ang=Math.PI*(1-value/100); // 180deg..0
  const nx=cx+R*Math.cos(ang), ny=cy-R*Math.sin(ang);
  const arc=(from,to,color,sw)=>{ const a0=Math.PI*(1-from/100), a1=Math.PI*(1-to/100);
    const x0=cx+R*Math.cos(a0), y0=cy-R*Math.sin(a0), x1=cx+R*Math.cos(a1), y1=cy-R*Math.sin(a1);
    const large=(to-from)>50?1:0; return <path d={`M ${x0} ${y0} A ${R} ${R} 0 ${large} 1 ${x1} ${y1}`} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round"/>; };
  const label = value>=60?"BULLISH":value>=40?"NEUTRAL":"BEARISH";
  const col = value>=60?"var(--green)":value>=40?"var(--gold)":"var(--red)";
  return (
    <div className="gauge-wrap">
      <svg width={w} height={h+30}>
        <path d={`M ${cx-R} ${cy} A ${R} ${R} 0 0 1 ${cx+R} ${cy}`} fill="none" stroke="var(--surface-3)" strokeWidth="13" strokeLinecap="round"/>
        {arc(0,33,"var(--red)",13)}{arc(34,66,"var(--gold)",13)}{arc(67,100,"var(--green)",13)}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--text)" strokeWidth="3" strokeLinecap="round"/>
        <circle cx={cx} cy={cy} r="6" fill="var(--text)"/>
        <text x={cx} y={cy-R-2} textAnchor="middle" className="mono" style={{fontSize:26,fontWeight:700,fill:col}}>{value}</text>
      </svg>
      <span className="gauge-label" style={{color:col}}>{label}</span>
      <span style={{fontSize:11.5,color:"var(--text-3)"}}>News sentiment score</span>
    </div>
  );
}

// Diverging bar chart (FII/DII)
function FlowBars({ data, h=210 }){
  const wrapRef=useRefC(null); const [w,setW]=useStateC(620);
  useEffectC(()=>{ const el=wrapRef.current; if(!el) return; const ro=new ResizeObserver(()=>setW(el.clientWidth)); ro.observe(el); setW(el.clientWidth); return ()=>ro.disconnect(); },[]);
  const all=data.flatMap(d=>[d.fii,d.dii]); const max=Math.max(...all.map(Math.abs))*1.1;
  const padB=24, padT=8; const innerH=h-padB-padT; const zero=padT+innerH/2;
  const groupW=w/data.length; const bw=Math.min(15, groupW*0.3);
  const barH=v=>(Math.abs(v)/max)*(innerH/2);
  const [hov,setHov]=useStateC(null);
  return (
    <div ref={wrapRef} style={{position:"relative"}}>
      <svg width={w} height={h} style={{display:"block"}}>
        <line x1="0" x2={w} y1={zero} y2={zero} stroke="var(--border-strong)" strokeWidth="1"/>
        {data.map((d,i)=>{ const cx=i*groupW+groupW/2;
          return <g key={i} onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)} style={{cursor:"pointer"}}>
            <rect x={cx-bw-2} y={d.fii>=0?zero-barH(d.fii):zero} width={bw} height={barH(d.fii)} rx="2" fill="var(--accent)" opacity={hov==null||hov===i?.95:.4}/>
            <rect x={cx+2} y={d.dii>=0?zero-barH(d.dii):zero} width={bw} height={barH(d.dii)} rx="2" fill="var(--gold)" opacity={hov==null||hov===i?.95:.4}/>
            <text x={cx} y={h-7} textAnchor="middle" style={{fontSize:10.5,fill:"var(--text-3)"}}>{d.day}</text>
          </g>; })}
      </svg>
      {hov!=null && <div style={{position:"absolute",top:4,left:Math.min(hov*groupW,w-160),background:"var(--surface-3)",border:"1px solid var(--border-strong)",borderRadius:8,padding:"7px 10px",fontSize:11.5,pointerEvents:"none",boxShadow:"var(--shadow-md)",minWidth:140}}>
        <div className="row between" style={{gap:14}}><span style={{color:"var(--accent-2)"}}>● FII</span><span className="mono" style={{color:data[hov].fii>=0?"var(--green)":"var(--red)"}}>{DATA.fmt.signed(data[hov].fii,0)} Cr</span></div>
        <div className="row between" style={{gap:14,marginTop:3}}><span style={{color:"var(--gold)"}}>● DII</span><span className="mono" style={{color:data[hov].dii>=0?"var(--green)":"var(--red)"}}>{DATA.fmt.signed(data[hov].dii,0)} Cr</span></div>
      </div>}
      <div className="row gap-lg" style={{justifyContent:"center",marginTop:6,fontSize:11.5,color:"var(--text-2)"}}>
        <span><span style={{color:"var(--accent-2)"}}>●</span> FII Net</span>
        <span><span style={{color:"var(--gold)"}}>●</span> DII Net</span>
      </div>
    </div>
  );
}

window.Charts = { Sparkline, AreaChart, Donut, Gauge, FlowBars };
Object.assign(window, { Sparkline, AreaChart, Donut, Gauge, FlowBars });
