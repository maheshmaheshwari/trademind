/* ===== App shell: Sidebar + Navbar ===== */
const NAV = [
  { id:"dashboard", label:"Dashboard", icon:"dashboard" },
  { id:"signals", label:"AI Signals", icon:"signals" },
  { id:"autopilot", label:"AI Authorized", icon:"brain" },
  { id:"market", label:"Market Overview", icon:"market" },
  { id:"portfolio", label:"Portfolio", icon:"portfolio" },
  { id:"trades", label:"Trades & Orders", icon:"trades" },
];

function Sidebar({ page, setPage, collapsed }){
  return (
    <aside className={"sidebar "+(collapsed?"collapsed":"")}>
      <div className="sb-brand">
        <span className="sb-logo">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></svg>
        </span>
        <span className="sb-brand-name">Trade<b>Mind</b></span>
      </div>
      <nav className="sb-nav">
        <div className="sb-section-label">Trading</div>
        {NAV.map(n=>(
          <button key={n.id} className={"sb-item "+(page===n.id?"active":"")} onClick={()=>setPage(n.id)} title={n.label}>
            <Icon name={n.icon} size={20}/>
            <span className="sb-label">{n.label}</span>
          </button>
        ))}
        <div className="sb-section-label">Account</div>
        <button className={"sb-item "+(page==="watchlist"?"active":"")} onClick={()=>setPage("watchlist")} title="Watchlist">
          <Icon name="bookmark" size={20}/><span className="sb-label">Watchlist</span>
        </button>
        <button className={"sb-item "+(page==="settings"?"active":"")} onClick={()=>setPage("settings")} title="Settings">
          <Icon name="settings" size={20}/><span className="sb-label">Settings</span>
        </button>
      </nav>
      <div className="sb-foot">
        <div className="sb-user" onClick={()=>setPage("settings")}>
          <span className="avatar" style={{width:34,height:34,borderRadius:10,fontSize:13}}>AK</span>
          <div className="sb-foot-txt col" style={{gap:0}}>
            <span className="nm">Arjun Kapoor</span>
            <span className="em">arjun@trademind.ai</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

function Navbar({ collapsed, setCollapsed, theme, setTheme, marketOpen, onLogout }){
  const [menu,setMenu]=useStateU(false);
  const [notifOpen,setNotifOpen]=useStateU(false);
  const [notifs,setNotifs]=useStateU(DATA.NOTIFS);
  const unread=notifs.filter(n=>n.unread).length;
  function markAllRead(){ setNotifs(ns=>ns.map(n=>({...n,unread:false}))); }
  function goSettings(tab){ setMenu(false); setNotifOpen(false); window.__openSettings?window.__openSettings(tab):window.__setPage&&window.__setPage("settings"); }
  return (
    <header className="navbar">
      <button className="nav-collapse" onClick={()=>setCollapsed(c=>!c)} title="Toggle sidebar"><Icon name="panelLeft" size={18}/></button>
      <div className="nav-search">
        <Icon name="search" size={17}/>
        <input placeholder="Search stocks, signals, sectors…"/>
        <kbd>⌘K</kbd>
      </div>
      <div className="nav-spacer"/>
      <div className="nav-right">
        <div className={"mkt-status "+(marketOpen?"open":"closed")}>
          <span className="dot"/>{marketOpen?"MARKET OPEN":"MARKET CLOSED"}
          <span className="hide-sm" style={{color:"var(--text-3)",fontWeight:500,fontSize:11.5,fontFamily:"var(--font-mono)"}}>NSE · 15:24</span>
        </div>
        <div style={{position:"relative"}}>
          <button className="icon-btn" title="Notifications" onClick={()=>setNotifOpen(o=>!o)}><Icon name="bell" size={19}/>{unread>0 && <span className="badge-dot"/>}</button>
          {notifOpen && <>
            <div style={{position:"fixed",inset:0,zIndex:9}} onClick={()=>setNotifOpen(false)}/>
            <div className="notif-pop">
              <div className="notif-head">
                <div className="row gap-sm"><span style={{fontWeight:700,fontSize:14.5}}>Notifications</span>{unread>0 && <span className="pill" style={{color:"var(--accent-2)",background:"var(--accent-soft)",border:"none"}}>{unread} new</span>}</div>
                <button className="notif-link" onClick={markAllRead}>Mark all read</button>
              </div>
              <div className="notif-list">
                {notifs.map(n=>(
                  <div key={n.id} className={"notif-item "+(n.unread?"unread":"")}>
                    <span className="notif-ic" style={{color:n.color,background:n.color+"1f"}}><Icon name={n.icon} size={16}/></span>
                    <div className="col" style={{gap:2,flex:1,minWidth:0}}>
                      <div className="row between" style={{gap:8}}><span style={{fontWeight:600,fontSize:13,whiteSpace:"nowrap"}}>{n.title}</span><span className="dim" style={{fontSize:11,whiteSpace:"nowrap",flexShrink:0}}>{n.time}</span></div>
                      <span style={{fontSize:12,color:"var(--text-2)",lineHeight:1.4}}>{n.msg}</span>
                    </div>
                  </div>
                ))}
              </div>
              <button className="notif-foot" onClick={()=>goSettings("notifications")}><Icon name="settings" size={15}/>Notification settings</button>
            </div>
          </>}
        </div>
        <button className="icon-btn" onClick={()=>setTheme(theme==="dark"?"light":"dark")} title="Toggle theme">
          <Icon name={theme==="dark"?"sun":"moon"} size={19}/>
        </button>
        <div style={{position:"relative"}}>
          <button className="avatar" onClick={()=>setMenu(m=>!m)}>AK</button>
          {menu && <>
            <div style={{position:"fixed",inset:0,zIndex:9}} onClick={()=>setMenu(false)}/>
            <div style={{position:"absolute",right:0,top:46,width:210,background:"var(--surface)",border:"1px solid var(--border-strong)",borderRadius:13,boxShadow:"var(--shadow-lg)",zIndex:10,overflow:"hidden",padding:6}}>
              <div style={{padding:"10px 12px"}}><div style={{fontWeight:600,fontSize:13.5}}>Arjun Kapoor</div><div style={{fontSize:11.5,color:"var(--text-3)"}}>Pro · Paper + Live</div></div>
              <div className="divider"/>
              {[["user","Profile","profile"],["settings","Preferences","appearance"],["bell","Notifications","notifications"],["shield","Security","security"]].map(m=>(
                <button key={m[1]} className="sb-item" style={{height:38}} onClick={()=>goSettings(m[2])}><Icon name={m[0]} size={17}/><span>{m[1]}</span></button>
              ))}
              <div className="divider"/>
              <button className="sb-item" style={{height:38,color:"var(--red)"}} onClick={()=>{setMenu(false);onLogout();}}><Icon name="logout" size={17}/><span>Log out</span></button>
            </div>
          </>}
        </div>
      </div>
    </header>
  );
}

Object.assign(window, { Sidebar, Navbar, NAV });
