/* ===== Main App ===== */
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "density": "balanced",
  "signalStyle": "rich",
  "accent": "#3B82F6"
}/*EDITMODE-END*/;

function App(){
  const [t,setTweak]=useTweaks(TWEAK_DEFAULTS);
  const [page,setPage]=useStateU("dashboard");
  const [settingsTab,setSettingsTab]=useStateU("profile");
  const [loggedIn,setLoggedIn]=useStateU(true);
  const [collapsed,setCollapsed]=useStateU(false);
  const [drawerStock,setDrawerStock]=useStateU(null);
  const toast=useToast();

  // apply theme + density + accent to root
  useEffectU(()=>{ const r=document.documentElement; r.dataset.theme=t.theme; r.dataset.density=t.density;
    r.style.setProperty("--accent", t.accent);
    // derive a soft accent
    r.style.setProperty("--accent-soft", t.accent+"22");
  },[t.theme,t.density,t.accent]);

  // expose nav helpers for in-page links
  useEffectU(()=>{ window.__setPage=setPage; window.__openStock=setDrawerStock;
    window.__openSettings=(tab)=>{ if(tab) setSettingsTab(tab); setPage("settings"); }; },[]);

  const openStock=(s)=>setDrawerStock(s);
  const pageProps={ openStock, toast, tweaks:t, setPage };

  if(!loggedIn){
    return <AuthPage onAuth={()=>{setLoggedIn(true);setPage("dashboard");toast({type:"success",title:"Welcome to TradeMind",msg:"Signed in · paper trading active"});}} theme={t.theme} setTheme={(v)=>setTweak("theme",v)}/>;
  }

  return (
    <div className="app">
      <div className="app-bg"/>
      <Sidebar page={page} setPage={setPage} collapsed={collapsed}/>
      <div className="main">
        <Navbar collapsed={collapsed} setCollapsed={setCollapsed} theme={t.theme} setTheme={(v)=>setTweak("theme",v)}
          marketOpen={true} onLogout={()=>{setLoggedIn(false);}}/>
        <div className="content">
          <div className="content-inner">
            {page==="dashboard" && <Dashboard {...pageProps}/>}
            {page==="signals" && <SignalsPage {...pageProps}/>}
            {page==="autopilot" && <AutopilotPage {...pageProps}/>}
            {page==="market" && <MarketPage {...pageProps}/>}
            {page==="portfolio" && <PortfolioPage {...pageProps}/>}
            {page==="trades" && <TradesPage {...pageProps}/>}
            {page==="watchlist" && <WatchlistPage {...pageProps}/>}
            {page==="settings" && <SettingsPage toast={toast} tweaks={t} setTweak={setTweak} onLogout={()=>setLoggedIn(false)} initialTab={settingsTab}/>}
          </div>
        </div>
      </div>

      {drawerStock && <StockDrawer stock={drawerStock} onClose={()=>setDrawerStock(null)} toast={toast}/>}

      <TweaksPanel title="Tweaks">
        <TweakSection label="Theme"/>
        <TweakRadio label="Mode" value={t.theme} options={["dark","light"]} onChange={v=>setTweak("theme",v)}/>
        <TweakColor label="Accent" value={t.accent} options={["#3B82F6","#8B5CF6","#14B8A6","#F59E0B"]} onChange={v=>setTweak("accent",v)}/>
        <TweakSection label="Layout"/>
        <TweakRadio label="Density" value={t.density} options={["compact","balanced","comfy"]} onChange={v=>setTweak("density",v)}/>
        <TweakSection label="Signal cards"/>
        <TweakRadio label="Style" value={t.signalStyle} options={["rich","compact","bold"]} onChange={v=>setTweak("signalStyle",v)}/>
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <ToastProvider><App/></ToastProvider>
);
