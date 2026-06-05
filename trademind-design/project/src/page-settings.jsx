/* ===== Settings Page ===== */
function Toggle({ on, onChange }){
  return <button className={"tgl "+(on?"on":"")} onClick={()=>onChange(!on)}><span className="tgl-knob"/></button>;
}

function SettingsPage({ toast, tweaks, setTweak, onLogout, initialTab }){
  const [tab,setTab]=useStateU(initialTab||"profile");
  useEffectU(()=>{ if(initialTab) setTab(initialTab); },[initialTab]);
  const [prefs,setPrefs]=useStateU({ tradeAlerts:true, signalAlerts:true, priceAlerts:true, news:false, weekly:true, email:true, push:true, sms:false });
  const set=(k,v)=>setPrefs(p=>({...p,[k]:v}));

  const TABS=[["profile","Profile","user"],["brokers","Brokers","shield"],["notifications","Notifications","bell"],["appearance","Appearance","sun"],["security","Security","lock"]];
  const BROKERS=[
    { name:"Angel One", status:"connected", detail:"GTT + live orders · synced 2m ago", color:"var(--green)" },
    { name:"Zerodha Kite", status:"connected", detail:"Holdings import · synced 14m ago", color:"var(--green)" },
    { name:"Upstox", status:"disconnected", detail:"Not connected", color:"var(--text-3)" },
    { name:"Groww", status:"disconnected", detail:"Not connected", color:"var(--text-3)" },
  ];

  return (
    <div className="page-fade">
      <div className="page-head">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-sub">Manage your account, brokers, and preferences</p>
        </div>
      </div>

      <div className="settings-layout">
        {/* side tabs */}
        <div className="settings-nav">
          {TABS.map(([id,label,ic])=>(
            <button key={id} className={"sb-item "+(tab===id?"active":"")} onClick={()=>setTab(id)}><Icon name={ic} size={19}/><span>{label}</span></button>
          ))}
          <div className="divider" style={{margin:"8px 0"}}/>
          <button className="sb-item" style={{color:"var(--red)"}} onClick={onLogout}><Icon name="logout" size={19}/><span>Log out</span></button>
        </div>

        {/* panel */}
        <div className="col" style={{gap:"calc(16px * var(--u))",minWidth:0}}>
          {tab==="profile" && <>
            <Card title="Profile" sub="Your personal information" icon="user">
              <div className="card-pad" style={{display:"flex",flexDirection:"column",gap:16}}>
                <div className="row gap-sm" style={{alignItems:"center"}}>
                  <span className="avatar" style={{width:64,height:64,borderRadius:18,fontSize:24}}>AK</span>
                  <div className="col" style={{gap:5}}>
                    <button className="btn btn-ghost btn-sm">Change photo</button>
                    <span className="dim" style={{fontSize:11.5}}>JPG or PNG · max 2MB</span>
                  </div>
                </div>
                <div className="settings-grid">
                  <div className="form-row"><label>Full name</label><input defaultValue="Arjun Kapoor"/></div>
                  <div className="form-row"><label>Email</label><input defaultValue="arjun@trademind.ai" type="email"/></div>
                  <div className="form-row"><label>Phone</label><input defaultValue="+91 98765 43210"/></div>
                  <div className="form-row"><label>PAN</label><input defaultValue="ABCDE1234F"/></div>
                </div>
              </div>
            </Card>
            <Card title="Trading Preferences" sub="Defaults for new orders" icon="trades">
              <div className="card-pad settings-grid">
                <div className="form-row"><label>Default account</label><select defaultValue="Paper"><option>Paper</option><option>Live</option></select></div>
                <div className="form-row"><label>Default order type</label><select><option>Market</option><option>Limit</option><option>GTT</option></select></div>
                <div className="form-row"><label>Risk per trade</label><select defaultValue="2% of capital"><option>1% of capital</option><option>2% of capital</option><option>5% of capital</option></select></div>
                <div className="form-row"><label>Base currency</label><select><option>INR (₹)</option></select></div>
              </div>
            </Card>
            <div className="row" style={{justifyContent:"flex-end",gap:10}}>
              <button className="btn btn-ghost">Cancel</button>
              <button className="btn btn-primary" onClick={()=>toast({type:"success",title:"Profile saved",msg:"Your changes have been updated"})}><Icon name="check" size={17}/>Save changes</button>
            </div>
          </>}

          {tab==="brokers" && (
            <Card title="Broker Connections" sub="Link your brokers for live data & execution" icon="shield">
              <div className="card-pad col" style={{gap:11}}>
                {BROKERS.map(b=>(
                  <div key={b.name} className="broker-row">
                    <div className="row gap-sm">
                      <span className="broker-logo" style={{color:b.color,background:b.color+"1f"}}><Icon name="shield" size={20}/></span>
                      <div className="col" style={{gap:1}}>
                        <span style={{fontWeight:600,fontSize:14}}>{b.name}</span>
                        <span className="dim" style={{fontSize:12}}>{b.detail}</span>
                      </div>
                    </div>
                    {b.status==="connected"
                      ? <div className="row gap-sm"><span className="pill" style={{color:"var(--green)",background:"var(--green-soft)",border:"none"}}><Icon name="checkCircle" size={12}/>Connected</span><button className="btn btn-ghost btn-sm" onClick={()=>toast({type:"info",title:`${b.name} disconnected`})}>Disconnect</button></div>
                      : <button className="btn btn-primary btn-sm" onClick={()=>toast({type:"success",title:`${b.name} connected`,msg:"Authorize in the broker popup to finish"})}><Icon name="plus" size={15}/>Connect</button>}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {tab==="notifications" && (
            <Card title="Notifications" sub="Choose what you get alerted about" icon="bell">
              <div className="card-pad col" style={{gap:0}}>
                {[["tradeAlerts","Trade executions","When an order is filled or closed"],
                  ["signalAlerts","New AI signals","When a watchlist stock gets a new BUY/SELL"],
                  ["priceAlerts","Price alerts","When a stock crosses your set thresholds"],
                  ["news","Breaking news","High-impact market news & events"],
                  ["weekly","Weekly digest","Performance summary every Monday"]].map(([k,t,d],i)=>(
                  <div key={k} className="pref-row" style={{borderTop:i?"1px solid var(--border)":"none"}}>
                    <div className="col" style={{gap:1}}><span style={{fontWeight:600,fontSize:13.5}}>{t}</span><span className="dim" style={{fontSize:12}}>{d}</span></div>
                    <Toggle on={prefs[k]} onChange={v=>set(k,v)}/>
                  </div>
                ))}
                <div className="divider" style={{margin:"14px 0 4px"}}/>
                <span className="field-label" style={{padding:"6px 0"}}>Delivery channels</span>
                <div className="row gap-sm wrap">
                  {[["email","Email"],["push","Push"],["sms","SMS"]].map(([k,t])=>(
                    <button key={k} className={"chan-btn "+(prefs[k]?"on":"")} onClick={()=>set(k,!prefs[k])}>
                      {prefs[k] && <Icon name="check" size={14}/>}{t}
                    </button>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {tab==="appearance" && (
            <Card title="Appearance" sub="Customize how TradeMind looks" icon="sun">
              <div className="card-pad col" style={{gap:18}}>
                <div className="col" style={{gap:9}}>
                  <span className="field-label">Theme</span>
                  <div className="row gap-sm">
                    {["dark","light"].map(m=>(
                      <button key={m} className={"theme-swatch "+(tweaks.theme===m?"on":"")} onClick={()=>setTweak("theme",m)}>
                        <span className="ts-prev" style={{background:m==="dark"?"#0A0E1A":"#EEF1F8"}}><span style={{background:m==="dark"?"#111827":"#fff"}}/><span style={{background:m==="dark"?"#111827":"#fff"}}/></span>
                        <span style={{textTransform:"capitalize",fontSize:13,fontWeight:600}}>{m}{tweaks.theme===m && <Icon name="check" size={14} style={{marginLeft:6,color:"var(--accent-2)",verticalAlign:"middle"}}/>}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="col" style={{gap:9}}>
                  <span className="field-label">Accent color</span>
                  <div className="row gap-sm">
                    {["#3B82F6","#8B5CF6","#14B8A6","#F59E0B"].map(c=>(
                      <button key={c} className={"accent-dot "+(tweaks.accent===c?"on":"")} style={{background:c}} onClick={()=>setTweak("accent",c)}>{tweaks.accent===c && <Icon name="check" size={15}/>}</button>
                    ))}
                  </div>
                </div>
                <div className="col" style={{gap:9}}>
                  <span className="field-label">Density</span>
                  <div className="seg">{["compact","balanced","comfy"].map(d=><button key={d} className={tweaks.density===d?"on":""} onClick={()=>setTweak("density",d)} style={{textTransform:"capitalize"}}>{d}</button>)}</div>
                </div>
                <div className="col" style={{gap:9}}>
                  <span className="field-label">Signal card style</span>
                  <div className="seg">{["rich","compact","bold"].map(d=><button key={d} className={tweaks.signalStyle===d?"on":""} onClick={()=>setTweak("signalStyle",d)} style={{textTransform:"capitalize"}}>{d}</button>)}</div>
                </div>
              </div>
            </Card>
          )}

          {tab==="security" && <>
            <Card title="Password" sub="Update your password" icon="lock">
              <div className="card-pad settings-grid">
                <div className="form-row"><label>Current password</label><input type="password" defaultValue="••••••••"/></div>
                <div/>
                <div className="form-row"><label>New password</label><input type="password" placeholder="••••••••"/></div>
                <div className="form-row"><label>Confirm new</label><input type="password" placeholder="••••••••"/></div>
              </div>
              <div className="card-pad" style={{paddingTop:0}}><button className="btn btn-primary" onClick={()=>toast({type:"success",title:"Password updated"})}>Update password</button></div>
            </Card>
            <Card title="Two-Factor Authentication" sub="Extra layer of security" icon="shield">
              <div className="card-pad">
                <div className="pref-row" style={{padding:0}}>
                  <div className="col" style={{gap:1}}><span style={{fontWeight:600,fontSize:13.5}}>Authenticator app</span><span className="dim" style={{fontSize:12}}>Use Google Authenticator or Authy</span></div>
                  <span className="pill" style={{color:"var(--green)",background:"var(--green-soft)",border:"none"}}><Icon name="checkCircle" size={12}/>Enabled</span>
                </div>
              </div>
            </Card>
            <Card title="Active Sessions" sub="Devices signed into your account" icon="eye">
              <div className="card-pad col" style={{gap:11}}>
                {[["MacBook Pro · Mumbai","Current session · Chrome","var(--green)",true],["iPhone 15 · Mumbai","2 hours ago · TradeMind app",null,false]].map(([dev,meta,c,cur])=>(
                  <div key={dev} className="pref-row" style={{padding:"4px 0"}}>
                    <div className="col" style={{gap:1}}><span style={{fontWeight:600,fontSize:13.5}}>{dev}</span><span className="dim" style={{fontSize:12}}>{meta}</span></div>
                    {cur ? <span className="pill" style={{color:"var(--green)",background:"var(--green-soft)",border:"none"}}>Active now</span> : <button className="btn btn-danger btn-sm" onClick={()=>toast({type:"info",title:"Session revoked"})}>Revoke</button>}
                  </div>
                ))}
              </div>
            </Card>
          </>}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SettingsPage });
