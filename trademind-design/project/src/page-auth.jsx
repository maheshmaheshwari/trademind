/* ===== Auth (login / register) ===== */
function AuthPage({ onAuth, theme, setTheme }){
  const D=DATA, F=D.fmt;
  const [mode,setMode]=useStateU("login");
  const [email,setEmail]=useStateU("arjun@trademind.ai");
  const [pw,setPw]=useStateU("••••••••");
  const [name,setName]=useStateU("");
  const [busy,setBusy]=useStateU(false);
  const [err,setErr]=useStateU("");
  const emailOk=/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email);
  function submit(e){ e.preventDefault();
    if(!emailOk){ setErr("Enter a valid email address"); return; }
    if(pw.length<6){ setErr("Password must be at least 6 characters"); return; }
    if(mode==="register"&&!name.trim()){ setErr("Please enter your name"); return; }
    setErr(""); setBusy(true); setTimeout(()=>{ setBusy(false); onAuth(); },800);
  }
  const preview=D.TOP_SIGNALS.slice(0,3);
  return (
    <div className="auth-grid" style={{minHeight:"100%",display:"grid",gridTemplateColumns:"1.1fr 1fr",background:"var(--bg)"}}>
      {/* showcase */}
      <div className="auth-show hide-sm" style={{position:"relative",overflow:"hidden",borderRight:"1px solid var(--border)",padding:"48px",display:"flex",flexDirection:"column",justifyContent:"space-between",background:"linear-gradient(160deg,var(--surface) 0%,var(--bg) 100%)"}}>
        <div style={{position:"absolute",top:"-12%",right:"-8%",width:420,height:420,borderRadius:"50%",background:"radial-gradient(circle,rgba(59,130,246,.22),transparent 70%)",pointerEvents:"none"}}/>
        <div className="row gap-sm" style={{position:"relative"}}>
          <span className="sb-logo"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></svg></span>
          <span className="sb-brand-name" style={{fontSize:19}}>Trade<b>Mind</b> <span style={{color:"var(--text-3)",fontWeight:600,fontSize:13}}>AI</span></span>
        </div>
        <div style={{position:"relative",maxWidth:440}}>
          <div className="pill" style={{color:"var(--gold)",background:"var(--gold-soft)",border:"none",marginBottom:18}}><Icon name="sparkle" size={13}/>AI-powered · Nifty 500</div>
          <h1 style={{fontSize:38,fontWeight:700,letterSpacing:"-.03em",lineHeight:1.12,margin:"0 0 14px"}}>Trade smarter with<br/>signals you can trust.</h1>
          <p style={{fontSize:15,color:"var(--text-2)",lineHeight:1.55,margin:0}}>Machine-learning signals across 498 stocks, real-time sentiment, and broker-synced execution — all in one terminal.</p>
          <div className="col" style={{gap:9,marginTop:26}}>
            {preview.map(s=>(
              <div key={s.id} className="row between" style={{padding:"11px 14px",background:"color-mix(in srgb,var(--surface) 70%,transparent)",border:"1px solid var(--border)",borderRadius:12,backdropFilter:"blur(8px)"}}>
                <SymbolCell s={s}/>
                <div className="row gap-sm"><Sparkline data={s.spark} color={s.change>=0?"var(--green)":"var(--red)"} w={70} h={26}/><SignalBadge signal={s.signal}/><span className="mono" style={{fontWeight:700,fontSize:13,color:"var(--green)"}}>{s.confidence}%</span></div>
              </div>
            ))}
          </div>
        </div>
        <div className="row gap-lg" style={{position:"relative",color:"var(--text-3)",fontSize:12.5}}>
          <span><b className="num" style={{color:"var(--text-2)"}}>498</b> stocks tracked</span>
          <span><b className="num" style={{color:"var(--text-2)"}}>68%</b> signal win rate</span>
          <span><b className="num" style={{color:"var(--text-2)"}}>15min</b> refresh</span>
        </div>
      </div>

      {/* form */}
      <div style={{display:"flex",flexDirection:"column",padding:"28px 32px",position:"relative"}}>
        <div className="row between">
          <span/>
          <button className="icon-btn" onClick={()=>setTheme(theme==="dark"?"light":"dark")}><Icon name={theme==="dark"?"sun":"moon"} size={19}/></button>
        </div>
        <div style={{flex:1,display:"flex",alignItems:"center",justifyContent:"center"}}>
          <div style={{width:"100%",maxWidth:380}}>
            <div className="col" style={{alignItems:"center",gap:14,marginBottom:26}}>
              <span className="sb-logo" style={{width:52,height:52,borderRadius:15}}><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></svg></span>
              <div className="col" style={{alignItems:"center",gap:4}}>
                <h2 style={{margin:0,fontSize:24,fontWeight:700,letterSpacing:"-.02em"}}>{mode==="login"?"Welcome back":"Create your account"}</h2>
                <p style={{margin:0,fontSize:13.5,color:"var(--text-2)"}}>{mode==="login"?"Sign in to your TradeMind terminal":"Start trading smarter in minutes"}</p>
              </div>
            </div>

            <button className="btn btn-ghost" style={{width:"100%",height:46,marginBottom:18}} onClick={onAuth}>
              <Icon name="google" size={18}/>Continue with Google
            </button>
            <div className="row gap-sm" style={{margin:"0 0 18px"}}><div className="divider" style={{flex:1}}/><span style={{fontSize:11.5,color:"var(--text-3)",fontWeight:500}}>OR</span><div className="divider" style={{flex:1}}/></div>

            <form onSubmit={submit}>
              {mode==="register" && (
                <div className="form-row"><label>Full name</label>
                  <div style={{position:"relative"}}><Icon name="user" size={17} style={{position:"absolute",left:13,top:14,color:"var(--text-3)"}}/><input style={{paddingLeft:40}} value={name} onChange={e=>setName(e.target.value)} placeholder="Arjun Kapoor"/></div>
                </div>
              )}
              <div className="form-row"><label>Email</label>
                <div style={{position:"relative"}}><Icon name="mail" size={17} style={{position:"absolute",left:13,top:14,color:"var(--text-3)"}}/><input style={{paddingLeft:40}} value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@email.com" type="email"/></div>
              </div>
              <div className="form-row"><label>Password</label>
                <div style={{position:"relative"}}><Icon name="lock" size={17} style={{position:"absolute",left:13,top:14,color:"var(--text-3)"}}/><input style={{paddingLeft:40}} value={pw} onChange={e=>setPw(e.target.value)} placeholder="••••••••" type="password"/></div>
              </div>
              {err && <div className="row gap-sm" style={{color:"var(--red)",fontSize:12.5,marginBottom:12}}><Icon name="alert" size={15}/>{err}</div>}
              {mode==="login" && <div className="row between" style={{marginBottom:16}}>
                <label className="row gap-sm" style={{fontSize:12.5,color:"var(--text-2)",cursor:"pointer"}}><input type="checkbox" defaultChecked style={{accentColor:"var(--accent)"}}/>Remember me</label>
                <a style={{fontSize:12.5,color:"var(--accent-2)",cursor:"pointer",textDecoration:"none"}}>Forgot password?</a>
              </div>}
              <button className="btn btn-primary" type="submit" style={{width:"100%",height:46}} disabled={busy}>
                {busy ? <Icon name="refresh" size={17} style={{animation:"spin 1s linear infinite"}}/> : <Icon name={mode==="login"?"logout":"plus"} size={17}/>}
                {busy?"Signing in…":mode==="login"?"Sign in":"Create account"}
              </button>
            </form>

            <p style={{textAlign:"center",fontSize:13,color:"var(--text-2)",marginTop:22}}>
              {mode==="login"?"New to TradeMind? ":"Already have an account? "}
              <a style={{color:"var(--accent-2)",fontWeight:600,cursor:"pointer",textDecoration:"none"}} onClick={()=>{setMode(mode==="login"?"register":"login");setErr("");}}>{mode==="login"?"Create an account":"Sign in"}</a>
            </p>
            <p style={{textAlign:"center",fontSize:11,color:"var(--text-3)",marginTop:26}}>Paper trading enabled by default · SEBI-registered broker integration</p>
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { AuthPage });
