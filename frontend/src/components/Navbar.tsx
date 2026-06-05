import { useState, useRef, useEffect } from 'react';

function relativeTime(isoStr: string): string {
  const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
  if (diff < 60)   return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
import { useNavigate } from 'react-router-dom';
import { PanelLeft, Search, Bell, Sun, Moon, User, Settings, Shield, LogOut } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { useGetNotificationsQuery, useMarkNotificationsReadMutation } from '../services/tradeMindApiService';

interface NavbarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Navbar({ collapsed, onToggle }: NavbarProps) {
  const { user, logout } = useAuth();
  const { theme, toggleTheme, density, setDensity } = useTheme();
  const navigate = useNavigate();
  const [menuOpen,  setMenuOpen]  = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const menuRef  = useRef<HTMLDivElement>(null);
  const notifRef = useRef<HTMLDivElement>(null);

  const { data: notifData } = useGetNotificationsQuery(undefined, { pollingInterval: 60000 });
  const [markRead] = useMarkNotificationsReadMutation();
  const unread: number = (notifData as any)?.unread ?? 0;

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (menuRef.current  && !menuRef.current.contains(e.target as Node))  setMenuOpen(false);
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) setNotifOpen(false);
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const initials = user?.display_name
    ? user.display_name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  // Determine market open (9:15 AM – 3:30 PM IST Mon–Fri)
  const now = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  const day = now.getDay();
  const mins = now.getHours() * 60 + now.getMinutes();
  const marketOpen = day >= 1 && day <= 5 && mins >= 555 && mins <= 930;

  function handleLogout() {
    setMenuOpen(false);
    logout();
    navigate('/');
  }

  return (
    <header
      className="flex-shrink-0 flex items-center gap-3.5 border-b border-line bg-surface/70 backdrop-blur-[14px] relative z-[4]"
      style={{ height: 'var(--navbar-h)', padding: '0 calc(24px * var(--u))' }}
    >
      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="w-9 h-9 rounded-[9px] border border-line bg-transparent text-ink-2 grid place-items-center hover:bg-surface-hover hover:text-ink transition-colors flex-shrink-0"
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <PanelLeft size={18} />
      </button>

      {/* Brand — only visible when sidebar is collapsed */}
      {collapsed && (
        <div className="flex items-center gap-2 flex-shrink-0">
          <span
            className="w-[30px] h-[30px] rounded-[9px] grid place-items-center flex-shrink-0"
            style={{ background: 'linear-gradient(135deg,var(--accent),#1E40AF)', boxShadow: '0 4px 14px rgba(59,130,246,.4)' }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/>
            </svg>
          </span>
          <span className="font-bold text-[15px] tracking-tight text-ink whitespace-nowrap">
            Trade<b style={{ color: 'var(--accent-2)' }}>Mind</b>
          </span>
        </div>
      )}

      {/* Search */}
      <div className="relative flex-1 max-w-[440px]">
        <Search size={17} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" />
        <input
          placeholder="Search stocks, signals, sectors…"
          className="w-full h-10 pl-10 pr-14 rounded-[11px] border border-line bg-surface-2 text-ink text-[13.5px] font-sans outline-none transition-colors placeholder:text-ink-3 focus:border-accent focus:bg-surface"
        />
        <kbd className="absolute right-3 top-1/2 -translate-y-1/2 font-mono text-[11px] text-ink-3 border border-line rounded-[5px] px-1.5 py-px">⌘K</kbd>
      </div>

      <div className="flex-1" />

      {/* Right actions */}
      <div className="flex items-center gap-2.5">
        {/* Market status */}
        <div className={`flex items-center gap-2 h-9 px-3 rounded-full text-[12.5px] font-semibold border ${marketOpen ? 'text-gain bg-gain-soft border-line' : 'text-loss bg-loss-soft border-line'}`}>
          <span className={`w-2 h-2 rounded-full ${marketOpen ? 'bg-gain animate-pulse-dot' : 'bg-loss'}`} />
          {marketOpen ? 'MARKET OPEN' : 'MARKET CLOSED'}
          <span className="hidden sm:inline text-ink-3 font-normal font-mono text-[11.5px]">NSE</span>
        </div>

        {/* Notifications */}
        <div className="relative hidden sm:block" ref={notifRef}>
          <button
            onClick={() => { setNotifOpen(o => !o); if (!notifOpen && unread > 0) markRead(); }}
            className="w-[38px] h-[38px] rounded-[10px] border border-line bg-transparent text-ink-2 grid place-items-center hover:bg-surface-hover hover:text-ink transition-colors relative">
            <Bell size={19} />
            {unread > 0 && <span className="absolute top-[7px] right-[8px] w-[7px] h-[7px] rounded-full bg-gold border-2 border-surface" />}
          </button>
          {notifOpen && (
            <div className="notif-pop">
              <div className="notif-head">
                <span className="text-[14px] font-semibold text-ink">Notifications</span>
                {unread > 0 && <span className="inline-flex items-center h-[20px] px-[8px] rounded-full text-[11px] font-bold bg-gold-soft text-gold">{unread} new</span>}
              </div>
              <div className="notif-list">
                {((notifData as any)?.data ?? []).slice(0, 6).map((n: any) => (
                  <div key={n.id} className={`notif-item ${!n.is_read ? 'unread' : ''}`}>
                    <span className="notif-ic" style={{ background: (n.color || '#3B82F6') + '22', color: n.color || '#3B82F6' }}>{n.icon}</span>
                    <div className="flex flex-col gap-[2px] min-w-0">
                      <span className="text-[13px] font-semibold text-ink truncate">{n.title}</span>
                      <span className="text-[12px] text-ink-2 truncate">{n.message}</span>
                      <span className="text-[11px] text-ink-3">{n.created_at ? relativeTime(n.created_at) : ''}</span>
                    </div>
                  </div>
                ))}
                {!(notifData as any)?.data?.length && (
                  <div className="text-center py-8 text-[13px] text-ink-3">No notifications yet</div>
                )}
              </div>
              <button className="notif-foot" onClick={() => { navigate('/settings', { state: { tab: 'notifications' } }); setNotifOpen(false); }}>
                <Settings size={14} /> Notification settings
              </button>
            </div>
          )}
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="w-[38px] h-[38px] rounded-[10px] border border-line bg-transparent text-ink-2 grid place-items-center hover:bg-surface-hover hover:text-ink transition-colors"
          title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
        >
          {theme === 'dark' ? <Sun size={19} /> : <Moon size={19} />}
        </button>

        {/* Avatar + dropdown */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(o => !o)}
            className="w-[38px] h-[38px] rounded-[11px] bg-gradient-to-br from-indigo-500 to-accent grid place-items-center font-bold text-[14px] text-white flex-shrink-0"
          >
            {initials}
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-[46px] bg-surface border border-line-strong rounded-[13px] shadow-lg z-10 overflow-hidden p-1.5" style={{ width: 230 }}>
              {/* User info */}
              <div className="px-3 py-2.5">
                <div className="font-semibold text-[13.5px] text-ink">{user?.display_name ?? 'User'}</div>
                <div className="text-[11.5px] text-ink-3 mt-0.5">{user?.username} · Paper trading</div>
              </div>

              <div className="h-px bg-line mx-1.5 my-1" />

              {/* Density control */}
              <div className="px-3 py-2">
                <div className="text-[10.5px] font-semibold tracking-[.08em] uppercase text-ink-3 mb-2">Density</div>
                <div className="flex gap-1">
                  {(['compact', 'balanced', 'comfy'] as const).map(d => (
                    <button
                      key={d}
                      onClick={() => setDensity(d)}
                      className={`flex-1 h-7 rounded-[7px] text-[11.5px] font-semibold border-none cursor-pointer capitalize transition-colors ${
                        density === d
                          ? 'bg-accent text-white'
                          : 'bg-transparent text-ink-3 hover:text-ink hover:bg-surface-hover'
                      }`}
                    >
                      {d[0].toUpperCase() + d.slice(1, 3)}
                    </button>
                  ))}
                </div>
              </div>

              <div className="h-px bg-line mx-1.5 my-1" />

              {/* Nav links */}
              <button
                onClick={() => { navigate('/settings'); setMenuOpen(false); }}
                className="flex items-center gap-3 w-full h-[38px] px-3 rounded-[9px] text-[14px] font-medium text-ink-2 hover:bg-surface-hover hover:text-ink transition-colors border-none bg-transparent"
              >
                <Settings size={17} /> Preferences
              </button>
              <button
                onClick={() => setMenuOpen(false)}
                className="flex items-center gap-3 w-full h-[38px] px-3 rounded-[9px] text-[14px] font-medium text-ink-2 hover:bg-surface-hover hover:text-ink transition-colors border-none bg-transparent"
              >
                <User size={17} /> Profile
              </button>
              <button
                onClick={() => setMenuOpen(false)}
                className="flex items-center gap-3 w-full h-[38px] px-3 rounded-[9px] text-[14px] font-medium text-ink-2 hover:bg-surface-hover hover:text-ink transition-colors border-none bg-transparent"
              >
                <Shield size={17} /> Security
              </button>

              <div className="h-px bg-line mx-1.5 my-1" />
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 w-full h-[38px] px-3 rounded-[9px] text-[14px] font-medium text-loss hover:bg-loss-soft transition-colors border-none bg-transparent"
              >
                <LogOut size={17} /> Log out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
