import { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard, LineChart, PieChart,
  ArrowLeftRight, Bookmark, Settings, PanelLeft,
  TrendingUp,
} from 'lucide-react';
import Navbar from './Navbar';
import { useAuth } from '../AuthContext';

const NAV = [
  { id: 'dashboard', path: '/dashboard', label: 'Dashboard',      Icon: LayoutDashboard },
  { id: 'signals',   path: '/signals',   label: 'AI Signals',     Icon: TrendingUp },
  { id: 'market',    path: '/market',    label: 'Market Overview', Icon: LineChart },
  { id: 'portfolio', path: '/portfolio', label: 'Portfolio',      Icon: PieChart },
  { id: 'trades',    path: '/orders',    label: 'Trades & Orders', Icon: ArrowLeftRight },
];

const ACCOUNT = [
  { id: 'watchlist', path: '/watchlist', label: 'Watchlist', Icon: Bookmark },
  { id: 'settings',  path: '/settings',  label: 'Settings',  Icon: Settings },
];

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();

  const isActive = (path: string) => location.pathname.startsWith(path);

  const initials = user?.display_name
    ? user.display_name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      {/* radial gradient bg */}
      <div className="fixed inset-0 pointer-events-none z-0 bg-[radial-gradient(1200px_700px_at_78%_-8%,rgba(59,130,246,.10),transparent_60%)]" />

      {/* ── Sidebar ── */}
      <aside
        className="flex-shrink-0 bg-surface border-r border-line flex flex-col relative z-[5] transition-[width] duration-[260ms] ease-[cubic-bezier(.4,0,.2,1)] overflow-hidden"
        style={{ width: collapsed ? 'var(--sidebar-w-collapsed)' : 'var(--sidebar-w)' }}
      >
        {/* Brand */}
        <div className="flex items-center gap-2.5 border-b border-line overflow-hidden flex-shrink-0" style={{ height: 'var(--navbar-h)', padding: '0 18px' }}>
          <span className="w-[34px] h-[34px] flex-shrink-0 rounded-[10px] grid place-items-center bg-[linear-gradient(135deg,var(--accent),#1E40AF)] shadow-[0_4px_14px_rgba(59,130,246,.4)]">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/>
            </svg>
          </span>
          <span className="font-bold text-[17px] tracking-tight whitespace-nowrap text-ink transition-opacity duration-200" style={{ opacity: collapsed ? 0 : 1 }}>
            Trade<b className="text-accent-2">Mind</b>
          </span>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-3 flex flex-col gap-0.5 overflow-y-auto overflow-x-hidden">
          <span
            className="text-[10.5px] font-semibold tracking-[.08em] uppercase text-ink-3 px-3 pt-3 pb-1.5 whitespace-nowrap transition-opacity duration-200"
            style={{ opacity: collapsed ? 0 : 1 }}
          >
            Trading
          </span>

          {NAV.map(({ id, path, label, Icon }) => {
            const active = isActive(path);
            return (
              <button
                key={id}
                onClick={() => navigate(path)}
                title={collapsed ? label : undefined}
                className={`relative flex items-center gap-3 h-[42px] px-3 rounded-[9px] font-medium text-[14px] whitespace-nowrap w-full text-left border-none transition-colors duration-150
                  ${active
                    ? 'bg-accent-soft text-accent-2 sb-item-active'
                    : 'bg-transparent text-ink-2 hover:bg-surface-hover hover:text-ink'
                  }`}
              >
                <Icon size={20} className="flex-shrink-0" strokeWidth={active ? 2.2 : 1.8} />
                <span className="transition-opacity duration-200 whitespace-nowrap overflow-hidden" style={{ opacity: collapsed ? 0 : 1 }}>
                  {label}
                </span>
              </button>
            );
          })}

          <span
            className="text-[10.5px] font-semibold tracking-[.08em] uppercase text-ink-3 px-3 pt-4 pb-1.5 whitespace-nowrap transition-opacity duration-200"
            style={{ opacity: collapsed ? 0 : 1 }}
          >
            Account
          </span>

          {ACCOUNT.map(({ id, path, label, Icon }) => {
            const active = isActive(path);
            return (
              <button
                key={id}
                onClick={() => navigate(path)}
                title={collapsed ? label : undefined}
                className={`relative flex items-center gap-3 h-[42px] px-3 rounded-[9px] font-medium text-[14px] whitespace-nowrap w-full text-left border-none transition-colors duration-150
                  ${active
                    ? 'bg-accent-soft text-accent-2 sb-item-active'
                    : 'bg-transparent text-ink-2 hover:bg-surface-hover hover:text-ink'
                  }`}
              >
                <Icon size={20} className="flex-shrink-0" strokeWidth={active ? 2.2 : 1.8} />
                <span className="transition-opacity duration-200" style={{ opacity: collapsed ? 0 : 1 }}>
                  {label}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Footer user */}
        <div className="border-t border-line p-3 flex-shrink-0">
          <div className="flex items-center gap-2.5 p-2 rounded-[9px] cursor-pointer hover:bg-surface-hover transition-colors duration-150" onClick={() => navigate('/settings')}>
            <div className="w-[34px] h-[34px] rounded-[10px] bg-gradient-to-br from-indigo-500 to-accent grid place-items-center font-bold text-[13px] text-white flex-shrink-0">
              {initials}
            </div>
            <div className="flex flex-col gap-0 overflow-hidden transition-opacity duration-200" style={{ opacity: collapsed ? 0 : 1 }}>
              <span className="font-semibold text-[13px] text-ink whitespace-nowrap">{user?.display_name ?? 'User'}</span>
              <span className="text-[11.5px] text-ink-3 whitespace-nowrap truncate">{user?.username ?? ''}</span>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 min-w-0 flex flex-col relative z-[1]">
        <Navbar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
        <main className="flex-1 overflow-y-auto overflow-x-hidden">
          <div className="dp dgap max-w-[1480px] mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

// Sidebar collapse toggle button exposed for Navbar
export { PanelLeft };
