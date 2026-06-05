/* ===== Icons — consistent 24px stroke set ===== */
const Ic = ({ d, fill, size, sw = 2, children, ...p }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill={fill || "none"}
       stroke={fill ? "none" : "currentColor"} strokeWidth={sw}
       strokeLinecap="round" strokeLinejoin="round" {...p}>
    {d ? <path d={d} /> : children}
  </svg>
);

const Icons = {
  dashboard: (p) => <Ic {...p}><rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/></Ic>,
  signals: (p) => <Ic {...p}><path d="M3 17l5-5 4 3 8-9"/><path d="M21 6v4h-4"/></Ic>,
  market: (p) => <Ic {...p}><path d="M3 3v18h18"/><path d="M7 14l3-4 3 2 4-6"/><circle cx="20" cy="6" r="1.4" fill="currentColor" stroke="none"/></Ic>,
  portfolio: (p) => <Ic {...p}><path d="M21 12a9 9 0 1 1-9-9v9z"/><path d="M12 3a9 9 0 0 1 9 9h-9z" opacity=".4" fill="currentColor" stroke="none"/></Ic>,
  trades: (p) => <Ic {...p}><path d="M7 8h13"/><path d="M17 5l3 3-3 3"/><path d="M17 16H4"/><path d="M7 13l-3 3 3 3"/></Ic>,
  wallet: (p) => <Ic {...p}><rect x="3" y="6" width="18" height="14" rx="3"/><path d="M3 10h18"/><circle cx="16.5" cy="14" r="1.3" fill="currentColor" stroke="none"/></Ic>,
  search: (p) => <Ic {...p}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></Ic>,
  bell: (p) => <Ic {...p}><path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></Ic>,
  sun: (p) => <Ic {...p}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></Ic>,
  moon: (p) => <Ic {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></Ic>,
  menu: (p) => <Ic {...p}><path d="M4 6h16M4 12h16M4 18h16"/></Ic>,
  panelLeft: (p) => <Ic {...p}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></Ic>,
  refresh: (p) => <Ic {...p}><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 4v4h-4"/></Ic>,
  plus: (p) => <Ic {...p}><path d="M12 5v14M5 12h14"/></Ic>,
  arrowUp: (p) => <Ic {...p}><path d="M12 19V5M6 11l6-6 6 6"/></Ic>,
  arrowDown: (p) => <Ic {...p}><path d="M12 5v14M6 13l6 6 6-6"/></Ic>,
  arrowUpRight: (p) => <Ic {...p}><path d="M7 17 17 7M8 7h9v9"/></Ic>,
  trendUp: (p) => <Ic {...p}><path d="M3 17l6-6 4 4 8-8"/><path d="M21 11V7h-4"/></Ic>,
  trendDown: (p) => <Ic {...p}><path d="M3 7l6 6 4-4 8 8"/><path d="M21 13v4h-4"/></Ic>,
  chevDown: (p) => <Ic {...p}><path d="M6 9l6 6 6-6"/></Ic>,
  chevUp: (p) => <Ic {...p}><path d="M6 15l6-6 6 6"/></Ic>,
  chevLeft: (p) => <Ic {...p}><path d="M15 18l-6-6 6-6"/></Ic>,
  chevRight: (p) => <Ic {...p}><path d="M9 18l6-6-6-6"/></Ic>,
  chevsUpDown: (p) => <Ic {...p}><path d="M7 15l5 5 5-5M7 9l5-5 5 5"/></Ic>,
  x: (p) => <Ic {...p}><path d="M18 6 6 18M6 6l12 12"/></Ic>,
  check: (p) => <Ic {...p}><path d="M20 6 9 17l-5-5"/></Ic>,
  checkCircle: (p) => <Ic {...p}><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.3 2.3 4.7-5"/></Ic>,
  alert: (p) => <Ic {...p}><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/></Ic>,
  sliders: (p) => <Ic {...p}><path d="M4 6h11M19 6h1M4 12h3M11 12h9M4 18h8M16 18h4"/><circle cx="17" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="14" cy="18" r="2"/></Ic>,
  filter: (p) => <Ic {...p}><path d="M3 5h18l-7 8v6l-4 2v-8z"/></Ic>,
  download: (p) => <Ic {...p}><path d="M12 4v11M7 11l5 5 5-5"/><path d="M4 20h16"/></Ic>,
  calendar: (p) => <Ic {...p}><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4M16 3v4"/></Ic>,
  clock: (p) => <Ic {...p}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></Ic>,
  target: (p) => <Ic {...p}><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/></Ic>,
  shield: (p) => <Ic {...p}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></Ic>,
  sparkle: (p) => <Ic {...p}><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/></Ic>,
  brain: (p) => <Ic {...p}><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5 3 3 0 0 0 2 4 3 3 0 0 0 5 1V4.5A2.5 2.5 0 0 0 9 4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 1 5 3 3 0 0 1-2 4 3 3 0 0 1-5 1"/></Ic>,
  layers: (p) => <Ic {...p}><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></Ic>,
  pie: (p) => <Ic {...p}><path d="M12 3v9h9a9 9 0 1 0-9-9z"/><path d="M21 12a9 9 0 0 1-9 9 9 9 0 0 1-9-9"/></Ic>,
  flow: (p) => <Ic {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20v-3"/></Ic>,
  news: (p) => <Ic {...p}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h7M7 12h10M7 16h6"/></Ic>,
  logout: (p) => <Ic {...p}><path d="M9 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3"/><path d="M16 17l5-5-5-5M21 12H9"/></Ic>,
  settings: (p) => <Ic {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 6.7 19l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H2a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4 6.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V2a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H22a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/></Ic>,
  eye: (p) => <Ic {...p}><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></Ic>,
  bookmark: (p) => <Ic {...p}><path d="M6 3h12v18l-6-4-6 4z"/></Ic>,
  lock: (p) => <Ic {...p}><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></Ic>,
  mail: (p) => <Ic {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/></Ic>,
  user: (p) => <Ic {...p}><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></Ic>,
  google: (p) => <svg viewBox="0 0 24 24" width={p.size} height={p.size}><path fill="#4285F4" d="M22.5 12.2c0-.7-.1-1.4-.2-2H12v3.9h5.9a5 5 0 0 1-2.2 3.3v2.7h3.6c2.1-2 3.2-4.8 3.2-7.9z"/><path fill="#34A853" d="M12 23c2.9 0 5.4-1 7.2-2.7l-3.6-2.7c-1 .7-2.3 1-3.6 1-2.8 0-5.1-1.9-6-4.4H2.3v2.8A11 11 0 0 0 12 23z"/><path fill="#FBBC05" d="M6 14.2a6.6 6.6 0 0 1 0-4.2V7.2H2.3a11 11 0 0 0 0 9.8z"/><path fill="#EA4335" d="M12 5.4c1.6 0 3 .5 4.1 1.6l3.1-3.1A11 11 0 0 0 2.3 7.2L6 10c.9-2.5 3.2-4.4 6-4.4z"/></svg>,
};
window.Icons = Icons;
