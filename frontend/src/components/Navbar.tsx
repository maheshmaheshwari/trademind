import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Search, Bell, Settings, TrendingUp, LogOut, Sun, Moon } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';

export default function Navbar() {
    const location = useLocation();
    const navigate = useNavigate();
    const { user, logout } = useAuth();
    const { theme, toggleTheme } = useTheme();

    const isActive = (path: string) => location.pathname === path;

    const navLinks = [
        { path: '/dashboard', label: 'Dashboard' },
        { path: '/portfolio', label: 'Portfolio' },
        { path: '/market', label: 'Market' },
        { path: '/orders', label: 'Orders' },
        { path: '/signals', label: 'AI Picks' },
    ];

    const initials = user?.display_name
        ? user.display_name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
        : 'U';

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    return (
        <header className="sticky top-0 z-50 w-full border-b border-slate-800 dark:border-slate-800 light:border-slate-200 bg-background-dark/80 dark:bg-background-dark/80 light:bg-white/80 backdrop-blur-md">
            <div className="px-6 md:px-10 py-3 flex items-center justify-between whitespace-nowrap">
                {/* Logo */}
                <Link to="/dashboard" className="flex items-center gap-3 text-white dark:text-white light:text-slate-900">
                    <div className="size-8 text-primary flex items-center justify-center">
                        <TrendingUp className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-bold leading-tight tracking-tight">TradeMind AI</h2>
                </Link>

                {/* Search */}
                {/* <div className="hidden lg:flex items-center gap-2 bg-surface-dark dark:bg-surface-dark light:bg-slate-100 border border-slate-700 dark:border-slate-700 light:border-slate-200 rounded-xl px-4 py-2 ml-8 flex-1 max-w-md">
                    <Search className="w-4 h-4 text-slate-400 dark:text-slate-400 light:text-slate-500" />
                    <input
                        type="text"
                        placeholder="Search stocks, indices..."
                        className="bg-transparent text-sm text-white dark:text-white light:text-slate-900 placeholder-slate-400 dark:placeholder-slate-400 light:placeholder-slate-500 outline-none flex-1"
                    />
                </div> */}

                {/* Nav links */}
                <nav className="hidden lg:flex items-center gap-8 ml-8">
                    {navLinks.map(({ path, label }) => (
                        <Link
                            key={path}
                            to={path}
                            className={`text-sm font-medium transition-colors ${isActive(path)
                                ? 'text-primary font-bold'
                                : 'text-slate-300 dark:text-slate-300 light:text-slate-600 hover:text-primary'
                                }`}
                        >
                            {label}
                        </Link>
                    ))}
                </nav>

                {/* Actions */}
                <div className="hidden lg:flex items-center gap-4 ml-8">
                    <button className="p-2 text-slate-400 dark:text-slate-400 light:text-slate-500 hover:text-white dark:hover:text-white light:hover:text-slate-900 transition-colors relative">
                        <Bell className="w-5 h-5" />
                        <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"></span>
                    </button>
                    <Link to="/settings/risk" className="p-2 text-slate-400 dark:text-slate-400 light:text-slate-500 hover:text-white dark:hover:text-white light:hover:text-slate-900 transition-colors">
                        <Settings className="w-5 h-5" />
                    </Link>
                    <button
                        onClick={toggleTheme}
                        className="p-2 text-slate-400 dark:text-slate-400 light:text-slate-500 hover:text-yellow-400 dark:hover:text-yellow-400 light:hover:text-amber-500 transition-all duration-300"
                        title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                    >
                        {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>
                    <button onClick={handleLogout} className="p-2 text-slate-400 dark:text-slate-400 light:text-slate-500 hover:text-red-400 transition-colors" title="Logout">
                        <LogOut className="w-5 h-5" />
                    </button>
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-purple-500 ring-2 ring-primary/20 flex items-center justify-center text-white font-bold text-sm">
                        {initials}
                    </div>
                </div>

                {/* Mobile menu */}
                <button className="lg:hidden text-white dark:text-white light:text-slate-900">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                </button>
            </div>
        </header>
    );
}
