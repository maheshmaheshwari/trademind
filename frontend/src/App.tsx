import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './AuthContext';
import { ThemeProvider } from './ThemeContext';
import { ToastProvider } from './components/ui';
import Layout from './components/Layout';
import AuthPage      from './pages/AuthPage';
import DashboardPage from './pages/DashboardPage';
import AISignalsPage from './pages/AISignalsPage';
import MarketPage    from './pages/MarketPage';
import PortfolioPage from './pages/PortfolioPage';
import TradesPage    from './pages/TradesPage';
import WatchlistPage from './pages/WatchlistPage';
import SettingsPage  from './pages/SettingsPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return (
    <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center text-[var(--text)]">
      Loading…
    </div>
  );
  if (!user) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  const { user, isLoading } = useAuth();
  if (isLoading) return (
    <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center text-[var(--text)] text-[18px]">
      Loading…
    </div>
  );

  return (
    <Routes>
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <AuthPage />} />
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/signals"   element={<AISignalsPage />} />
        <Route path="/market"    element={<MarketPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="/orders"    element={<TradesPage />} />
        <Route path="/watchlist" element={<WatchlistPage />} />
        <Route path="/settings"  element={<SettingsPage />} />
        {/* Legacy redirect */}
        <Route path="/settings/risk" element={<Navigate to="/settings" replace />} />
      </Route>
      <Route path="/trade/:symbol"  element={<Navigate to="/signals" replace />} />
      <Route path="/market/:symbol" element={<Navigate to="/market"  replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <ToastProvider>
            <AppRoutes />
          </ToastProvider>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}
