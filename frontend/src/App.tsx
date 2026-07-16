import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './AuthContext';
import { ThemeProvider } from './ThemeContext';
import { ToastProvider } from './components/ui';
import { ErrorBoundary } from './components/ErrorBoundary';
import Layout from './components/Layout';

const AuthPage      = lazy(() => import('./pages/AuthPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const AISignalsPage = lazy(() => import('./pages/AISignalsPage'));
const AutopilotPage = lazy(() => import('./pages/AutopilotPage'));
const MarketPage    = lazy(() => import('./pages/MarketPage'));
const PortfolioPage = lazy(() => import('./pages/PortfolioPage'));
const TradesPage    = lazy(() => import('./pages/TradesPage'));
const WatchlistPage = lazy(() => import('./pages/WatchlistPage'));
const SettingsPage  = lazy(() => import('./pages/SettingsPage'));
const BacktestPage  = lazy(() => import('./pages/BacktestPage'));
const StockPage     = lazy(() => import('./pages/StockPage'));

function PageLoader() {
  return (
    <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center text-[var(--text)]">
      Loading…
    </div>
  );
}

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
    <Suspense fallback={<PageLoader />}>
    <Routes>
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <AuthPage />} />
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/signals"   element={<AISignalsPage />} />
        <Route path="/autopilot" element={<AutopilotPage />} />
        <Route path="/market"    element={<MarketPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="/orders"    element={<TradesPage />} />
        <Route path="/watchlist" element={<WatchlistPage />} />
        <Route path="/backtest"  element={<BacktestPage />} />
        <Route path="/settings"  element={<SettingsPage />} />
        <Route path="/stocks/:symbol" element={<StockPage />} />
        {/* Legacy redirect */}
        <Route path="/settings/risk" element={<Navigate to="/settings" replace />} />
      </Route>
      <Route path="/trade/:symbol"  element={<Navigate to="/signals" replace />} />
      <Route path="/market/:symbol" element={<Navigate to="/market"  replace />} />
    </Routes>
    </Suspense>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ThemeProvider>
          <AuthProvider>
            <ToastProvider>
              <AppRoutes />
            </ToastProvider>
          </AuthProvider>
        </ThemeProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
