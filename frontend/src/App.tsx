import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './AuthContext';
import { ThemeProvider } from './ThemeContext';
import Layout from './components/Layout';
import AuthPage from './pages/AuthPage';
import DashboardPage from './pages/DashboardPage';
import AISignalsPage from './pages/AISignalsPage';
import TradeExecutionPage from './pages/TradeExecutionPage';
import PortfolioPage from './pages/PortfolioPage';
import OrderHistoryPage from './pages/OrderHistoryPage';
import RiskSettingsPage from './pages/RiskSettingsPage';
import MarketPage from './pages/MarketPage';
import StockDetailPage from './pages/StockDetailPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return <div className="min-h-screen bg-background-dark flex items-center justify-center text-white">Loading...</div>;
  if (!user) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return <div className="min-h-screen bg-background-dark flex items-center justify-center text-white text-lg">Loading...</div>;
  }

  return (
    <Routes>
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <AuthPage />} />
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/signals" element={<AISignalsPage />} />
        <Route path="/market" element={<MarketPage />} />
        <Route path="/market/:symbol" element={<StockDetailPage />} />
        <Route path="/trade/:symbol" element={<TradeExecutionPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="/orders" element={<OrderHistoryPage />} />
        <Route path="/settings/risk" element={<RiskSettingsPage />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
