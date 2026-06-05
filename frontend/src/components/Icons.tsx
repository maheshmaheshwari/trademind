import type { SVGProps } from 'react';
import {
  LayoutDashboard, TrendingUp, TrendingDown, LineChart, PieChart,
  ArrowLeftRight, Wallet, Search, Bell, Sun, Moon, Menu, PanelLeft,
  RefreshCw, Plus, ArrowUp, ArrowDown, ArrowUpRight,
  ChevronDown, ChevronUp, ChevronLeft, ChevronRight, ChevronsUpDown,
  X, Check, CheckCircle, AlertCircle, SlidersHorizontal, Filter,
  Download, Calendar, Clock, Target, Shield, Sparkles, Brain,
  Layers, BarChart2, Newspaper, LogOut, Settings, Eye, Bookmark,
  Lock, Mail, User, type LucideProps,
} from 'lucide-react';

export type IconName =
  | 'dashboard' | 'signals' | 'market' | 'portfolio' | 'trades'
  | 'wallet' | 'search' | 'bell' | 'sun' | 'moon' | 'menu' | 'panelLeft'
  | 'refresh' | 'plus' | 'arrowUp' | 'arrowDown' | 'arrowUpRight'
  | 'trendUp' | 'trendDown' | 'chevDown' | 'chevUp' | 'chevLeft' | 'chevRight'
  | 'chevsUpDown' | 'x' | 'check' | 'checkCircle' | 'alert' | 'sliders'
  | 'filter' | 'download' | 'calendar' | 'clock' | 'target' | 'shield'
  | 'sparkle' | 'brain' | 'layers' | 'pie' | 'flow' | 'news' | 'logout'
  | 'settings' | 'eye' | 'bookmark' | 'lock' | 'mail' | 'user' | 'google';

type LucideComponent = React.ComponentType<LucideProps>;

const lucideMap: Partial<Record<IconName, LucideComponent>> = {
  dashboard:    LayoutDashboard,
  signals:      TrendingUp,
  market:       LineChart,
  portfolio:    PieChart,
  trades:       ArrowLeftRight,
  wallet:       Wallet,
  search:       Search,
  bell:         Bell,
  sun:          Sun,
  moon:         Moon,
  menu:         Menu,
  panelLeft:    PanelLeft,
  refresh:      RefreshCw,
  plus:         Plus,
  arrowUp:      ArrowUp,
  arrowDown:    ArrowDown,
  arrowUpRight: ArrowUpRight,
  trendUp:      TrendingUp,
  trendDown:    TrendingDown,
  chevDown:     ChevronDown,
  chevUp:       ChevronUp,
  chevLeft:     ChevronLeft,
  chevRight:    ChevronRight,
  chevsUpDown:  ChevronsUpDown,
  x:            X,
  check:        Check,
  checkCircle:  CheckCircle,
  alert:        AlertCircle,
  sliders:      SlidersHorizontal,
  filter:       Filter,
  download:     Download,
  calendar:     Calendar,
  clock:        Clock,
  target:       Target,
  shield:       Shield,
  sparkle:      Sparkles,
  brain:        Brain,
  layers:       Layers,
  pie:          PieChart,
  flow:         BarChart2,
  news:         Newspaper,
  logout:       LogOut,
  settings:     Settings,
  eye:          Eye,
  bookmark:     Bookmark,
  lock:         Lock,
  mail:         Mail,
  user:         User,
};

// Google brand icon — not in lucide, kept as inline SVG
function GoogleIcon({ size = 20, ...p }: { size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} {...p}>
      <path fill="#4285F4" d="M22.5 12.2c0-.7-.1-1.4-.2-2H12v3.9h5.9a5 5 0 0 1-2.2 3.3v2.7h3.6c2.1-2 3.2-4.8 3.2-7.9z"/>
      <path fill="#34A853" d="M12 23c2.9 0 5.4-1 7.2-2.7l-3.6-2.7c-1 .7-2.3 1-3.6 1-2.8 0-5.1-1.9-6-4.4H2.3v2.8A11 11 0 0 0 12 23z"/>
      <path fill="#FBBC05" d="M6 14.2a6.6 6.6 0 0 1 0-4.2V7.2H2.3a11 11 0 0 0 0 9.8z"/>
      <path fill="#EA4335" d="M12 5.4c1.6 0 3 .5 4.1 1.6l3.1-3.1A11 11 0 0 0 2.3 7.2L6 10c.9-2.5 3.2-4.4 6-4.4z"/>
    </svg>
  );
}

export interface IconProps {
  name: IconName;
  size?: number;
  className?: string;
  strokeWidth?: number;
}

export function Icon({ name, size = 20, className, strokeWidth = 2 }: IconProps) {
  if (name === 'google') return <GoogleIcon size={size} className={className} />;
  const Component = lucideMap[name];
  if (!Component) return null;
  return <Component size={size} className={className} strokeWidth={strokeWidth} />;
}
