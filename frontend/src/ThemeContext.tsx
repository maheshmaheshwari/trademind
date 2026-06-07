import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

type Theme = 'dark' | 'light';
type Density = 'compact' | 'balanced' | 'comfy';
export type SignalStyle = 'rich' | 'compact' | 'bold';

interface ThemeContextType {
  theme: Theme;
  density: Density;
  signalStyle: SignalStyle;
  toggleTheme: () => void;
  setDensity: (d: Density) => void;
  setSignalStyle: (s: SignalStyle) => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  density: 'balanced',
  signalStyle: 'rich',
  toggleTheme: () => {},
  setDensity: () => {},
  setSignalStyle: () => {},
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('trademind-theme');
    return saved === 'light' ? 'light' : 'dark';
  });

  const [density, setDensityState] = useState<Density>(() => {
    const saved = localStorage.getItem('trademind-density');
    return (saved === 'compact' || saved === 'comfy') ? saved : 'balanced';
  });

  const [signalStyle, setSignalStyleState] = useState<SignalStyle>(() => {
    const saved = localStorage.getItem('trademind-signal-style');
    return (saved === 'compact' || saved === 'bold') ? saved : 'rich';
  });

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    root.classList.toggle('dark', theme === 'dark');
    root.classList.toggle('light', theme === 'light');
    localStorage.setItem('trademind-theme', theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute('data-density', density);
    localStorage.setItem('trademind-density', density);
  }, [density]);

  useEffect(() => {
    localStorage.setItem('trademind-signal-style', signalStyle);
  }, [signalStyle]);

  const toggleTheme   = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  const setDensity    = (d: Density) => setDensityState(d);
  const setSignalStyle = (s: SignalStyle) => setSignalStyleState(s);

  return (
    <ThemeContext.Provider value={{ theme, density, signalStyle, toggleTheme, setDensity, setSignalStyle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
