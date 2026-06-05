import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

type Theme = 'dark' | 'light';
type Density = 'compact' | 'balanced' | 'comfy';

interface ThemeContextType {
  theme: Theme;
  density: Density;
  toggleTheme: () => void;
  setDensity: (d: Density) => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  density: 'balanced',
  toggleTheme: () => {},
  setDensity: () => {},
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

  useEffect(() => {
    const root = document.documentElement;
    // data-theme drives CSS variables + Tailwind's dark: selector
    root.setAttribute('data-theme', theme);
    // Keep dark class in sync for any legacy Tailwind dark: usage
    root.classList.toggle('dark', theme === 'dark');
    root.classList.toggle('light', theme === 'light');
    localStorage.setItem('trademind-theme', theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute('data-density', density);
    localStorage.setItem('trademind-density', density);
  }, [density]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  const setDensity = (d: Density) => setDensityState(d);

  return (
    <ThemeContext.Provider value={{ theme, density, toggleTheme, setDensity }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
