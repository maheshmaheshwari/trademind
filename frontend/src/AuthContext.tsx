import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import { getMe, clearToken } from './api';
import { resetLoggingOut } from './services/tradeMindInterceptor';

interface User {
    id: number;
    username: string;
    display_name: string;
    virtual_balance: number;
    virtual_invested: number;
    total_pnl: number;
    win_count: number;
    loss_count: number;
    mode: string;
    email?: string;
    phone?: string;
    avatar_url?: string;
    totp_enabled?: boolean;
    default_account?: string;
    currency?: string;
}

interface AuthContextType {
    user: User | null;
    setUser: (user: User | null) => void;
    login: (user: User) => void;
    logout: () => void;
    refreshUser: () => Promise<void>;
    isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('trademind_token') ?? sessionStorage.getItem('trademind_token');
        if (token) {
            getMe()
                .then((u) => setUser(u))
                .catch(() => {
                    clearToken();
                    setUser(null);
                })
                .finally(() => setIsLoading(false));
        } else {
            setIsLoading(false);
        }
    }, []);

    const logout = useCallback(() => {
        setUser(null);
        clearToken();
    }, []);

    // Interceptor fires this event on 401 — auto-logout
    useEffect(() => {
        window.addEventListener('trademind:unauthorized', logout);
        return () => window.removeEventListener('trademind:unauthorized', logout);
    }, [logout]);

    const login = (u: User) => {
        setUser(u);
        resetLoggingOut();
    };

    const refreshUser = async () => {
        try {
            const u = await getMe();
            setUser(u);
        } catch {
            // noop
        }
    };

    return (
        <AuthContext.Provider value={{ user, setUser, login, logout, refreshUser, isLoading }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
}
