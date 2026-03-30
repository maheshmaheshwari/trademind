import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getMe, clearToken } from './api';

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
        // Check if we have a JWT token and validate it
        const token = localStorage.getItem('trademind_token');
        if (token) {
            getMe()
                .then((u) => {
                    setUser(u);
                    localStorage.setItem('trademind_user', JSON.stringify(u));
                })
                .catch(() => {
                    // Token expired or invalid — try cached user
                    const stored = localStorage.getItem('trademind_user');
                    if (stored) {
                        try { setUser(JSON.parse(stored)); } catch { }
                    }
                    clearToken();
                })
                .finally(() => setIsLoading(false));
        } else {
            setIsLoading(false);
        }
    }, []);

    const login = (u: User) => {
        setUser(u);
        localStorage.setItem('trademind_user', JSON.stringify(u));
    };

    const logout = () => {
        setUser(null);
        localStorage.removeItem('trademind_user');
        clearToken();
    };

    const refreshUser = async () => {
        try {
            const u = await getMe();
            setUser(u);
            localStorage.setItem('trademind_user', JSON.stringify(u));
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
