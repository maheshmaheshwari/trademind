import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { CheckCircle, AlertCircle, Info } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastData {
  type?: ToastType;
  title: string;
  msg?: string;
  duration?: number;
}

interface ToastItem extends ToastData {
  id: string;
  out?: boolean;
}

type PushFn = (t: ToastData) => void;

const ToastCtx = createContext<PushFn>(() => {});
export const useToast = () => useContext(ToastCtx);

const icons: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error:   AlertCircle,
  info:    Info,
};

const iconBg: Record<ToastType, string> = {
  success: 'bg-gain-soft text-gain',
  error:   'bg-loss-soft text-loss',
  info:    'bg-accent-soft text-accent-2',
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const push = useCallback((t: ToastData) => {
    const id = Math.random().toString(36).slice(2);
    const dur = t.duration ?? 3400;
    setToasts(x => [...x, { ...t, id }]);
    setTimeout(() => setToasts(x => x.map(o => o.id === id ? { ...o, out: true } : o)), dur);
    setTimeout(() => setToasts(x => x.filter(o => o.id !== id)), dur + 300);
  }, []);

  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="fixed bottom-5 left-1/2 -translate-x-1/2 z-[200] flex flex-col gap-2.5 items-center pointer-events-none">
        {toasts.map(t => {
          const type = t.type ?? 'info';
          const Icon = icons[type];
          return (
            <div
              key={t.id}
              className={`pointer-events-auto min-w-[300px] max-w-[420px] flex items-center gap-3 px-3.5 py-3 rounded-[13px] bg-surface-2 border border-line-strong shadow-lg ${t.out ? 'animate-toast-out' : 'animate-toast-in'}`}
            >
              <span className={`w-[34px] h-[34px] rounded-[9px] grid place-items-center flex-shrink-0 ${iconBg[type]}`}>
                <Icon size={18} />
              </span>
              <div className="flex flex-col gap-0">
                <span className="font-semibold text-[13.5px] text-ink">{t.title}</span>
                {t.msg && <span className="text-[12px] text-ink-2 mt-0.5">{t.msg}</span>}
              </div>
            </div>
          );
        })}
      </div>
    </ToastCtx.Provider>
  );
}
