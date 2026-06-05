import { TrendingUp, TrendingDown } from 'lucide-react';

interface DeltaProps {
  value: number;
  suffix?: string;
  size?: number;
  showIcon?: boolean;
  abs?: boolean;
}

export function Delta({ value, suffix = '%', size = 13, showIcon = true, abs = false }: DeltaProps) {
  const pos = value >= 0;
  const display = (pos ? '+' : '') + (abs ? Math.abs(value) : value).toFixed(2) + suffix;
  return (
    <span
      className="inline-flex items-center gap-0.5 font-semibold font-mono"
      style={{ color: pos ? '#10B981' : '#EF4444', fontSize: size }}
    >
      {showIcon && (pos
        ? <TrendingUp size={14} strokeWidth={2} />
        : <TrendingDown size={14} strokeWidth={2} />
      )}
      {display}
    </span>
  );
}
