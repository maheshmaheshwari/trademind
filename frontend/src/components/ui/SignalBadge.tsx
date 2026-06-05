import { TrendingUp, TrendingDown, Target } from 'lucide-react';

export type Signal = 'BUY' | 'SELL' | 'HOLD';

interface SignalBadgeProps {
  signal: Signal;
  size?: 'sm' | 'md';
}

const config: Record<Signal, { cls: string; Icon: typeof TrendingUp }> = {
  BUY:  { cls: 'bg-gain-soft text-gain',  Icon: TrendingUp },
  SELL: { cls: 'bg-loss-soft text-loss',  Icon: TrendingDown },
  HOLD: { cls: 'bg-gold-soft text-gold',  Icon: Target },
};

export function SignalBadge({ signal, size = 'md' }: SignalBadgeProps) {
  const { cls, Icon } = config[signal] ?? config.HOLD;
  return (
    <span className={`inline-flex items-center gap-1 rounded-[7px] font-bold tracking-wide ${cls} ${size === 'sm' ? 'h-5 px-1.5 text-[10.5px]' : 'h-[23px] px-2 text-[11.5px]'}`}>
      <Icon size={size === 'sm' ? 10 : 12} strokeWidth={2.5} />
      {signal}
    </span>
  );
}
