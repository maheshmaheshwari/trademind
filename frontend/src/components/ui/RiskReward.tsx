interface RiskRewardProps {
  /** reward-to-risk ratio (reward ÷ risk); e.g. 2.5 means reward is 2.5× the risk */
  value: number | null | undefined;
  size?: number;
  showLabel?: boolean;
}

/**
 * Displays the trade's reward:risk as "1 : X.X" (risk : reward).
 * Green when the payoff is favourable (reward ≥ 1.5× risk), amber when marginal
 * (≥ 1×), muted red below 1×, and a dash when there's no active trade (HOLD).
 */
export function RiskReward({ value, size = 13, showLabel = false }: RiskRewardProps) {
  if (value == null || !(value > 0)) {
    return <span className="text-ink-3 font-mono" style={{ fontSize: size }}>—</span>;
  }
  const col = value >= 1.5 ? 'var(--green)' : value >= 1 ? 'var(--gold)' : 'var(--red)';
  return (
    <span className="inline-flex items-center gap-1 font-mono font-semibold tabular-nums"
      style={{ color: col, fontSize: size }}
      title={`Reward is ${value.toFixed(2)}× the risk`}>
      {showLabel && <span className="text-ink-3 font-sans font-medium">R:R</span>}
      1 : {value.toFixed(1)}
    </span>
  );
}
