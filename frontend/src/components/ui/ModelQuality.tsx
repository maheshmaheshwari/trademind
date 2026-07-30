import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';

interface ModelQualityProps {
  accuracy?: number | null;   // model test accuracy %
  precision?: number | null;  // model test precision %
  size?: number;
  showText?: boolean;         // show the label, not just the icon
}

/**
 * Badge showing how reliable the model behind a signal is, so a user doesn't
 * act on a confident call from a coin-flip model. Tiers match the recommender's
 * quality floor (reliable = accuracy ≥ 70% AND precision ≥ 60%):
 *   - Reliable  (green)  — safe to trust
 *   - Caution   (amber)  — borderline (≥60% both)
 *   - Low       (red)    — model is barely better than chance; confidence is not enough
 */
export function ModelQuality({ accuracy, precision, size = 13, showText = true }: ModelQualityProps) {
  if (accuracy == null || precision == null) {
    return <span className="text-ink-3 font-mono" style={{ fontSize: size }}>—</span>;
  }
  const reliable = accuracy >= 70 && precision >= 60;
  const caution = !reliable && accuracy >= 60 && precision >= 60;
  const col = reliable ? 'var(--green)' : caution ? 'var(--gold)' : 'var(--red)';
  const label = reliable ? 'Reliable' : caution ? 'Caution' : 'Low reliability';
  const Icon = reliable ? ShieldCheck : caution ? ShieldAlert : ShieldX;
  return (
    <span className="inline-flex items-center gap-1 font-semibold"
      style={{ color: col, fontSize: size }}
      title={`Model test accuracy ${accuracy}% · precision ${precision}%`}>
      <Icon size={size + 1} strokeWidth={2} />
      {showText && label}
    </span>
  );
}
