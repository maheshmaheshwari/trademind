interface ConfProps {
  value: number; // 0–100
}

export function Conf({ value }: ConfProps) {
  const color =
    value >= 80 ? '#10B981' :
    value >= 65 ? '#3B82F6' :
    '#F59E0B';
  return (
    <div className="flex items-center gap-2">
      <div className="conf-track flex-1 min-w-[46px]">
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: color }} />
      </div>
      <span className="font-mono text-[12px] font-semibold min-w-[34px] text-right text-ink">{value}%</span>
    </div>
  );
}
