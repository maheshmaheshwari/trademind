const COLORS = [
  '#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6',
  '#06B6D4','#F97316','#EC4899','#14B8A6','#6366F1',
];

export function symColor(symbol: string): string {
  let h = 0;
  for (let i = 0; i < symbol.length; i++) h = (h * 31 + symbol.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
}

interface SymbolCellProps {
  symbol: string;
  name?: string;
  sector?: string;
  showSector?: boolean;
}

export function SymbolCell({ symbol, name, sector, showSector = true }: SymbolCellProps) {
  const color = symColor(symbol);
  return (
    <div className="flex items-center gap-2.5">
      <span
        className="w-8 h-8 rounded-[9px] grid place-items-center font-bold text-[12px] flex-shrink-0 tracking-tight"
        style={{ background: color + '22', color }}
      >
        {symbol.slice(0, 2)}
      </span>
      <div className="flex flex-col gap-0">
        <span className="font-semibold text-[13.5px] text-ink leading-none">{symbol}</span>
        {showSector && (sector || name) && (
          <span className="text-[11.5px] text-ink-3 mt-0.5">{sector || name}</span>
        )}
      </div>
    </div>
  );
}
