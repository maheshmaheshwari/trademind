import { ChevronLeft, ChevronRight } from 'lucide-react';

interface PagerProps {
  page: number;       // 1-indexed
  pages: number;
  total: number;
  perPage: number;
  onPage: (p: number) => void;
  label?: string;
}

export function Pager({ page, pages, total, perPage, onPage, label = 'rows' }: PagerProps) {
  const from = (page - 1) * perPage + 1;
  const to   = Math.min(page * perPage, total);

  const nums: number[] = [];
  const start = Math.max(1, Math.min(page - 1, pages - 2));
  for (let i = start; i <= Math.min(start + 2, pages); i++) nums.push(i);

  if (pages <= 1 && total <= perPage) {
    return (
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-line flex-wrap">
        <span className="text-[12.5px] text-ink-2">{total} {label}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-line flex-wrap">
      <span className="text-[12.5px] text-ink-2">
        Showing <b className="font-mono text-ink">{from}–{to}</b> of <b className="font-mono text-ink">{total}</b> {label}
      </span>

      <div className="flex gap-1">
        <PgBtn onClick={() => onPage(page - 1)} disabled={page === 1}>
          <ChevronLeft size={15} />
        </PgBtn>

        {start > 1 && (
          <>
            <PgBtn onClick={() => onPage(1)}>1</PgBtn>
            {start > 2 && <span className="pg-btn border-none cursor-default text-ink-3">…</span>}
          </>
        )}

        {nums.map(n => (
          <PgBtn key={n} active={n === page} onClick={() => onPage(n)}>{n}</PgBtn>
        ))}

        {start + 2 < pages && (
          <>
            <span className="pg-btn border-none cursor-default text-ink-3">…</span>
            <PgBtn onClick={() => onPage(pages)}>{pages}</PgBtn>
          </>
        )}

        <PgBtn onClick={() => onPage(page + 1)} disabled={page === pages}>
          <ChevronRight size={15} />
        </PgBtn>
      </div>
    </div>
  );
}

function PgBtn({ children, active, disabled, onClick }: {
  children: React.ReactNode;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`min-w-8 h-8 px-2 rounded-lg border text-[13px] font-semibold grid place-items-center transition-colors
        ${active
          ? 'bg-accent text-white border-accent'
          : 'border-line bg-transparent text-ink-2 hover:bg-surface-hover hover:text-ink disabled:opacity-40 disabled:cursor-not-allowed'
        }`}
    >
      {children}
    </button>
  );
}
