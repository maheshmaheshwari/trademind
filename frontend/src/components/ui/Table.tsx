import { useState } from 'react';
import { ChevronsUpDown, ChevronUp, ChevronDown } from 'lucide-react';

// ─── useSort hook ──────────────────────────────────────────────────────────

export interface SortState {
  key: string;
  dir: 'asc' | 'desc';
}

export function useSort(initialKey: string, initialDir: 'asc' | 'desc' = 'desc') {
  const [sort, setSort] = useState<SortState>({ key: initialKey, dir: initialDir });

  const toggle = (key: string) =>
    setSort(s => s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' });

  function apply<T extends Record<string, unknown>>(arr: T[]): T[] {
    const { key, dir } = sort;
    if (!key) return arr;
    return [...arr].sort((a, b) => {
      const x = a[key], y = b[key];
      let cmp = 0;
      if (typeof x === 'string' && typeof y === 'string') cmp = x.localeCompare(y);
      else if (typeof x === 'number' && typeof y === 'number') cmp = x - y;
      return dir === 'asc' ? cmp : -cmp;
    });
  }

  return { sort, toggle, apply };
}

// ─── Th — sortable table header ────────────────────────────────────────────

interface ThProps {
  label: string;
  sortKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  align?: 'left' | 'right' | 'center';
  width?: string | number;
}

export function Th({ label, sortKey, sort, onToggle, align = 'left', width }: ThProps) {
  const active = sort.key === sortKey;
  const Icon = active ? (sort.dir === 'asc' ? ChevronUp : ChevronDown) : ChevronsUpDown;
  return (
    <th
      onClick={() => onToggle(sortKey)}
      className={`text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 px-3.5 border-b border-line sticky top-0 bg-surface z-[1] cursor-pointer select-none whitespace-nowrap hover:text-ink transition-colors ${active ? 'text-accent-2' : ''}`}
      style={{ textAlign: align, width, paddingTop: 'calc(11px * var(--u))', paddingBottom: 'calc(11px * var(--u))' }}
    >
      {label}
      <span className="inline-flex align-middle ml-1 opacity-60">
        <Icon size={13} strokeWidth={active ? 2.5 : 1.5} className={active ? 'text-accent-2' : ''} />
      </span>
    </th>
  );
}

// ─── Plain th (non-sortable) ───────────────────────────────────────────────

interface PlainThProps {
  children: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  width?: string | number;
}

export function PlainTh({ children, align = 'left', width }: PlainThProps) {
  return (
    <th
      className="text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 px-3.5 border-b border-line sticky top-0 bg-surface z-[1] whitespace-nowrap"
      style={{ textAlign: align, width, paddingTop: 'calc(11px * var(--u))', paddingBottom: 'calc(11px * var(--u))' }}
    >
      {children}
    </th>
  );
}

// ─── Td ────────────────────────────────────────────────────────────────────

interface TdProps {
  children: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  mono?: boolean;
  className?: string;
}

export function Td({ children, align = 'left', mono = false, className = '' }: TdProps) {
  return (
    <td
      className={`px-3.5 border-b border-line whitespace-nowrap text-[13px] ${mono ? 'font-mono' : ''} ${className}`}
      style={{ textAlign: align, paddingTop: 'calc(12px * var(--u))', paddingBottom: 'calc(12px * var(--u))' }}
    >
      {children}
    </td>
  );
}
