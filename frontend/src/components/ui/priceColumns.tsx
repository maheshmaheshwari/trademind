/**
 * The four prices that describe a trade, as one reusable column set.
 *
 * Every table that shows an order or a position renders the same block in the
 * same order — Entry → Current → Sell → Stop Loss — so a row means the same
 * thing wherever you read it. Defined once here rather than per page, because
 * the interesting rule is easy to get subtly wrong in eight places:
 *
 *   The SELL column is the ACTUAL sale price once a trade is closed, and the
 *   TARGET (a projection) while it is still open. Those are different claims,
 *   so they are not rendered identically — a projection is shown muted with a
 *   "target" marker, a real fill is shown solid. Collapsing the two would let
 *   an expected price read as money actually made.
 *
 * `sold` comes from the backend (trading_engine.get_bracket_levels), which sets
 * it only for an EXECUTED SQUARE_OFF leg — a PENDING stop or target row is an
 * instruction, not a fill.
 */
import type { ReactNode } from 'react';
import type { DataTableColumn, DataTableRow } from './DataTable';

function inr(n: number | null | undefined, dec = 2): string {
  if (n == null) return '—';
  return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function Muted({ children }: { children: ReactNode }) {
  return <span className="text-ink-3">{children}</span>;
}

/** How to pull the four prices out of whatever shape the row happens to be. */
export interface PriceAccessors<T> {
  entry: (row: T) => number | null | undefined;
  current: (row: T) => number | null | undefined;
  /** Realised sale price. Return null/undefined while the trade is still open. */
  sold?: (row: T) => number | null | undefined;
  /** Projected sale price — shown when `sold` yields nothing. */
  target: (row: T) => number | null | undefined;
  stopLoss: (row: T) => number | null | undefined;
}

// Constrained to DataTableRow (Record<string, any>) for the same reason
// DataTable is: interfaces get no implicit index signature, so the stricter
// Record<string, unknown> would reject every type in src/types.
export function priceColumns<T extends DataTableRow>(
  get: PriceAccessors<T>,
): DataTableColumn<T>[] {
  return [
    {
      id: 'entry_price',
      header: 'Entry',
      align: 'right',
      mono: true,
      accessor: r => get.entry(r) ?? undefined,
      cell: r => {
        const v = get.entry(r);
        return v == null ? <Muted>—</Muted> : <span>{inr(v)}</span>;
      },
    },
    {
      id: 'current_price',
      header: 'Current',
      align: 'right',
      mono: true,
      accessor: r => get.current(r) ?? undefined,
      cell: r => {
        const v = get.current(r);
        if (v == null) return <Muted>—</Muted>;
        const entry = get.entry(r);
        // Coloured against the entry, so the column reads as "is this trade up
        // or down" at a glance rather than as a bare number.
        const cls = entry == null ? '' : v >= entry ? 'text-gain' : 'text-loss';
        return <span className={cls}>{inr(v)}</span>;
      },
    },
    {
      id: 'sell_price',
      header: 'Sell',
      align: 'right',
      mono: true,
      accessor: r => (get.sold?.(r) ?? get.target(r)) ?? undefined,
      cell: r => {
        const actual = get.sold?.(r);
        if (actual != null) {
          // A real fill: solid, and marked so it cannot be mistaken for the target.
          return (
            <span className="inline-flex flex-col items-end leading-tight">
              <span className="text-gain font-semibold">{inr(actual)}</span>
              <span className="text-[10.5px] text-ink-3">sold</span>
            </span>
          );
        }
        const projected = get.target(r);
        if (projected == null) return <Muted>—</Muted>;
        return (
          <span className="inline-flex flex-col items-end leading-tight">
            <span className="text-ink-2">{inr(projected)}</span>
            <span className="text-[10.5px] text-ink-3">target</span>
          </span>
        );
      },
    },
    {
      id: 'stop_loss',
      header: 'Stop Loss',
      align: 'right',
      mono: true,
      accessor: r => get.stopLoss(r) ?? undefined,
      cell: r => {
        const v = get.stopLoss(r);
        return v == null ? <Muted>—</Muted> : <span className="text-loss">{inr(v)}</span>;
      },
    },
  ];
}
