/**
 * TradeMind AI — Shared data table
 *
 * One table implementation for every list page, built on material-react-table
 * (TanStack Table + MUI under the hood) but styled to be visually identical to
 * the hand-rolled Tailwind tables it replaces. Pages describe columns and data;
 * nothing about the look is decided per page any more.
 *
 * Pagination is MRT's own (bottom toolbar, `paginationDisplayMode: 'pages'`),
 * restyled to the app's tokens rather than replaced.
 *
 * The two loading states are distinct and must not be collapsed into one:
 *
 *   * `isLoading` — first load, nothing to show yet. Renders our shape-aware
 *     <SkeletonRows> (avatar in the first column, button pills in the last),
 *     which MRT's generic cell skeletons cannot reproduce.
 *   * `isFetching` — a refetch behind data already on screen. Keeps the rows
 *     visible and shows MRT's progress bar, so a background refresh never
 *     blanks a table the user is reading. RTK Query reports both, and passing
 *     only `isLoading` would make every refetch invisible.
 *
 * MUI's own chrome (Paper elevation, sort arrows, column action menus,
 * density/fullscreen toggles) is stripped below. Colors come from the CSS
 * custom properties in index.css, so light/dark switching keeps working
 * through ThemeContext with no extra wiring.
 */
import { useMemo, type ReactNode } from 'react';
import {
  MaterialReactTable,
  useMaterialReactTable,
  type MRT_ColumnDef,
  type MRT_SortingState,
} from 'material-react-table';
import { ChevronsUpDown, ChevronUp, ChevronDown } from 'lucide-react';

import { SkeletonRows } from './Skeleton';

type Align = 'left' | 'right' | 'center';

/**
 * Row shape. Deliberately `Record<string, any>` — MRT's own MRT_RowData — and
 * NOT `Record<string, unknown>`: TypeScript only grants an implicit index
 * signature to type ALIASES, so every interface in src/types (AllSignal,
 * Holding, Trade, WatchlistItem…) fails the stricter constraint at the call
 * site with "Index signature for type 'string' is missing".
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type DataTableRow = Record<string, any>;

export interface DataTableColumn<T> {
  /** Unique column id. Doubles as the property read from the row when no `accessor` is given. */
  id: string;
  header: string;
  /** Value the column SORTS on. Defaults to `row[id]`. Give this when the cell renders something derived. */
  accessor?: (row: T) => unknown;
  /** Cell contents. Defaults to the accessor value. */
  cell?: (row: T) => ReactNode;
  align?: Align;
  /** Tabular figures — for any column holding numbers. */
  mono?: boolean;
  /** Default true. */
  sortable?: boolean;
  size?: number;
  minSize?: number;
}

export interface DataTableProps<T extends DataTableRow> {
  columns: DataTableColumn<T>[];
  data: T[];
  /** First load — no data yet. Renders skeleton rows. Pass RTK Query's isLoading. */
  isLoading?: boolean;
  /** Refetch with data already on screen. Shows a progress bar, keeps the rows. Pass RTK Query's isFetching. */
  isFetching?: boolean;
  skeletonRows?: number;
  emptyMessage?: ReactNode;
  onRowClick?: (row: T) => void;
  /** Column id to sort by on first render. */
  initialSort?: { id: string; desc?: boolean };
  /** Omit to render every row on one page with no pagination controls. */
  pagination?: { perPage: number; perPageOptions?: number[] };
  /** Stable React key per row. Falls back to row index. */
  getRowId?: (row: T) => string;
}

// ─── Header ────────────────────────────────────────────────────────────────
// MRT's TableSortLabel still handles the click (its icon is hidden in
// `headCellSx`); this only draws the chevron, so the tri-state indicator —
// ChevronsUpDown when unsorted, single chevron when active — survives, which
// MUI's own sort arrow does not do.

function SortIndicator({ dir, sortable }: { dir: false | 'asc' | 'desc'; sortable: boolean }) {
  if (!sortable) return null;
  const Icon = dir === 'asc' ? ChevronUp : dir === 'desc' ? ChevronDown : ChevronsUpDown;
  const active = dir !== false;
  return (
    <span className="inline-flex align-middle ml-1 opacity-60">
      <Icon size={13} strokeWidth={active ? 2.5 : 1.5} className={active ? 'text-accent-2' : ''} />
    </span>
  );
}

// ─── Slot styling ──────────────────────────────────────────────────────────
// Mirrors the Th/PlainTh/Td classes these tables used before. Padding stays on
// the `var(--u)` scale so the table keeps tracking the app's density setting.

const headCellSx = (align: Align, sortable: boolean) => ({
  fontSize: '11px',
  fontWeight: 600,
  letterSpacing: '.04em',
  textTransform: 'uppercase' as const,
  color: 'var(--text-3)',
  padding: 'calc(11px * var(--u)) 14px',
  borderBottom: '1px solid var(--border)',
  background: 'var(--surface)',
  whiteSpace: 'nowrap' as const,
  verticalAlign: 'middle',
  textAlign: align,
  cursor: sortable ? 'pointer' : 'default',
  transition: 'color .15s var(--ease)',
  '&:hover': sortable ? { color: 'var(--text)' } : {},
  // MRT renders its own MUI sort arrow inside the label — hide it and keep the
  // label's click handling, so our chevron above is the only indicator drawn.
  '& .MuiTableSortLabel-icon': { display: 'none' },
  '& .MuiTableSortLabel-root': { color: 'inherit' },
  '& .MuiTableSortLabel-root:hover': { color: 'inherit' },
  '& .MuiTableSortLabel-root.Mui-active': { color: 'inherit' },
  '& .Mui-TableHeadCell-Content': { justifyContent: align === 'right' ? 'flex-end' : align === 'center' ? 'center' : 'flex-start' },
});

const bodyCellSx = (align: Align, mono: boolean) => ({
  fontSize: '13px',
  fontFamily: mono ? 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)' : 'inherit',
  fontVariantNumeric: mono ? ('tabular-nums' as const) : ('normal' as const),
  color: 'var(--text)',
  padding: 'calc(12px * var(--u)) 14px',
  borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap' as const,
  textAlign: align,
  verticalAlign: 'middle',
});

export function DataTable<T extends DataTableRow>({
  columns,
  data,
  isLoading = false,
  isFetching = false,
  skeletonRows = 8,
  emptyMessage = 'No rows to show.',
  onRowClick,
  initialSort,
  pagination,
  getRowId,
}: DataTableProps<T>) {
  const mrtColumns = useMemo<MRT_ColumnDef<T>[]>(
    () =>
      columns.map(c => {
        const align: Align = c.align ?? 'left';
        const sortable = c.sortable !== false;
        return {
          id: c.id,
          header: c.header,
          accessorFn: (row: T) => (c.accessor ? c.accessor(row) : row[c.id]),
          enableSorting: sortable,
          size: c.size,
          minSize: c.minSize,
          Header: ({ column }) => (
            <span className="inline-flex items-center whitespace-nowrap">
              {c.header}
              <SortIndicator dir={column.getIsSorted()} sortable={sortable} />
            </span>
          ),
          ...(c.cell ? { Cell: ({ row }) => c.cell!(row.original) } : {}),
          muiTableHeadCellProps: { sx: headCellSx(align, sortable) },
          muiTableBodyCellProps: { sx: bodyCellSx(align, c.mono ?? false) },
        } satisfies MRT_ColumnDef<T>;
      }),
    [columns],
  );

  const initialSorting: MRT_SortingState = initialSort
    ? [{ id: initialSort.id, desc: initialSort.desc ?? true }]
    : [];

  const table = useMaterialReactTable<T>({
    columns: mrtColumns,
    data,
    // Everything MUI adds that this design does not have.
    enableColumnActions: false,
    enableColumnFilters: false,
    enableGlobalFilter: false,
    enableTopToolbar: false,
    // Bottom toolbar carries MRT's pagination and its progress bar. With the top
    // toolbar off it is the only place MRT can draw progress, which is why this
    // stays on even for an unpaginated table that is refetching.
    enableBottomToolbar: true,
    enableDensityToggle: false,
    enableFullScreenToggle: false,
    enableHiding: false,
    enableColumnDragging: false,
    enableColumnOrdering: false,
    enableRowSelection: false,
    enableStickyHeader: true,
    enablePagination: !!pagination,
    paginationDisplayMode: 'pages',
    layoutMode: 'semantic',
    ...(getRowId ? { getRowId: (row: T) => getRowId(row) } : {}),
    // isLoading is handled by the skeleton branch below, so MRT is only told
    // about isFetching — otherwise it would draw its own cell skeletons on top.
    state: { showProgressBars: isFetching },
    initialState: {
      sorting: initialSorting,
      ...(pagination ? { pagination: { pageIndex: 0, pageSize: pagination.perPage } } : {}),
    },
    muiTablePaperProps: {
      elevation: 0,
      sx: { background: 'transparent', boxShadow: 'none', borderRadius: 0, border: 'none' },
    },
    muiBottomToolbarProps: {
      sx: {
        background: 'transparent',
        boxShadow: 'none',
        borderTop: '1px solid var(--border)',
        minHeight: 'unset',
        '& .MuiBox-root': { alignItems: 'center' },
      },
    },
    muiLinearProgressProps: {
      sx: {
        height: 2,
        background: 'transparent',
        '& .MuiLinearProgress-bar': { background: 'var(--accent)' },
      },
    },
    // Matches Pager's button language: 32px, 8px radius, 1px line border,
    // accent fill when active.
    muiPaginationProps: {
      shape: 'rounded',
      variant: 'outlined',
      color: 'primary',
      showRowsPerPage: !!pagination?.perPageOptions,
      ...(pagination?.perPageOptions ? { rowsPerPageOptions: pagination.perPageOptions } : {}),
      sx: {
        '& .MuiPaginationItem-root': {
          minWidth: 32,
          height: 32,
          borderRadius: '8px',
          border: '1px solid var(--border)',
          color: 'var(--text-2)',
          fontSize: '13px',
          fontWeight: 600,
          background: 'transparent',
        },
        '& .MuiPaginationItem-root:hover': { background: 'var(--surface-hover)', color: 'var(--text)' },
        '& .MuiPaginationItem-root.Mui-selected': {
          background: 'var(--accent)',
          borderColor: 'var(--accent)',
          color: '#fff',
        },
        '& .MuiPaginationItem-root.Mui-selected:hover': { background: 'var(--accent)' },
        '& .MuiPaginationItem-ellipsis': { border: 'none', color: 'var(--text-3)' },
        '& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows': {
          fontSize: '12.5px',
          color: 'var(--text-2)',
        },
      },
    },
    muiTableContainerProps: {
      className: 'overflow-x-auto',
      sx: { maxHeight: 'none', background: 'transparent' },
    },
    muiTableProps: {
      sx: { borderCollapse: 'collapse', width: '100%', tableLayout: 'auto' },
    },
    muiTableHeadRowProps: { sx: { boxShadow: 'none', background: 'var(--surface)' } },
    muiTableBodyRowProps: ({ row }) => ({
      hover: false,
      onClick: onRowClick ? () => onRowClick(row.original) : undefined,
      sx: {
        cursor: onRowClick ? 'pointer' : 'default',
        background: 'transparent',
        transition: 'background-color .15s var(--ease)',
        '&:hover': { background: 'var(--surface-2)' },
        '&:hover td': { background: 'var(--surface-2)' },
      },
    }),
    renderEmptyRowsFallback: () => (
      <div className="text-center py-[50px] px-5 text-ink-3 text-[13px]">{emptyMessage}</div>
    ),
  });

  // First load only: render the plain skeleton table rather than MRT, so the
  // shape-aware SkeletonRows design is preserved exactly. The header is drawn
  // statically so the table does not reflow when data lands. A REFETCH does not
  // come through here — it keeps the real rows and gets the progress bar.
  if (isLoading) {
    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr>
              {(columns ?? []).map(c => (
                <th
                  key={c?.id}
                  className="text-[11px] font-semibold tracking-[.04em] uppercase text-ink-3 px-3.5 border-b border-line sticky top-0 bg-surface z-[1] whitespace-nowrap"
                  style={{
                    textAlign: c?.align ?? 'left',
                    paddingTop: 'calc(11px * var(--u))',
                    paddingBottom: 'calc(11px * var(--u))',
                  }}
                >
                  {c?.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <SkeletonRows cols={columns.length} rows={skeletonRows} />
          </tbody>
        </table>
      </div>
    );
  }

  return <MaterialReactTable table={table} />;
}
