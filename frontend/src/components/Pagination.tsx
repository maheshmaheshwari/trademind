import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface PaginationProps {
    page: number;
    pageSize: number;
    total: number;
    onPageChange: (page: number) => void;
    onPageSizeChange: (size: number) => void;
    pageSizeOptions?: number[];
    loading?: boolean;
}

export default function Pagination({
    page,
    pageSize,
    total,
    onPageChange,
    onPageSizeChange,
    pageSizeOptions = [10, 25, 50, 100],
    loading = false,
}: PaginationProps) {
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const start = total > 0 ? page * pageSize + 1 : 0;
    const end = Math.min((page + 1) * pageSize, total);

    // Generate visible page numbers (show max 5 pages around current)
    const getPageNumbers = () => {
        const pages: (number | '...')[] = [];
        if (totalPages <= 7) {
            for (let i = 0; i < totalPages; i++) pages.push(i);
        } else {
            pages.push(0);
            if (page > 3) pages.push('...');
            const rangeStart = Math.max(1, page - 1);
            const rangeEnd = Math.min(totalPages - 2, page + 1);
            for (let i = rangeStart; i <= rangeEnd; i++) pages.push(i);
            if (page < totalPages - 4) pages.push('...');
            pages.push(totalPages - 1);
        }
        return pages;
    };

    return (
        <div className="flex items-center justify-between px-5 py-3.5 border-t border-slate-700 bg-surface-dark">
            {/* Left: Rows per page */}
            <div className="flex items-center gap-2">
                <span className="text-slate-400 text-xs font-medium">Rows per page:</span>
                <select
                    value={pageSize}
                    onChange={e => { onPageSizeChange(Number(e.target.value)); onPageChange(0); }}
                    className="bg-transparent text-white text-xs font-semibold px-1 py-1 focus:outline-none cursor-pointer border-b border-slate-600 hover:border-primary transition-colors"
                >
                    {pageSizeOptions.map(n => <option key={n} value={n} className="bg-slate-800 text-white">{n}</option>)}
                </select>
            </div>

            {/* Center: Record count */}
            <div className="text-slate-400 text-xs font-medium tracking-wide">
                {loading ? (
                    <span className="text-slate-500">Loading...</span>
                ) : (
                    <span>Showing <span className="text-white font-semibold">{start}–{end}</span> of <span className="text-white font-semibold">{total.toLocaleString()}</span></span>
                )}
            </div>

            {/* Right: Page navigation */}
            <div className="flex items-center gap-1">
                {/* First */}
                <button
                    disabled={page === 0 || loading}
                    onClick={() => onPageChange(0)}
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-20 disabled:cursor-not-allowed transition-all"
                    title="First page"
                >
                    <ChevronsLeft className="w-4 h-4" />
                </button>

                {/* Prev */}
                <button
                    disabled={page === 0 || loading}
                    onClick={() => onPageChange(page - 1)}
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-20 disabled:cursor-not-allowed transition-all"
                    title="Previous page"
                >
                    <ChevronLeft className="w-4 h-4" />
                </button>

                {/* Page Numbers */}
                <div className="flex items-center gap-0.5 mx-1">
                    {getPageNumbers().map((p, i) =>
                        p === '...' ? (
                            <span key={`dots-${i}`} className="w-8 h-8 flex items-center justify-center text-slate-600 text-xs">•••</span>
                        ) : (
                            <button
                                key={p}
                                onClick={() => onPageChange(p as number)}
                                disabled={loading}
                                className={`w-8 h-8 rounded-lg text-xs font-semibold transition-all ${p === page
                                    ? 'bg-primary text-white shadow-lg shadow-primary/25'
                                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                                    }`}
                            >
                                {(p as number) + 1}
                            </button>
                        )
                    )}
                </div>

                {/* Next */}
                <button
                    disabled={page >= totalPages - 1 || loading}
                    onClick={() => onPageChange(page + 1)}
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-20 disabled:cursor-not-allowed transition-all"
                    title="Next page"
                >
                    <ChevronRight className="w-4 h-4" />
                </button>

                {/* Last */}
                <button
                    disabled={page >= totalPages - 1 || loading}
                    onClick={() => onPageChange(totalPages - 1)}
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-20 disabled:cursor-not-allowed transition-all"
                    title="Last page"
                >
                    <ChevronsRight className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}
