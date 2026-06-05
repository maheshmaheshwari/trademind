/**
 * TradeMind AI — Shared formatting utilities
 *
 * Use these helpers instead of copy-pasting formatting logic across pages.
 */

/**
 * Format a number as Indian Rupees with commas.
 * @example inr(1234567.89) → "₹12,34,567.89"
 */
export function inr(n: number, dec = 2): string {
  return '₹' + Number(n).toLocaleString('en-IN', {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

/**
 * Format a large number compactly in Indian units (Cr / L).
 * @example inrCompact(12345678) → "₹1.23 Cr"
 */
export function inrCompact(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}

/**
 * Format a duration in minutes as a human-readable "ago" string.
 * @example fmtAgo(90) → "1h ago"
 */
export function fmtAgo(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}m ago`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m > 0 ? `${h}h ${m}m ago` : `${h}h ago`;
}

/**
 * Format a Date or ISO string as "ago" using actual elapsed minutes.
 * @example fmtDateAgo(new Date(Date.now() - 300000)) → "5m ago"
 */
export function fmtDateAgo(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const minutes = (Date.now() - d.getTime()) / 60000;
  return fmtAgo(minutes);
}
