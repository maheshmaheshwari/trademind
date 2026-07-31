import { useState } from 'react';
import { CalendarDays, ChevronDown, AlertTriangle, CheckCircle2 } from 'lucide-react';
import {
  useGetMarketHolidaysQuery,
  useGetDataFreshnessQuery,
  type MarketHoliday,
} from '../services/tradeMindApiService';

/**
 * Trading-calendar banner shown above every page.
 *
 * Two things, in priority order:
 *  1. A price-data warning when /api/market/data-freshness reports missing or
 *     partially-collected trading days — i.e. the DB is behind the NSE
 *     calendar, so signals are being computed on stale bars.
 *  2. Otherwise the calendar state: today's holiday, or the next one coming up.
 *
 * Silent when the market is trading normally and the data is current — a
 * banner that is always on stops being read.
 */

const DAY_FMT: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short' };

function fmtDate(iso?: string | null): string {
  if (!iso) return '—';
  const d = new Date(`${iso}T00:00:00`);
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-IN', DAY_FMT);
}

function daysAwayLabel(n?: number): string {
  if (n === undefined || n === null) return '';
  if (n === 0) return 'today';
  if (n === 1) return 'tomorrow';
  return `in ${n} days`;
}

export default function MarketBanner() {
  const [open, setOpen] = useState(false);

  const { data: cal } = useGetMarketHolidaysQuery({ upcoming: 8 });
  const { data: fresh } = useGetDataFreshnessQuery(60);

  const today = cal?.today;
  const upcoming = (cal?.upcoming ?? []).filter(h => !h?.is_weekend);
  const next: MarketHoliday | undefined = upcoming[0];

  // Freshness problems worth interrupting for. "stale" alone during market
  // hours is normal (today's bar lands at 15:35 IST), so only missing/partial
  // days and a multi-day lag are surfaced.
  const missing = (fresh?.missing_dates ?? []);
  const partial = (fresh?.partial_dates ?? []);
  const unexpected = (fresh?.unexpected_dates ?? []);
  const staleBy = fresh?.stale_by_days ?? 0;
  const dataProblem =
    fresh?.status === 'no_data' ||
    missing.length > 0 || partial.length > 0 || (staleBy ?? 0) > 1;

  const holidayToday = today?.is_holiday ? today?.holiday_name : null;

  // Nothing to say — trading day, data current, next holiday far off.
  if (!dataProblem && !holidayToday && (next?.days_away ?? 99) > 10) return null;

  const tone = dataProblem
    ? { wrap: 'border-loss/40 bg-loss-soft', text: 'text-loss', Icon: AlertTriangle }
    : holidayToday
      ? { wrap: 'border-gold/40 bg-gold-soft', text: 'text-gold', Icon: CalendarDays }
      : { wrap: 'border-line bg-surface-2', text: 'text-ink-2', Icon: CalendarDays };

  const { Icon } = tone;

  return (
    <div className={`rounded-[12px] border ${tone.wrap} mb-4`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 bg-transparent border-none cursor-pointer text-left"
      >
        <Icon size={17} className={`${tone.text} flex-shrink-0`} />

        <div className="flex-1 min-w-0 flex flex-wrap items-center gap-x-2 gap-y-0.5">
          {dataProblem ? (
            <>
              <span className={`text-[13px] font-semibold ${tone.text}`}>
                Price data behind the NSE calendar
              </span>
              <span className="text-[12.5px] text-ink-2">
                latest bar {fmtDate(fresh?.latest_price_date)} · expected{' '}
                {fmtDate(fresh?.last_trading_day)}
                {missing.length > 0 && ` · ${missing.length} trading day${missing.length > 1 ? 's' : ''} missing`}
                {partial.length > 0 && ` · ${partial.length} partially collected`}
              </span>
            </>
          ) : holidayToday ? (
            <>
              <span className={`text-[13px] font-semibold ${tone.text}`}>
                Market closed today — {holidayToday}
              </span>
              <span className="text-[12.5px] text-ink-2">
                NSE trading resumes {fmtDate(today?.next_trading_day)}
              </span>
            </>
          ) : (
            <>
              <span className="text-[13px] font-semibold text-ink">
                Next NSE holiday · {fmtDate(next?.date)} — {next?.description}
              </span>
              <span className="text-[12.5px] text-ink-3">
                {next?.weekday} · {daysAwayLabel(next?.days_away)}
              </span>
            </>
          )}
        </div>

        <ChevronDown
          size={16}
          className={`text-ink-3 flex-shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3 grid gap-4 md:grid-cols-2">
          {/* ── Upcoming holidays ── */}
          <div>
            <div className="text-[10.5px] font-semibold tracking-[.08em] uppercase text-ink-3 mb-2">
              Upcoming NSE holidays
            </div>
            {(cal?.upcoming ?? []).length === 0 ? (
              <div className="text-[12.5px] text-ink-3">
                No holidays stored for the rest of the calendar year.
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {(cal?.upcoming ?? []).map(h => (
                  <div key={h?.date} className="flex items-center gap-2 text-[12.5px]">
                    <span className="font-mono text-ink-2 w-[62px] flex-shrink-0">
                      {fmtDate(h?.date)}
                    </span>
                    <span className="text-ink truncate flex-1">{h?.description}</span>
                    <span className="text-ink-3 flex-shrink-0">
                      {h?.is_weekend ? `${h?.weekday} · weekend` : daysAwayLabel(h?.days_away)}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {(cal?.years_covered ?? []).length > 0 && (
              <div className="text-[11px] text-ink-3 mt-2">
                Calendar loaded for {(cal?.years_covered ?? []).join(', ')} · source: nseindia.com
              </div>
            )}
          </div>

          {/* ── Price-date verification ── */}
          <div>
            <div className="text-[10.5px] font-semibold tracking-[.08em] uppercase text-ink-3 mb-2">
              Price-date check
            </div>
            <div className="flex items-center gap-2 text-[12.5px] mb-1.5">
              {fresh?.status === 'ok' ? (
                <CheckCircle2 size={15} className="text-gain flex-shrink-0" />
              ) : (
                <AlertTriangle size={15} className="text-gold flex-shrink-0" />
              )}
              <span className="text-ink">
                {fresh?.status === 'ok'
                  ? `All ${fresh?.trading_days_present ?? 0} trading days present`
                  : fresh?.status === 'no_calendar'
                    ? 'No holiday calendar stored yet'
                    : `${fresh?.trading_days_present ?? 0} of ${fresh?.trading_days_expected ?? 0} trading days present`}
              </span>
            </div>

            <div className="text-[12.5px] text-ink-2">
              Latest stored bar: <span className="font-mono">{fmtDate(fresh?.latest_price_date)}</span>
              {' · '}last trading day: <span className="font-mono">{fmtDate(fresh?.last_trading_day)}</span>
            </div>

            {missing.length > 0 && (
              <div className="text-[12.5px] text-loss mt-1.5">
                Missing: {missing.slice(0, 6).map(d => fmtDate(d)).join(', ')}
                {missing.length > 6 && ` +${missing.length - 6} more`}
              </div>
            )}
            {partial.length > 0 && (
              <div className="text-[12.5px] text-gold mt-1.5">
                Partial: {partial.slice(0, 4).map(p => `${fmtDate(p?.date)} (${p?.symbols} symbols)`).join(', ')}
              </div>
            )}
            {unexpected.length > 0 && (
              <div className="text-[12.5px] text-gold mt-1.5">
                Bars dated on a non-trading day:{' '}
                {unexpected.slice(0, 4).map(u => `${fmtDate(u?.date)} (${u?.holiday || u?.reason})`).join(', ')}
              </div>
            )}
            {(fresh?.uncovered_years ?? []).length > 0 && (
              <div className="text-[11px] text-ink-3 mt-2">
                Not checked for {(fresh?.uncovered_years ?? []).join(', ')} — no holiday calendar for{' '}
                {(fresh?.uncovered_years ?? []).length > 1 ? 'those years' : 'that year'}.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
