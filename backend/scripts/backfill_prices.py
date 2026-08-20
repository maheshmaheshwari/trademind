"""
Backfill EOD prices + technical indicators for a range of trading days.

The daily `eod_data` scheduler job only ever collects "today". When it misses a
day — the Space was asleep at 15:35 IST, or Angel One throttled the run — that
day is gone: the recovery queue re-runs the job, but the job still collects
today, and the coverage gate measures today, so a replay can never repair a
past date. This script is the repair path.

It is date-driven rather than "last N days from now", idempotent, and skips
work that is already done, so it is safe to re-run.

Usage (from backend/, with venv active):

    # one day
    python scripts/backfill_prices.py --date 2026-08-18

    # a range (inclusive); non-trading days are dropped automatically
    python scripts/backfill_prices.py --from 2026-08-18 --to 2026-08-19

    # the last N calendar days, resolved to trading days
    python scripts/backfill_prices.py --days 7

    # see what would happen, touch nothing
    python scripts/backfill_prices.py --from 2026-08-18 --to 2026-08-19 --dry-run

    # a few symbols only
    python scripts/backfill_prices.py --date 2026-08-18 --symbols RELIANCE TCS

Useful flags:
    --force             re-fetch and overwrite days that are already present
    --prices-only       skip the indicator pass
    --indicators-only   recompute indicators from prices already in the DB
    --rate-limit-secs   seconds between Angel requests (default 0.6, adaptive)

Skip rule: a symbol whose target days are all present in `prices` costs no API
call at all. With --force nothing is skipped.
"""

import argparse
import logging
import os
import sys
import time
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

# Allow both `python scripts/backfill_prices.py` and `import scripts.backfill_prices`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import (  # noqa: E402
    _execute,
    get_connection,
    get_all_prices_df,
    init_database,
    insert_indicators,
    insert_prices_batch,
    release_connection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_prices")

# Angel One throttles hard and without warning. Two back-to-back getCandleData
# calls with no gap are enough to earn
#   "Access denied because of exceeding access rate"
# (reproduced 2026-08-20). The daily job treats a throttled symbol as a failure
# and defers it to a single bulk retry at the end; if the throttle is still on
# when that retry runs, every symbol fails twice and the day collects 0 rows.
# Here we instead back off and retry the *same* symbol immediately, and widen
# the inter-request gap for the rest of the run, so one throttled window slows
# the backfill down rather than emptying it.
DEFAULT_RATE_LIMIT_SECS = float(os.environ.get("ANGEL_RATE_LIMIT_SECS", "0.6"))
RATE_LIMIT_BACKOFFS = (5, 15, 45)      # seconds to wait before each same-symbol retry
RATE_LIMIT_GAP_GROWTH = 1.5            # multiply the steady-state gap after a throttle
RATE_LIMIT_GAP_CEILING = 3.0           # …but never slower than this

_RATE_LIMIT_MARKERS = ("exceeding access rate", "access denied", "too many requests")

# sma_200 needs 200 bars; ask for plenty of slack around the target window.
INDICATOR_LOOKBACK_DAYS = 500


def _is_rate_limited(err: str) -> bool:
    e = (err or "").lower()
    return any(m in e for m in _RATE_LIMIT_MARKERS)


# ──────────────────────────────────────────────────────────────────────────────
# Target days
# ──────────────────────────────────────────────────────────────────────────────

def resolve_target_days(args) -> List[date_type]:
    """Turn --date / --from/--to / --days into the list of NSE trading days.

    Weekends and exchange holidays are dropped here, so the rest of the script
    never has to reason about them and a holiday can't be reported as a gap.
    """
    from analysis.trading_calendar import (
        last_expected_trading_day,
        today_ist,
        trading_days_between,
    )

    if args.date:
        start = end = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.from_date or args.to_date:
        if not (args.from_date and args.to_date):
            raise SystemExit("--from and --to must be given together")
        start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    else:
        end = today_ist()
        start = end - timedelta(days=args.days)

    if start > end:
        raise SystemExit(f"--from {start} is after --to {end}")

    # Never target a day the market hasn't finished yet: before the 15:35 IST
    # EOD window there is no complete daily candle, and asking for one is how
    # a run "succeeds" with zero rows.
    cutoff = last_expected_trading_day()
    if end > cutoff:
        logger.info("Trimming end %s → %s (no complete EOD candle after that yet)", end, cutoff)
        end = cutoff

    days = trading_days_between(start, end)
    if not days:
        raise SystemExit(f"No NSE trading days between {start} and {end}")
    return days


# ──────────────────────────────────────────────────────────────────────────────
# What's already there
# ──────────────────────────────────────────────────────────────────────────────

def _existing_pairs(table: str, start: date_type, end: date_type) -> Set[Tuple[str, str]]:
    """{(symbol, 'YYYY-MM-DD')} already present in `table` over the window.

    One bulk query rather than a per-symbol probe — 500 round trips to Timescale
    Cloud costs minutes on its own.
    """
    date_col = "date"
    where_interval = " AND interval = '1d'" if table == "prices" else ""
    conn = get_connection()
    try:
        cur = _execute(
            conn,
            f"SELECT symbol, {date_col} FROM {table} "
            f"WHERE {date_col} >= ? AND {date_col} <= ?{where_interval}",
            (start.isoformat(), end.isoformat()),
        )
        return {(r[0], str(r[1])[:10]) for r in cur.fetchall()}
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────────────────────
# Price backfill
# ──────────────────────────────────────────────────────────────────────────────

def backfill_prices(token_map: Dict, target_days: List[date_type],
                    force: bool, rate_limit: float, dry_run: bool) -> Dict:
    from scripts.update_stocks_angel import angel_login, fetch_candles

    start, end = target_days[0], target_days[-1]
    want = {d.isoformat() for d in target_days}

    present = set() if force else _existing_pairs("prices", start, end)

    todo: List[Tuple[str, Dict, Set[str]]] = []
    for symbol, info in token_map.items():
        ns = f"{symbol}.NS"
        missing = {d for d in want if (ns, d) not in present}
        if missing:
            todo.append((symbol, info, missing))

    skipped = len(token_map) - len(todo)
    print(f"\n📅 Target trading days ({len(target_days)}): "
          f"{', '.join(d.isoformat() for d in target_days)}")
    print(f"📊 Symbols: {len(token_map)} total — {len(todo)} need data, {skipped} already complete")

    if not todo:
        print("✅ Nothing to fetch — every symbol already has every target day.")
        return {"fetched": 0, "rows": 0, "failed": [], "skipped": skipped, "touched": {}}

    if dry_run:
        for symbol, _, missing in todo[:20]:
            print(f"   would fetch {symbol:15s} {sorted(missing)}")
        if len(todo) > 20:
            print(f"   ... and {len(todo) - 20} more")
        return {"fetched": 0, "rows": 0, "failed": [], "skipped": skipped, "touched": {}}

    # Angel wants IST wall-clock strings. Pad the window by a day on each side:
    # the API is inclusive-ish at the edges and a wider window costs nothing,
    # while rows outside `want` are filtered out before insert anyway.
    from_str = (start - timedelta(days=1)).strftime("%Y-%m-%d 09:15")
    to_str = (end + timedelta(days=1)).strftime("%Y-%m-%d 15:30")

    api = angel_login()
    gap = rate_limit
    total_rows = 0
    failed: List[str] = []
    touched: Dict[str, Set[str]] = {}

    try:
        for idx, (symbol, info, missing) in enumerate(todo, 1):
            rows = None
            for attempt in range(len(RATE_LIMIT_BACKOFFS) + 1):
                try:
                    rows = fetch_candles(
                        api, symbol=symbol, token=info["token"], exchange="NSE",
                        from_date=from_str, to_date=to_str,
                    )
                    break
                except Exception as exc:
                    if _is_rate_limited(str(exc)) and attempt < len(RATE_LIMIT_BACKOFFS):
                        wait = RATE_LIMIT_BACKOFFS[attempt]
                        gap = min(gap * RATE_LIMIT_GAP_GROWTH, RATE_LIMIT_GAP_CEILING)
                        logger.warning(
                            "[%d/%d] %s throttled — waiting %ds, gap now %.2fs (retry %d)",
                            idx, len(todo), symbol, wait, gap, attempt + 1,
                        )
                        time.sleep(wait)
                        continue
                    logger.error("[%d/%d] %s FAILED: %s", idx, len(todo), symbol, exc)
                    failed.append(symbol)
                    rows = None
                    break

            if rows:
                # Only the days we actually asked for. Without this a padded
                # window would silently rewrite neighbouring days too, which
                # makes --date 2026-08-18 a lie.
                keep = [r for r in rows if r[2] in missing]
                if keep:
                    inserted = insert_prices_batch(keep, sync=False)
                    total_rows += inserted
                    touched[f"{symbol}.NS"] = {r[2] for r in keep}
                    logger.info("[%d/%d] %-15s +%d rows %s",
                                idx, len(todo), symbol, inserted, sorted({r[2] for r in keep}))
                else:
                    logger.info("[%d/%d] %-15s no candles for the target day(s)",
                                idx, len(todo), symbol)

            if idx % 50 == 0:
                print(f"  ⏳ {idx}/{len(todo)} — {total_rows} rows, {len(failed)} failed")
            time.sleep(gap)
    finally:
        try:
            api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
        except Exception:
            pass

    return {"fetched": len(todo), "rows": total_rows, "failed": failed,
            "skipped": skipped, "touched": touched}


# ──────────────────────────────────────────────────────────────────────────────
# Indicator backfill
# ──────────────────────────────────────────────────────────────────────────────

def backfill_indicators(symbols: List[str], target_days: List[date_type],
                        force: bool, dry_run: bool) -> Dict:
    """Compute technical indicators for each target day and upsert one row per day.

    `analysis.signals.process_stock` only ever writes `df.iloc[-1]` — the latest
    bar — so it cannot repair a historical date. Here the same `calculate_all`
    runs over the full history and every target day is written.

    `signal` / `signal_strength` are deliberately left NULL: those are ML model
    output produced by scripts/generate_trades.py against the *current* bar, and
    backfilling them for a past date would fabricate a recommendation that was
    never made. That is also why an existing indicator row is skipped unless
    --force — insert_indicators upserts every column, so rewriting a row that
    already carries a signal would null it out.
    """
    import pandas as pd
    from analysis.indicators import calculate_all

    start, end = target_days[0], target_days[-1]
    want = {d.isoformat() for d in target_days}

    present = set() if force else _existing_pairs("technical_indicators", start, end)

    todo = []
    for ns in symbols:
        missing = {d for d in want if (ns, d) not in present}
        if missing:
            todo.append((ns, missing))

    print(f"\n🔬 Indicators: {len(todo)} symbol(s) need rows, "
          f"{len(symbols) - len(todo)} already complete")

    if not todo:
        print("✅ Nothing to compute — indicators already present for every target day.")
        return {"symbols": 0, "rows": 0, "failed": []}

    if dry_run:
        for ns, missing in todo[:20]:
            print(f"   would compute {ns:18s} {sorted(missing)}")
        if len(todo) > 20:
            print(f"   ... and {len(todo) - 20} more")
        return {"symbols": 0, "rows": 0, "failed": []}

    written = 0
    failed: List[str] = []
    conn = get_connection()
    try:
        for idx, (ns, missing) in enumerate(todo, 1):
            try:
                prices = get_all_prices_df(ns, days=INDICATOR_LOOKBACK_DAYS)
                if not prices or len(prices) < 14:
                    logger.warning("[%d/%d] %s skipped — only %d price rows",
                                   idx, len(todo), ns, len(prices) if prices else 0)
                    failed.append(ns)
                    continue

                df = calculate_all(pd.DataFrame(prices))
                df["_d"] = df["date"].astype(str).str[:10]

                rows_for_symbol = 0
                for _, row in df[df["_d"].isin(missing)].iterrows():
                    ok = insert_indicators(ns, row["_d"], {
                        k: _safe_float(row.get(k)) for k in (
                            "rsi_14", "macd", "macd_signal", "macd_hist",
                            "bb_upper", "bb_middle", "bb_lower",
                            "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
                            "atr_14", "adx_14", "stoch_k", "stoch_d", "obv",
                            "support_1", "support_2", "support_3",
                            "resistance_1", "resistance_2", "resistance_3",
                        )
                    }, conn=conn)
                    # insert_indicators logs and returns False rather than
                    # raising. On Postgres a failed statement aborts the whole
                    # transaction, so every later insert on this connection
                    # would fail too — bail out and reset instead of writing
                    # into a dead transaction.
                    if not ok:
                        raise RuntimeError(f"insert_indicators rejected {ns} {row['_d']}")
                    rows_for_symbol += 1

                conn.commit()
                written += rows_for_symbol
                if rows_for_symbol == 0:
                    logger.warning("[%d/%d] %s — no price bar for %s, indicators not written",
                                   idx, len(todo), ns, sorted(missing))
                    failed.append(ns)
            except Exception as exc:
                conn.rollback()
                logger.error("[%d/%d] %s indicator failure: %s", idx, len(todo), ns, exc)
                failed.append(ns)

            if idx % 50 == 0:
                print(f"  ⏳ {idx}/{len(todo)} — {written} indicator rows")
    finally:
        release_connection(conn)

    return {"symbols": len(todo), "rows": written, "failed": failed}


def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f      # drop NaN
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Coverage report
# ──────────────────────────────────────────────────────────────────────────────

def report_coverage(target_days: List[date_type]) -> None:
    start, end = target_days[0], target_days[-1]
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT c.d,
                   (SELECT COUNT(*) FROM prices p
                     WHERE p.interval='1d' AND p.date = c.d) AS px,
                   (SELECT COUNT(*) FROM technical_indicators t
                     WHERE t.date = c.d) AS ind
              FROM (SELECT generate_series(?::date, ?::date, '1 day')::date AS d) c
             ORDER BY c.d
        """, (start.isoformat(), end.isoformat()))
        want = {d.isoformat() for d in target_days}
        print(f"\n{'='*60}")
        print("📈 Coverage after backfill")
        print(f"{'date':<14}{'prices':>10}{'indicators':>14}")
        for d, px, ind in cur.fetchall():
            ds = str(d)[:10]
            if ds not in want:
                continue
            print(f"{ds:<14}{px:>10,}{ind:>14,}")
    except Exception as exc:
        logger.warning("Coverage report unavailable: %s", exc)
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill EOD prices and technical indicators for a range of trading days.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_argument_group("date selection (pick one)")
    g.add_argument("--date", help="single day, YYYY-MM-DD")
    g.add_argument("--from", dest="from_date", help="range start, YYYY-MM-DD (inclusive)")
    g.add_argument("--to", dest="to_date", help="range end, YYYY-MM-DD (inclusive)")
    g.add_argument("--days", type=int, default=7,
                   help="fallback: last N calendar days ending today (default 7)")

    ap.add_argument("--symbols", nargs="+", help="only these symbols (without .NS)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch/recompute days that are already present")
    ap.add_argument("--prices-only", action="store_true", help="skip the indicator pass")
    ap.add_argument("--indicators-only", action="store_true",
                    help="recompute indicators from prices already in the DB")
    ap.add_argument("--rate-limit-secs", type=float, default=DEFAULT_RATE_LIMIT_SECS,
                    help=f"seconds between Angel requests (default {DEFAULT_RATE_LIMIT_SECS})")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen, change nothing")
    args = ap.parse_args(argv)

    if args.prices_only and args.indicators_only:
        raise SystemExit("--prices-only and --indicators-only are mutually exclusive")

    init_database()
    target_days = resolve_target_days(args)

    from scripts.update_stocks_angel import load_token_map
    token_map = load_token_map()
    if args.symbols:
        wanted = {s.upper().removesuffix(".NS") for s in args.symbols}
        token_map = {k: v for k, v in token_map.items() if k.upper() in wanted}
        if not token_map:
            raise SystemExit(f"None of {sorted(wanted)} are in the Angel token map")

    price_result = {"rows": 0, "failed": [], "skipped": 0, "touched": {}}
    if not args.indicators_only:
        price_result = backfill_prices(
            token_map, target_days, args.force, args.rate_limit_secs, args.dry_run
        )

    ind_result = {"rows": 0, "failed": [], "symbols": 0}
    if not args.prices_only:
        # Indicators are recomputed for every requested symbol, not only the
        # ones whose prices changed in this run: a day may have had prices all
        # along and be missing only its indicator row (exactly the state a
        # half-finished EOD chain leaves behind).
        symbols = [f"{s}.NS" for s in token_map]
        ind_result = backfill_indicators(symbols, target_days, args.force, args.dry_run)

    print(f"\n{'='*60}")
    print("✅ Backfill complete" if not args.dry_run else "🔍 Dry run — nothing written")
    print(f"   Price rows written:      {price_result['rows']:,}")
    print(f"   Indicator rows written:  {ind_result['rows']:,}")
    print(f"   Symbols skipped (had data): {price_result['skipped']:,}")
    if price_result["failed"]:
        print(f"   ⚠️  Price failures ({len(price_result['failed'])}): "
              f"{', '.join(price_result['failed'][:15])}"
              + (" ..." if len(price_result["failed"]) > 15 else ""))
    if ind_result["failed"]:
        print(f"   ⚠️  Indicator failures ({len(ind_result['failed'])}): "
              f"{', '.join(ind_result['failed'][:15])}"
              + (" ..." if len(ind_result["failed"]) > 15 else ""))

    if not args.dry_run:
        report_coverage(target_days)
        print("\n   Signals are NOT regenerated by this script — run "
              "`python scripts/generate_trades.py` once coverage looks right.")

    return 1 if (price_result["failed"] or ind_result["failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
