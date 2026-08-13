"""
TradeMind — Historical Corporate-Announcement Backfill (NSE + BSE)

One entry point for deepening news history behind the 2023-01-01 wall the
news_sentiment table currently starts at. Drives the two exchange collectors
and the single FinBERT scoring pass, and sizes the CI shard matrices.

WHY THESE TWO SOURCES
    GDELT is not an option — not because of a date floor, but because it does
    not answer at all. Measured 2026-08-13: three fetches (RELIANCE 2019-03,
    TCS 2019-03, INFY 2021-06) each returned 429 on the first call, exhausted
    the full 60/120/180s retry ladder, and yielded zero articles; average 411s
    per request for nothing. The first call was throttled with no prior traffic,
    so this is not a burnt quota. news_sentiment has never held a single
    GDELT-sourced row — an earlier version of this docstring called it "the
    collector that produced most of our media rows", which the database
    contradicts (media rows come from RSS, Economic Times, yfinance and
    alphavantage). Whether its pre-2017 date floor is real is now untestable
    and moot.

    Announcements are also the better input for this model: exact timestamps,
    an explicit symbol on every row, and no headline-to-company matching to get
    wrong — unlike media archives, where the matching is where the accuracy
    goes.

    Consequence worth knowing: pre-2023 history is announcements-only, while
    2023-onward carries announcements AND press coverage. news_count therefore
    means something different either side of that boundary.

WINDOWS — THE TWO SOURCES TILE, THEY DO NOT OVERLAP
    BSE  2010-01-01 .. 2017-12-31   materially deeper coverage before 2018
    NSE  2018-01-01 .. today        overlaps the daily job on purpose, to top
                                    up whatever it missed (both build URLs via
                                    _ann_url, so re-fetches dedupe away)

    From 2018 the two sources OVERLAP rather than tile — see BSE_TO for the
    measurement behind that. Each exchange holds announcements the other does
    not, so running both maximises coverage; the price is inflated count
    features (news_count and friends) on dates where both carry the same event.
    Within a source, re-fetches still dedupe on uq_news_url_pubdate.

    Topping up 2023+ needs --no-skip. The skip check only inspects a symbol's
    OLDEST stored row, so 60% of symbols already satisfy it from the 2018
    backfill and get skipped. Interior gaps — 2026-06 and 2026-07 collected at
    roughly half the usual monthly rate — are invisible to any min/max coverage
    check, so only a full re-fetch fills them.

    An earlier version of this docstring said NSE's API "bottoms out around
    2018". That is wrong — NSE returns 2012 Q1 announcements for TCS, INFY,
    HDFCBANK and RELIANCE, and BSE serves post-2018 fine (63 rows for TCS in
    2019 Q1). Both archives cover the whole range; the split is a coverage
    choice, not a limit. Measured head-to-head on the same symbol and quarter,
    BSE is equal or richer before 2018, sometimes by a wide margin:

        symbol    period   BSE   NSE
        TCS       2012Q1    32    19
        TCS       2015Q1    45    40
        INFY      2012Q1    12    12
        RELIANCE  2012Q1    26    18
        RELIANCE  2015Q1   171    33

    What the tiling is actually for: a company files the same event to both
    exchanges under different URLs, so uq_news_url_pubdate cannot dedupe across
    sources. Overlapping windows would store that event twice and double its
    weight in the daily sentiment aggregate. Keep the windows disjoint wherever
    the boundary sits. Override with --from-date/--to-date only on a fresh
    database, where there is nothing above to collide with.

FETCH IN PARALLEL, SCORE ONCE
    Shards store rows unscored. score_pending_news() claims every globally
    unscored row, so two shards scoring concurrently would process the same
    headlines twice; it also keeps torch out of the fetch shards entirely.

Usage:
    # plan (emits GitHub Actions output lines)
    python scripts/backfill_announcements.py --plan bse

    # one shard of a parallel fetch
    python scripts/backfill_announcements.py --source bse --shard 3/9 --fetch-only

    # single scoring pass once every shard has landed
    python scripts/backfill_announcements.py --score-only

    # everything, serially, on one machine
    python scripts/backfill_announcements.py --source both
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import get_active_universe  # noqa: E402

logger = logging.getLogger(__name__)

# ── Default windows ───────────────────────────────────────────────────────────

BSE_FROM = "2010-01-01"
# Runs to today, same as NSE. The windows deliberately OVERLAP from 2018 on —
# an earlier version of this file forbade that, on the grounds that a company
# files the same event to both exchanges and the resulting duplicate would
# double-weight it. Measured, that is only half true:
#
#   TCS      2019Q1  BSE 63 rows / 43 dates, NSE 55 / 41  -> 40 shared,  3 BSE-only
#   RELIANCE 2019Q1  BSE 61 rows / 39 dates, NSE 33 / 23  -> 23 shared, 16 BSE-only
#
# BSE carries announcements on dates NSE has nothing for — 41% of RELIANCE's
# dates. That is coverage, not duplication.
#
# The cost is real but narrower than first claimed: avg_sentiment is a mean, so
# a near-duplicate row barely moves it. What inflates on shared dates is the
# COUNT features — news_count, positive_count/negative_count, mkt_news_count.
# Anything reading those across the 2018 boundary is comparing a one-exchange
# count to a two-exchange count.
BSE_TO   = date.today().isoformat()
NSE_FROM = "2018-01-01"
# Runs to today, not to a fixed 2022-12-31. The daily job covers 2023 onward,
# but it covers it imperfectly: 2026-06 and 2026-07 landed ~2,000 announcements
# each against a ~3,300/month baseline, roughly half the daily rate of the
# months either side. Carrying the window to the present lets a backfill top up
# whatever the daily job missed. Safe to overlap the daily job because both
# build URLs through _ann_url(), so re-fetched rows collide on
# uq_news_url_pubdate and are dropped.
#
# BSE_TO stays at 2017-12-31 and must not follow: BSE and NSE give the same
# filing different URLs, so the index cannot dedupe across sources and
# overlapping windows would double-weight every dual-filed event.
NSE_TO   = date.today().isoformat()

# ── Shard sizing ──────────────────────────────────────────────────────────────
# BSE is the expensive one: one calendar-year window per symbol per year, each
# paged 50 rows at a time, so its cost scales with the window length. Measured
# 24.7s/symbol over TCS/INFY/ITC/WIPRO for 2010-2017 (8 years).
#
# Halved from 60 to 30 when BSE_TO moved from 2017-12-31 to today: the window
# went from 8 years to ~16.6, doubling per-symbol cost, which would have taken a
# 60-symbol shard from ~25 min to ~52 min against the job's 90-minute timeout —
# a 1.7x margin. 30/shard restores roughly the original ~26 min and 3.4x margin.
# max-parallel is unchanged, so this adds waves rather than concurrent load on
# BSE.
#
# NSE answers a whole multi-year range in one request per symbol (~4s including
# the politeness sleep) regardless of how long that range is, so widening its
# window does not change its sizing.
SYMBOLS_PER_BSE_SHARD = 30
SYMBOLS_PER_NSE_SHARD = 125


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _universe() -> List[str]:
    """Bare NSE tickers for the current Nifty 500 constituents."""
    return [s.replace(".NS", "") for s in get_active_universe()]


def _shard(symbols: List[str], spec: Optional[str]) -> List[str]:
    """Round-robin slice i/N. Round-robin, not contiguous, so a shard's cost is
    an average of the universe rather than whatever its alphabetical block
    happens to contain."""
    if not spec:
        return symbols
    i, n = (int(x) for x in spec.split("/"))
    return symbols[i - 1::n]


# ── Planning ──────────────────────────────────────────────────────────────────

def shard_plan(source: str, per_shard: Optional[int] = None) -> Dict:
    """How many shards this source needs, as the workflow matrix needs it."""
    symbols = _universe()
    if per_shard is None:
        per_shard = SYMBOLS_PER_BSE_SHARD if source == "bse" else SYMBOLS_PER_NSE_SHARD
    count = len(symbols)
    shards = max(1, -(-count // per_shard)) if count else 0
    return {"count": count, "shards": shards,
            "matrix": list(range(1, shards + 1))}


# ── Steps ─────────────────────────────────────────────────────────────────────

def run_bse(symbols: List[str], from_date: date, to_date: date,
            skip_covered: bool = True) -> Dict:
    from collectors.bse_announcements_collector import backfill
    logger.info("BSE: %d symbols, %s to %s", len(symbols), from_date, to_date)
    return backfill(symbols, from_date=from_date, to_date=to_date,
                    skip_covered=skip_covered)


def run_nse(symbols: List[str], from_date: date, to_date: date,
            skip_covered: bool = True, score: bool = False) -> Dict:
    from collectors.nse_announcements_collector import backfill_all
    logger.info("NSE: %d symbols, %s to %s", len(symbols), from_date, to_date)
    # The NSE API speaks dd-mm-yyyy; everything else here is ISO.
    return backfill_all(
        from_date=from_date.strftime("%d-%m-%Y"),
        to_date=to_date.strftime("%d-%m-%Y"),
        symbols=symbols,
        skip_existing=skip_covered,
        score=score,
        # Never DDL from a shard — concurrent CREATE/ALTER against the same
        # catalog rows raises "tuple concurrently updated". The workflow's
        # prepare job bootstraps the schema once, single-threaded.
        ensure_schema=False,
    )


def run_scoring(batch_limit: int, shard: Optional[str] = None) -> int:
    """FinBERT over every unscored row, whatever collector wrote it.

    With `shard` ("i/N") this claims only MOD(id, N) = i-1, so shards can run
    concurrently without processing each other's headlines. Unsharded it takes
    the whole backlog, which is correct for the nightly job and hopeless for a
    backfill: measured ~2.7 rows/sec on a CPU runner, 315k rows is ~32 hours.
    """
    from collectors.gdelt_collector import score_pending_news
    total = 0
    while True:
        scored = score_pending_news(batch_limit, shard=shard)
        total += scored
        if scored < batch_limit:
            break
        logger.info("scored %d so far%s…", total, f" (shard {shard})" if shard else "")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Historical NSE/BSE announcement backfill")
    p.add_argument("--source", choices=["nse", "bse", "both"], default="both",
                   help="which exchange archive to fetch (default both)")
    p.add_argument("--from-date", type=str, default=None,
                   help="override start date YYYY-MM-DD (default per source: "
                        f"BSE {BSE_FROM}, NSE {NSE_FROM})")
    p.add_argument("--to-date", type=str, default=None,
                   help="override end date YYYY-MM-DD (default per source: "
                        f"BSE {BSE_TO}, NSE {NSE_TO})")
    p.add_argument("--symbol", type=str, default=None,
                   help="restrict to one symbol, e.g. HDFCBANK")
    p.add_argument("--shard", type=str, default=None, metavar="i/N",
                   help="process only shard i of N")
    p.add_argument("--fetch-only", action="store_true",
                   help="store rows unscored — required in parallel shards")
    p.add_argument("--score-only", action="store_true",
                   help="skip fetching, only run FinBERT over unscored rows")
    p.add_argument("--no-skip", action="store_true",
                   help="re-fetch symbols whose stored history already reaches "
                        "--from-date. REQUIRED to top up 2023+ NSE gaps: the skip "
                        "check only looks at a symbol's OLDEST stored row, so 60%% "
                        "of symbols already satisfy it from the 2018 backfill and "
                        "would be skipped. Interior gaps (2026-06/07 ran at half "
                        "the usual rate) are invisible to any min/max coverage "
                        "check, so a full re-fetch is the only way to fill them")
    p.add_argument("--score-batch", type=int, default=500,
                   help="headlines per FinBERT batch (default 500). Smaller than "
                        "the scheduler's 2000 on purpose: eight scoring shards "
                        "write concurrently, and a 2000-row UPDATE is both more "
                        "likely to exceed the 30s statement_timeout and more "
                        "expensive to lose when it does")
    p.add_argument("--score-shard", type=str, default=None, metavar="i/N",
                   help="score only rows where MOD(id, N) = i-1. Scoring shards "
                        "may run concurrently; fetch shards may not score at all")
    p.add_argument("--plan", choices=["nse", "bse"], default=None,
                   help="print count/shards/matrix as GitHub Actions output lines and exit")
    p.add_argument("--symbols-per-shard", type=int, default=None,
                   help="--plan: symbols per shard")
    args = p.parse_args()

    # Plan mode emits machine-readable output only — logging would corrupt
    # $GITHUB_OUTPUT.
    if args.plan:
        plan = shard_plan(args.plan, args.symbols_per_shard)
        print(f"count={plan['count']}")
        print(f"shards={plan['shards']}")
        print(f"matrix={json.dumps(plan['matrix'])}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.score_only:
        total = run_scoring(args.score_batch, shard=args.score_shard)
        logger.info("Scoring pass complete: %d headlines scored", total)
        return

    symbols = _universe()
    if args.symbol:
        symbols = [s for s in symbols if s.upper() == args.symbol.upper()]
        if not symbols:
            logger.error("Symbol %s is not in the active universe", args.symbol)
            sys.exit(1)
    symbols = _shard(symbols, args.shard)
    if not symbols:
        logger.info("Shard %s is empty — nothing to do", args.shard)
        return
    logger.info("%d symbol(s) in scope%s", len(symbols),
                f" (shard {args.shard})" if args.shard else "")

    skip_covered = not args.no_skip
    results = {}

    if args.source in ("bse", "both"):
        results["bse"] = run_bse(
            symbols,
            _parse_date(args.from_date or BSE_FROM),
            _parse_date(args.to_date or BSE_TO),
            skip_covered=skip_covered,
        )

    if args.source in ("nse", "both"):
        results["nse"] = run_nse(
            symbols,
            _parse_date(args.from_date or NSE_FROM),
            _parse_date(args.to_date or NSE_TO),
            skip_covered=skip_covered,
            score=False,          # scoring is always a separate pass
        )

    # Name the skipped symbols, don't just count them. A sharded run splits the
    # universe across 13 jobs, so a symbol silently dropped in shard 7 is
    # invisible unless its shard says which one it was.
    for src, res in results.items():
        unmapped = res.get("unmapped") or []
        empty    = res.get("empty") or []
        failed   = res.get("failed") or []
        logger.info("%s: %s rows over %s symbols "
                    "(unmapped %d, no-data %d, failed %d)",
                    src.upper(), res.get("rows"), res.get("processed"),
                    len(unmapped), len(empty), len(failed))
        if unmapped:
            # Expected for NSE-only listings (BSE Ltd itself, CDSL) — 2 of 500
            # at time of writing. A sudden jump here means the scrip list
            # changed shape, not that the universe did.
            logger.info("%s unmapped (no scrip code): %s",
                        src.upper(), ", ".join(unmapped))
        if failed:
            # These are real losses — a symbol that errored fetched nothing.
            logger.warning("%s failed (%d): %s",
                           src.upper(), len(failed), ", ".join(failed))

    if not args.fetch_only:
        total = run_scoring(args.score_batch, shard=args.score_shard)
        logger.info("Scoring pass complete: %d headlines scored", total)
    else:
        logger.info("Fetch-only — run --score-only once every shard has finished")

    # A shard that lost 40 symbols to DB errors used to exit 0 and show a green
    # check — the whole point of running this is the rows, so per-symbol
    # failures have to fail the job, not just the "could not reach the API at
    # all" case. Re-running is safe and cheap: covered symbols are skipped and
    # uq_news_url_pubdate makes any re-fetch a no-op.
    failed_symbols = sorted({s for r in results.values()
                               for s in (r.get("failed") or [])})
    if failed_symbols:
        logger.error("%d symbol(s) failed — re-run this shard to retry them: %s",
                     len(failed_symbols), ", ".join(failed_symbols))
    if failed_symbols or any(r.get("status") == "error" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
