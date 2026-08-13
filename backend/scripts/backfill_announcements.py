"""
TradeMind — Historical Corporate-Announcement Backfill (NSE + BSE)

One entry point for deepening news history behind the 2023-01-01 wall the
news_sentiment table currently starts at. Drives the two exchange collectors
and the single FinBERT scoring pass, and sizes the CI shard matrices.

WHY THESE TWO SOURCES
    GDELT cannot go deeper. Its DOC 2.0 API rejects any start date before
    2017 outright ("Invalid query start date" — a 2016-01-01 query fails while
    2017-01-02 returns articles), so the collector that produced most of our
    media rows has no earlier data to give.

    Announcements are also the better input for this model: exact timestamps,
    an explicit symbol on every row, and no headline-to-company matching to get
    wrong — unlike media archives, where the matching is where the accuracy
    goes.

WINDOWS — THE TWO SOURCES TILE, THEY DO NOT OVERLAP
    BSE  2010-01-01 .. 2017-12-31   archive reaches 2010
    NSE  2018-01-01 .. 2022-12-31   NSE's API bottoms out around 2018;
                                    2023-01-01 onward is already collected
    A company files the same event to both exchanges, so overlapping the
    windows would store every announcement twice and double its weight in the
    daily sentiment aggregate. The defaults tile deliberately. Override with
    --from-date/--to-date on a fresh database, where there is nothing above to
    collide with.

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
BSE_TO   = "2017-12-31"
NSE_FROM = "2018-01-01"
NSE_TO   = "2022-12-31"

# ── Shard sizing ──────────────────────────────────────────────────────────────
# BSE is the expensive one: eight calendar-year windows per symbol, each paged
# 50 rows at a time. Measured 24.7s/symbol over TCS/INFY/ITC/WIPRO (4,678 rows
# for 2010-2017), so all 500 is ~3.2h serially — past a comfortable job length.
# 60/shard puts a shard at ~25 min. NSE answers a whole multi-year range in one
# request per symbol (~4s including the politeness sleep), ~35 min for 500, so
# it splits four ways just to stay under an hour.
SYMBOLS_PER_BSE_SHARD = 60
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
    )


def run_scoring(batch_limit: int) -> int:
    """FinBERT over every unscored row, whatever collector wrote it."""
    from collectors.gdelt_collector import score_pending_news
    total = 0
    while True:
        scored = score_pending_news(batch_limit)
        total += scored
        if scored < batch_limit:
            break
        logger.info("scored %d so far…", total)
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
                   help="re-fetch symbols whose stored history already reaches --from-date")
    p.add_argument("--score-batch", type=int, default=2000,
                   help="headlines per FinBERT batch (default 2000)")
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
        total = run_scoring(args.score_batch)
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
        total = run_scoring(args.score_batch)
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
