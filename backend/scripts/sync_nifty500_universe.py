"""
Sync the tracked stock universe to the current NSE Nifty 500 constituent list.

Background
----------
The tracked universe had drifted ~7% from the live index: 35 current
constituents were completely absent (no Angel token, no prices, no model) while
33 removed names were still being collected daily and were still model-eligible
for signal generation. `nifty_constituents` also carried one wrong sector
(ABREL: "Forest Materials" -> "Realty"), two stale company names, and three
orphan rows with no token/price/model at all.

This script re-syncs every layer of the universe from the NSE CSV, in order:

    1. compare       what has to be ADDED and REMOVED, before touching anything
    2. tokens        NSE CSV  -> data/angel_tokens.json   (Angel scrip master)
    3. constituents  NSE CSV  -> nifty_constituents table (names + sectors)
    4. prices        Angel One -> prices table            (chunked backfill)
    5. indicators    prices    -> technical_indicators
    6. news          GDELT     -> news_sentiment + FinBERT scoring
    7. retire        archive dropped names AND delete them from the HF store
    8. report        final coverage + add/remove reconciliation

Steps are independent and re-runnable; use --steps to run a subset.

It deliberately STOPS BEFORE MODEL TRAINING. Everything here prepares the
inputs a model needs — tokens, prices, indicators, news/sentiment — and the
weekly retrain (.github/workflows/weekly-retrain.yml) is what trains them.
That workflow enumerates `SELECT DISTINCT symbol FROM prices`, so any symbol
this script backfills is picked up on the next Friday run with no extra
wiring, and model_training already skips symbols with < 100 training rows.
Training here as well would just fit the same models twice, on the same data,
by two different code paths.

Angel One quirks this script works around
-----------------------------------------
* **2000-day cap on ONE_DAY candles.** A request for 2010-01-01 -> today does
  NOT error - it silently returns only the last ~2000 calendar days. Measured:
  five symbols all reported "first candle 2021-02-08", which is exactly
  today - 2000d, not their listing date. History is therefore fetched in
  CHUNK_DAYS-sized windows walking backwards.
* **Rate limit is far tighter than the documented 3 req/s.** 0.35s between
  historical calls gets "Access denied because of exceeding access rate"
  almost immediately, and 2.5s still trips AB1021 "Too many requests"
  intermittently. Default here is 3.0s with exponential backoff, and a
  rate-limited response is retried rather than being recorded as "no data"
  (that false negative is exactly what an earlier probe hit on CANHLIFE).
* **Corporate actions are not reliably pre-adjusted.** Angel patches historical
  candles for splits/bonuses only when their data team gets to it, so some
  symbols are adjusted and some are not. Nothing is adjusted at ingest time -
  model_training.apply_corporate_action_adjustments() detects the ex-date price
  ratio and skips symbols Angel already fixed, so adjustment stays in one place.

News backfill
-------------
New constituents have no news history, so their sentiment features are empty
until GDELT is backfilled. bootstrap_gdelt(only_missing=True) targets exactly
the symbols with no news rows, and score_pending_news() then runs FinBERT over
whatever is unscored. GDELT is rate-limited to one request per 12s per
(symbol, month), so this is the slowest step by far — budget hours, not
minutes, and tune the window with --news-from-year.

Note historical_news_collector.backfill_stock_sentiment() is NOT used: it
calls conn.execute() with INSERT OR REPLACE (SQLite syntax, and psycopg2
connections have no .execute()) and targets news_daily_sentiment, which is a
continuous aggregate and cannot be inserted into. The GDELT collector writes
through insert_news() into the news_sentiment hypertable, and the aggregate
refreshes itself hourly.

Usage
-----
    cd backend && source venv/bin/activate

    python scripts/sync_nifty500_universe.py --dry-run          # plan only
    python scripts/sync_nifty500_universe.py --steps compare    # just the diff
    python scripts/sync_nifty500_universe.py                    # all steps
    python scripts/sync_nifty500_universe.py --steps prices,indicators
    python scripts/sync_nifty500_universe.py --steps news --news-from-year 2023

Writes to the DB in .env (prod) unless APP_ENV=test is set.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyotp  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from SmartApi import SmartConnect  # noqa: E402

from database.db import (  # noqa: E402
    _execute,
    get_connection,
    insert_prices_batch,
    release_connection,
    upsert_nifty_constituents,
)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# ── Config ────────────────────────────────────────────────────────────────────

NSE_CSV = os.path.join(os.path.dirname(_BACKEND_DIR), "ind_nifty500list.csv")
TOKENS_FILE = os.path.join(_BACKEND_DIR, "data", "angel_tokens.json")
FINAL_DIR = os.path.join(_BACKEND_DIR, "final_models")
RETIRED_DIR = os.path.join(_BACKEND_DIR, "model_archives", "removed_from_index")
LOG_DIR = os.path.join(_BACKEND_DIR, "logs")

SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

# Angel One truncates ONE_DAY requests at ~2000 calendar days; stay under it.
CHUNK_DAYS = 1900
HISTORY_START = "2010-01-01"
RATE_LIMIT_SECS = 3.0          # historical API; see module docstring
MAX_RETRIES = 5
BACKOFF_BASE_SECS = 10

ALL_STEPS = ["compare", "tokens", "constituents", "prices", "indicators",
             "news", "retire", "report"]

# GDELT: 1 request per (symbol, month). Two years keeps a full-universe news
# backfill inside a CI job's 6h cap; older news barely moves a sentiment feature.
NEWS_FROM_YEAR_DEFAULT = date.today().year - 2

# Shard sizing for --news-plan. At ~31 months x 12s a symbol costs ~6 min, so 6
# symbols is ~37 min of GDELT per shard — well inside a job's 6h cap with room
# for a much deeper --news-from-year. MAX_NEWS_SHARDS keeps a big index review
# from fanning out to hundreds of runners (GitHub caps a matrix at 256 anyway).
SYMBOLS_PER_NEWS_SHARD = 6
MAX_NEWS_SHARDS = 20

# Price backfill needs the same treatment. Measured on the 2026-07 intake:
# 35 symbols took 26 minutes, i.e. ~45s each (up to 4 chunked requests at
# RATE_LIMIT_SECS apart, plus retries, plus inserting ~4k rows). A first run on
# an empty DB would be all 500 symbols -> ~6.3h, past a GitHub job's 6h cap. At
# 60 per shard a shard is ~45 min and 500 symbols fan out to 9 shards.
SYMBOLS_PER_PRICE_SHARD = 60
MAX_PRICE_SHARDS = 12

# Indicators are sharded too, even though a symbol is nominally ~2.5s. Cost is
# NOT uniform: a symbol missing 15 dates and one missing its entire 4108-date
# history both require computing every indicator across the full price series
# (they are all lookbacks — sma_200 alone needs 200 prior bars), and the second
# then writes 4108 rows instead of 15. The first repair pass is the expensive
# one: all 500 symbols need work, ~49.6k rows in total.
#
# 50 per shard keeps a shard in the low minutes even if it draws several
# full-history symbols, so no shard is anywhere near the 6h cap and one slow
# shard cannot hold up the rest. Steady state is far cheaper — symbols whose
# dates all match are skipped without computing anything.
SYMBOLS_PER_INDICATOR_SHARD = 50
MAX_INDICATOR_SHARDS = 12

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"sync_universe_{datetime.now():%Y-%m-%d_%H%M%S}.log")

logger = logging.getLogger("sync_universe")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)
_fh = logging.FileHandler(LOG_PATH)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
logger.addHandler(_fh)
logger.propagate = False


def step_banner(n: int, name: str, detail: str = "") -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"STEP {n}/{len(ALL_STEPS)} — {name.upper()}{('  ·  ' + detail) if detail else ''}")
    logger.info("=" * 78)


# ── Shared helpers ────────────────────────────────────────────────────────────

def read_nse_csv() -> List[Dict]:
    """The NSE Nifty 500 CSV as [{symbol, name, sector, isin, series}] (.NS-suffixed)."""
    with open(NSE_CSV) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        sym = r["Symbol"].strip()
        out.append({
            "symbol": f"{sym}.NS",
            "base": sym,
            "name": r["Company Name"].strip(),
            "sector": r["Industry"].strip(),
            "series": r["Series"].strip(),
            "isin": r["ISIN Code"].strip(),
        })
    return out


def load_tokens() -> Dict:
    with open(TOKENS_FILE) as f:
        return json.load(f)


def angel_login() -> SmartConnect:
    api_key = os.getenv("ANGEL_API_KEY", "")
    client_id = os.getenv("ANGEL_CLIENT_ID", "")
    mpin = os.getenv("ANGEL_MPIN", "") or os.getenv("ANGEL_PASSWORD", "")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
    if not all([api_key, client_id, mpin, totp_secret]):
        logger.error("Angel One credentials missing in .env")
        sys.exit(1)

    api = SmartConnect(api_key=api_key)
    data = api.generateSession(client_id, mpin, pyotp.TOTP(totp_secret).now())
    if not data.get("status"):
        logger.error(f"Angel One login failed: {data.get('message')}")
        sys.exit(1)
    logger.info(f"Angel One session established (client {client_id})")
    return api


def angel_logout(api: SmartConnect) -> None:
    try:
        api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
    except Exception:
        pass


def _is_rate_limited(err: Exception) -> bool:
    msg = str(err).lower()
    return "exceeding access rate" in msg or "too many requests" in msg or "ab1021" in msg


def fetch_candles_chunk(api: SmartConnect, token: str, frm: date, to: date) -> List[list]:
    """One ONE_DAY candle request with backoff. Rate-limit errors are retried,
    never swallowed into an empty result (that would look like 'stock has no
    history' and silently drop a valid symbol)."""
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": f"{frm:%Y-%m-%d} 09:15",
        "todate": f"{to:%Y-%m-%d} 15:30",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = api.getCandleData(params)
            if not resp.get("status"):
                msg = str(resp.get("message", ""))
                if "rate" in msg.lower() or "too many" in msg.lower():
                    raise RuntimeError(msg)
                return []
            return resp.get("data") or []
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE_SECS * (attempt + 1) if _is_rate_limited(e) else 2
            logger.warning(f"    retry {attempt + 1}/{MAX_RETRIES - 1} in {wait}s ({str(e)[:70]})")
            time.sleep(wait)
    return []


def fetch_full_history(api: SmartConnect, base: str, token: str,
                       start: date, end: date) -> List[tuple]:
    """Walk backwards in CHUNK_DAYS windows until the listing date is reached.

    Angel's 2000-day cap makes a single wide request silently lossy, so the
    range is split. Once a chunk comes back starting later than the window it
    was asked for, that chunk contains the listing date and there is nothing
    older to fetch - stop rather than spending requests on empty windows.
    """
    symbol_ns = f"{base}.NS"
    seen: Dict[str, tuple] = {}
    win_end = end

    while win_end >= start:
        win_start = max(start, win_end - timedelta(days=CHUNK_DAYS))
        candles = fetch_candles_chunk(api, token, win_start, win_end)
        time.sleep(RATE_LIMIT_SECS)

        if not candles:
            logger.info(f"    {win_start} -> {win_end}: no candles (pre-listing) — stopping")
            break

        for ts, o, h, l, c, v in candles:
            d = ts[:10]
            seen[d] = (symbol_ns, "NSE", d, None,
                       round(float(o), 2), round(float(h), 2),
                       round(float(l), 2), round(float(c), 2),
                       int(v), "1d")

        first_seen = min(candles, key=lambda x: x[0])[0][:10]
        logger.info(f"    {win_start} -> {win_end}: {len(candles):5} candles (first {first_seen})")

        # Listing date is inside this window -> no older data exists.
        if first_seen > (win_start + timedelta(days=5)).isoformat():
            logger.info(f"    listing date reached ({first_seen}) — stopping")
            break

        win_end = win_start - timedelta(days=1)

    return [seen[d] for d in sorted(seen)]


def db_symbols_with_prices() -> set:
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d'")
        return {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)


def db_row_count_before(symbol: str, cutoff: str) -> int:
    conn = get_connection()
    try:
        cur = _execute(conn,
                       "SELECT COUNT(*) FROM prices WHERE symbol=? AND interval='1d' AND date <= ?",
                       (symbol, cutoff))
        return cur.fetchone()[0]
    finally:
        release_connection(conn)


def deactivate_signals(symbols: List[str], dry_run: bool = False) -> int:
    """Mark every live trade_signals row for these symbols is_active = FALSE.

    Retiring a model does NOT retire its signals on its own, and the two have to
    happen together. insert_trade_signals_batch() only deactivates symbols that
    appear in the batch it is handed, and generate_trades.py builds that batch by
    enumerating final_models/*.pkl — so archiving a de-indexed model is precisely
    what strands its last signals at is_active = TRUE permanently. The symbol can
    never appear in a future batch, so nothing ever clears the flag, and the API
    keeps serving a stock that left the index.

    Matches both stored forms: trade_signals holds 'SYMBOL.NS' while the pkl
    filenames are bare 'SYMBOL'.
    """
    if not symbols:
        return 0

    pats = list(symbols) + [f"{s}.NS" for s in symbols]
    placeholders = ",".join(["?"] * len(pats))
    conn = get_connection()
    try:
        if dry_run:
            cur = _execute(conn,
                           f"SELECT COUNT(*) FROM trade_signals "
                           f"WHERE is_active = TRUE AND symbol IN ({placeholders})",
                           tuple(pats))
            return cur.fetchone()[0]

        cur = _execute(conn,
                       f"UPDATE trade_signals SET is_active = FALSE "
                       f"WHERE is_active = TRUE AND symbol IN ({placeholders})",
                       tuple(pats))
        conn.commit()
        return cur.rowcount
    finally:
        release_connection(conn)


# ── Step 1: compare ───────────────────────────────────────────────────────────

def _norm(sym: str) -> str:
    """Canonical '.NS' form. final_models holds BOTH conventions on disk
    ({SYM}_final.pkl and {SYM}.NS_final.pkl — see the dedup note in
    generate_trades.py), so comparing raw filenames reports every bare-named
    model as a stranger that must be removed."""
    return sym if sym.endswith(".NS") else f"{sym}.NS"


def _is_stock(sym: str) -> bool:
    """False for the non-stock rows that legitimately live in these tables:
    index symbols (^NSEI, ^BSESN, ^INDIAVIX, ^CRSLDX) in prices, and the
    market-wide news buckets (MARKET:RBI, MARKET:SEBI, ...) in news_sentiment.
    Treating them as constituents would flag them for removal forever."""
    return not sym.startswith("^") and not sym.startswith("MARKET:")


def _universe_layers() -> List[Tuple[str, set, bool]]:
    """(layer name, symbols it holds, whether extras are actually removed).

    `enforced=False` layers keep history for de-indexed names on purpose —
    prices/indicators/news cost little and stay useful for backtests, so their
    "to REMOVE" count is informational, not a to-do list. Only the constituent
    list, the token map and the model set are trimmed to the index.
    """
    tokens = {_norm(k) for k in load_tokens()} if os.path.exists(TOKENS_FILE) else set()
    models = {_norm(f.replace("_final.pkl", "")) for f in os.listdir(FINAL_DIR)
              if f.endswith("_final.pkl")} if os.path.isdir(FINAL_DIR) else set()

    conn = get_connection()
    try:
        # Active only: deactivated rows are the audit trail of what left the
        # index, and counting them here would report every past removal as a
        # symbol still needing removal.
        cur = _execute(conn, "SELECT symbol FROM nifty_constituents WHERE is_active = TRUE")
        consts = {r[0] for r in cur.fetchall()}
        cur = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d'")
        px = {r[0] for r in cur.fetchall()}
        cur = _execute(conn, "SELECT DISTINCT symbol FROM technical_indicators")
        inds = {r[0] for r in cur.fetchall()}
        cur = _execute(conn, "SELECT DISTINCT symbol FROM news_sentiment WHERE symbol IS NOT NULL")
        news = {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)

    return [
        ("nifty_constituents", consts, True),
        ("angel_tokens.json", tokens, True),
        ("final_models", models, True),
        ("prices", px, False),
        ("technical_indicators", inds, False),
        ("news_sentiment", news, False),
    ]


def step_compare(nse: List[Dict], label: str = "before") -> Dict:
    """What has to be ADDED and what has to be REMOVED, per layer.

    Run before any writes so the plan is on record, and again at the end so the
    two can be diffed - a clean second pass is the proof the sync worked.
    """
    n = 1 if label == "before" else len(ALL_STEPS)
    step_banner(n, "compare", f"NSE list vs tracked universe ({label})")

    csv_syms = {s["symbol"] for s in nse}
    layers = _universe_layers()

    logger.info(f"NSE Nifty 500 list: {len(csv_syms)} symbols")
    logger.info("")
    logger.info(f"  {'layer':<22} {'have':>6} {'to ADD':>8} {'to REMOVE':>10}   {'action'}")
    logger.info(f"  {'-' * 64}")

    out = {}
    for name, have, enforced in layers:
        have_stocks = {_norm(s) for s in have if _is_stock(s)}
        to_add = sorted(csv_syms - have_stocks)
        to_remove = sorted(have_stocks - csv_syms)
        out[name] = {"to_add": to_add, "to_remove": to_remove,
                     "have": len(have_stocks), "enforced": enforced}
        note = "trimmed to index" if enforced else "history kept"
        logger.info(f"  {name:<22} {len(have_stocks):>6} {len(to_add):>8} "
                    f"{len(to_remove):>10}   {note}")

    for name, d in out.items():
        if d["to_add"]:
            logger.info("")
            logger.info(f"  {name} — ADD ({len(d['to_add'])}): "
                        f"{[s.replace('.NS', '') for s in d['to_add']]}")
        if d["to_remove"] and d["enforced"]:
            logger.info(f"  {name} — REMOVE ({len(d['to_remove'])}): "
                        f"{[s.replace('.NS', '') for s in d['to_remove']]}")

    # Only enforced layers count toward "in sync" — de-indexed price/indicator/
    # news history is retained on purpose and would otherwise never reach zero.
    enf = [d for d in out.values() if d["enforced"]]
    total_add = len(set().union(*[set(d["to_add"]) for d in enf]) or set())
    total_remove = len(set().union(*[set(d["to_remove"]) for d in enf]) or set())
    logger.info("")
    logger.info(f"  enforced layers — to add: {total_add}  ·  to remove: {total_remove}")
    if label == "after" and total_add == 0 and total_remove == 0:
        logger.info("  universe is fully in sync")
    return out


# ── Step 2: tokens ────────────────────────────────────────────────────────────

def step_tokens(nse: List[Dict], dry_run: bool) -> Dict:
    """Resolve every CSV symbol to an Angel instrument token, rewrite angel_tokens.json."""
    step_banner(2, "tokens", "NSE CSV -> data/angel_tokens.json")

    logger.info("Downloading Angel One scrip master...")
    resp = requests.get(SCRIP_MASTER_URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    instruments = resp.json()
    nse_eq = {
        i["symbol"][:-3]: i
        for i in instruments
        if i.get("exch_seg") == "NSE" and i.get("symbol", "").endswith("-EQ")
    }
    logger.info(f"Scrip master: {len(instruments):,} instruments, {len(nse_eq):,} NSE-EQ")

    old = load_tokens() if os.path.exists(TOKENS_FILE) else {}
    token_map, missing = {}, []
    for s in nse:
        inst = nse_eq.get(s["base"])
        if not inst:
            missing.append(s["base"])
            continue
        token_map[s["base"]] = {
            "token": inst["token"],
            "trading_symbol": inst["symbol"],
            "name": s["name"],
            "sector": s["sector"],
            "isin": s["isin"],
        }

    added = sorted(set(token_map) - set(old))
    removed = sorted(set(old) - set(token_map))
    changed = sorted(k for k in set(token_map) & set(old)
                     if old[k].get("token") != token_map[k]["token"])

    logger.info(f"Resolved {len(token_map)}/{len(nse)} symbols  ·  unresolved: {len(missing)}")
    if missing:
        logger.warning(f"  UNRESOLVED: {missing}")
    logger.info(f"  + added   : {len(added)}  {added if added else ''}")
    logger.info(f"  - removed : {len(removed)}  {removed if removed else ''}")
    logger.info(f"  ~ retoken : {len(changed)}  {changed if changed else ''}")

    if dry_run:
        logger.info("DRY-RUN: angel_tokens.json not written")
    else:
        if os.path.exists(TOKENS_FILE):
            backup = TOKENS_FILE.replace(".json", f".bak-{datetime.now():%Y%m%d_%H%M%S}.json")
            shutil.copy2(TOKENS_FILE, backup)
            logger.info(f"Backed up previous token map -> {os.path.basename(backup)}")
        with open(TOKENS_FILE, "w") as f:
            json.dump(token_map, f, indent=2)
        logger.info(f"Wrote {len(token_map)} tokens -> data/angel_tokens.json")

    return {"resolved": len(token_map), "missing": missing, "added": added, "removed": removed}


# ── Step 3: constituents ──────────────────────────────────────────────────────

def step_constituents(nse: List[Dict], dry_run: bool) -> Dict:
    """Make nifty_constituents exactly match the NSE list (fixes sectors/names, drops orphans)."""
    step_banner(3, "constituents", "NSE CSV -> nifty_constituents")

    conn = get_connection()
    try:
        # Only active rows count as "in the table" — a previously deactivated
        # symbol must read as absent so it is re-added (and reactivated by the
        # upsert) if NSE puts it back, and so to_drop never re-lists names that
        # were already deactivated by an earlier run.
        cur = _execute(conn, "SELECT symbol, name, sector FROM nifty_constituents "
                             "WHERE is_active = TRUE")
        existing = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    finally:
        release_connection(conn)

    csv_map = {s["symbol"]: s for s in nse}
    to_add = sorted(set(csv_map) - set(existing))
    to_drop = sorted(set(existing) - set(csv_map))
    sector_fix = [(s, existing[s][1], csv_map[s]["sector"])
                  for s in sorted(set(csv_map) & set(existing))
                  if (existing[s][1] or "").strip() != csv_map[s]["sector"]]
    name_fix = [(s, existing[s][0], csv_map[s]["name"])
                for s in sorted(set(csv_map) & set(existing))
                if (existing[s][0] or "").strip() != csv_map[s]["name"]]

    logger.info(f"Table has {len(existing)} rows · NSE list has {len(csv_map)}")
    logger.info(f"  + insert     : {len(to_add)}")
    logger.info(f"  - delete     : {len(to_drop)}  {to_drop if to_drop else ''}")
    logger.info(f"  ~ sector fix : {len(sector_fix)}")
    for s, o, n in sector_fix:
        logger.info(f"      {s:16} {o!r} -> {n!r}")
    logger.info(f"  ~ name fix   : {len(name_fix)}")
    for s, o, n in name_fix:
        logger.info(f"      {s:16} {o!r} -> {n!r}")

    if dry_run:
        logger.info("DRY-RUN: no DB writes")
        return {"added": to_add, "dropped": to_drop, "sector_fix": sector_fix, "name_fix": name_fix}

    written = upsert_nifty_constituents(
        [{"symbol": s["symbol"], "name": s["name"], "sector": s["sector"]} for s in nse]
    )
    logger.info(f"Upserted {written} constituents")

    if to_drop:
        conn = get_connection()
        try:
            for s in to_drop:
                _execute(conn,
                         "UPDATE nifty_constituents SET is_active = FALSE, removed_at = NOW() "
                         "WHERE symbol = ? AND is_active = TRUE",
                         (s,))
            conn.commit()
        finally:
            release_connection(conn)
        logger.info(f"Deactivated {len(to_drop)} rows no longer in the index")

    # upsert_nifty_constituents clears the sector cache, but the deactivation
    # above happens after it - clear again so a live process can't serve
    # dropped names.
    try:
        from database import db as _db
        _db._sector_map_cache.clear()
    except Exception:
        pass

    conn = get_connection()
    try:
        cur = _execute(conn,
                       "SELECT COUNT(*) FILTER (WHERE is_active), COUNT(*) "
                       "FROM nifty_constituents")
        active, total = cur.fetchone()
        logger.info(f"nifty_constituents now has {active} active "
                    f"({total - active} deactivated, {total} rows total)")
    finally:
        release_connection(conn)

    return {"added": to_add, "dropped": to_drop, "sector_fix": sector_fix, "name_fix": name_fix}


# ── Step 4: prices ────────────────────────────────────────────────────────────

def step_prices(nse: List[Dict], dry_run: bool, only: Optional[List[str]] = None,
                shard: Optional[Tuple[int, int]] = None) -> Dict:
    """Backfill full available history for constituents with no price data.

    Shardable: at ~45s a symbol (chunked requests + rate limit + retries + a
    ~4k-row insert), a first run against an empty DB is ~500 symbols and would
    run past a CI job's 6h cap in one process. Each symbol is independent and
    writes straight to the DB, so round-robin splitting needs no merge step.
    """
    tag = f" shard {shard[0]}/{shard[1]}" if shard else ""
    step_banner(4, "prices", f"Angel One -> prices{tag}")

    tokens = load_tokens()
    have = db_symbols_with_prices()
    targets = [s for s in nse if s["symbol"] not in have]
    if only:
        targets = [s for s in targets if s["base"] in only or s["symbol"] in only]
    if shard:
        idx, total = shard
        targets = targets[idx - 1::total]

    logger.info(f"Constituents with price data : {len(nse) - len([s for s in nse if s['symbol'] not in have])}")
    logger.info(f"Constituents to backfill     : {len(targets)}")
    if not targets:
        logger.info("Nothing to backfill")
        return {"backfilled": {}, "failed": []}
    logger.info(f"  {[t['base'] for t in targets]}")

    if dry_run:
        logger.info("DRY-RUN: no Angel calls, no DB writes")
        return {"backfilled": {}, "failed": []}

    start = datetime.strptime(HISTORY_START, "%Y-%m-%d").date()
    end = date.today()
    api = angel_login()
    results, failed = {}, []

    try:
        for idx, s in enumerate(targets, 1):
            base = s["base"]
            info = tokens.get(base)
            if not info:
                logger.error(f"[{idx}/{len(targets)}] {base}: no token — skipped (run --steps tokens)")
                failed.append(base)
                continue

            logger.info(f"[{idx}/{len(targets)}] {base} (token {info['token']})")
            try:
                rows = fetch_full_history(api, base, info["token"], start, end)
            except Exception as e:
                logger.error(f"    FAILED: {str(e)[:120]}")
                failed.append(base)
                continue

            if not rows:
                logger.warning(f"    no candles returned — nothing inserted")
                results[base] = 0
                continue

            inserted = insert_prices_batch(rows, sync=False)
            results[base] = inserted
            logger.info(f"    inserted {inserted} rows  ({rows[0][2]} -> {rows[-1][2]})")
    finally:
        angel_logout(api)

    total = sum(results.values())
    logger.info(f"Backfill complete: {len(results)} symbols, {total:,} rows, {len(failed)} failed")
    if failed:
        logger.warning(f"  FAILED: {failed}")
    return {"backfilled": results, "failed": failed}


# ── Step 5: indicators ────────────────────────────────────────────────────────


def step_indicators(nse: List[Dict], dry_run: bool, only: Optional[List[str]] = None,
                    shard: Optional[Tuple[int, int]] = None) -> Dict:
    """Compute technical indicators for constituents whose coverage is short.

    Cheap next to the price backfill (~2.5s a symbol, so ~21 min for a full
    500), but it takes the same shard so it can run in the same job as the
    prices slice it depends on.

    Selection is by COVERAGE, not presence. It used to be
    `nse - (SELECT DISTINCT symbol FROM technical_indicators)`, which asked only
    "does this symbol have any indicator row at all". A newly added constituent
    picks up exactly ONE row on its first EOD, from calculate_indicators_job
    computing the latest bar — so it satisfied that test forever while having no
    history at all. 19 new constituents sat at 1 row against 151-4109 price rows,
    and every re-run reported "nothing to compute". Unlike sentiment, which
    prefetch_all_data fills with 0.0 when absent, missing indicators arrive as
    NaN across real features (RSI, MACD, Bollinger, SMA, ATR, ADX, Stoch, OBV),
    so this silently poisoned training for those names.
    """
    tag = f" shard {shard[0]}/{shard[1]}" if shard else ""
    step_banner(5, "indicators", f"prices -> technical_indicators{tag}")

    from collectors.backfill_indicators_historical import find_missing_indicator_dates

    # Partition the CSV list BEFORE asking which symbols have gaps. Shards run
    # concurrently and close gaps as they go, so "symbols with gaps" shrinks
    # while they run — striding that list would give each shard a different view
    # of it and let symbols fall between the strides entirely. The constituent
    # list is fixed for the run, so slicing it first gives every shard a stable,
    # disjoint slice it exclusively owns. It also scopes the gap query to the
    # shard's own symbols.
    csv_syms = [s["symbol"] for s in nse]
    if only:
        csv_syms = [t for t in csv_syms if t in only or t.replace(".NS", "") in only]
    if shard:
        idx, total = shard
        csv_syms = csv_syms[idx - 1::total]

    gaps = find_missing_indicator_dates(csv_syms)
    targets = [s for s in csv_syms if s in gaps]

    for t in targets[:20]:
        d = gaps[t]
        logger.info(f"      {t:16} {len(d):>5} missing date(s)  {d[0]} .. {d[-1]}")
    if len(targets) > 20:
        logger.info(f"      ... and {len(targets) - 20} more")

    logger.info(f"Constituents with missing indicator dates: {len(targets)} "
                f"({sum(len(gaps[t]) for t in targets)} dates)")
    if not targets:
        logger.info("Nothing to compute")
        return {"ok": [], "failed": []}
    logger.info(f"  {[t.replace('.NS', '') for t in targets]}")

    if dry_run:
        logger.info("DRY-RUN: no indicator computation")
        return {"ok": [], "failed": []}

    # backfill_symbol, NOT process_stock. process_stock computes indicators over
    # 400 days and then stores only df.iloc[-1] — one row, today's, which
    # already exists. It is the reason these gaps accumulate in the first place,
    # so calling it here detected 5257 missing dates and then wrote nothing,
    # reporting "50 ok" in 3 seconds (run 30807471826).
    from collectors.backfill_indicators_historical import backfill_symbol

    ok, failed = [], []
    rows_written = 0
    for idx, sym in enumerate(targets, 1):
        want = {d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                for d in gaps[sym]}
        try:
            n = backfill_symbol(sym, only_dates=want)
            if n:
                ok.append(sym)
                rows_written += n
                logger.info(f"[{idx}/{len(targets)}] {sym:18} filled {n}/{len(want)} date(s)")
            else:
                failed.append(sym)
                # backfill_symbol swallows its own exceptions and returns 0, so
                # the reason is already logged above and is NOT inferable here.
                # This used to guess "(likely < 14 price bars)", which sent
                # AMBER.NS and BANKBARODA.NS's real cause — "out of memory" from
                # Timescale — off in entirely the wrong direction.
                logger.warning(f"[{idx}/{len(targets)}] {sym:18} wrote 0 rows "
                               f"— see the logged error above for the cause")
        except Exception as e:
            failed.append(sym)
            logger.error(f"[{idx}/{len(targets)}] {sym:18} FAILED: {str(e)[:100]}")

    logger.info(f"Indicators backfilled: {len(ok)} ok, {len(failed)} failed, "
                f"{rows_written} row(s) written")
    return {"ok": ok, "failed": failed}


# ── News shard planning ───────────────────────────────────────────────────────

def shard_plan(nse: List[Dict], what: str, per_shard: Optional[int] = None) -> Dict:
    """How many shards a step needs, given how much work it actually has.

    Printed as GitHub Actions output lines so a matrix can size itself. Both
    sharded steps scale with the intake, which is only known at run time: a
    3-stock review should not spin up a fixed 6 runners, and a first run against
    an empty DB (500 symbols) must not be crammed into one job that then blows
    the 6h cap. Emits an empty matrix when there is no work, so the job skips.
    """
    csv_syms = {s["symbol"] for s in nse}
    conn = get_connection()
    try:
        if what == "news":
            cur = _execute(conn,
                           "SELECT DISTINCT symbol FROM news_sentiment WHERE symbol IS NOT NULL")
            cap, default_per = MAX_NEWS_SHARDS, SYMBOLS_PER_NEWS_SHARD
        elif what == "indicators":
            # "have" = every price date already has an indicator row. Same
            # date-level predicate step_indicators uses, so the plan can never
            # disagree with what the step then does.
            cur = _execute(conn, """
                SELECT DISTINCT p.symbol FROM prices p
                WHERE p.interval = '1d'
                  AND NOT EXISTS (
                        SELECT 1 FROM prices p2
                        LEFT JOIN technical_indicators t
                               ON t.symbol = p2.symbol AND t.date = p2.date
                         WHERE p2.symbol = p.symbol AND p2.interval = '1d'
                           AND t.date IS NULL)
            """)
            cap, default_per = MAX_INDICATOR_SHARDS, SYMBOLS_PER_INDICATOR_SHARD
        else:
            cur = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d'")
            cap, default_per = MAX_PRICE_SHARDS, SYMBOLS_PER_PRICE_SHARD
        have = {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)

    per_shard = per_shard or default_per
    missing = sorted(csv_syms - have)
    n = len(missing)
    shards = min(cap, max(1, -(-n // per_shard))) if n else 0
    return {
        "count": n,
        "shards": shards,
        "matrix": list(range(1, shards + 1)),
        "symbols": [s.replace(".NS", "") for s in missing],
    }


# ── Step 6: news ──────────────────────────────────────────────────────────────

def step_news(nse: List[Dict], dry_run: bool, from_year: int,
              score_only: bool = False, fetch_only: bool = False,
              shard: Optional[Tuple[int, int]] = None) -> Dict:
    """Backfill GDELT news for constituents with none, then score it with FinBERT.

    Sentiment is one of the 96 model features, so a symbol with no news trains
    on a hole. Targets exactly the symbols that have no news rows yet.

    Slow by construction: GDELT is throttled to one request per 12s per
    (symbol, month), so cost is roughly symbols x months x 12s. Two knobs bound
    it - --news-from-year narrows the window, and --shard i/N splits the symbol
    list across parallel jobs.

    Sharding is safe because every article goes straight into the DB through
    insert_news() - there are no per-shard artifacts to merge afterwards, unlike
    the model retrain where each shard produces .pkl files. Shards do NOT score,
    though: score_pending_news() claims globally-unscored rows, so concurrent
    shards would score the same headlines repeatedly. Fetch in parallel
    (--news-fetch-only), score once at the end (--news-score-only).
    """
    tag = f" shard {shard[0]}/{shard[1]}" if shard else ""
    step_banner(6, "news", f"GDELT from {from_year} -> news_sentiment + FinBERT{tag}")

    conn = get_connection()
    try:
        cur = _execute(conn,
                       "SELECT DISTINCT symbol FROM news_sentiment WHERE symbol IS NOT NULL")
        have_news = {r[0] for r in cur.fetchall()}
        cur = _execute(conn, "SELECT COUNT(*) FROM news_sentiment WHERE sentiment IS NULL")
        unscored = cur.fetchone()[0]
    finally:
        release_connection(conn)

    csv_syms = {s["symbol"] for s in nse}
    missing = sorted(csv_syms - have_news)

    if shard:
        idx, total = shard
        # Round-robin so shards stay balanced however long the list is.
        missing = missing[idx - 1::total]

    months = max(1, (date.today().year - from_year) * 12 + date.today().month)
    est_hours = len(missing) * months * 12 / 3600

    logger.info(f"Constituents with news : {len(have_news & csv_syms)}")
    logger.info(f"Constituents missing   : {len(missing)}{tag}")
    logger.info(f"Unscored headlines     : {unscored:,}")
    if missing:
        logger.info(f"  {[s.replace('.NS', '') for s in missing]}")
        logger.info(f"  estimated GDELT time: ~{est_hours:.1f}h "
                    f"({len(missing)} symbols x ~{months} months x 12s)")

    if dry_run:
        logger.info("DRY-RUN: no GDELT fetch, no scoring")
        return {"collected": [], "failed": [], "scored": 0}

    collected, failed = [], []
    if missing and not score_only:
        from collectors.gdelt_collector import GdeltUnavailable, bootstrap_gdelt
        # Per-symbol rather than only_missing=True so one symbol's failure
        # cannot abort the rest of the shard, and progress is logged per symbol.
        for i, sym in enumerate(missing, 1):
            base = sym.replace(".NS", "")
            logger.info(f"[{i}/{len(missing)}] GDELT {base} from {from_year}-01 ...")
            try:
                bootstrap_gdelt(from_year=from_year, from_month=1, only_symbol=base)
                collected.append(sym)
            except GdeltUnavailable:
                # Deliberately NOT swallowed by the per-symbol handler below.
                # That handler exists so one bad symbol doesn't sink the shard,
                # but a wholesale 429 block is not per-symbol — every remaining
                # symbol would fail the same way, ~6 min per month apiece. Abort
                # the step so the job fails in minutes instead of hours.
                logger.error(f"    ABORTING: GDELT is refusing traffic "
                             f"(stopped at {base}, {i}/{len(missing)})")
                raise
            except Exception as e:
                failed.append(sym)
                logger.error(f"    {base} FAILED: {str(e)[:120]}")
    elif score_only:
        logger.info("--news-score-only: skipping GDELT fetch")

    total_scored = 0
    if fetch_only:
        logger.info("--news-fetch-only: leaving scoring to the finalize pass")
    else:
        from collectors.gdelt_collector import score_pending_news
        while True:
            try:
                n = score_pending_news(batch_limit=2000)
            except Exception as e:
                logger.error(f"Scoring failed: {str(e)[:150]}")
                break
            if not n:
                break
            total_scored += n
            logger.info(f"  scored {n} headlines (running total {total_scored:,})")

    logger.info(f"News step complete: {len(collected)} symbols fetched, "
                f"{len(failed)} failed, {total_scored:,} headlines scored")
    if failed:
        logger.warning(f"  FAILED: {[s.replace('.NS', '') for s in failed]}")
    return {"collected": collected, "failed": failed, "scored": total_scored}


# ── Step 7: retire ────────────────────────────────────────────────────────────

def step_retire(nse: List[Dict], dry_run: bool, push_remote: bool = True) -> Dict:
    """Archive models for symbols no longer in the index.

    generate_trades.py enumerates final_models/*.pkl, so a leftover .pkl keeps a
    de-indexed stock signal-eligible. Moving the file is what actually retires
    it locally. Price history is intentionally left in the DB - it costs little
    and stays available for backtests.

    Also deactivates the retired names' live trade_signals rows (see
    deactivate_signals) - the API serves is_active = TRUE, and removing the .pkl
    is what makes those rows unreachable by the normal deactivation path, so
    they would otherwise stay served forever.

    LOCAL ARCHIVING IS ONLY HALF THE JOB, so remote deletion is the DEFAULT.
    model_store.upload_all() is add-only, so an encrypted copy left on the Hub
    is re-downloaded by sync_models() and production keeps trading a stock that
    left the index. Deletion is a commit, so it stays recoverable via
    sync_models(revision=<pre-delete>). --no-retire-remote opts out.
    """
    step_banner(7, "retire",
                "archive + delete de-indexed models, deactivate their signals")

    csv_syms = {s["symbol"] for s in nse}
    pkls = [f for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")]
    stale = sorted(f for f in pkls
                   if f.replace("_final.pkl", "") not in csv_syms
                   and f.replace("_final.pkl", "") + ".NS" not in csv_syms)

    logger.info(f"Models on disk: {len(pkls)} · to retire: {len(stale)}")
    for f in stale:
        logger.info(f"      {f.replace('_final.pkl', '')}")

    symbols = [f.replace("_final.pkl", "") for f in stale]

    if dry_run:
        n = deactivate_signals(symbols, dry_run=True)
        logger.info(f"DRY-RUN: no files moved · {n} live signal row(s) would be deactivated")
        return {"retired": stale, "signals_deactivated": n}

    if stale:
        os.makedirs(RETIRED_DIR, exist_ok=True)
        for f in stale:
            shutil.move(os.path.join(FINAL_DIR, f), os.path.join(RETIRED_DIR, f))
        logger.info(f"Moved {len(stale)} models -> model_archives/removed_from_index/")

    remaining = len([f for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")])
    logger.info(f"final_models now holds {remaining} models")

    deactivated = deactivate_signals(symbols)
    if deactivated:
        logger.info(f"Deactivated {deactivated} live trade_signals row(s) for retired names")

    if stale and push_remote:
        from scripts.model_store import delete_models
        n = delete_models(symbols, commit_message="retire names dropped from Nifty 500")
        logger.info(f"Deleted {n} model(s) from the HF model repo")
    elif stale:
        logger.warning("--no-retire-remote: LOCAL ONLY — these models remain on the "
                       "HF Hub and will be re-synced onto production, which will keep "
                       "generating signals for them. Finish with: python "
                       f"scripts/model_store.py delete {' '.join(symbols[:3])} ...")

    return {"retired": stale, "remote_deleted": bool(stale and push_remote),
            "signals_deactivated": deactivated}


# ── Step 8: report ────────────────────────────────────────────────────────────

def step_report(nse: List[Dict]) -> Dict:
    """Closing comparison — the same add/remove diff as step 1, run again.

    Deliberately the identical calculation rather than a separate "coverage"
    view: if step 1 said ADD 35 / REMOVE 33 and step 8 says ADD 0 / REMOVE 0,
    that pair IS the proof the sync did what it set out to do. A differently
    computed summary could agree while the underlying sets still disagreed.
    """
    out = step_compare(nse, label="after")

    enf = [d for d in out.values() if d["enforced"]]
    leftover_add = sorted(set().union(*[set(d["to_add"]) for d in enf]) or set())
    leftover_rm = sorted(set().union(*[set(d["to_remove"]) for d in enf]) or set())

    logger.info("")
    if not leftover_add and not leftover_rm:
        logger.info("RESULT: every layer matches the NSE Nifty 500 list exactly.")
    else:
        logger.warning(f"RESULT: {len(leftover_add)} symbol(s) still to add, "
                       f"{len(leftover_rm)} still to remove.")
        if leftover_add:
            logger.warning(f"  still missing somewhere: "
                           f"{[s.replace('.NS', '') for s in leftover_add]}")
        if leftover_rm:
            logger.warning(f"  still present somewhere: "
                           f"{[s.replace('.NS', '') for s in leftover_rm]}")
        logger.info("  (final_models gaps are expected until the weekly retrain runs)")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RATE_LIMIT_SECS

    p = argparse.ArgumentParser(description="Sync tracked universe to the NSE Nifty 500 list")
    p.add_argument("--steps", default=",".join(ALL_STEPS),
                   help=f"comma-separated subset of: {','.join(ALL_STEPS)}")
    p.add_argument("--dry-run", action="store_true", help="plan only, no writes")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="restrict prices/indicators to these symbols")
    p.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECS,
                   help=f"seconds between Angel historical calls (default {RATE_LIMIT_SECS}). "
                        "Angel's quota tightens under sustained load — raise this for a "
                        "retry pass over symbols that failed with 'exceeding access rate'")
    p.add_argument("--news-from-year", type=int, default=NEWS_FROM_YEAR_DEFAULT,
                   help=f"first year of GDELT news to backfill (default "
                        f"{NEWS_FROM_YEAR_DEFAULT}). Cost is symbols x months x 12s, "
                        "so an earlier year gets expensive fast")
    p.add_argument("--news-score-only", action="store_true",
                   help="step 6: skip the GDELT fetch, only run FinBERT over "
                        "headlines already collected but unscored")
    p.add_argument("--news-fetch-only", action="store_true",
                   help="step 6: fetch from GDELT but do not score — use in "
                        "parallel shards, then score once with --news-score-only")
    p.add_argument("--shard", default=None, metavar="i/N",
                   help="step 6: process only shard i of N (round-robin over the "
                        "symbols needing news), e.g. --shard 3/6")
    p.add_argument("--plan", choices=["prices", "news", "indicators"], default=None,
                   help="print how many shards this step needs as GitHub Actions "
                        "output lines (count/shards/matrix) and exit — lets the "
                        "workflow matrix size itself to the actual intake")
    p.add_argument("--symbols-per-shard", type=int, default=None,
                   help=f"--plan: symbols per shard (default "
                        f"{SYMBOLS_PER_PRICE_SHARD} for prices, "
                        f"{SYMBOLS_PER_NEWS_SHARD} for news)")
    p.add_argument("--no-retire-remote", action="store_true",
                   help="step 7: keep retired models on the HF store. Retirement is "
                        "then LOCAL ONLY and production will re-sync and keep trading "
                        "the de-indexed stocks")
    args = p.parse_args()

    # Plan mode short-circuits: it emits machine-readable output only, so the
    # usual banner/logging would corrupt $GITHUB_OUTPUT.
    if args.plan:
        plan = shard_plan(read_nse_csv(), args.plan, args.symbols_per_shard)
        print(f"count={plan['count']}")
        print(f"shards={plan['shards']}")
        print(f"matrix={json.dumps(plan['matrix'])}")
        print(f"symbols={json.dumps(plan['symbols'])}")
        return

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        p.error(f"unknown step(s): {bad}. valid: {ALL_STEPS}")

    shard = None
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/", 1))
        except ValueError:
            p.error(f"--shard must look like i/N, got {args.shard!r}")
        if not 1 <= i <= n:
            p.error(f"--shard i must be within 1..N, got {args.shard}")
        shard = (i, n)

    target_db = os.getenv("PGHOST", "sqlite")
    logger.info("=" * 78)
    logger.info("NIFTY 500 UNIVERSE SYNC")
    logger.info("=" * 78)
    logger.info(f"  source CSV : {NSE_CSV}")
    logger.info(f"  target DB  : {target_db}  (APP_ENV={os.getenv('APP_ENV', 'prod')})")
    logger.info(f"  steps      : {steps}")
    logger.info(f"  dry-run    : {args.dry_run}")
    logger.info(f"  log file   : {LOG_PATH}")

    if args.rate_limit != RATE_LIMIT_SECS:
        logger.info(f"  rate limit : {args.rate_limit}s between historical calls "
                    f"(default {RATE_LIMIT_SECS}s)")
        RATE_LIMIT_SECS = args.rate_limit

    nse = read_nse_csv()
    non_eq = [s["base"] for s in nse if s["series"] != "EQ"]
    logger.info(f"  NSE list   : {len(nse)} symbols"
                + (f"  (non-EQ series: {non_eq})" if non_eq else "  (all series EQ)"))

    t0 = time.time()
    out = {}
    if "compare" in steps:
        out["compare_before"] = step_compare(nse, label="before")
    if "tokens" in steps:
        out["tokens"] = step_tokens(nse, args.dry_run)
    if "constituents" in steps:
        out["constituents"] = step_constituents(nse, args.dry_run)
    if "prices" in steps:
        out["prices"] = step_prices(nse, args.dry_run, args.symbols, shard)
    if "indicators" in steps:
        out["indicators"] = step_indicators(nse, args.dry_run, args.symbols, shard)
    if "news" in steps:
        out["news"] = step_news(nse, args.dry_run, args.news_from_year,
                                args.news_score_only, args.news_fetch_only, shard)
    if "retire" in steps:
        out["retire"] = step_retire(nse, args.dry_run, not args.no_retire_remote)
    if "report" in steps:
        out["report"] = step_report(nse)

    # Tell CI whether anything the Space actually loads has changed, so it can
    # decide whether a restart is warranted. The Space re-reads the model set and
    # the token map on boot, so only those matter: an indicators-only run
    # changes neither, and restarting for it is pure disruption — every restart
    # replays overdue jobs on startup, and that catch-up re-runs EOD collection
    # and burns Angel One's rate quota, which then starves the genuinely
    # scheduled run later the same day (2026-08-05: 224 "exceeding access rate"
    # errors at 11:00 after a restart, then only 479/500 prices at 15:35).
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        tok = out.get("tokens", {})
        retired = len(out.get("retire", {}).get("retired", []) or [])
        tokens_changed = bool(tok.get("added") or tok.get("removed"))
        models_changed = retired > 0 or tokens_changed
        try:
            with open(gh_out, "a") as fh:
                fh.write(f"retired={retired}\n")
                fh.write(f"models_changed={'true' if models_changed else 'false'}\n")
            logger.info(f"CI outputs: retired={retired} models_changed={models_changed}")
        except Exception as exc:
            logger.warning(f"Could not write GITHUB_OUTPUT: {exc}")

    logger.info("")
    logger.info("=" * 78)
    logger.info(f"DONE in {time.time() - t0:.0f}s — log: {LOG_PATH}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
