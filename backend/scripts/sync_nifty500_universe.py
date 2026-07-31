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

    1. tokens        NSE CSV  -> data/angel_tokens.json   (Angel scrip master)
    2. constituents  NSE CSV  -> nifty_constituents table (names + sectors)
    3. prices        Angel One -> prices table            (chunked backfill)
    4. indicators    prices    -> technical_indicators
    5. train         prices    -> final_models/*.pkl      (gated, see below)
    6. retire        archive models for names dropped from the index
    7. report        final coverage summary

Steps are independent and re-runnable; use --steps to run a subset.

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

Training gate
-------------
A symbol is only trained if it has >= --min-train-rows (default 100) daily bars
at or before TRAIN_END. That mirrors model_training's own `len(Xtr) < 100`
skip, but applied up-front so recent IPOs don't burn a training slot to fail.
Many of the 35 new names are 2025-26 listings with far less than that.

Usage
-----
    cd backend && source venv/bin/activate

    python scripts/sync_nifty500_universe.py --dry-run          # plan only
    python scripts/sync_nifty500_universe.py                    # all steps
    python scripts/sync_nifty500_universe.py --steps prices,indicators
    python scripts/sync_nifty500_universe.py --steps train --min-train-rows 150

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

ALL_STEPS = ["tokens", "constituents", "prices", "indicators", "train", "retire", "report"]

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


# ── Step 1: tokens ────────────────────────────────────────────────────────────

def step_tokens(nse: List[Dict], dry_run: bool) -> Dict:
    """Resolve every CSV symbol to an Angel instrument token, rewrite angel_tokens.json."""
    step_banner(1, "tokens", "NSE CSV -> data/angel_tokens.json")

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


# ── Step 2: constituents ──────────────────────────────────────────────────────

def step_constituents(nse: List[Dict], dry_run: bool) -> Dict:
    """Make nifty_constituents exactly match the NSE list (fixes sectors/names, drops orphans)."""
    step_banner(2, "constituents", "NSE CSV -> nifty_constituents")

    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT symbol, name, sector FROM nifty_constituents")
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
                _execute(conn, "DELETE FROM nifty_constituents WHERE symbol=?", (s,))
            conn.commit()
        finally:
            release_connection(conn)
        logger.info(f"Deleted {len(to_drop)} rows no longer in the index")

    # upsert_nifty_constituents clears the sector cache, but the delete above
    # happens after it - clear again so a live process can't serve dropped names.
    try:
        from database import db as _db
        _db._sector_map_cache.clear()
    except Exception:
        pass

    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT COUNT(*) FROM nifty_constituents")
        logger.info(f"nifty_constituents now has {cur.fetchone()[0]} rows")
    finally:
        release_connection(conn)

    return {"added": to_add, "dropped": to_drop, "sector_fix": sector_fix, "name_fix": name_fix}


# ── Step 3: prices ────────────────────────────────────────────────────────────

def step_prices(nse: List[Dict], dry_run: bool, only: Optional[List[str]] = None) -> Dict:
    """Backfill full available history for constituents with no price data."""
    step_banner(3, "prices", "Angel One -> prices")

    tokens = load_tokens()
    have = db_symbols_with_prices()
    targets = [s for s in nse if s["symbol"] not in have]
    if only:
        targets = [s for s in targets if s["base"] in only or s["symbol"] in only]

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


# ── Step 4: indicators ────────────────────────────────────────────────────────

def step_indicators(nse: List[Dict], dry_run: bool, only: Optional[List[str]] = None) -> Dict:
    """Compute technical indicators for constituents that don't have any yet."""
    step_banner(4, "indicators", "prices -> technical_indicators")

    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT DISTINCT symbol FROM technical_indicators")
        have_ind = {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)

    have_px = db_symbols_with_prices()
    targets = [s["symbol"] for s in nse if s["symbol"] in have_px and s["symbol"] not in have_ind]
    if only:
        targets = [t for t in targets if t in only or t.replace(".NS", "") in only]

    logger.info(f"Constituents missing indicators: {len(targets)}")
    if not targets:
        logger.info("Nothing to compute")
        return {"ok": [], "failed": []}
    logger.info(f"  {[t.replace('.NS', '') for t in targets]}")

    if dry_run:
        logger.info("DRY-RUN: no indicator computation")
        return {"ok": [], "failed": []}

    from analysis.signals import process_stock

    ok, failed = [], []
    for idx, sym in enumerate(targets, 1):
        try:
            res = process_stock(sym, days=400)
            if res:
                ok.append(sym)
                logger.info(f"[{idx}/{len(targets)}] {sym:18} {res.get('signal', '?'):10} "
                            f"strength={res.get('strength', 0)}")
            else:
                failed.append(sym)
                logger.warning(f"[{idx}/{len(targets)}] {sym:18} returned no result "
                               f"(likely < 14 bars)")
        except Exception as e:
            failed.append(sym)
            logger.error(f"[{idx}/{len(targets)}] {sym:18} FAILED: {str(e)[:100]}")

    logger.info(f"Indicators computed: {len(ok)} ok, {len(failed)} failed")
    return {"ok": ok, "failed": failed}


# ── Step 5: train ─────────────────────────────────────────────────────────────

def step_train(nse: List[Dict], dry_run: bool, min_rows: int,
               only: Optional[List[str]] = None) -> Dict:
    """Train models for constituents that have none — gated on history depth."""
    from scripts.retrain_walk_forward import TRAIN_END, TEST_START, train_one

    step_banner(5, "train", f"gate: >= {min_rows} daily bars on/before {TRAIN_END}")
    logger.info(f"Walk-forward split: train -> {TRAIN_END} | test {TEST_START} -> present")

    have_px = db_symbols_with_prices()
    existing_models = {f.replace("_final.pkl", "") for f in os.listdir(FINAL_DIR)
                       if f.endswith("_final.pkl")}
    targets = [s["symbol"] for s in nse
               if s["symbol"] in have_px and s["symbol"] not in existing_models]
    if only:
        targets = [t for t in targets if t in only or t.replace(".NS", "") in only]

    logger.info(f"Constituents without a model: {len(targets)}")
    if not targets:
        logger.info("Nothing to train")
        return {"trained": [], "skipped": [], "failed": []}

    eligible, skipped = [], []
    for sym in sorted(targets):
        n = db_row_count_before(sym, TRAIN_END)
        (eligible if n >= min_rows else skipped).append((sym, n))

    logger.info(f"  eligible ({len(eligible)}):")
    for sym, n in eligible:
        logger.info(f"      {sym:18} {n:5} bars <= {TRAIN_END}")
    logger.info(f"  skipped — insufficient history ({len(skipped)}):")
    for sym, n in skipped:
        logger.info(f"      {sym:18} {n:5} bars <= {TRAIN_END}  (need {min_rows})")

    if dry_run:
        logger.info("DRY-RUN: no training")
        return {"trained": [], "skipped": skipped, "failed": []}

    trained, failed, results = [], [], []
    for idx, (sym, n) in enumerate(eligible, 1):
        logger.info(f"[{idx}/{len(eligible)}] training {sym} ({n} bars)...")
        try:
            r = train_one(sym)
            results.append(r)
            if r.get("status") == "ok":
                trained.append(sym)
                logger.info(f"    OK  {r.get('best_model')} {r.get('horizon')} "
                            f"acc={r.get('accuracy', 0):.1%} prec={r.get('precision', 0):.1%} "
                            f"[{r.get('quality_tier')}]")
            else:
                failed.append(sym)
                logger.warning(f"    {r.get('status')}: {str(r.get('error', ''))[:100]}")
        except Exception as e:
            failed.append(sym)
            logger.error(f"    FAILED: {str(e)[:120]}")

    # Metrics belong in the DB, not retrain_results.csv (see CLAUDE.md).
    if results:
        try:
            from database.db import insert_model_training_stats
            run_id = f"universe_sync_{datetime.now():%Y%m%d_%H%M%S}"
            written = insert_model_training_stats(run_id, results)
            logger.info(f"Recorded {written} training rows in model_training_stats (run {run_id})")
        except Exception as e:
            logger.warning(f"Could not record training stats: {str(e)[:100]}")

    logger.info(f"Training complete: {len(trained)} trained, {len(skipped)} gated out, "
                f"{len(failed)} failed")
    return {"trained": trained, "skipped": skipped, "failed": failed}


# ── Step 6: retire ────────────────────────────────────────────────────────────

def step_retire(nse: List[Dict], dry_run: bool, push_remote: bool = False) -> Dict:
    """Archive models for symbols no longer in the index.

    generate_trades.py enumerates final_models/*.pkl, so a leftover .pkl keeps a
    de-indexed stock signal-eligible. Moving the file is what actually retires
    it locally. Price history is intentionally left in the DB - it costs little
    and stays available for backtests.

    LOCAL ARCHIVING IS ONLY HALF THE JOB. model_store.upload_all() is add-only,
    so the encrypted copies stay on the Hub and sync_models() re-downloads them
    onto production. --retire-remote also deletes them from the Hub (recoverable
    - it's a commit, so sync_models(revision=...) restores). Without that flag
    this step warns and leaves production untouched.
    """
    step_banner(6, "retire", "archive models for de-indexed names")

    csv_syms = {s["symbol"] for s in nse}
    pkls = [f for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")]
    stale = sorted(f for f in pkls
                   if f.replace("_final.pkl", "") not in csv_syms
                   and f.replace("_final.pkl", "") + ".NS" not in csv_syms)

    logger.info(f"Models on disk: {len(pkls)} · to retire: {len(stale)}")
    for f in stale:
        logger.info(f"      {f.replace('_final.pkl', '')}")

    if dry_run:
        logger.info("DRY-RUN: no files moved")
        return {"retired": stale}

    if stale:
        os.makedirs(RETIRED_DIR, exist_ok=True)
        for f in stale:
            shutil.move(os.path.join(FINAL_DIR, f), os.path.join(RETIRED_DIR, f))
        logger.info(f"Moved {len(stale)} models -> model_archives/removed_from_index/")

    remaining = len([f for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")])
    logger.info(f"final_models now holds {remaining} models")

    symbols = [f.replace("_final.pkl", "") for f in stale]
    if stale and push_remote:
        from scripts.model_store import delete_models
        n = delete_models(symbols, commit_message="retire names dropped from Nifty 500")
        logger.info(f"Deleted {n} model(s) from the HF model repo")
    elif stale:
        logger.warning("LOCAL ONLY — these models remain on the HF Hub and will be "
                       "re-synced onto production, which will keep generating signals "
                       "for them. Re-run with --retire-remote (or: python "
                       f"scripts/model_store.py delete {' '.join(symbols[:3])} ...) to "
                       "complete retirement.")

    return {"retired": stale, "remote_deleted": bool(stale and push_remote)}


# ── Step 7: report ────────────────────────────────────────────────────────────

def step_report(nse: List[Dict]) -> Dict:
    """Final coverage across every layer of the universe."""
    step_banner(7, "report", "coverage")

    csv_syms = {s["symbol"] for s in nse}
    tokens = {f"{k}.NS" for k in load_tokens()} if os.path.exists(TOKENS_FILE) else set()
    have_px = db_symbols_with_prices()
    models = {f.replace("_final.pkl", "") for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")}
    models |= {m + ".NS" for m in models if not m.endswith(".NS")}

    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT symbol FROM nifty_constituents")
        consts = {r[0] for r in cur.fetchall()}
        cur = _execute(conn, "SELECT DISTINCT symbol FROM technical_indicators")
        inds = {r[0] for r in cur.fetchall()}
    finally:
        release_connection(conn)

    layers = [
        ("nifty_constituents", consts),
        ("angel_tokens.json", tokens),
        ("prices", have_px),
        ("technical_indicators", inds),
        ("final_models", models),
    ]

    logger.info(f"NSE Nifty 500 list: {len(csv_syms)} symbols")
    logger.info("")
    logger.info(f"  {'layer':<22} {'covered':>9} {'missing':>9} {'extra':>7}")
    logger.info(f"  {'-' * 50}")
    summary = {}
    for name, have in layers:
        covered = len(csv_syms & have)
        missing = sorted(csv_syms - have)
        extra = sorted(have - csv_syms)
        # indices (^NSEI etc.) are legitimately outside the constituent list
        extra = [e for e in extra if not e.startswith("^")]
        summary[name] = {"covered": covered, "missing": missing, "extra": extra}
        logger.info(f"  {name:<22} {covered:>4}/{len(csv_syms):<4} {len(missing):>9} {len(extra):>7}")

    for name, d in summary.items():
        if d["missing"]:
            logger.info("")
            logger.info(f"  {name} missing ({len(d['missing'])}): "
                        f"{[m.replace('.NS', '') for m in d['missing']]}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RATE_LIMIT_SECS

    p = argparse.ArgumentParser(description="Sync tracked universe to the NSE Nifty 500 list")
    p.add_argument("--steps", default=",".join(ALL_STEPS),
                   help=f"comma-separated subset of: {','.join(ALL_STEPS)}")
    p.add_argument("--dry-run", action="store_true", help="plan only, no writes")
    p.add_argument("--min-train-rows", type=int, default=100,
                   help="minimum daily bars on/before TRAIN_END to train (default 100)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="restrict prices/indicators/train to these symbols")
    p.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECS,
                   help=f"seconds between Angel historical calls (default {RATE_LIMIT_SECS}). "
                        "Angel's quota tightens under sustained load — raise this for a "
                        "retry pass over symbols that failed with 'exceeding access rate'")
    p.add_argument("--retire-remote", action="store_true",
                   help="step 6: also delete retired models from the HF model repo "
                        "(without this, retirement is local-only and production keeps them)")
    args = p.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        p.error(f"unknown step(s): {bad}. valid: {ALL_STEPS}")

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
    if "tokens" in steps:
        out["tokens"] = step_tokens(nse, args.dry_run)
    if "constituents" in steps:
        out["constituents"] = step_constituents(nse, args.dry_run)
    if "prices" in steps:
        out["prices"] = step_prices(nse, args.dry_run, args.symbols)
    if "indicators" in steps:
        out["indicators"] = step_indicators(nse, args.dry_run, args.symbols)
    if "train" in steps:
        out["train"] = step_train(nse, args.dry_run, args.min_train_rows, args.symbols)
    if "retire" in steps:
        out["retire"] = step_retire(nse, args.dry_run, args.retire_remote)
    if "report" in steps:
        out["report"] = step_report(nse)

    logger.info("")
    logger.info("=" * 78)
    logger.info(f"DONE in {time.time() - t0:.0f}s — log: {LOG_PATH}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
