"""
TradeMind — Angel One Historical Price Backfill

Downloads 3 years of daily OHLCV data (Jan 2023 → present) for all 499
Nifty 500 stocks via Angel One SmartAPI and stores in the prices table.

Corner cases handled:
  - Rate limiting: exponential backoff retry on 429/access denied
  - Session expiry: auto re-login every 150 stocks + on auth errors
  - JSON parse errors: retry with backoff
  - Network timeouts: retry up to MAX_RETRIES times
  - Bad OHLCV data: skip rows with None/zero/negative prices
  - Delisted stocks: log and continue
  - Resume: saves progress to failed_stocks.json, supports --retry-failed
  - Duplicate rows: ON CONFLICT DO NOTHING in DB

Usage:
    PYTHONPATH=. python collectors/backfill_prices_angel.py
    PYTHONPATH=. python collectors/backfill_prices_angel.py --from-year 2023
    PYTHONPATH=. python collectors/backfill_prices_angel.py --symbol HDFCBANK
    PYTHONPATH=. python collectors/backfill_prices_angel.py --retry-failed
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pyotp
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_database, insert_prices_batch

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROM_YEAR             = 2023
FROM_MONTH            = 1
SLEEP_BETWEEN_STOCKS  = 1.0     # seconds between stocks
SLEEP_BETWEEN_CHUNKS  = 1.2     # seconds between chunk API calls (avoid rate limit)
RELOGIN_EVERY         = 150     # re-authenticate every N stocks
MAX_RETRIES           = 3       # max retries per chunk on failure
RETRY_BASE_SLEEP      = 5.0     # base sleep on first retry (doubles each time)
RATE_LIMIT_SLEEP      = 10.0    # extra sleep when rate limited

_TOKENS_FILE     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "angel_tokens.json")
_FAILED_FILE     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backfill_failed.json")


# ---------------------------------------------------------------------------
# Angel One session
# ---------------------------------------------------------------------------

class AngelSession:
    def __init__(self):
        self.api_key     = os.getenv("ANGEL_API_KEY", "")
        self.client_id   = os.getenv("ANGEL_CLIENT_ID", "")
        self.password    = os.getenv("ANGEL_MPIN", "") or os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
        self.smart_api   = None
        self._logged_in  = False

    def login(self, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                from SmartApi import SmartConnect
                self.smart_api = SmartConnect(api_key=self.api_key)
                totp = pyotp.TOTP(self.totp_secret).now()
                data = self.smart_api.generateSession(self.client_id, self.password, totp)
                if data.get("status"):
                    self._logged_in = True
                    logger.info(f"Angel One login OK — {self.client_id}")
                    return True
                msg = data.get("message", "Unknown")
                logger.warning(f"Login attempt {attempt+1}/{retries} failed: {msg}")
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Login attempt {attempt+1}/{retries} error: {e}")
                time.sleep(3 * (attempt + 1))
        logger.error("All login attempts failed")
        return False

    def logout(self):
        try:
            if self.smart_api and self._logged_in:
                self.smart_api.terminateSession(self.client_id)
                self._logged_in = False
        except Exception:
            pass

    def get_candles(self, token: str, exchange: str,
                    from_date: str, to_date: str) -> Optional[List]:
        """
        Fetch daily candles for one date chunk with retry + backoff.

        Returns list of candles, [] for no data, None for auth error
        (caller should re-login on None).
        """
        if not self._logged_in:
            return None

        params = {
            "exchange":    exchange,
            "symboltoken": token,
            "interval":    "ONE_DAY",
            "fromdate":    from_date,
            "todate":      to_date,
        }

        for attempt in range(MAX_RETRIES):
            try:
                data = self.smart_api.getCandleData(params)

                # Auth error — signal caller to re-login
                msg = str(data.get("message", "") or data.get("errorcode", "")).lower()
                if "invalid token" in msg or "session expired" in msg or "unauthorized" in msg:
                    logger.warning("Session expired — signalling re-login")
                    self._logged_in = False
                    return None

                # Rate limit — wait longer and retry
                if "rate" in msg or "access denied" in msg or "exceeding" in msg:
                    wait = RATE_LIMIT_SLEEP * (attempt + 1)
                    logger.warning(f"Rate limited — sleeping {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue

                if not data.get("status") or not data.get("data"):
                    # No data for this chunk (holiday period / delisted) — not an error
                    return []

                return data["data"]

            except Exception as e:
                err = str(e).lower()

                # Rate limit in exception message
                if "rate" in err or "access denied" in err or "exceeding" in err:
                    wait = RATE_LIMIT_SLEEP * (attempt + 1)
                    logger.warning(f"Rate limit exception — sleeping {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue

                # JSON parse / network error — retry with backoff
                wait = RETRY_BASE_SLEEP * (2 ** attempt)
                logger.warning(f"Chunk error (attempt {attempt+1}/{MAX_RETRIES}): {e} — retrying in {wait}s")
                time.sleep(wait)

        logger.error(f"Chunk failed after {MAX_RETRIES} retries: {from_date[:10]} → {to_date[:10]}")
        return []


# ---------------------------------------------------------------------------
# Date chunk builder
# ---------------------------------------------------------------------------

def build_chunks(from_year: int, from_month: int) -> List[Tuple[str, str]]:
    """Split from_year/month → today into ~365-day chunks for Angel One API."""
    chunks = []
    start  = datetime(from_year, from_month, 1)
    end    = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=364), end)
        chunks.append((
            cursor.strftime("%Y-%m-%d 09:15"),
            chunk_end.strftime("%Y-%m-%d 15:30"),
        ))
        cursor = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Per-symbol backfill
# ---------------------------------------------------------------------------

def backfill_symbol(session: AngelSession, symbol: str, token_info: dict,
                    chunks: List[Tuple[str, str]]) -> Tuple[int, bool]:
    """
    Download all chunks for one symbol and batch-insert into DB.
    Returns (rows_inserted, needs_relogin).
    """
    symbol_ns = f"{symbol}.NS"
    token     = token_info["token"]
    exchange  = token_info.get("exchange", "NSE")
    total_inserted = 0

    for from_dt, to_dt in chunks:
        candles = session.get_candles(token, exchange, from_dt, to_dt)

        # None = auth error, caller must re-login
        if candles is None:
            return total_inserted, True

        if not candles:
            time.sleep(SLEEP_BETWEEN_CHUNKS)
            continue

        rows = []
        for candle in candles:
            try:
                ts, o, h, l, c, v = candle

                # Parse timestamp — handle tz-aware and naive formats
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
                except ValueError:
                    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                date_str = dt.strftime("%Y-%m-%d")

                o, h, l, c = float(o), float(h), float(l), float(c)
                v = int(v)

                # Skip bad data: zero/negative prices
                if c <= 0 or o <= 0 or h <= 0 or l <= 0:
                    continue

                # Skip malformed OHLC (high < low)
                if h < l:
                    continue

                rows.append((symbol_ns, exchange, date_str, None,
                              round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                              v, "1d"))

            except Exception as e:
                logger.debug(f"Bad candle for {symbol} — skipping: {e}")
                continue

        if rows:
            inserted = insert_prices_batch(rows)
            total_inserted += inserted

        time.sleep(SLEEP_BETWEEN_CHUNKS)

    return total_inserted, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backfill_all(from_year: int = FROM_YEAR, from_month: int = FROM_MONTH,
                 symbol_filter: str = None, retry_failed: bool = False):
    init_database()

    if not os.path.exists(_TOKENS_FILE):
        print(f"❌ {_TOKENS_FILE} not found — run map_tokens.py first")
        sys.exit(1)

    with open(_TOKENS_FILE) as f:
        token_map = json.load(f)

    # --retry-failed: only process symbols from previous failed run
    if retry_failed:
        if not os.path.exists(_FAILED_FILE):
            print("No failed stocks file found. Run full backfill first.")
            sys.exit(0)
        with open(_FAILED_FILE) as f:
            failed_prev = json.load(f)
        symbols = [(s, token_map[s]) for s in failed_prev if s in token_map]
        print(f"\n🔁 Retrying {len(symbols)} previously failed stocks...\n")
    elif symbol_filter:
        symbols = [(s, t) for s, t in token_map.items() if s.upper() == symbol_filter.upper()]
        if not symbols:
            print(f"Symbol {symbol_filter} not found in angel_tokens.json")
            sys.exit(1)
    else:
        symbols = list(token_map.items())

    chunks = build_chunks(from_year, from_month)
    print(f"\n📅 Date chunks ({len(chunks)} total):")
    for f, t in chunks:
        print(f"   {f[:10]} → {t[:10]}")
    print(f"\n📊 Backfilling {len(symbols)} stocks | "
          f"sleep: {SLEEP_BETWEEN_CHUNKS}s/chunk, {SLEEP_BETWEEN_STOCKS}s/stock\n")

    session = AngelSession()
    if not session.login():
        print("❌ Angel One login failed — check credentials in .env")
        sys.exit(1)

    total_rows = 0
    failed     = []

    for idx, (symbol, token_info) in enumerate(tqdm(symbols, desc="Backfilling", unit="stock")):

        # Scheduled re-login every RELOGIN_EVERY stocks
        if idx > 0 and idx % RELOGIN_EVERY == 0:
            logger.info(f"Scheduled re-login at stock {idx}...")
            session.logout()
            time.sleep(3)
            if not session.login():
                logger.error("Re-login failed — stopping")
                # Save remaining symbols as failed
                remaining = [s for s, _ in symbols[idx:]]
                failed.extend(remaining)
                break

        try:
            rows, needs_relogin = backfill_symbol(session, symbol, token_info, chunks)

            # Session expired mid-symbol — re-login and retry once
            if needs_relogin:
                logger.warning(f"Session expired at {symbol} — re-logging in")
                session.logout()
                time.sleep(3)
                if session.login():
                    rows, needs_relogin = backfill_symbol(session, symbol, token_info, chunks)
                    if needs_relogin:
                        logger.error(f"{symbol}: still auth error after re-login — skipping")
                        failed.append(symbol)
                        continue
                else:
                    logger.error("Re-login failed — stopping")
                    failed.extend([s for s, _ in symbols[idx:]])
                    break

            total_rows += rows

        except Exception as e:
            logger.error(f"{symbol}: unexpected error — {e}")
            failed.append(symbol)

        time.sleep(SLEEP_BETWEEN_STOCKS)

    session.logout()

    # Save failed list for --retry-failed
    if failed:
        with open(_FAILED_FILE, "w") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Failed stocks saved to {_FAILED_FILE}")

    print(f"\n{'='*60}")
    print(f"✅ Done: {len(symbols) - len(failed)}/{len(symbols)} stocks")
    print(f"📊 Total new rows inserted: {total_rows:,}")
    if failed:
        print(f"❌ Failed ({len(failed)}): {', '.join(failed[:20])}")
        print(f"   Re-run failed: PYTHONPATH=. python collectors/backfill_prices_angel.py --retry-failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Angel One 3-year price backfill")
    parser.add_argument("--from-year",    type=int,  default=FROM_YEAR,  help="Start year (default 2023)")
    parser.add_argument("--from-month",   type=int,  default=FROM_MONTH, help="Start month (default 1)")
    parser.add_argument("--symbol",       type=str,  default=None,       help="Single symbol e.g. HDFCBANK")
    parser.add_argument("--retry-failed", action="store_true",           help="Retry stocks that failed in last run")
    args = parser.parse_args()

    backfill_all(
        from_year=args.from_year,
        from_month=args.from_month,
        symbol_filter=args.symbol,
        retry_failed=args.retry_failed,
    )
