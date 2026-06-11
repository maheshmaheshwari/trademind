"""
TradeMind AI — LTP Fetcher (Angel One)

Fetches Last Traded Price (LTP) for stocks using Angel One's ltpData API.
Used by the price monitor for intraday SL/target checking.

Also supports fetching 30-min candle data for intraday chart display.
"""

import logging
import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Load full token map
_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "angel_tokens.json"
)
_TOKEN_MAP = {}
try:
    with open(_TOKENS_FILE) as f:
        _TOKEN_MAP = json.load(f)
except FileNotFoundError:
    logger.warning("angel_tokens.json not found for LTP fetcher")


def _get_angel_session():
    """Create and return an authenticated Angel One SmartConnect session."""
    try:
        from SmartApi import SmartConnect
        import pyotp

        api_key = os.getenv("ANGEL_API_KEY")
        client_id = os.getenv("ANGEL_CLIENT_ID")
        password = os.getenv("ANGEL_MPIN") or os.getenv("ANGEL_PASSWORD")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        if not all([api_key, client_id, password, totp_secret]):
            logger.error("Missing Angel One credentials in .env")
            return None

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_id, password, totp)

        if not session or session.get("status") is False:
            logger.error(f"Angel One login failed: {session}")
            return None

        logger.info("Angel One LTP session created successfully")
        return smart_api

    except Exception as e:
        logger.error(f"Error creating Angel One session: {e}")
        return None


def fetch_ltp_batch(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch Last Traded Price for a list of stock symbols.
    
    Args:
        symbols: List like ["TCS.NS", "RELIANCE.NS", "INFY.NS"]
        
    Returns:
        Dict mapping symbol → LTP price, e.g. {"TCS.NS": 2637.40, ...}
    """
    if not symbols:
        return {}

    smart_api = _get_angel_session()
    if not smart_api:
        logger.error("Cannot fetch LTP — Angel One session failed")
        return {}

    results = {}
    
    for symbol in symbols:
        short_name = symbol.replace(".NS", "").upper()
        token_info = _TOKEN_MAP.get(short_name)
        
        if not token_info:
            logger.warning(f"No token found for {short_name}")
            continue
        
        try:
            ltp_data = smart_api.ltpData(
                exchange="NSE",
                tradingsymbol=token_info.get("trading_symbol", f"{short_name}-EQ"),
                symboltoken=token_info["token"]
            )
            
            if ltp_data and ltp_data.get("status") and ltp_data.get("data"):
                ltp = float(ltp_data["data"].get("ltp", 0))
                if ltp > 0:
                    results[symbol] = ltp
                    logger.debug(f"LTP {symbol}: ₹{ltp}")
            else:
                logger.warning(f"No LTP data for {symbol}: {ltp_data}")
                
            # Rate limiting — Angel One allows ~3 req/sec
            time.sleep(0.35)
            
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
    
    logger.info(f"Fetched LTP for {len(results)}/{len(symbols)} symbols")
    return results


def fetch_ltp_single(symbol: str) -> Optional[float]:
    """Fetch LTP for a single symbol. Returns None on failure."""
    results = fetch_ltp_batch([symbol])
    return results.get(symbol)


def fetch_intraday_30min(symbols: List[str] = None) -> int:
    """
    Fetch 30-min candle data for symbols and save to DB.
    
    If symbols is None, fetches for all symbols with open positions.
    Returns the number of candles saved.
    """
    from database.db import get_connection, release_connection, insert_prices_batch
    
    if symbols is None:
        conn = get_connection()
        from database.db import _execute as _ex
        rows = _ex(conn, "SELECT DISTINCT symbol FROM positions").fetchall()
        release_connection(conn)
        symbols = [r[0] for r in rows] if rows else []
    
    if not symbols:
        logger.info("No symbols to fetch intraday data for")
        return 0

    smart_api = _get_angel_session()
    if not smart_api:
        return 0

    total_saved = 0
    
    for symbol in symbols:
        short_name = symbol.replace(".NS", "").upper()
        token_info = _TOKEN_MAP.get(short_name)
        
        if not token_info:
            continue
        
        try:
            from_date = datetime.now().strftime("%Y-%m-%d 09:15")
            to_date = datetime.now().strftime("%Y-%m-%d 15:30")
            
            params = {
                "exchange": "NSE",
                "symboltoken": token_info["token"],
                "interval": "THIRTY_MINUTE",
                "fromdate": from_date,
                "todate": to_date,
            }
            
            data = smart_api.getCandleData(params)
            
            if not data.get("status") or not data.get("data"):
                logger.warning(f"No 30-min data for {short_name}")
                continue
            
            db_rows = []
            for candle in data["data"]:
                ts, o, h, l, c, v = candle
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                
                db_rows.append((
                    symbol, "NSE", date_str, time_str,
                    round(float(o), 2), round(float(h), 2),
                    round(float(l), 2), round(float(c), 2),
                    int(v), "30m",
                ))
            
            if db_rows:
                inserted = insert_prices_batch(db_rows)
                total_saved += inserted
                logger.info(f"30-min data: {short_name} — {inserted} candles saved")
            
            time.sleep(0.35)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching 30-min data for {short_name}: {e}")
    
    logger.info(f"Total 30-min candles saved: {total_saved}")
    return total_saved


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    # Test LTP fetch
    prices = fetch_ltp_batch(["TCS.NS", "RELIANCE.NS", "INFY.NS"])
    for sym, price in prices.items():
        print(f"  {sym}: ₹{price:,.2f}")
