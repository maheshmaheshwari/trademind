"""
Nifty 500 AI — Angel One SmartAPI Live Data Collector

Fetches real-time and historical candle data via Angel One SmartAPI.
Provides live market prices, intraday data, and historical OHLCV candles.

Prerequisites:
    1. Angel One Demat account (free)
    2. SmartAPI app created at https://smartapi.angelone.in/
    3. TOTP enabled on your Angel One account
    4. Credentials in .env:
       ANGEL_API_KEY=your_api_key
       ANGEL_SECRET_KEY=your_secret
       ANGEL_CLIENT_ID=your_client_id (e.g. M123456)
       ANGEL_PASSWORD=your_password
       ANGEL_TOTP_SECRET=your_totp_manual_key

Usage:
    from collectors.angel_collector import AngelCollector

    angel = AngelCollector()
    if angel.login():
        # Get live price
        ltp = angel.get_ltp("TCS")
        print(f"TCS live price: ₹{ltp}")

        # Get historical candles
        candles = angel.get_historical("TCS", interval="ONE_DAY", days=30)
        angel.logout()
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

from database.db import insert_prices_batch, init_database

load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================
# Angel One symbol token mapping
# SmartAPI uses numeric tokens, not NSE symbols.
# This maps common Nifty 50 stocks to their tokens.
# Full list: https://margincalculator.angelone.in/OpenAPI/files/OpenAPIScripMaster.json
# ==========================================
ANGEL_TOKENS = {
    "TCS": {"token": "11536", "exchange": "NSE", "symbol": "TCS-EQ"},
    "RELIANCE": {"token": "2885", "exchange": "NSE", "symbol": "RELIANCE-EQ"},
    "HDFCBANK": {"token": "1333", "exchange": "NSE", "symbol": "HDFCBANK-EQ"},
    "INFY": {"token": "1594", "exchange": "NSE", "symbol": "INFY-EQ"},
    "ICICIBANK": {"token": "4963", "exchange": "NSE", "symbol": "ICICIBANK-EQ"},
    "SBIN": {"token": "3045", "exchange": "NSE", "symbol": "SBIN-EQ"},
    "ITC": {"token": "1660", "exchange": "NSE", "symbol": "ITC-EQ"},
    "KOTAKBANK": {"token": "1922", "exchange": "NSE", "symbol": "KOTAKBANK-EQ"},
    "HINDUNILVR": {"token": "1394", "exchange": "NSE", "symbol": "HINDUNILVR-EQ"},
    "BAJFINANCE": {"token": "317", "exchange": "NSE", "symbol": "BAJFINANCE-EQ"},
    "BHARTIARTL": {"token": "10604", "exchange": "NSE", "symbol": "BHARTIARTL-EQ"},
    "TATAMOTORS": {"token": "3456", "exchange": "NSE", "symbol": "TATAMOTORS-EQ"},
    "TATASTEEL": {"token": "3499", "exchange": "NSE", "symbol": "TATASTEEL-EQ"},
    "MARUTI": {"token": "10999", "exchange": "NSE", "symbol": "MARUTI-EQ"},
    "SUNPHARMA": {"token": "3351", "exchange": "NSE", "symbol": "SUNPHARMA-EQ"},
    "WIPRO": {"token": "3787", "exchange": "NSE", "symbol": "WIPRO-EQ"},
    "HCLTECH": {"token": "7229", "exchange": "NSE", "symbol": "HCLTECH-EQ"},
    "TECHM": {"token": "13538", "exchange": "NSE", "symbol": "TECHM-EQ"},
    "AXISBANK": {"token": "5900", "exchange": "NSE", "symbol": "AXISBANK-EQ"},
    "NTPC": {"token": "11630", "exchange": "NSE", "symbol": "NTPC-EQ"},
    "POWERGRID": {"token": "14977", "exchange": "NSE", "symbol": "POWERGRID-EQ"},
    "COALINDIA": {"token": "20374", "exchange": "NSE", "symbol": "COALINDIA-EQ"},
    "ONGC": {"token": "2475", "exchange": "NSE", "symbol": "ONGC-EQ"},
    "TITAN": {"token": "3506", "exchange": "NSE", "symbol": "TITAN-EQ"},
    "ASIANPAINT": {"token": "236", "exchange": "NSE", "symbol": "ASIANPAINT-EQ"},
    "ULTRACEMCO": {"token": "11532", "exchange": "NSE", "symbol": "ULTRACEMCO-EQ"},
    "DRREDDY": {"token": "881", "exchange": "NSE", "symbol": "DRREDDY-EQ"},
    "CIPLA": {"token": "694", "exchange": "NSE", "symbol": "CIPLA-EQ"},
    "M&M": {"token": "2031", "exchange": "NSE", "symbol": "M&M-EQ"},
    "ADANIENT": {"token": "25", "exchange": "NSE", "symbol": "ADANIENT-EQ"},
    # Index tokens
    "NIFTY50": {"token": "99926000", "exchange": "NSE", "symbol": "NIFTY"},
    "NIFTY500": {"token": "99926004", "exchange": "NSE", "symbol": "NIFTY 500"},
    "SENSEX": {"token": "99919000", "exchange": "BSE", "symbol": "SENSEX"},
}

# Interval mapping for SmartAPI
INTERVAL_MAP = {
    "1min": "ONE_MINUTE",
    "3min": "THREE_MINUTE",
    "5min": "FIVE_MINUTE",
    "10min": "TEN_MINUTE",
    "15min": "FIFTEEN_MINUTE",
    "30min": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}


class AngelCollector:
    """
    Angel One SmartAPI client for live and historical market data.

    Usage:
        angel = AngelCollector()
        if angel.login():
            ltp = angel.get_ltp("TCS")
            candles = angel.get_historical("RELIANCE", interval="ONE_DAY", days=30)
            angel.logout()
    """

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.secret_key = os.getenv("ANGEL_SECRET_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.password = os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        self._logged_in = False

    def _is_configured(self) -> bool:
        """Check if all required credentials are present."""
        required = [self.api_key, self.client_id, self.password, self.totp_secret]
        return all(v and "your_" not in v for v in required)

    def login(self) -> bool:
        """
        Login to Angel One SmartAPI using TOTP authentication.

        Returns:
            True if login succeeded, False otherwise.
        """
        if not self._is_configured():
            logger.warning(
                "Angel One credentials not configured. "
                "Fill in ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET in .env"
            )
            print("⚠️  Angel One not configured. Fill credentials in .env file:")
            print("   ANGEL_CLIENT_ID=your_client_id (e.g. M123456)")
            print("   ANGEL_PASSWORD=your_angel_one_password")
            print("   ANGEL_TOTP_SECRET=your_totp_manual_key")
            return False

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP code
            totp = pyotp.TOTP(self.totp_secret).now()

            # Login
            data = self.smart_api.generateSession(
                self.client_id,
                self.password,
                totp,
            )

            if data.get("status"):
                self.auth_token = data["data"]["jwtToken"]
                self.feed_token = self.smart_api.getfeedToken()
                self._logged_in = True
                logger.info(f"✅ Angel One login successful for {self.client_id}")
                print(f"✅ Angel One connected — Client: {self.client_id}")
                return True
            else:
                logger.error(f"Angel One login failed: {data.get('message', 'Unknown error')}")
                print(f"❌ Angel One login failed: {data.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Angel One login error: {e}")
            print(f"❌ Angel One login error: {e}")
            return False

    def logout(self) -> None:
        """Logout from Angel One API."""
        if self.smart_api and self._logged_in:
            try:
                self.smart_api.terminateSession(self.client_id)
                self._logged_in = False
                logger.info("Angel One session terminated")
            except Exception as e:
                logger.warning(f"Error during logout: {e}")

    def get_ltp(self, stock_name: str) -> Optional[float]:
        """
        Get the Last Traded Price (live) for a stock.

        Args:
            stock_name: Short name like "TCS", "RELIANCE", "SBIN"

        Returns:
            Current price as float, or None on error.

        Example:
            price = angel.get_ltp("TCS")
            print(f"TCS: ₹{price}")
        """
        if not self._logged_in:
            logger.error("Not logged in. Call login() first.")
            return None

        token_info = ANGEL_TOKENS.get(stock_name.upper())
        if not token_info:
            logger.error(f"Unknown stock: {stock_name}. Add it to ANGEL_TOKENS dict.")
            return None

        try:
            data = self.smart_api.ltpData(
                exchange=token_info["exchange"],
                tradingsymbol=token_info["symbol"],
                symboltoken=token_info["token"],
            )

            if data.get("status"):
                ltp = data["data"]["ltp"]
                return float(ltp)
            else:
                logger.error(f"LTP failed for {stock_name}: {data.get('message')}")
                return None

        except Exception as e:
            logger.error(f"Error getting LTP for {stock_name}: {e}")
            return None

    def get_historical(
        self,
        stock_name: str,
        interval: str = "ONE_DAY",
        days: int = 30,
        save_to_db: bool = True,
    ) -> List[Dict]:
        """
        Get historical candle data from Angel One.

        Args:
            stock_name: Short name like "TCS", "RELIANCE"
            interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                      THIRTY_MINUTE, ONE_HOUR, ONE_DAY
            days: Number of days of history
            save_to_db: Whether to save to the database

        Returns:
            List of candle dicts with date, open, high, low, close, volume.
        """
        if not self._logged_in:
            logger.error("Not logged in. Call login() first.")
            return []

        token_info = ANGEL_TOKENS.get(stock_name.upper())
        if not token_info:
            logger.error(f"Unknown stock: {stock_name}")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
            to_date = datetime.now().strftime("%Y-%m-%d 15:30")

            params = {
                "exchange": token_info["exchange"],
                "symboltoken": token_info["token"],
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date,
            }

            data = self.smart_api.getCandleData(params)

            if not data.get("status") or not data.get("data"):
                logger.error(f"No candle data for {stock_name}: {data.get('message')}")
                return []

            candles = []
            db_rows = []

            for candle in data["data"]:
                # Angel One candle format: [timestamp, open, high, low, close, volume]
                ts, o, h, l, c, v = candle

                # Parse timestamp
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S") if interval != "ONE_DAY" else None

                candle_dict = {
                    "date": date_str,
                    "time": time_str,
                    "open": round(float(o), 2),
                    "high": round(float(h), 2),
                    "low": round(float(l), 2),
                    "close": round(float(c), 2),
                    "volume": int(v),
                }
                candles.append(candle_dict)

                # Map interval for DB
                db_interval = "1d" if interval == "ONE_DAY" else interval.lower()
                yf_symbol = f"{stock_name}.NS"

                db_rows.append((
                    yf_symbol, "NSE", date_str, time_str,
                    candle_dict["open"], candle_dict["high"],
                    candle_dict["low"], candle_dict["close"],
                    candle_dict["volume"], db_interval,
                ))

            # Save to DB
            if save_to_db and db_rows:
                inserted = insert_prices_batch(db_rows)
                logger.info(f"Angel One: {stock_name} — {inserted} candles saved")

            return candles

        except Exception as e:
            logger.error(f"Error getting historical data for {stock_name}: {e}")
            return []

    def get_multiple_ltp(self, stock_names: List[str]) -> Dict[str, float]:
        """
        Get live prices for multiple stocks at once.

        Args:
            stock_names: List of short names ["TCS", "RELIANCE", "SBIN"]

        Returns:
            Dict mapping stock name to LTP, e.g. {"TCS": 3450.0, "RELIANCE": 1420.5}
        """
        prices = {}
        for name in stock_names:
            ltp = self.get_ltp(name)
            if ltp is not None:
                prices[name] = ltp
        return prices


def collect_live_prices() -> Dict[str, float]:
    """
    Convenience function: login, fetch all prices, logout.

    Returns:
        Dict of stock_name → live price.
    """
    angel = AngelCollector()
    if not angel.login():
        return {}

    try:
        stock_names = list(ANGEL_TOKENS.keys())
        # Exclude index symbols
        stock_names = [s for s in stock_names if s not in ["NIFTY50", "NIFTY500", "SENSEX"]]

        prices = angel.get_multiple_ltp(stock_names)
        print(f"\n📊 Live Prices ({len(prices)} stocks):")
        for name, price in sorted(prices.items()):
            print(f"   {name:15s}: ₹{price:,.2f}")
        return prices

    finally:
        angel.logout()


# ==========================================
# Quick test
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()

    angel = AngelCollector()
    if angel.login():
        # Test live price
        ltp = angel.get_ltp("TCS")
        if ltp:
            print(f"\n📈 TCS Live Price: ₹{ltp:,.2f}")

        # Test historical data
        print("\n📊 Fetching 30 days of RELIANCE candles...")
        candles = angel.get_historical("RELIANCE", interval="ONE_DAY", days=30)
        if candles:
            print(f"   Got {len(candles)} candles")
            print(f"   Latest: {candles[-1]}")

        angel.logout()
    else:
        print("\nTo use Angel One, fill in your credentials in .env:")
        print("   ANGEL_CLIENT_ID=M123456")
        print("   ANGEL_PASSWORD=your_password")
        print("   ANGEL_TOTP_SECRET=your_totp_key")
