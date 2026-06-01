"""
Retry failed stocks from a previous import run.
Usage: python retry_failed.py SYM1 SYM2 ...
"""
import json, logging, os, sys, time
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from database.db import get_connection, init_database, insert_prices_batch, get_latest_date, _execute
from update_stocks_angel import fetch_candles

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOKENS_FILE = os.path.join(os.path.dirname(__file__), "data", "angel_tokens.json")
RATE_LIMIT_SECS = 5.0  # longer delay for persistent rate-limit failures

def angel_login():
    api_key   = os.getenv("ANGEL_API_KEY", "")
    client_id = os.getenv("ANGEL_CLIENT_ID", "")
    mpin      = os.getenv("ANGEL_MPIN", "") or os.getenv("ANGEL_PASSWORD", "")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
    smart_api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    data = smart_api.generateSession(client_id, mpin, totp)
    if not data.get("status"):
        print(f"❌ Login failed: {data.get('message')}")
        sys.exit(1)
    print(f"✅ Angel One connected — {client_id}")
    return smart_api

init_database()
token_map = json.load(open(TOKENS_FILE))

if len(sys.argv) == 3 and sys.argv[1] == "--file":
    failed_symbols = open(sys.argv[2]).read().split()
else:
    failed_symbols = sys.argv[1:]

if not failed_symbols:
    print("No symbols provided.")
    sys.exit(0)

# Filter token map to only requested symbols
targets = {s: token_map[s] for s in failed_symbols if s in token_map}
missing = [s for s in failed_symbols if s not in token_map]
if missing:
    print(f"⚠️  Not in token map: {missing}")

print(f"\n🔁 Retrying {len(targets)} stocks...\n")
smart_api = angel_login()

success, failed, total_rows = 0, 0, 0
failed_list = []

for idx, (symbol, info) in enumerate(targets.items(), 1):
    ns_symbol = f"{symbol}.NS"
    latest_str = get_latest_date(ns_symbol)
    from datetime import datetime, timedelta, date as date_type
    if latest_str:
        latest = datetime.strptime(latest_str, "%Y-%m-%d").date()
        days_missing = (date_type.today() - latest).days + 1
    else:
        days_missing = 400

    try:
        rows = fetch_candles(smart_api, symbol=symbol, token=info["token"], exchange="NSE", days=days_missing)
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ["token", "session", "invalid", "unauthorized"]):
            logger.warning("Session expired, reconnecting...")
            smart_api = angel_login()
            try:
                rows = fetch_candles(smart_api, symbol=symbol, token=info["token"], exchange="NSE", days=days_missing)
            except Exception as e2:
                logger.error(f"[{idx}/{len(targets)}] {symbol} failed after reconnect: {e2}")
                failed += 1; failed_list.append(symbol)
                time.sleep(RATE_LIMIT_SECS); continue
        else:
            logger.warning(f"[{idx}/{len(targets)}] {symbol} FAILED: {e}")
            failed += 1; failed_list.append(symbol)
            time.sleep(RATE_LIMIT_SECS); continue

    if rows:
        inserted = insert_prices_batch(rows, sync=False)
        total_rows += inserted
        success += 1
        logger.info(f"[{idx}/{len(targets)}] {symbol:15s} +{inserted} rows")
    else:
        success += 1
        logger.info(f"[{idx}/{len(targets)}] {symbol:15s} already up to date")

    time.sleep(RATE_LIMIT_SECS)

try:
    smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
except Exception:
    pass

print(f"\n{'='*60}")
print(f"✅ Retry complete!")
print(f"   Successful: {success}  |  Failed: {failed}  |  New rows: {total_rows}")
if failed_list:
    print(f"   Still failing: {failed_list}")

conn = get_connection()
r = _execute(conn, "SELECT COUNT(DISTINCT symbol) as syms, COUNT(*) as rows, MAX(date) as newest FROM prices WHERE interval = '1d'").fetchone()
print(f"\n📊 DB state: {r[0]} symbols, {r[1]:,} rows, latest date: {r[2]}")
conn.close()
