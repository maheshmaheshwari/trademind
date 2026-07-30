"""
Seed the nifty_constituents DB table from the local data.nifty500_full module.

The Nifty 500 constituent list (symbol/name/sector) used to be imported directly
from data/nifty500_full.py at runtime — but data/ is gitignored and excluded
from the HF Space deploy, so /api/market/sectors crashed in production
(ModuleNotFoundError). The list now lives in the DB; this one-shot script loads
it there. data/nifty500_full.py remains the local source of truth (regenerated
from NSE); re-run this after it changes.

Run against test first, then prod (APP_ENV controls the target):
    APP_ENV=test python scripts/seed_nifty_constituents.py
    python scripts/seed_nifty_constituents.py          # prod
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nifty500_full import NIFTY_500_STOCKS  # noqa: E402  (local source list)
from database.db import upsert_nifty_constituents, get_nifty_constituents  # noqa: E402

n = upsert_nifty_constituents(NIFTY_500_STOCKS)
total = len(get_nifty_constituents())
print(f"upserted {n} constituents · table now has {total} rows")
