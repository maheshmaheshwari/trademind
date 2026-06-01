"""Check the latest stock data dates in the database."""
from collections import Counter
from database.db import get_connection, _execute, _rows_to_dicts

conn = get_connection()

print("=== OVERALL STATS ===")
overall = _execute(conn,
    "SELECT COUNT(DISTINCT symbol) as total_symbols, MIN(date) as earliest, MAX(date) as latest, COUNT(*) as total_rows FROM prices WHERE interval = '1d'"
).fetchone()
print(f"  Total symbols: {overall[0]}")
print(f"  Earliest date: {overall[1]}")
print(f"  Latest date:   {overall[2]}")
print(f"  Total rows:    {overall[3]}")

print()
print("=== STOCKS WITH OLDEST DATA (bottom 10) ===")
rows = _execute(conn,
    "SELECT symbol, MAX(date) as latest_date FROM prices WHERE interval = '1d' GROUP BY symbol ORDER BY latest_date ASC LIMIT 10"
).fetchall()
for r in rows:
    print(f"  {r[0]:20s} | Last: {r[1]}")

print()
print("=== STOCKS WITH NEWEST DATA (top 10) ===")
rows2 = _execute(conn,
    "SELECT symbol, MAX(date) as latest_date FROM prices WHERE interval = '1d' GROUP BY symbol ORDER BY latest_date DESC LIMIT 10"
).fetchall()
for r in rows2:
    print(f"  {r[0]:20s} | Last: {r[1]}")

print()
print("=== DATE DISTRIBUTION (how many stocks per latest date) ===")
rows3 = _execute(conn,
    "SELECT MAX(date) as latest_date, COUNT(*) as num_stocks FROM prices WHERE interval = '1d' GROUP BY symbol HAVING MAX(date) IS NOT NULL"
).fetchall()
date_counts = Counter()
for r in rows3:
    date_counts[r[0]] += 1
for date, count in sorted(date_counts.items(), reverse=True)[:10]:
    print(f"  {date}: {count} stocks")

print()
print("=== TRADE SIGNALS ===")
try:
    sig = _execute(conn, "SELECT MIN(generated_date), MAX(generated_date), COUNT(*) FROM trade_signals").fetchone()
    print(f"  Earliest: {sig[0]}")
    print(f"  Latest:   {sig[1]}")
    print(f"  Total:    {sig[2]}")
except Exception:
    print("  No trade_signals table found")

conn.close()
