"""
One-time migration: add volume tracking columns to trade_signals table.

Adds:
  - recommended_volume INTEGER  — max qty across all users for this signal
  - consumed_volume INTEGER DEFAULT 0 — total qty placed so far

Run once:
    cd backend && source venv/bin/activate && python database/migrate_db_volumes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection

def migrate():
    conn = get_connection()
    try:
        # Check if columns already exist
        cols = [row[1] for row in conn.execute("PRAGMA table_info(trade_signals)").fetchall()]

        if "recommended_volume" not in cols:
            conn.execute("ALTER TABLE trade_signals ADD COLUMN recommended_volume INTEGER")
            print("✅ Added recommended_volume column")
        else:
            print("ℹ️  recommended_volume already exists")

        if "consumed_volume" not in cols:
            conn.execute("ALTER TABLE trade_signals ADD COLUMN consumed_volume INTEGER DEFAULT 0")
            print("✅ Added consumed_volume column")
        else:
            print("ℹ️  consumed_volume already exists")

        conn.commit()
        print("✅ Migration complete!")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
