"""
Nifty 500 AI — Main Entry Point (CLI)

Command-line interface for managing the data pipeline.

Commands:
    python main.py setup      — Create database + download 2 years of data
    python main.py collect    — Run one full collection cycle manually
    python main.py server     — Start FastAPI server on port 8000
    python main.py schedule   — Start automated scheduler
    python main.py status     — Show database statistics
    python main.py signals    — Print today's top 10 BUY signals
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO")

# Initialise date-based rotating file logging (writes to logs/YYYY-MM-DD.log)
from api.logging_setup import setup_logging
setup_logging(log_dir="logs", level=log_level)

logger = logging.getLogger("nifty500-ai")


def cmd_setup():
    """
    First-time setup: create database and download 2 years of historical data.
    This takes approximately 15-20 minutes for 50 stocks.
    """
    print("\n" + "=" * 60)
    print("🚀 Nifty 500 AI — First Time Setup")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n📦 Step 1: Creating database...")
    from database.db import init_database
    init_database()

    # Step 2: Download historical price data (2 years)
    print("\n📊 Step 2: Downloading 2 years of price data...")
    print("   This will take about 15-20 minutes. Please be patient.\n")
    from collectors.price_collector import collect_all_stocks, collect_index_data
    collect_all_stocks(interval="1d", period="2y")

    # Step 3: Download index data
    print("\n📈 Step 3: Downloading index data...")
    collect_index_data(period="2y")

    # Step 4: Calculate indicators
    print("\n🔬 Step 4: Calculating technical indicators...")
    from analysis.signals import process_all_stocks
    process_all_stocks()

    # Step 5: Show status
    print("\n📊 Step 5: Final status...")
    cmd_status()

    print("\n✅ Setup complete! Next steps:")
    print("   1. Start the API server:  python main.py server")
    print("   2. Open dashboard:        http://localhost:8000/docs")
    print("   3. Start scheduler:       python main.py schedule")
    print("=" * 60 + "\n")


def cmd_collect():
    """Run one full manual collection cycle."""
    print("\n🔄 Running full collection cycle...\n")

    # Collect prices
    from collectors.price_collector import collect_eod_data, collect_index_data
    collect_eod_data()
    collect_index_data(period="5d")

    # Calculate indicators
    from analysis.signals import process_all_stocks
    process_all_stocks()

    # Collect news (optional — needs API key)
    try:
        from collectors.news_collector import collect_all_news
        collect_all_news()
    except Exception as e:
        print(f"⚠️  News collection skipped: {e}")

    # Collect FII/DII
    try:
        from collectors.fii_collector import collect_fii_dii_data
        collect_fii_dii_data()
    except Exception as e:
        print(f"⚠️  FII/DII collection skipped: {e}")

    print("\n✅ Collection cycle complete!")


def cmd_server():
    """Start the FastAPI server."""
    port = int(os.getenv("PORT", 8000))

    print(f"\n🚀 Starting Nifty 500 AI API Server on port {port}...")
    print(f"   📖 API Docs:  http://localhost:{port}/docs")
    print(f"   🔄 ReDoc:     http://localhost:{port}/redoc")
    print(f"   ❤️  Health:    http://localhost:{port}/api/health")
    print(f"\n   Press Ctrl+C to stop.\n")

    from api.logging_setup import get_uvicorn_log_config
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        # Pass a custom log_config so uvicorn does NOT reset the root logger.
        # Our DailyFileHandler (added in server.py + startup event) then
        # survives alongside uvicorn's own uvicorn/uvicorn.error loggers.
        log_config=get_uvicorn_log_config(log_level),
        access_log=False,   # our middleware handles access logging
    )


def cmd_schedule():
    """Start the automated scheduler."""
    print("\n📅 Starting Nifty 500 AI Scheduler...\n")
    from scheduler.jobs import start_scheduler
    start_scheduler()


def cmd_status():
    """Show database statistics and last update times."""
    from database.db import get_db_stats, get_connection, release_connection, _execute

    print("\n📊 Nifty 500 AI — Database Status")
    print("=" * 50)

    stats = get_db_stats()
    for table, count in stats.items():
        print(f"  {table:25s}: {count:>8,} rows")

    # Show latest price date
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT MAX(date) as latest FROM prices WHERE interval='1d'").fetchone()
        if row and row[0]:
            print(f"\n  📅 Latest price data: {row[0]}")

        row = _execute(conn, "SELECT MAX(date) as latest FROM technical_indicators").fetchone()
        if row and row[0]:
            print(f"  📅 Latest indicators: {row[0]}")

        row = _execute(conn, "SELECT COUNT(DISTINCT symbol) as count FROM prices WHERE interval='1d'").fetchone()
        if row:
            print(f"  📈 Stocks tracked:    {row[0]}")
    finally:
        release_connection(conn)

    print("=" * 50 + "\n")


def cmd_signals():
    """Print today's top 10 BUY signals."""
    from database.db import get_top_signals

    print("\n🟢 Top 10 BUY Signals")
    print("=" * 70)

    buys = get_top_signals(signal_type="BUY", limit=10)
    if buys:
        print(f"  {'Symbol':15s} {'Signal':12s} {'Confidence':>10s}  {'Target':>10s}  {'Stop Loss':>10s}")
        print(f"  {'-'*15} {'-'*12} {'-'*10}  {'-'*10}  {'-'*10}")
        for s in buys:
            symbol = s.get("symbol", "?")
            signal = s.get("signal", "?")
            confidence = s.get("confidence", 0)
            target = s.get("target_price")
            stop = s.get("stop_loss")
            target_str = f"₹{target:.2f}" if target else "—"
            stop_str = f"₹{stop:.2f}" if stop else "—"
            print(f"  {symbol:15s} {signal:12s} {confidence:>9.1f}%  {target_str:>10s}  {stop_str:>10s}")
    else:
        print("  No BUY signals found. Run: python main.py collect")

    print("\n🔴 Top 10 SELL Signals")
    print("=" * 70)

    sells = get_top_signals(signal_type="SELL", limit=10)
    if sells:
        print(f"  {'Symbol':15s} {'Signal':12s} {'Confidence':>10s}")
        print(f"  {'-'*15} {'-'*12} {'-'*10}")
        for s in sells:
            symbol = s.get("symbol", "?")
            signal = s.get("signal", "?")
            confidence = s.get("confidence", 0)
            print(f"  {symbol:15s} {signal:12s} {confidence:>9.1f}%")
    else:
        print("  No SELL signals found.")

    print("=" * 70 + "\n")


def main():
    """Parse command line arguments and run the appropriate command."""
    if len(sys.argv) < 2:
        print("\n📌 Nifty 500 AI Trading Data Pipeline")
        print("=" * 45)
        print("\nUsage: python main.py <command>\n")
        print("Commands:")
        print("  setup     — First-time setup (create DB + download data)")
        print("  collect   — Run one manual collection cycle")
        print("  server    — Start FastAPI server on port 8000")
        print("  schedule  — Start automated scheduler")
        print("  status    — Show database statistics")
        print("  signals   — Print today's top BUY/SELL signals")
        print()
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        "setup": cmd_setup,
        "collect": cmd_collect,
        "server": cmd_server,
        "schedule": cmd_schedule,
        "status": cmd_status,
        "signals": cmd_signals,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"\n❌ Unknown command: '{command}'")
        print(f"   Available: {', '.join(commands.keys())}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
