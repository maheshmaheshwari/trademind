#!/usr/bin/env python3
"""
Clear Dummy Data Script

Clears existing dummy/seed data from the database before running full scrapes.
Preserves stock records but clears OHLC, indicators, and financial data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import sessionmaker
from app.database import sync_engine
from app.models import Stock, OHLCData, TechnicalIndicator, FinancialData, GrowthMetrics


def clear_data(preserve_stocks: bool = True, confirm: bool = False):
    """Clear dummy data from database."""
    
    Session = sessionmaker(bind=sync_engine)
    session = Session()
    
    try:
        # Count existing records
        print("=== Current Database Records ===")
        stock_count = session.query(Stock).count()
        ohlc_count = session.query(OHLCData).count()
        indicator_count = session.query(TechnicalIndicator).count()
        financial_count = session.query(FinancialData).count()
        growth_count = session.query(GrowthMetrics).count()
        
        print(f"Stocks:              {stock_count}")
        print(f"OHLC Data:           {ohlc_count}")
        print(f"Technical Indicators: {indicator_count}")
        print(f"Financial Data:      {financial_count}")
        print(f"Growth Metrics:      {growth_count}")
        print()
        
        if not confirm:
            print("⚠️  This will DELETE all the above data (except Stocks if --preserve-stocks).")
            response = input("Are you sure? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        # Delete data
        print("\n🗑️  Deleting data...")
        
        # Clear in order (respect foreign keys)
        deleted_indicators = session.query(TechnicalIndicator).delete()
        print(f"   Deleted {deleted_indicators} technical indicators")
        
        deleted_financial = session.query(FinancialData).delete()
        print(f"   Deleted {deleted_financial} financial data records")
        
        deleted_growth = session.query(GrowthMetrics).delete()
        print(f"   Deleted {deleted_growth} growth metrics records")
        
        deleted_ohlc = session.query(OHLCData).delete()
        print(f"   Deleted {deleted_ohlc} OHLC data records")
        
        if not preserve_stocks:
            deleted_stocks = session.query(Stock).delete()
            print(f"   Deleted {deleted_stocks} stock records")
        
        session.commit()
        print("\n✅ Data cleared successfully!")
        
        # Show remaining
        print("\n=== Remaining Records ===")
        print(f"Stocks: {session.query(Stock).count()}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear dummy data from database')
    parser.add_argument('--preserve-stocks', action='store_true', default=True,
                        help='Keep stock records (default: True)')
    parser.add_argument('--include-stocks', action='store_true',
                        help='Also delete stock records')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    preserve_stocks = not args.include_stocks
    clear_data(preserve_stocks=preserve_stocks, confirm=args.yes)
