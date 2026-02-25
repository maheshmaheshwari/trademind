#!/usr/bin/env python3
"""
Technical Indicators Calculator

Calculates historical technical indicators from OHLC data for all stocks
and stores them in the database. Uses pandas-ta for indicator calculations.

Usage:
    # Calculate indicators for all stocks with existing OHLC data
    python scripts/calculate_indicators.py
    
    # Specific stocks only
    python scripts/calculate_indicators.py --symbols RELIANCE TCS INFY
    
    # Limit number of stocks (for testing)
    python scripts/calculate_indicators.py --limit 5
    
    # Recalculate (overwrite existing)
    python scripts/calculate_indicators.py --recalculate
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_ta as ta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from app.database import sync_engine
from app.models import Stock, OHLCData, TechnicalIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calculate_indicators.log')
    ]
)
logger = logging.getLogger(__name__)


def load_ohlc_dataframe(stock_id: int, session: Session) -> Optional[pd.DataFrame]:
    """Load OHLC data for a stock into a pandas DataFrame."""
    ohlc_records = session.query(OHLCData).filter(
        OHLCData.stock_id == stock_id
    ).order_by(OHLCData.date).all()
    
    if not ohlc_records:
        return None
    
    data = {
        'date': [r.date for r in ohlc_records],
        'open': [r.open for r in ohlc_records],
        'high': [r.high for r in ohlc_records],
        'low': [r.low for r in ohlc_records],
        'close': [r.close for r in ohlc_records],
        'volume': [r.volume for r in ohlc_records]
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for the DataFrame."""
    
    # Returns
    df['returns_1d'] = df['close'].pct_change(1) * 100
    df['returns_5d'] = df['close'].pct_change(5) * 100
    df['returns_20d'] = df['close'].pct_change(20) * 100
    
    # RSI
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # EMAs
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    
    # Volatility (20-day standard deviation of returns)
    df['volatility_20'] = df['returns_1d'].rolling(window=20).std()
    
    # ATR
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_14'] = atr
    
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None and 'ADX_14' in adx.columns:
        df['adx_14'] = adx['ADX_14']
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        if 'MACD_12_26_9' in macd.columns:
            df['macd'] = macd['MACD_12_26_9']
        if 'MACDs_12_26_9' in macd.columns:
            df['macd_signal'] = macd['MACDs_12_26_9']
        if 'MACDh_12_26_9' in macd.columns:
            df['macd_histogram'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        if 'BBU_20_2.0' in bbands.columns:
            df['bb_upper'] = bbands['BBU_20_2.0']
        if 'BBM_20_2.0' in bbands.columns:
            df['bb_middle'] = bbands['BBM_20_2.0']
        if 'BBL_20_2.0' in bbands.columns:
            df['bb_lower'] = bbands['BBL_20_2.0']
        if 'BBB_20_2.0' in bbands.columns:
            df['bb_width'] = bbands['BBB_20_2.0']
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Market regime (based on EMA crossovers and trend)
    def determine_regime(row):
        if pd.isna(row.get('ema_50')) or pd.isna(row.get('ema_200')):
            return None
        if row['ema_50'] > row['ema_200'] * 1.02:  # 2% above
            return 'Bull'
        elif row['ema_50'] < row['ema_200'] * 0.98:  # 2% below
            return 'Bear'
        else:
            return 'Sideways'
    
    df['market_regime'] = df.apply(determine_regime, axis=1)
    
    return df


def save_indicators_to_db(stock_id: int, df: pd.DataFrame, session: Session, recalculate: bool = False) -> int:
    """Save calculated indicators to the database."""
    saved_count = 0
    
    for idx, row in df.iterrows():
        indicator_date = idx if isinstance(idx, date) else idx.date()
        
        # Skip rows with too many NaN values (early days without enough history)
        if pd.isna(row.get('rsi_14')) and pd.isna(row.get('ema_20')):
            continue
        
        # Check if record exists
        existing = session.query(TechnicalIndicator).filter(
            TechnicalIndicator.stock_id == stock_id,
            TechnicalIndicator.date == indicator_date
        ).first()
        
        if existing and not recalculate:
            continue
        
        if existing:
            # Update existing
            indicator = existing
        else:
            # Create new
            indicator = TechnicalIndicator(stock_id=stock_id, date=indicator_date)
            session.add(indicator)
        
        # Set all indicator values
        indicator.returns_1d = row.get('returns_1d') if pd.notna(row.get('returns_1d')) else None
        indicator.returns_5d = row.get('returns_5d') if pd.notna(row.get('returns_5d')) else None
        indicator.returns_20d = row.get('returns_20d') if pd.notna(row.get('returns_20d')) else None
        indicator.rsi_14 = row.get('rsi_14') if pd.notna(row.get('rsi_14')) else None
        indicator.ema_20 = row.get('ema_20') if pd.notna(row.get('ema_20')) else None
        indicator.ema_50 = row.get('ema_50') if pd.notna(row.get('ema_50')) else None
        indicator.ema_200 = row.get('ema_200') if pd.notna(row.get('ema_200')) else None
        indicator.volatility_20 = row.get('volatility_20') if pd.notna(row.get('volatility_20')) else None
        indicator.atr_14 = row.get('atr_14') if pd.notna(row.get('atr_14')) else None
        indicator.adx_14 = row.get('adx_14') if pd.notna(row.get('adx_14')) else None
        indicator.macd = row.get('macd') if pd.notna(row.get('macd')) else None
        indicator.macd_signal = row.get('macd_signal') if pd.notna(row.get('macd_signal')) else None
        indicator.macd_histogram = row.get('macd_histogram') if pd.notna(row.get('macd_histogram')) else None
        indicator.bb_upper = row.get('bb_upper') if pd.notna(row.get('bb_upper')) else None
        indicator.bb_middle = row.get('bb_middle') if pd.notna(row.get('bb_middle')) else None
        indicator.bb_lower = row.get('bb_lower') if pd.notna(row.get('bb_lower')) else None
        indicator.bb_width = row.get('bb_width') if pd.notna(row.get('bb_width')) else None
        indicator.volume_sma_20 = row.get('volume_sma_20') if pd.notna(row.get('volume_sma_20')) else None
        indicator.volume_ratio = row.get('volume_ratio') if pd.notna(row.get('volume_ratio')) else None
        indicator.market_regime = row.get('market_regime')
        
        saved_count += 1
    
    return saved_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Calculate technical indicators from OHLC data')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of stocks')
    parser.add_argument('--recalculate', action='store_true', help='Recalculate and overwrite existing')
    
    args = parser.parse_args()
    
    # Create database session
    Session = sessionmaker(bind=sync_engine)
    session = Session()
    
    try:
        # Get stocks to process
        query = session.query(Stock)
        
        if args.symbols:
            query = query.filter(Stock.symbol.in_(args.symbols))
        
        stocks = query.all()
        
        if args.limit:
            stocks = stocks[:args.limit]
        
        logger.info(f"Processing {len(stocks)} stocks for technical indicators")
        logger.info("=" * 50)
        
        success_count = 0
        fail_count = 0
        total_indicators = 0
        
        for i, stock in enumerate(stocks):
            logger.info(f"[{i+1}/{len(stocks)}] Calculating indicators for {stock.symbol}...")
            
            # Load OHLC data
            df = load_ohlc_dataframe(stock.id, session)
            
            if df is None or len(df) < 50:  # Need at least 50 days for meaningful indicators
                logger.warning(f"  Skipped: Insufficient OHLC data ({len(df) if df is not None else 0} days)")
                fail_count += 1
                continue
            
            try:
                # Calculate indicators
                df = calculate_indicators(df)
                
                # Save to database
                saved = save_indicators_to_db(stock.id, df, session, args.recalculate)
                session.commit()
                
                success_count += 1
                total_indicators += saved
                logger.info(f"  ✓ Saved {saved} indicator records ({len(df)} days of data)")
                
            except Exception as e:
                session.rollback()
                logger.error(f"  ✗ Error: {e}")
                fail_count += 1
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("Technical indicator calculation complete!")
        logger.info(f"Stocks: {len(stocks)} | Success: {success_count} | Failed: {fail_count}")
        logger.info(f"Total indicator records saved: {total_indicators}")
        logger.info("=" * 50)
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
