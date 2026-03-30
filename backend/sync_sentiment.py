import sys
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from scheduler.jobs import collect_news_job
from database.db import get_remote_turso_connection

def run_sync():
    print("1. Reading collected data from local DB...")
    local_conn = sqlite3.connect("nifty500.db")
    local_cursor = local_conn.cursor()
    
    local_cursor.execute("SELECT headline, source, published_at, symbol, sentiment, confidence, url FROM news_sentiment")
    news_rows = local_cursor.fetchall()
    
    local_cursor.execute('''SELECT date, symbol, avg_sentiment, news_count, positive_count, 
                            negative_count, neutral_count, max_positive, max_negative, 
                            avg_confidence, source FROM news_daily_sentiment''')
    daily_rows = local_cursor.fetchall()
    
    print(f"Found {len(news_rows)} news items and {len(daily_rows)} daily summaries.")
    
    if not news_rows and not daily_rows:
        print("Nothing to sync.")
        return
        
    print("2. Connecting to Turso...")
    remote_conn = get_remote_turso_connection()
    remote_cursor = remote_conn.cursor()
    
    # Insert news_sentiment
    if news_rows:
        print(f"Pushing {len(news_rows)} rows to news_sentiment table in Turso...")
        for row in news_rows:
            try:
                # We use INSERT OR IGNORE if there is a unique constraint, else REPLACE or just INSERT
                remote_cursor.execute('''
                    INSERT INTO news_sentiment 
                    (headline, source, published_at, symbol, sentiment, confidence, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', row)
            except Exception as e:
                # If unique constraint fails, it just skips
                pass
                
    # Insert news_daily_sentiment
    if daily_rows:
        print(f"Pushing {len(daily_rows)} rows to news_daily_sentiment table in Turso...")
        for row in daily_rows:
            try:
                remote_cursor.execute('''
                    INSERT INTO news_daily_sentiment
                    (date, symbol, avg_sentiment, news_count, positive_count, negative_count, neutral_count, max_positive, max_negative, avg_confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', row)
            except Exception as e:
                try:
                    # Update if it already exists
                    remote_cursor.execute('''
                        UPDATE news_daily_sentiment SET 
                        avg_sentiment=?, news_count=?, positive_count=?, negative_count=?, 
                        neutral_count=?, max_positive=?, max_negative=?, avg_confidence=?, source=?
                        WHERE date=? AND symbol=?
                    ''', (row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[0], row[1]))
                except:
                    pass
                
    remote_conn.commit()
    print("✅ Sync complete!")

if __name__ == "__main__":
    run_sync()
