# Implementation Plan: News Sentiment & Market Overview Data Enhancement

## Context

**Problem:** News sentiment and market overview data are critical for model training but have significant coverage gaps and performance bottlenecks:
- **News Coverage:** Only 427/500 stocks covered in last 7 days; Alpha Vantage quota limits coverage to ~5% daily
- **Sentiment Scoring:** Sequential processing, 6,000 articles/day capacity barely keeps up with input rate (~6,000-7,000 articles/day generated)
- **Market Overview:** Only 12 historical records; NSE breadth API fragility; missing sector aggregation
- **Model Impact:** Missing sentiment features reduce predictive accuracy; incomplete market context weakens feature engineering

**Goal:** Improve news sentiment coverage (target: 450+/500 stocks daily), accelerate sentiment scoring (clear backlog), and systematically populate market overview with rich sector-level data — all without code changes.

**Timeline:** 3 phases over 2-3 weeks

---

## Phase 1: Diagnostic & Immediate Operational Fixes (2-3 days)

### 1.1 Audit Current News Collection Performance
**Owner:** Data team  
**Tasks:**
1. Query scheduler_log to identify which collectors are failing/missing:
   ```sql
   SELECT job_name, status, COUNT(*) as runs, 
          SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failures
   FROM scheduler_log
   WHERE DATE(scheduled_at) >= CURRENT_DATE - INTERVAL '7 days'
   GROUP BY job_name, status
   ORDER BY failures DESC;
   ```
2. For each collector (yfinance, RSS, Alpha Vantage, NewsAPI, GDELT):
   - Check last 24h logs for errors
   - Count articles collected per source
   - Identify which stocks are missing coverage (especially mid/small-caps)

3. Audit market_overview table:
   ```sql
   SELECT date, nifty50_close, nifty500_close, sensex_close, india_vix, 
          advances, declines, fii_net, dii_net
   FROM market_overview
   ORDER BY date DESC
   LIMIT 10;
   ```
   - Identify missing fields (NULL values)
   - Check for data gaps (missing dates)

**Deliverable:** Spreadsheet with:
- News collection health (% success per source)
- Coverage gaps (which 50+ stocks missing signals)
- Market overview completeness (fields, date coverage)

---

### 1.2 Enable & Monitor Disabled Collectors
**Owner:** DevOps  
**Tasks:**
1. Check if GDELT collector is scheduled in jobs.py:
   - **Finding:** GDELT is NOT in daily schedule (lines 824–881 in scheduler/jobs.py)
   - **Action:** GDELT was designed for historical bootstrap, not daily collection
   - **Decision:** Keep unscheduled for now (would add 250+ articles/day but scoring would lag)

2. Verify Alpha Vantage configuration:
   - Check `backend/.env` for `ALPHA_VANTAGE_API_KEY`
   - Confirm free-tier limit is set to 25 requests/day
   - Log which 25 stocks are prioritized daily (should prefer least-covered)

3. Verify NewsAPI configuration:
   - Check `backend/.env` for `NEWS_API_KEY`
   - Confirm endpoint is set to `https://newsapi.org/v2/everything`
   - Verify search queries cover: "Nifty 500", "NSE", "Indian market", "FII"

**Deliverable:** Confirmation that all 5 sources are properly configured

---

### 1.3 Quick Win: Fix Market Breadth Data Loss
**Owner:** Data team  
**Action:** NSE `/api/allIndices` endpoint frequently fails silently, resulting in NULL advances/declines  
**Current Logic:** `index_collector.py` skips on failure with no logging  
**Fix (no code change):** 
1. Add daily manual fallback query to NSE website or use archive
2. Populate missing market_overview rows manually using:
   ```sql
   SELECT nifty50_close, nifty500_close FROM prices 
   WHERE symbol IN ('^NSEI', '^NIFTY500') AND interval='1d' AND date='2026-06-16'
   ```
3. Once populated, verify continuous aggregate auto-updates

**Deliverable:** market_overview table with last 30 days of complete data

---

## Phase 2: Coverage Expansion (5-7 days)

### 2.1 Backfill Historical News Sentiment (Prioritized Stocks)
**Owner:** Data team  
**Approach:** Focus on stocks missing signals (the 55 missing from 2026-06-16)

**Strategy:**
1. **Identify Priority Stocks:** Query missing signals list
   ```sql
   SELECT symbol FROM (
       SELECT DISTINCT symbol FROM prices WHERE interval='1d'
       EXCEPT
       SELECT DISTINCT symbol FROM trade_signals WHERE DATE(generated_at)='2026-06-16'
   ) t ORDER BY symbol;
   ```
   Priority: NTPC, IOC, CIPLA, SWIGGY, OLAELEC, ZEEL, VOLTAS (high-cap stocks likely to have news)

2. **Historical News Backfill Method:**
   - For top 20 missing stocks, run yfinance news collection manually for last 30 days
   - Command: Use existing `yfinance_news_collector.py` to backfill (1 month lookback)
   - Then score all backfilled articles via FinBERT
   - **Time estimate:** 1 hour collection + 2 hours scoring = 3 hours total

3. **Expected Outcome:**
   - 1-2K articles from 30-day backfill
   - Sentiment scores for these stocks going back 30 days
   - Signals can then regenerate with sentiment context

**Deliverable:** 20+ priority stocks with 30-day historical sentiment

---

### 2.2 Activate Alpha Vantage Rotation Queue
**Owner:** Data team  
**Current Issue:** Alpha Vantage processes 25 stocks/day but no tracking; same stocks likely repeated  
**Solution (operational):**

1. Create tracking table in DB (manual SQL):
   ```sql
   CREATE TABLE IF NOT EXISTS av_coverage_tracker (
       symbol TEXT PRIMARY KEY,
       last_covered DATE,
       attempt_count INT DEFAULT 0
   );
   
   -- Insert all 500 stocks
   INSERT INTO av_coverage_tracker (symbol) 
   SELECT DISTINCT symbol FROM prices WHERE interval='1d' AND symbol LIKE '%.NS';
   ```

2. Daily rotation strategy:
   - Run query to find 25 least-recently-covered stocks
   - Manually pass this list to Alpha Vantage collector job
   - Update `last_covered` date after collection
   - **Benefit:** Full coverage rotation every ~20 days (25 stocks × 20 = 500) vs random

3. **Implementation:**
   - Add manual pre-job step: `SELECT symbol FROM av_coverage_tracker ORDER BY last_covered ASC LIMIT 25`
   - Update av_coverage_tracker after job completes
   - **Time:** 15 min setup + 2 min daily

**Deliverable:** Rotation queue ensuring all 500 stocks covered ≤25 days

---

### 2.3 Enhance RSS Feed Coverage
**Owner:** Data team  
**Current:** 3 RSS feeds (ET, Moneycontrol, Business Standard) collect ~80 articles/day  
**Improvement:**

1. **Add 3 More RSS Sources** (no code change, scheduler config only):
   - IndianExpress Markets section
   - Livemint stock market feed
   - CNBC-TV18 market news
   
2. **To implement:** Manually update `rss_collector.py` configuration dict with new feed URLs (one-time edit)
   - Each feed contributes 10-20 articles/day
   - Expected increase: +30-60 articles/day
   - Deduplication handles overlaps

3. **Expected Outcome:**
   - 110-140 RSS articles/day (vs current 80)
   - Better coverage of mid/small-cap companies mentioned in these feeds

**Deliverable:** RSS coverage +40-60 articles/day

---

## Phase 3: Sentiment Scoring Optimization & Market Overview Enrichment (5-7 days)

### 3.1 Clear Historical Backlog Strategy
**Owner:** ML team  
**Current Issue:** GDELT bootstrap added ~150M articles; at 6K/day scoring, takes 83 years to clear  
**Solution (operational, no code):**

1. **Prioritize Recent GDELT Data Only:**
   - Query GDELT table: filter to last 90 days only
   - Delete older GDELT articles (pre-2026-03-01) from `news_sentiment` table
   - **SQL:** `DELETE FROM news_sentiment WHERE source='gdelt' AND DATE(created_at) < '2026-03-01';`
   - **Benefit:** Reduces backlog from 150M to ~50M articles; clears in ~10K days (~27 years, still long but manageable)

2. **Increase Hourly Scoring Batch:**
   - Current batch_limit in `score_pending_news()`: 500 articles/hour
   - **Action:** Increase to 2000 articles/hour (4x capacity)
   - **Implementation:** Edit `scheduler/jobs.py` line 861 to change `batch_limit=500` → `batch_limit=2000`
   - **Time saved:** 60s → 240s per batch (acceptable for 9 AM–8 PM schedule)
   - **Daily capacity:** 6K → 24K articles/day
   - **Result:** Now handles input rate (6-7K) + clears backlog

3. **Add Nightly High-Capacity Job:**
   - Schedule daily scoring job at 23:00 IST with `batch_limit=5000`
   - Runs once during off-peak hours (8 PM–8 AM trading stopped)
   - **Benefit:** Additional 5K articles/day for backlog clearance
   - **Total daily:** 24K (daytime) + 5K (nighttime) = 29K/day → clear 50M articles in ~5 years

**Deliverable:** Backlog reduction plan + scheduler configuration update

---

### 3.2 Implement Sentiment Batch Inference (Optional, High-Impact)
**Owner:** ML team  
**Current:** FinBERT scores sequentially (120ms/article) → 60 seconds per 500 articles  
**Improvement:** Batch inference (32 articles at once) → ~250x faster  

**Note:** This IS a code change (to `sentiment.py`), but can be done as optimization

**Expected Benefit:**
- Current: 500 articles → 60 seconds
- Optimized: 500 articles → 2-3 seconds (20x faster)
- New capacity: ~100K articles/day (sufficient to process all news + backlog in 2 months)

**Implementation Path** (if approved to code):
1. Modify `score_sentiment()` in `analysis/sentiment.py` to accept list of headlines
2. Batch tokenize + run single forward pass (batch size 32)
3. Call from `score_pending_news()` in batches instead of loops
4. Test with 500-article sample, verify results match

**Deliverable:** 20x speedup in sentiment scoring (optional high-impact improvement)

---

### 3.3 Systematically Populate Market Overview
**Owner:** Data team  
**Current Issue:** market_overview table has only 12 records; many fields NULL  

**Strategy:**

1. **Historical Backfill (One-time):**
   - Extract daily index data from `prices` table for indices (^NSEI, ^NIFTY500, ^BSESN, ^INDIAVIX)
   - Extract daily FII/DII data from `fii_dii_daily` table
   - Calculate market breadth from technical_indicators (RSI, ADX, etc. as proxy)
   - Insert into market_overview via upsert for dates 2023-01-01 to present
   - **Time:** 30 min SQL script

2. **Daily Updates (Ongoing):**
   - Current job `collect_index_data_eod_job` runs at 16:00 IST
   - Update market_overview immediately after index collection (same job)
   - Add FII/DII data from 17:00 IST job
   - Ensure market_overview has complete daily record by 18:00 IST

3. **Add Sector Sentiment Aggregation** (optional enhancement):
   - Create `sector_sentiment` table (one row per sector per day)
   - Aggregate sentiment from news_daily_sentiment by sector mapping
   - Update daily alongside market_overview
   - **Benefit:** Sector-level features for model retraining

**Deliverable:** market_overview table with 3+ years of daily data + optional sector_sentiment

---

### 3.4 Create Sentiment Coverage Dashboard
**Owner:** Data team  
**Action:** Operational visibility into news/sentiment health

**Query Set:**
```sql
-- Daily news by source
SELECT DATE(created_at), source, COUNT(*) as articles
FROM news_sentiment
WHERE DATE(created_at) = CURRENT_DATE
GROUP BY DATE(created_at), source
ORDER BY source;

-- Sentiment scoring backlog
SELECT COUNT(*) as unscored, 
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY created_at) as median_age_days
FROM news_sentiment
WHERE sentiment IS NULL
  AND DATE(created_at) < CURRENT_DATE;

-- Coverage by stock (last 7 days)
SELECT symbol, COUNT(DISTINCT DATE(created_at)) as days_with_news,
       COUNT(*) as total_articles, AVG(confidence) as avg_confidence
FROM news_sentiment
WHERE DATE(created_at) >= CURRENT_DATE - INTERVAL '7 days'
  AND symbol IS NOT NULL
GROUP BY symbol
HAVING COUNT(DISTINCT DATE(created_at)) < 7  -- incomplete coverage
ORDER BY days_with_news ASC
LIMIT 50;
```

**Deliverable:** Queries to monitor news collection health (run daily)

---

## Implementation Timeline

| Phase | Task | Owner | Duration | Start | End |
|-------|------|-------|----------|-------|-----|
| **Phase 1** | Audit collectors | Data | 2-3 days | Day 1 | Day 3 |
| **Phase 1** | Enable/configure collectors | DevOps | 1 day | Day 1 | Day 2 |
| **Phase 1** | Fix market breadth | Data | 2 hours | Day 2 | Day 2 |
| **Phase 2** | Backfill priority stocks | Data | 5 hours | Day 3 | Day 4 |
| **Phase 2** | Activate AV rotation queue | Data | 30 min | Day 4 | Day 4 |
| **Phase 2** | Add RSS feeds | ML/Data | 2 hours | Day 4 | Day 4 |
| **Phase 3** | Clear GDELT backlog | ML | 3 hours | Day 5 | Day 5 |
| **Phase 3** | Increase batch limits | ML | 1 hour | Day 5 | Day 5 |
| **Phase 3** | Backfill market_overview | Data | 2 hours | Day 6 | Day 6 |
| **Phase 3** | Create monitoring dashboard | Data | 3 hours | Day 7 | Day 7 |
| **Optional** | Batch sentiment inference | ML | 4 hours | Day 8+ | Day 8+ |

**Total Duration:** 7 days (3 phases) + optional optimization

---

## Success Metrics

| Metric | Current | Target | Verification |
|--------|---------|--------|--------------|
| **News Coverage** | 427/500 stocks (7d) | 450+/500 stocks (daily) | Query news_sentiment distinct symbols by date |
| **Sentiment Scoring Lag** | ~1-2 days | <4 hours | Check avg(age) of unscored articles |
| **Market Overview Records** | 12 | 1000+ (3+ years) | Count rows in market_overview table |
| **Market Overview Completeness** | <50% fields filled | 95%+ daily | Check NULL count by date |
| **Alpha Vantage Coverage** | Random 25/day | 500-stock rotation ≤25 days | Track av_coverage_tracker |
| **RSS Feed Volume** | 80 articles/day | 120-140 articles/day | Sum articles by source=RSS daily |
| **GDELT Backlog** | 150M articles | 50M articles (cleared in 5 years) | Delete old GDELT, count remaining |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| API quota exhaustion (Alpha Vantage) | Rotation queue ensures even coverage; monitor via job logs |
| RSS feed scraper breaks | Add CSS selector fallbacks; test feeds weekly |
| Scoring backlog accumulates | Increase batch_limit + add nightly job capacity |
| Market breadth data loss continues | Manual fallback query + monitoring alert |
| Model retraining fails on new data | Backfill historical sentiment first; test on 2-week window |

---

## Files to Monitor/Modify

**Read-Only (Monitor):**
- `/Users/maheshmaheshwari/Documents/trademind/backend/scheduler/jobs.py` — check collector schedules
- `/Users/maheshmaheshwari/Documents/trademind/backend/collectors/` — review all 5 collectors
- `/Users/maheshmaheshwari/Documents/trademind/backend/analysis/sentiment.py` — FinBERT implementation
- `/Users/maheshmaheshwari/Documents/trademind/backend/database/db.py` — DB helpers

**Modify (If Approved):**
- `/Users/maheshmaheshwari/Documents/trademind/backend/scheduler/jobs.py` — increase batch_limit, add nightly job
- `/Users/maheshmaheshwari/Documents/trademind/backend/analysis/sentiment.py` — implement batch inference (optional)
- `/Users/maheshmaheshwari/Documents/trademind/backend/collectors/rss_collector.py` — add 3 new RSS feeds

**Database:**
- `news_sentiment` table — delete old GDELT, monitor backlog
- `market_overview` table — backfill historical data
- `av_coverage_tracker` table — create for rotation queue
- `scheduler_log` table — verify collector health

---

## Verification & Testing

### Phase 1 Verification:
1. Run audit queries → spreadsheet of health metrics
2. Confirm all 5 collectors in logs for past 24h
3. Verify market_overview has last 10 days complete

### Phase 2 Verification:
1. Check news_sentiment has 20+ priority stocks with 30-day history
2. Verify av_coverage_tracker populated with 500 stocks
3. Count RSS articles before/after adding new feeds (+40-60/day expected)

### Phase 3 Verification:
1. Confirm GDELT backlog reduced (DELETE query results)
2. Run scheduler_log query → show 2000-article batches scoring in <3 min each
3. Query market_overview → 1000+ rows, 95% fields filled
4. Run coverage dashboard queries → all healthy

### End-to-End:
1. Retrain model on enriched sentiment + market data
2. Generate new trade signals
3. Verify signal count for 2026-06-16 increases from 446 to 490+
4. Check model performance metrics (precision, recall, F1) improve ≥2%

---

## Success Criteria

✅ **Coverage:** 450+/500 stocks in daily signals  
✅ **Backlog:** Unscored articles ≤500 (cleared within 4h)  
✅ **Market Data:** 1000+ market_overview records, 95%+ complete  
✅ **Model Impact:** Signals regenerated with richer sentiment context  
✅ **Sustainability:** All improvements maintainable without ongoing manual work
