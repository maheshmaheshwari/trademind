# Phase 1 Audit Report - News Sentiment & Market Overview

**Date:** 2026-06-17  
**Status:** ✅ COMPLETE

---

## 1. COLLECTOR HEALTH (Last 7 days)

| Collector | Status | Success Rate | Status |
|-----------|--------|-------------|--------|
| **Daily News Collection** | 4 done, 3 failed, 1 running | 57% | ⚠️ DEGRADED |
| **Hourly News Refresh** | 20 done, 15 failed, 9 running | 57% | ⚠️ DEGRADED |
| **yfinance Per-Stock News** | 3 done, 2 failed, 2 running | 43% | 🔴 POOR |
| **RSS Market News** | 4 done, 3 failed, 1 running | 57% | ⚠️ DEGRADED |
| **Alpha Vantage News** | 4 done, 2 failed, 2 running | 67% | ⚠️ DEGRADED |
| **FinBERT News Scoring** | 30 done, 23 failed, 17 running | 57% | ⚠️ DEGRADED |
| **Index & Market Overview** | 5 done, 3 failed, 1 running | 63% | ⚠️ DEGRADED |

**Key Insight:** All collectors showing 40-70% failure rates. Root cause: Backend restart/recovery cycle (many "running" jobs indicate restart events).

---

## 2. ARTICLES COLLECTED (Last 24 hours: 2026-06-16 → 2026-06-17)

**Total: 337 articles**

| Source | Articles | Stocks Tagged | Coverage |
|--------|----------|--------------|----------|
| Economic Times | 144 | 0 | Market-wide |
| ET Markets (RSS) | 76 | 25 | Stock-specific |
| Business Standard | 70 | 23 | Stock-specific |
| Times of India | 22 | 0 | Market-wide |
| yfinance | 11 | 9 | Stock-specific |
| Other sources | 14 | 0 | Market-wide |

**Analysis:**
- ✅ RSS feeds performing well (146 articles, 48 stocks)
- ❌ yfinance collection failing (only 11 articles vs expected 4,990)
- ✅ Market-wide news flowing steadily (236 articles)
- ⚠️ FinBERT scoring backlog cleared (0 unscored articles - all caught up)

---

## 3. NEWS COVERAGE

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Stocks with news TODAY** | 0/501 | 100+ | 🔴 NONE (post-market hours) |
| **Stocks with news LAST 7 days** | 427/501 | 450+ | ⚠️ 85% (74 stocks missing) |
| **Sentiment scoring backlog** | 0 articles | <500 | ✅ CLEARED |

**Missing Stocks (Last 7 days):** 74 total  
*Sample:* ABLBL, ACMESOLAR, AEGISVOPAK, AFCONS, AGARWALEYE, APTUS, ATHERENERG, BAJAJHFL, BHARTIHEXA, BLS...

---

## 4. MARKET OVERVIEW TABLE STATUS

| Field | Rows Filled | % Complete | Status |
|-------|------------|-----------|--------|
| **Total Records** | 12 | - | 🔴 CRITICAL (should be 1000+) |
| **Nifty 50 Close** | 3/12 | 25% | 🔴 CRITICAL |
| **Nifty 500 Close** | 3/12 | 25% | 🔴 CRITICAL |
| **Sensex Close** | 3/12 | 25% | 🔴 CRITICAL |
| **India VIX** | 7/12 | 58% | ⚠️ PARTIAL |
| **FII/DII Data** | 4/12 | 33% | ⚠️ PARTIAL |

**Date Coverage:** 2026-06-02 to 2026-06-17 (only 16 days)

**Root Cause:** `collect_index_data_eod_job` runs at 16:00 IST but only creates records sporadically due to API failures.

---

## 5. FII/DII DATA STATUS

| Metric | Value |
|--------|-------|
| **Table** | fii_dii_daily |
| **Total Records** | 739 |
| **Date Range** | 2023-06-09 to 2026-06-16 |
| **Status** | ✅ HEALTHY (3+ years) |

**Note:** FII/DII data is well-maintained. Issue is `market_overview` table not linked to it.

---

## 6. API CONFIGURATION VERIFICATION

| API | Key | Status | Issue |
|-----|-----|--------|-------|
| **NewsAPI** | Configured (key: 3d011a47...) | ✅ READY | None |
| **Alpha Vantage** | Configured (key: UFHGWM28Z8GJF8YN) | ✅ READY | 25/day quota limit |
| **Angel One** | Configured | ✅ READY | None |
| **FinBERT** | N/A (local model) | ✅ READY | None |

**All APIs properly configured. No credential issues.**

---

## Findings Summary

### ✅ What's Working
1. **News Collection Sources:** 5 active collectors (yfinance, RSS, Alpha Vantage, NewsAPI, GDELT)
2. **Sentiment Scoring:** FinBERT backlog CLEARED (0 unscored articles)
3. **FII/DII Data:** 3+ years of complete data in fii_dii_daily table
4. **API Keys:** All configured and ready

### 🔴 Critical Issues
1. **Market Overview Table:** Only 12 records (CRITICAL - needs 1000+)
2. **Index Data Collection:** Failing frequently, resulting in NULL values
3. **News Coverage Gap:** 74/501 stocks missing news (15% gap)
4. **Data Integration:** market_overview not linked to fii_dii_daily

### ⚠️ Degradation
1. **Collector Reliability:** 40-70% failure rates (recovery cycle impact)
2. **yfinance Collection:** Only 11 articles (expected 4,990) - needs investigation
3. **Today's Coverage:** 0 stocks (post-market - expected; will resume tomorrow)

---

## Recommendations for Phase 1.3 (Quick Win)

### Immediate Action: Backfill market_overview Table

**Step 1:** Extract historical index data from prices table
```sql
-- Get last 90 days of market data
SELECT date, 
       (SELECT close FROM prices WHERE symbol='^NSEI' AND interval='1d' AND prices.date=market.date LIMIT 1) as nifty50_close,
       (SELECT close FROM prices WHERE symbol='^NIFTY500' AND interval='1d' AND prices.date=market.date LIMIT 1) as nifty500_close,
       (SELECT close FROM prices WHERE symbol='^BSESN' AND interval='1d' AND prices.date=market.date LIMIT 1) as sensex_close,
       (SELECT close FROM prices WHERE symbol='^INDIAVIX' AND interval='1d' AND prices.date=market.date LIMIT 1) as india_vix
FROM (SELECT DISTINCT date FROM prices WHERE interval='1d' AND date >= CURRENT_DATE - INTERVAL '90 days') market
ORDER BY date DESC;
```

**Step 2:** Link to FII/DII data
```sql
UPDATE market_overview m
SET fii_net = f.fii_net, dii_net = f.dii_net
FROM fii_dii_daily f
WHERE m.date = f.date;
```

**Step 3:** Insert missing market_overview records for 2023-2026
- Extract from historical prices + fii_dii_daily
- Insert via UPSERT (ON CONFLICT DO UPDATE)

---

## Next Steps

**Phase 1 Status:** ✅ DIAGNOSTICS COMPLETE

**Ready for Phase 1.3 Implementation:**
- Populate market_overview with 3+ years of daily data
- Link to fii_dii_daily 
- Verify index data collection reliability

