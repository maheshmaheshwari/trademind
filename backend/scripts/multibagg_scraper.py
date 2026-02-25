#!/usr/bin/env python3
"""
Multibagg.ai Web Scraper

Playwright-based scraper to extract financial data from multibagg.ai for Nifty 500 stocks.
Stores data in PostgreSQL database for TradeMind project.

Usage:
    # Test mode (5 stocks)
    python scripts/multibagg_scraper.py --test
    
    # Full run (all Nifty 500 stocks)
    python scripts/multibagg_scraper.py
    
    # Resume from a specific stock
    python scripts/multibagg_scraper.py --resume-from HDFCBANK
"""

import argparse
import asyncio
import csv
import logging
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import sync_engine
from app.models import Stock, FinancialData, GrowthMetrics, TechnicalIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multibagg_scraper.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StockFinancials:
    """Container for scraped stock financial data."""
    symbol: str
    name: str
    
    # Overview metrics
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    
    # CAGR metrics (from overview)
    sales_cagr_1y: Optional[float] = None
    sales_cagr_3y: Optional[float] = None
    sales_cagr_5y: Optional[float] = None
    sales_cagr_10y: Optional[float] = None
    
    profit_cagr_1y: Optional[float] = None
    profit_cagr_3y: Optional[float] = None
    profit_cagr_5y: Optional[float] = None
    profit_cagr_10y: Optional[float] = None
    
    roe_ttm: Optional[float] = None
    roe_3y: Optional[float] = None
    roe_5y: Optional[float] = None
    roe_10y: Optional[float] = None
    
    roce_ttm: Optional[float] = None
    roce_3y: Optional[float] = None
    roce_5y: Optional[float] = None
    roce_10y: Optional[float] = None
    
    # Yearly financials (dict of year -> data)
    yearly_revenue: dict[int, float] = None
    yearly_net_profit: dict[int, float] = None
    yearly_operating_profit: dict[int, float] = None
    yearly_eps: dict[int, float] = None
    
    # Technical indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    
    # EMAs
    ema_5: Optional[float] = None
    ema_10: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    ema_150: Optional[float] = None
    ema_200: Optional[float] = None
    
    # SMAs
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None
    sma_150: Optional[float] = None
    sma_200: Optional[float] = None
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    
    # Other indicators
    atr_14: Optional[float] = None
    stoch_rsi: Optional[float] = None
    cci_20: Optional[float] = None
    momentum: Optional[float] = None
    adx_14: Optional[float] = None
    
    def __post_init__(self):
        if self.yearly_revenue is None:
            self.yearly_revenue = {}
        if self.yearly_net_profit is None:
            self.yearly_net_profit = {}
        if self.yearly_operating_profit is None:
            self.yearly_operating_profit = {}
        if self.yearly_eps is None:
            self.yearly_eps = {}


class MultibaggScraper:
    """Playwright-based scraper for multibagg.ai."""
    
    BASE_URL = "https://www.multibagg.ai"
    SEARCH_URL = f"{BASE_URL}/screeners/screener/stocks/new-screen"
    
    # Rate limiting
    MIN_DELAY = 2.0  # seconds between requests
    MAX_DELAY = 5.0
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.stocks_processed = 0
        self.stocks_failed = 0
        self.stock_url_cache: dict[str, str] = {}  # symbol -> URL mapping
    
    async def __aenter__(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        
        # Create context with realistic user agent
        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.page = await context.new_page()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
    
    async def build_stock_url_cache(self, target_symbols: Optional[list[str]] = None) -> dict[str, str]:
        """
        Build a cache of stock symbols to URLs by scrolling through the screener table.
        This is done once at the start to enable fast direct navigation.
        
        Args:
            target_symbols: If provided, stop once all these symbols are found
        """
        logger.info("Building stock URL cache from screener table...")
        
        try:
            await self.page.goto(self.SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(4)
            
            await self.page.wait_for_selector('#screener-scroll-container, table', timeout=30000)
            
            scroll_container = await self.page.query_selector('#screener-scroll-container')
            found_symbols = set()
            target_set = set(target_symbols) if target_symbols else None
            no_new_count = 0
            max_no_new = 5  # Stop if no new stocks found after 5 scrolls
            
            while True:
                # Find all stock rows in current view
                rows = await self.page.query_selector_all('tr')
                prev_count = len(self.stock_url_cache)
                
                for row in rows:
                    try:
                        # Get the stock link
                        link = await row.query_selector('a[class*="TableRow_url"]')
                        if not link:
                            continue
                        
                        href = await link.get_attribute('href')
                        if not href or '/stocks/' not in href:
                            continue
                        
                        # Get all text in the row to find the symbol
                        row_text = await row.inner_text()
                        # Symbol is usually in column 4, but we'll look for known patterns
                        parts = row_text.split()
                        
                        # Find potential symbol (all caps, 2-20 chars)
                        for part in parts:
                            cleaned = part.strip()
                            if cleaned.isupper() and 2 <= len(cleaned) <= 20 and cleaned.isalnum():
                                self.stock_url_cache[cleaned] = f"{self.BASE_URL}{href}"
                                found_symbols.add(cleaned)
                                break
                    except:
                        continue
                
                # Check if we found all target symbols
                if target_set and target_set.issubset(found_symbols):
                    logger.info(f"Found all {len(target_set)} target symbols in cache")
                    break
                
                # Check if we're making progress
                if len(self.stock_url_cache) == prev_count:
                    no_new_count += 1
                    if no_new_count >= max_no_new:
                        logger.info(f"No new stocks found after {max_no_new} scrolls, stopping")
                        break
                else:
                    no_new_count = 0
                
                # Scroll down
                if scroll_container:
                    await scroll_container.evaluate('el => el.scrollBy(0, 1000)')
                else:
                    await self.page.evaluate('window.scrollBy(0, 1000)')
                await asyncio.sleep(1)
            
            logger.info(f"Built URL cache with {len(self.stock_url_cache)} stocks")
            return self.stock_url_cache
            
        except Exception as e:
            logger.error(f"Error building URL cache: {e}")
            return self.stock_url_cache
    
    async def _random_delay(self):
        """Add random delay to avoid rate limiting."""
        delay = random.uniform(self.MIN_DELAY, self.MAX_DELAY)
        await asyncio.sleep(delay)
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text, handling various formats."""
        if not text or text.strip() in ('', '-', 'N/A', 'NA', '--'):
            return None
        
        # Remove commas and spaces
        text = text.replace(',', '').replace(' ', '').strip()
        
        # Handle Cr (Crore) suffix
        if 'Cr' in text or 'cr' in text:
            text = re.sub(r'[Cc]r\.?', '', text)
            try:
                return float(text) * 10_000_000  # 1 Cr = 10 Million
            except ValueError:
                return None
        
        # Handle % suffix
        if '%' in text:
            text = text.replace('%', '')
        
        # Handle negative numbers with parentheses
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        
        try:
            return float(text)
        except ValueError:
            return None
    
    async def search_stock(self, symbol: str) -> Optional[str]:
        """Search for a stock and return its detail page URL."""
        try:
            # Navigate to screener page
            await self.page.goto(self.SEARCH_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)
            
            # Look for search input
            search_input = await self.page.query_selector('input[placeholder*="Search"]')
            if search_input:
                await search_input.fill(symbol)
                await asyncio.sleep(1.5)
                
                # Click on first result
                first_result = await self.page.query_selector('a[href*="/stocks/"]')
                if first_result:
                    href = await first_result.get_attribute('href')
                    if href:
                        return f"{self.BASE_URL}{href}" if not href.startswith('http') else href
            
            # Alternative: Navigate directly to stock page
            # Try common URL patterns
            stock_url = f"{self.BASE_URL}/stocks/{symbol.lower()}"
            response = await self.page.goto(stock_url, wait_until="networkidle", timeout=30000)
            if response and response.ok:
                return stock_url
            
            return None
            
        except PlaywrightTimeout:
            logger.warning(f"Timeout searching for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error searching for {symbol}: {e}")
            return None
    
    async def scrape_stock_page(self, symbol: str, name: str) -> Optional[StockFinancials]:
        """Scrape financial data for a stock using cached URL or screener table."""
        financials = StockFinancials(symbol=symbol, name=name)
        
        try:
            # Try to use cached URL first (much faster)
            if symbol in self.stock_url_cache:
                stock_url = self.stock_url_cache[symbol]
                await self.page.goto(stock_url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(3)
            else:
                # Fallback: Navigate to screener and find the stock
                screener_url = f"{self.BASE_URL}/screeners/screener/stocks/new-screen"
                await self.page.goto(screener_url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(4)
                
                await self.page.wait_for_selector('#screener-scroll-container, table', timeout=30000)
                
                # Find the stock link in the table
                stock_link = await self.page.query_selector(
                    f'a[class*="TableRow_url"]:has-text("{name.split()[0]}")'
                )
                
                if stock_link:
                    href = await stock_link.get_attribute('href')
                    if href:
                        stock_url = f"{self.BASE_URL}{href}" if not href.startswith('http') else href
                        self.stock_url_cache[symbol] = stock_url  # Cache for future use
                        await self.page.goto(stock_url, wait_until="domcontentloaded", timeout=60000)
                        await asyncio.sleep(3)
                else:
                    logger.warning(f"Could not find {symbol} in screener table")
                    return None
            
            # Verify we're on a stock page
            current_url = self.page.url
            if '/stocks/' not in current_url:
                logger.warning(f"Not on stock page for {symbol}: {current_url}")
                return None
            
            # Now we're on the stock detail page - scrape overview data
            html = await self.page.content()
            soup = BeautifulSoup(html, 'lxml')
            
            # Extract overview metrics (CAGR, ROE, ROCE, etc.)
            financials = await self._extract_overview_metrics(financials, soup)
            
            # Navigate to Financials tab
            financials_tab = await self.page.query_selector('button:has-text("Financials"), a:has-text("Financials"), [role="tab"]:has-text("Financials")')
            if financials_tab:
                await financials_tab.click()
                await asyncio.sleep(2)
                await self.page.wait_for_load_state("networkidle", timeout=15000)
                
                # Scrape P&L data
                html = await self.page.content()
                soup = BeautifulSoup(html, 'lxml')
                financials = await self._extract_financial_statements(financials, soup)
            
            # Navigate to Technicals tab for technical indicators
            # Using the correct selector - anchor link with href="#technicals"
            technicals_tab = await self.page.query_selector('a[href="#technicals"]')
            if technicals_tab:
                await technicals_tab.click()
                await asyncio.sleep(2)
                
                # Click "View detailed analysis" to open the modal with all indicators
                # The element is a span with class TechnicalModal_detailedAnalysis__*
                detailed_analysis_link = await self.page.query_selector('span[class*="TechnicalModal_detailedAnalysis"], .TechnicalModal_detailedAnalysis__DLhpG')
                if detailed_analysis_link:
                    await detailed_analysis_link.click()
                    await asyncio.sleep(2)
                    
                    # Wait for the modal to appear
                    await self.page.wait_for_selector('.SideModal_modal__dUDud, [class*="SideModal_modal"]', timeout=5000)
                    
                    # Scrape technical indicators from the modal
                    html = await self.page.content()
                    soup = BeautifulSoup(html, 'lxml')
                    financials = await self._extract_technical_indicators(financials, soup)
                    
                    # Close modal by pressing Escape
                    await self.page.keyboard.press('Escape')
                    await asyncio.sleep(0.5)
                else:
                    logger.debug(f"Could not find 'View detailed analysis' link for {symbol}")
            
            logger.info(f"✓ Successfully scraped {symbol}")
            return financials
            
        except PlaywrightTimeout:
            logger.warning(f"Timeout scraping {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {symbol}: {e}")
            return None
    
    async def _extract_overview_metrics(self, financials: StockFinancials, soup: BeautifulSoup) -> StockFinancials:
        """Extract overview metrics from stock page."""
        
        # The page shows metrics in a structured format
        # Look for Sales CAGR, Profit CAGR, ROE, ROCE sections
        
        page_text = soup.get_text()
        
        # Extract CAGR values using regex patterns
        # Sales CAGR pattern: "Sales CAGR" followed by 1Y, 3Y, 5Y, 10Y values
        sales_cagr_pattern = r'Sales\s+CAGR[^\d]*?([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%'
        sales_match = re.search(sales_cagr_pattern, page_text, re.I)
        if sales_match:
            financials.sales_cagr_1y = self._parse_number(sales_match.group(1))
            financials.sales_cagr_3y = self._parse_number(sales_match.group(2))
            financials.sales_cagr_5y = self._parse_number(sales_match.group(3))
            financials.sales_cagr_10y = self._parse_number(sales_match.group(4))
        
        # Profit CAGR pattern
        profit_cagr_pattern = r'Profit\s+CAGR[^\d]*?([\d.\-]+)%[^\d]*([\d.\-]+)%[^\d]*([\d.\-]+)%[^\d]*([\d.\-]+)%'
        profit_match = re.search(profit_cagr_pattern, page_text, re.I)
        if profit_match:
            financials.profit_cagr_1y = self._parse_number(profit_match.group(1))
            financials.profit_cagr_3y = self._parse_number(profit_match.group(2))
            financials.profit_cagr_5y = self._parse_number(profit_match.group(3))
            financials.profit_cagr_10y = self._parse_number(profit_match.group(4))
        
        # ROE pattern: "ROE" followed by TTM, 3Y, 5Y, 10Y values
        roe_pattern = r'\bROE\b[^\d]*?([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%'
        roe_match = re.search(roe_pattern, page_text, re.I)
        if roe_match:
            financials.roe_ttm = self._parse_number(roe_match.group(1))
            financials.roe_3y = self._parse_number(roe_match.group(2))
            financials.roe_5y = self._parse_number(roe_match.group(3))
            financials.roe_10y = self._parse_number(roe_match.group(4))
        
        # ROCE pattern
        roce_pattern = r'\bROCE\b[^\d]*?([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%[^\d]*([\d.]+)%'
        roce_match = re.search(roce_pattern, page_text, re.I)
        if roce_match:
            financials.roce_ttm = self._parse_number(roce_match.group(1))
            financials.roce_3y = self._parse_number(roce_match.group(2))
            financials.roce_5y = self._parse_number(roce_match.group(3))
            financials.roce_10y = self._parse_number(roce_match.group(4))
        
        # Market Cap
        market_cap_pattern = r'Market\s*Cap[^\d₹]*₹?([\d,.]+)\s*Cr'
        market_cap_match = re.search(market_cap_pattern, page_text, re.I)
        if market_cap_match:
            financials.market_cap = self._parse_number(market_cap_match.group(1) + ' Cr')
        
        # P/E Ratio
        pe_pattern = r'P/?E\s*Ratio[^\d]*?([\d.]+)'
        pe_match = re.search(pe_pattern, page_text, re.I)
        if pe_match:
            financials.pe_ratio = self._parse_number(pe_match.group(1))
        
        # P/B Ratio
        pb_pattern = r'P/?B\s*Ratio[^\d]*?([\d.]+)'
        pb_match = re.search(pb_pattern, page_text, re.I)
        if pb_match:
            financials.pb_ratio = self._parse_number(pb_match.group(1))
        
        # 52W High/Low
        high_52w_pattern = r'52W\s*High[^\d₹]*₹?([\d,.]+)'
        high_match = re.search(high_52w_pattern, page_text, re.I)
        if high_match:
            financials.high_52w = self._parse_number(high_match.group(1))
        
        low_52w_pattern = r'52W\s*Low[^\d₹]*₹?([\d,.]+)'
        low_match = re.search(low_52w_pattern, page_text, re.I)
        if low_match:
            financials.low_52w = self._parse_number(low_match.group(1))
        
        return financials
    
    async def _extract_technical_indicators(self, financials: StockFinancials, soup: BeautifulSoup) -> StockFinancials:
        """Extract technical indicators from the detailed analysis modal."""
        
        page_text = soup.get_text()
        
        # RSI - Full name: "Relative Strength Index (14)" followed by value
        rsi_pattern = r'Relative\s+Strength\s+Index\s*\(14\)[^\d]*?([+-]?[\d.]+)'
        rsi_match = re.search(rsi_pattern, page_text, re.I)
        if rsi_match:
            financials.rsi_14 = self._parse_number(rsi_match.group(1))
        
        # MACD - Full name: "MACD Level (12, 26)" followed by value
        macd_pattern = r'MACD\s+Level\s*\([^)]+\)[^\d\-]*?([+-]?[\d.]+)'
        macd_match = re.search(macd_pattern, page_text, re.I)
        if macd_match:
            financials.macd = self._parse_number(macd_match.group(1))
        
        # Stochastic RSI - "Stochastic RSI (14)" followed by value
        stoch_pattern = r'Stochastic\s+RSI\s*\(14\)[^\d]*?([+-]?[\d.]+)'
        stoch_match = re.search(stoch_pattern, page_text, re.I)
        if stoch_match:
            financials.stoch_rsi = self._parse_number(stoch_match.group(1))
        
        # CCI - "Commodity Channel Index (20)"
        cci_pattern = r'Commodity\s+Channel\s+Index\s*\(20\)[^\d\-]*?([+-]?[\d.]+)'
        cci_match = re.search(cci_pattern, page_text, re.I)
        if cci_match:
            financials.cci_20 = self._parse_number(cci_match.group(1))
        
        # ADX - "Average Directional Index (14)"
        adx_pattern = r'Average\s+Directional\s+Index\s*\(14\)[^\d]*?([+-]?[\d.]+)'
        adx_match = re.search(adx_pattern, page_text, re.I)
        if adx_match:
            financials.adx_14 = self._parse_number(adx_match.group(1))
        
        # ATR - "Average True Range" or similar
        atr_pattern = r'(?:Average\s+)?True\s+Range[^\d]*?([+-]?[\d.]+)'
        atr_match = re.search(atr_pattern, page_text, re.I)
        if atr_match:
            financials.atr_14 = self._parse_number(atr_match.group(1))
        
        # Momentum (10)
        momentum_pattern = r'Momentum\s*\(10\)[^\d\-]*?([+-]?[\d.]+)'
        momentum_match = re.search(momentum_pattern, page_text, re.I)
        if momentum_match:
            financials.momentum = self._parse_number(momentum_match.group(1))
        
        # Bollinger Bands (20, 2) - returns tuple "(lower, middle, upper)"
        bb_pattern = r'Bollinger\s+Bands\s*\([^)]+\)[^\(]*?\(([^)]+)\)'
        bb_match = re.search(bb_pattern, page_text, re.I)
        if bb_match:
            bb_values = bb_match.group(1).split(',')
            if len(bb_values) >= 3:
                financials.bb_lower = self._parse_number(bb_values[0].strip())
                financials.bb_middle = self._parse_number(bb_values[1].strip())
                financials.bb_upper = self._parse_number(bb_values[2].strip())
        
        # EMAs - "Exponential Moving Average (N)"
        ema_periods = [5, 10, 20, 50, 100, 150, 200]
        for period in ema_periods:
            ema_pattern = rf'Exponential\s+Moving\s+Average\s*\({period}\)[^\d]*?([+-]?[\d,.]+)'
            ema_match = re.search(ema_pattern, page_text, re.I)
            if ema_match:
                setattr(financials, f'ema_{period}', self._parse_number(ema_match.group(1)))
        
        # SMAs - "Simple Moving Average (N)"
        sma_periods = [5, 10, 20, 50, 100, 150, 200]
        for period in sma_periods:
            sma_pattern = rf'Simple\s+Moving\s+Average\s*\({period}\)[^\d]*?([+-]?[\d,.]+)'
            sma_match = re.search(sma_pattern, page_text, re.I)
            if sma_match:
                setattr(financials, f'sma_{period}', self._parse_number(sma_match.group(1)))
        
        return financials
    
    async def _extract_financial_statements(self, financials: StockFinancials, soup: BeautifulSoup) -> StockFinancials:
        """Extract yearly financial statement data."""
        
        # Look for Profit & Loss table
        tables = soup.find_all('table')
        
        for table in tables:
            header_row = table.find('tr')
            if not header_row:
                continue
            
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Extract years from headers (format: Mar 2024, Mar 2023, etc.)
            years = []
            for h in headers[1:]:  # Skip first column (row label)
                year_match = re.search(r'(\d{4})', h)
                if year_match:
                    years.append(int(year_match.group(1)))
            
            if not years:
                continue
            
            # Parse rows
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                
                label = cells[0].get_text().strip().lower()
                values = [self._parse_number(cell.get_text()) for cell in cells[1:]]
                
                # Map to appropriate fields
                if 'revenue' in label or 'sales' in label or 'total income' in label:
                    for i, (year, val) in enumerate(zip(years, values)):
                        if val is not None:
                            financials.yearly_revenue[year] = val
                
                elif 'net profit' in label or 'profit after tax' in label or 'pat' in label:
                    for i, (year, val) in enumerate(zip(years, values)):
                        if val is not None:
                            financials.yearly_net_profit[year] = val
                
                elif 'operating profit' in label or 'ebit' in label:
                    for i, (year, val) in enumerate(zip(years, values)):
                        if val is not None:
                            financials.yearly_operating_profit[year] = val
                
                elif 'eps' in label or 'earnings per share' in label:
                    for i, (year, val) in enumerate(zip(years, values)):
                        if val is not None:
                            financials.yearly_eps[year] = val
        
        return financials


def load_nifty500_stocks(csv_path: str) -> list[dict]:
    """Load Nifty 500 stocks from CSV file."""
    stocks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Symbol'):
                stocks.append({
                    'symbol': row['Symbol'].strip(),
                    'name': row.get('Company Name', '').strip(),
                    'industry': row.get('Industry', '').strip(),
                    'market_cap_category': row.get('Market Cap Category', '').strip(),
                })
    return stocks


def save_to_database(financials: StockFinancials, session: Session) -> bool:
    """Save scraped financial data to database."""
    try:
        # Find the stock in database
        stock = session.query(Stock).filter(Stock.symbol == financials.symbol).first()
        if not stock:
            logger.warning(f"Stock {financials.symbol} not found in database")
            return False
        
        today = date.today()
        
        # Save GrowthMetrics (CAGR snapshot)
        growth = session.query(GrowthMetrics).filter(
            GrowthMetrics.stock_id == stock.id,
            GrowthMetrics.snapshot_date == today
        ).first()
        
        if not growth:
            growth = GrowthMetrics(stock_id=stock.id, snapshot_date=today)
            session.add(growth)
        
        # Update growth metrics
        growth.sales_cagr_1y = financials.sales_cagr_1y
        growth.sales_cagr_3y = financials.sales_cagr_3y
        growth.sales_cagr_5y = financials.sales_cagr_5y
        growth.sales_cagr_10y = financials.sales_cagr_10y
        
        growth.profit_cagr_1y = financials.profit_cagr_1y
        growth.profit_cagr_3y = financials.profit_cagr_3y
        growth.profit_cagr_5y = financials.profit_cagr_5y
        growth.profit_cagr_10y = financials.profit_cagr_10y
        
        growth.roe_1y = financials.roe_ttm
        growth.roe_3y = financials.roe_3y
        growth.roe_5y = financials.roe_5y
        growth.roe_10y = financials.roe_10y
        
        growth.roce_1y = financials.roce_ttm
        growth.roce_3y = financials.roce_3y
        growth.roce_5y = financials.roce_5y
        growth.roce_10y = financials.roce_10y
        
        growth.market_cap = financials.market_cap
        growth.pe_ratio = financials.pe_ratio
        growth.pb_ratio = financials.pb_ratio
        growth.dividend_yield = financials.dividend_yield
        growth.high_52w = financials.high_52w
        growth.low_52w = financials.low_52w
        
        # Save yearly FinancialData
        all_years = set(financials.yearly_revenue.keys()) | set(financials.yearly_net_profit.keys())
        
        for year in all_years:
            fin_data = session.query(FinancialData).filter(
                FinancialData.stock_id == stock.id,
                FinancialData.fiscal_year == year
            ).first()
            
            if not fin_data:
                fin_data = FinancialData(stock_id=stock.id, fiscal_year=year)
                session.add(fin_data)
            
            # Update fields
            if year in financials.yearly_revenue:
                fin_data.revenue = financials.yearly_revenue[year]
            if year in financials.yearly_net_profit:
                fin_data.net_profit = financials.yearly_net_profit[year]
            if year in financials.yearly_operating_profit:
                fin_data.operating_profit = financials.yearly_operating_profit[year]
            if year in financials.yearly_eps:
                fin_data.eps = financials.yearly_eps[year]
        
        # Save TechnicalIndicator (current day snapshot)
        tech = session.query(TechnicalIndicator).filter(
            TechnicalIndicator.stock_id == stock.id,
            TechnicalIndicator.date == today
        ).first()
        
        if not tech:
            tech = TechnicalIndicator(stock_id=stock.id, date=today)
            session.add(tech)
        
        # Update technical indicators
        tech.rsi_14 = financials.rsi_14
        tech.ema_20 = financials.ema_20
        tech.ema_50 = financials.ema_50
        tech.ema_200 = financials.ema_200
        tech.volatility_20 = None  # Not available from multibagg
        tech.atr_14 = financials.atr_14
        tech.adx_14 = financials.adx_14
        tech.macd = financials.macd
        tech.bb_upper = financials.bb_upper
        tech.bb_middle = financials.bb_middle
        tech.bb_lower = financials.bb_lower
        
        session.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error saving {financials.symbol} to database: {e}")
        session.rollback()
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Scrape financial data from Multibagg.ai')
    parser.add_argument('--test', action='store_true', help='Test mode: scrape only 5 stocks')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of stocks to scrape')
    parser.add_argument('--resume-from', type=str, help='Resume from a specific stock symbol')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    parser.add_argument('--visible', action='store_true', help='Run browser in visible mode (for debugging)')
    
    args = parser.parse_args()
    
    headless = not args.visible
    limit = 5 if args.test else args.limit
    
    # Load Nifty 500 stocks
    csv_path = Path(settings.nifty500_symbols_file)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / csv_path
    
    stocks = load_nifty500_stocks(str(csv_path))
    logger.info(f"Loaded {len(stocks)} stocks from {csv_path}")
    
    # Handle resume
    if args.resume_from:
        resume_idx = None
        for i, s in enumerate(stocks):
            if s['symbol'] == args.resume_from:
                resume_idx = i
                break
        if resume_idx is not None:
            stocks = stocks[resume_idx:]
            logger.info(f"Resuming from {args.resume_from} ({len(stocks)} stocks remaining)")
        else:
            logger.warning(f"Stock {args.resume_from} not found, starting from beginning")
    
    # Apply limit
    if limit:
        stocks = stocks[:limit]
        logger.info(f"Limited to {len(stocks)} stocks")
    
    # Create database session
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()
    
    try:
        async with MultibaggScraper(headless=headless) as scraper:
            successful = 0
            failed = 0
            
            # Build URL cache for target stocks (much faster than scrolling for each)
            target_symbols = [stock['symbol'] for stock in stocks]
            await scraper.build_stock_url_cache(target_symbols=target_symbols)
            
            for i, stock in enumerate(stocks):
                symbol = stock['symbol']
                name = stock['name']
                
                logger.info(f"[{i+1}/{len(stocks)}] Scraping {symbol} ({name})...")
                
                # Add delay between stocks
                if i > 0:
                    await scraper._random_delay()
                
                # Scrape stock
                financials = await scraper.scrape_stock_page(symbol, name)
                
                if financials:
                    # Save to database
                    if save_to_database(financials, session):
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                
                # Progress update every 10 stocks
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(stocks)} | Success: {successful} | Failed: {failed}")
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Scraping complete!")
            logger.info(f"Total: {len(stocks)} | Success: {successful} | Failed: {failed}")
            logger.info(f"{'='*50}")
    
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
