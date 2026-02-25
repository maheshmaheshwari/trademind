"""
Nifty 500 AI — Stock Symbols List

Top 50 most important Nifty 500 stocks with yfinance symbols (.NS suffix)
and display names. These cover major large-cap stocks across all sectors.

Usage:
    from data.stocks_list import NIFTY_50_STOCKS, get_all_symbols
    for stock in NIFTY_50_STOCKS:
        print(f"{stock['symbol']} — {stock['name']}")
"""

from typing import Dict, List

# ==========================================
# Top 50 Nifty 500 stocks (most actively traded)
# Format: {"symbol": "TICKER.NS", "name": "Company Name", "sector": "Sector"}
# ==========================================
NIFTY_50_STOCKS: List[Dict[str, str]] = [
    # --- IT / Technology ---
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "IT"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "IT"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "IT"},
    {"symbol": "LTIMindtree.NS", "name": "LTIMindtree", "sector": "IT"},

    # --- Banking / Finance ---
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Banking"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Finance"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Finance"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Insurance"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Insurance"},

    # --- Oil & Gas / Energy ---
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"symbol": "NTPC.NS", "name": "NTPC", "sector": "Power"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corp", "sector": "Power"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Energy"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Logistics"},

    # --- FMCG / Consumer ---
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG"},
    {"symbol": "ITC.NS", "name": "ITC", "sector": "FMCG"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India", "sector": "FMCG"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "FMCG"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "FMCG"},

    # --- Automobile ---
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Automobile"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automobile"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automobile"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "sector": "Automobile"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Automobile"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Automobile"},

    # --- Metals & Mining ---
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Metals"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Metals"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Mining"},

    # --- Pharma / Healthcare ---
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical", "sector": "Pharma"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Labs", "sector": "Pharma"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Pharma"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "sector": "Healthcare"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma"},

    # --- Cement / Infrastructure ---
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Cement"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Cement"},
    {"symbol": "SHREECEM.NS", "name": "Shree Cement", "sector": "Cement"},
    {"symbol": "LARSENTOUB.NS", "name": "Larsen & Toubro", "sector": "Infrastructure"},

    # --- Telecom / Others ---
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Consumer"},
    {"symbol": "LTIM.NS", "name": "LTI Mindtree", "sector": "IT"},
]

# Fix: L&T yfinance symbol is LT.NS, not LARSENTOUB.NS
NIFTY_50_STOCKS[45] = {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Infrastructure"}

# Index symbols for market overview data
INDEX_SYMBOLS = {
    "NIFTY 500": "^CNX500",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "INDIA VIX": "^INDIAVIX",
}


def get_all_symbols() -> List[str]:
    """
    Get all stock symbols (just the ticker strings).

    Returns:
        List of symbol strings like ["TCS.NS", "RELIANCE.NS", ...]
    """
    return [stock["symbol"] for stock in NIFTY_50_STOCKS]


def get_stock_info(symbol: str) -> Dict[str, str]:
    """
    Get name and sector for a stock symbol.

    Args:
        symbol: Stock symbol, e.g. "TCS.NS"

    Returns:
        Dict with symbol, name, sector or empty dict if not found.
    """
    for stock in NIFTY_50_STOCKS:
        if stock["symbol"] == symbol:
            return stock
    return {}


def get_stocks_by_sector(sector: str) -> List[Dict[str, str]]:
    """
    Get all stocks in a given sector.

    Args:
        sector: Sector name, e.g. "IT", "Banking", "Pharma"

    Returns:
        List of stock dicts in that sector.
    """
    return [stock for stock in NIFTY_50_STOCKS if stock["sector"].lower() == sector.lower()]


def get_all_sectors() -> List[str]:
    """Get list of all unique sectors."""
    return list(set(stock["sector"] for stock in NIFTY_50_STOCKS))
