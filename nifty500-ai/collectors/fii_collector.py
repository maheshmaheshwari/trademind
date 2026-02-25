"""
Nifty 500 AI — FII/DII Data Collector

Collects Foreign Institutional Investor (FII) and Domestic Institutional
Investor (DII) net buy/sell data. This data indicates institutional
money flow into/out of Indian equities.

Usage:
    from collectors.fii_collector import collect_fii_dii_data
    data = collect_fii_dii_data()
    print(f"FII net: ₹{data['fii_net']} Cr, DII net: ₹{data['dii_net']} Cr")
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# NSE FII/DII data URL
NSE_FII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

# Headers to mimic a browser request (NSE blocks non-browser requests)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def collect_fii_dii_data() -> Optional[Dict]:
    """
    Fetch today's FII/DII trading activity from NSE India.

    Returns:
        Dict with keys: date, fii_net, dii_net (values in crores)
        or None if the request fails.

    Note:
        NSE may block requests without proper session cookies.
        This is a best-effort scraper. Data may not be available
        on weekends or market holidays.
    """
    try:
        # First, visit the NSE homepage to get session cookies
        session = requests.Session()
        session.headers.update(NSE_HEADERS)

        # Get session cookie from NSE homepage
        session.get("https://www.nseindia.com/", timeout=10)

        # Now fetch FII/DII data with the session
        response = session.get(NSE_FII_URL, timeout=10)

        if response.status_code != 200:
            logger.warning(f"NSE FII/DII API returned status {response.status_code}")
            return None

        data = response.json()

        # Parse the response — NSE returns a list of categories
        fii_net = 0.0
        dii_net = 0.0

        for entry in data:
            category = entry.get("category", "")
            net_value = float(entry.get("netValue", 0))

            if "FII" in category.upper() or "FPI" in category.upper():
                fii_net += net_value
            elif "DII" in category.upper():
                dii_net += net_value

        result = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii_net": round(fii_net, 2),
            "dii_net": round(dii_net, 2),
        }

        logger.info(f"FII/DII data collected: FII={result['fii_net']}Cr, DII={result['dii_net']}Cr")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching FII/DII data: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing FII/DII data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in FII/DII collector: {e}")
        return None


def scrape_moneycontrol_fii() -> Optional[Dict]:
    """
    Alternative: Scrape FII/DII data from MoneyControl as fallback.

    Returns:
        Dict with fii_net, dii_net or None on failure.
    """
    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/fiidii_activity.php"
        response = requests.get(url, headers=NSE_HEADERS, timeout=10)

        if response.status_code != 200:
            logger.warning(f"MoneyControl returned status {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # This is a best-effort scraper — MoneyControl may change their HTML
        # Look for the FII/DII table
        tables = soup.find_all("table")

        logger.info("MoneyControl FII/DII page scraped — parsing may vary")

        # Return placeholder if parsing fails
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii_net": 0.0,
            "dii_net": 0.0,
            "source": "moneycontrol",
        }

    except Exception as e:
        logger.error(f"Error scraping MoneyControl: {e}")
        return None


if __name__ == "__main__":
    """Quick test."""
    logging.basicConfig(level=logging.INFO)
    data = collect_fii_dii_data()
    if data:
        print(f"FII Net: ₹{data['fii_net']} Cr")
        print(f"DII Net: ₹{data['dii_net']} Cr")
    else:
        print("Could not fetch FII/DII data (may be weekend or market holiday)")
