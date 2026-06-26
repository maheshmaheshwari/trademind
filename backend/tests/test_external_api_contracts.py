"""
Contract tests for the parsing functions that scheduler jobs depend on for
external API data (Angel One candles/LTP, yfinance news). These never call
the real network — they feed fixtures shaped exactly like the real response
(see tests/fixtures/*.json, each annotated with the exact file/function its
shape mirrors) into the same parsing code the scheduler jobs use, and check
the DB ends up with the right rows. This is what actually breaks first when
Angel One/yfinance change their response shape — catching it here is cheaper
than catching it via a failed production EOD job.
"""
from conftest import load_fixture

from database.db import get_connection, release_connection, _execute, insert_prices_batch


class _FakeSmartApiCandles:
    def __init__(self, response):
        self._response = response

    def getCandleData(self, params):
        return self._response


class _FakeSmartApiLtp:
    def __init__(self, response):
        self._response = response

    def ltpData(self, exchange, tradingsymbol, symboltoken):
        return self._response


def test_angel_eod_candle_parsing_and_insert():
    from scripts.update_stocks_angel import fetch_candles

    fixture = load_fixture("angel_eod_candles")
    fixture = {k: v for k, v in fixture.items() if k != "_mirrors"}
    fake_api = _FakeSmartApiCandles(fixture)

    rows = fetch_candles(fake_api, symbol="RELIANCE", token="2885", exchange="NSE", days=5)

    assert len(rows) == 3
    symbol_ns, exchange, date, time_, o, h, l, c, v, interval = rows[-1]
    assert symbol_ns == "RELIANCE.NS"
    assert date == "2026-06-24"
    assert c == 1458.3
    assert interval == "1d"

    inserted = insert_prices_batch(rows)
    assert inserted >= 1

    conn = get_connection()
    try:
        row = _execute(
            conn, "SELECT close FROM prices WHERE symbol = ? AND date = ?",
            ("RELIANCE.NS", "2026-06-24"),
        ).fetchone()
    finally:
        release_connection(conn)
    assert row is not None
    assert row[0] == 1458.3


def test_angel_ltp_parsing():
    fixture = load_fixture("angel_ltp_response")
    fixture = {k: v for k, v in fixture.items() if k != "_mirrors"}
    fake_api = _FakeSmartApiLtp(fixture)

    ltp_data = fake_api.ltpData(exchange="NSE", tradingsymbol="RELIANCE-EQ", symboltoken="2885")
    assert ltp_data.get("status") is True
    ltp = float(ltp_data["data"].get("ltp", 0))
    assert ltp == 1458.3


def test_yfinance_news_parsing_and_insert(monkeypatch):
    import collectors.yfinance_news_collector as yfc

    fixture = load_fixture("yfinance_news_response")
    articles = fixture["news"]

    monkeypatch.setattr(yfc, "_fetch_news_with_timeout", lambda symbol: articles)

    inserted = yfc.collect_stock("RELIANCE.NS")
    assert inserted >= 1

    conn = get_connection()
    try:
        row = _execute(
            conn, "SELECT headline, source FROM news_sentiment WHERE symbol = ?",
            ("RELIANCE.NS",),
        ).fetchone()
    finally:
        release_connection(conn)
    assert row is not None
    assert "Q1 earnings beat" in row[0]
    assert row[1] == "yfinance"
