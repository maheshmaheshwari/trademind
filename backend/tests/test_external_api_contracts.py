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


# ── BSE corporate announcements ───────────────────────────────────────────────

class _FakeBsePages:
    """Stands in for requests.Session against the BSE announcements endpoint.

    Serves one canned page per `pageno` so the pagination loop in
    fetch_symbol_window can be exercised without the network.
    """

    def __init__(self, pages):
        self._pages = pages
        self.requested = []

    def get(self, url, params=None, timeout=None):
        self.requested.append(params)
        payload = self._pages[params["pageno"] - 1]
        return _FakeResponse(payload)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_bse_scrip_list_maps_nse_symbols():
    from collectors.bse_announcements_collector import parse_scrip_list

    fixture = load_fixture("bse_scrip_list")
    scrip_map = parse_scrip_list({"Table": fixture["Table"]})

    assert scrip_map["HDFCBANK"] == "500180"
    assert scrip_map["RELIANCE"] == "500325"
    # The live endpoint returns a bare array rather than {"Table": [...]}; both
    # shapes have to parse or the collector breaks on the real response.
    assert parse_scrip_list(fixture["Table"]) == scrip_map


def test_bse_announcement_parsing_and_insert():
    from collectors.bse_announcements_collector import (
        parse_announcements, store, total_for_range,
    )

    fixture = load_fixture("bse_announcements")
    payload = {k: v for k, v in fixture.items() if k != "_mirrors"}

    assert total_for_range(payload) == 58        # drives pagination

    rows = parse_announcements(payload, "HDFCBANK.NS")
    assert len(rows) == 3

    headline, source, published_at, symbol, sentiment, confidence, url = rows[0]
    assert source == "BSE"
    assert symbol == "HDFCBANK.NS"
    assert published_at.startswith("2010-")
    # Fetch and scoring are separate passes — a shard must not leave a score.
    assert sentiment is None and confidence is None
    # One URL per announcement, or the dedupe index collapses a whole day's
    # filings into a single row.
    assert "newsid=" in url
    assert len({r[6] for r in rows}) == 3

    assert store(rows) == 3

    conn = get_connection()
    try:
        stored = _execute(
            conn,
            "SELECT COUNT(*), COUNT(sentiment) FROM news_sentiment WHERE symbol = ? AND source = 'BSE'",
            ("HDFCBANK.NS",),
        ).fetchone()
    finally:
        release_connection(conn)
    assert stored == (3, 0)


def test_bse_refetch_does_not_duplicate():
    """uq_news_url_pubdate is what makes a re-run idempotent.

    The index lives in schema_pg.py but was missing there for a long time while
    present on prod, so this asserts the schema this suite builds actually has
    it — without it every "ON CONFLICT DO NOTHING" in the news collectors is a
    silent no-op and re-runs pile up duplicates.
    """
    from collectors.bse_announcements_collector import parse_announcements, store

    fixture = load_fixture("bse_announcements")
    payload = {k: v for k, v in fixture.items() if k != "_mirrors"}
    rows = parse_announcements(payload, "HDFCBANK.NS")

    store(rows)
    store(rows)

    conn = get_connection()
    try:
        count = _execute(
            conn, "SELECT COUNT(*) FROM news_sentiment WHERE symbol = ?", ("HDFCBANK.NS",),
        ).fetchone()[0]
    finally:
        release_connection(conn)
    assert count == 3


def test_bse_pagination_stops_at_row_count():
    """A short final page ends the loop instead of paging forever."""
    from collectors.bse_announcements_collector import fetch_symbol_window
    from datetime import date

    fixture = load_fixture("bse_announcements")
    page = {"Table": fixture["Table"], "Table1": [{"ROWCNT": 6}]}
    session = _FakeBsePages([page, page])

    rows = fetch_symbol_window(session, "500180", "HDFCBANK.NS",
                               date(2010, 1, 1), date(2010, 3, 31))

    # Page 1 returns 3 of a stated 6 rows — fewer than the 50-row page size, so
    # the range is exhausted and page 2 is never requested.
    assert len(rows) == 3
    assert len(session.requested) == 1
    assert session.requested[0]["strPrevDate"] == "20100101"
    assert session.requested[0]["strToDate"] == "20100331"


def test_nse_announcement_urls_are_unique_per_filing():
    """Same symbol, same day, different filings must not collide.

    They used to: every row shared one URL, and (url, published_at) is the
    dedupe key, so a day on which a company filed results AND a board-meeting
    notice AND a disclosure kept exactly one of them.
    """
    from collectors.nse_announcements_collector import _ann_url
    from datetime import datetime

    day = datetime(2018, 5, 2)
    a = _ann_url("HDFCBANK", day, "Board meeting intimation")
    b = _ann_url("HDFCBANK", day, "Audited financial results for Q4 FY18")

    assert a != b
    assert _ann_url("HDFCBANK", day, "Board meeting intimation") == a  # stable
