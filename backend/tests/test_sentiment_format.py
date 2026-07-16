"""
news_daily_sentiment cagg — numeric sentiment format (schema_pg.py).

The sentiment column's canonical format is a signed numeric string; the cagg
must aggregate those correctly and still tolerate legacy text labels. The old
definition silently zeroed every numeric row — these tests would have caught
that.
"""
import pytest

from database.db import get_connection, release_connection, _execute


SYM = "CAGGTEST.NS"


def _refresh_cagg(conn):
    old = conn.autocommit
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CALL refresh_continuous_aggregate('news_daily_sentiment', NULL, NULL)")
    conn.autocommit = old


class TestNewsDailySentimentCagg:
    def test_numeric_and_label_rows_aggregate_correctly(self):
        conn = get_connection()
        try:
            rows = [
                ("strong buy news", "0.85", 0.85),    # numeric positive
                ("bad results",     "-0.9", 0.9),     # numeric negative
                ("filing notice",   "0.0",  0.95),    # numeric neutral (FinBERT zero)
                ("legacy row",      "positive", 0.7), # legacy label — must still count
            ]
            for headline, sentiment, conf in rows:
                _execute(conn, """INSERT INTO news_sentiment
                    (headline, source, published_at, symbol, sentiment, confidence)
                    VALUES (?, 'test', '2026-07-10 09:00+00', ?, ?, ?)""",
                    (headline, SYM, sentiment, conf))
            conn.commit()
            _refresh_cagg(conn)

            cur = _execute(conn, """SELECT news_count, positive_count, negative_count,
                                           neutral_count, avg_sentiment
                                    FROM news_daily_sentiment WHERE symbol = ?""", (SYM,))
            r = cur.fetchone()
            assert r is not None, "cagg row missing for test symbol"
            news, pos, neg, neu, avg = r
            assert news == 4
            assert pos == 2   # 0.85 + legacy 'positive'
            assert neg == 1
            assert neu == 1
            # (0.85 - 0.9 + 0.0 + 0.7) / 4 = 0.1625
            assert avg == pytest.approx(0.1625, abs=1e-6)
        finally:
            release_connection(conn)
