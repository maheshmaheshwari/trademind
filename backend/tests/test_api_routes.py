"""
API route tests against the TEST database.

Pattern: seed the test DB the same way production data would arrive
(direct INSERT via the same helpers/SQL the app itself uses), then hit the
real route through TestClient and assert on the response — mirrors how
values actually enter the system (collectors/schedulers -> DB -> API).

A few routes are file-backed instead of DB-backed (signals/all,
backtest/summary) — those are tested by monkeypatching the route module's
path constant to a fixture file, so they never touch the real
backend/data/*.json used by the live app.
"""
import json
from datetime import datetime

from database.db import get_connection, release_connection, _execute

from conftest import load_fixture

_TODAY = datetime.now().strftime("%Y-%m-%d")


def _insert_trade_signal(conn, **kwargs):
    defaults = dict(
        symbol="TESTSTOCK.NS", name="Test Stock Ltd.", signal="BUY", confidence=85.0,
        trade_type="LONG", buy_price=100.0, target_price=110.0, stop_loss=95.0,
        risk_reward=2.0, expected_return_pct=10.0, current_price=100.0,
        atr_14=2.5, atr_pct=2.5, avg_daily_volume=500000, daily_turnover_cr=5.0,
        liquidity="HIGH", max_safe_qty=100, max_qty_per_user=10,
        max_investment_per_user=10000.0, min_qty=1, recommended_volume=100,
        consumed_volume=0, model_name="XGBoost_1M", model_horizon="1 Month",
        model_accuracy=82.0, model_precision=74.0, top_drivers='["RSI","MACD"]',
        sentiment=0.1, generated_date=_TODAY, generated_at=f"{_TODAY} 09:00:00",
        is_active=True,
    )
    defaults.update(kwargs)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    _execute(
        conn,
        f"INSERT INTO trade_signals ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


def test_sentiment_market_route(api_client):
    from database.db import insert_news
    insert_news(
        headline="Nifty hits fresh high on strong FII inflows",
        source="rss", published_at="2026-06-24 08:00:00", symbol=None,
        sentiment="positive", confidence=0.8,
    )

    resp = api_client.get("/api/sentiment/market")
    assert resp.status_code == 200
    body = resp.json()
    live_shape = load_fixture("api_sentiment_market")
    assert set(["score", "label", "article_count", "breakdown", "news"]) <= set(body.keys())
    assert set(["score", "label", "article_count", "breakdown", "news"]) <= set(live_shape.keys())
    assert any("Nifty hits fresh high" in n.get("headline", "") for n in body["news"])


def test_news_market_route(api_client):
    from database.db import insert_news
    insert_news(
        headline="RBI holds repo rate steady at policy meet",
        source="rss", published_at="2026-06-24 07:00:00", symbol=None,
    )

    resp = api_client.get("/api/news/market")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert any("RBI holds repo rate" in n.get("headline", "") for n in body["data"])


def test_signals_top_buys_route(api_client):
    conn = get_connection()
    try:
        _insert_trade_signal(conn, symbol="BUYSTOCK.NS", signal="BUY", confidence=95.0)
        _insert_trade_signal(conn, symbol="SELLSTOCK.NS", signal="SELL", confidence=90.0)
    finally:
        release_connection(conn)

    resp = api_client.get("/api/signals/top-buys")
    assert resp.status_code == 200
    symbols = [s["symbol"] for s in resp.json()["signals"]]
    assert "BUYSTOCK.NS" in symbols
    assert "SELLSTOCK.NS" not in symbols


def test_signals_top_sells_route(api_client):
    conn = get_connection()
    try:
        _insert_trade_signal(conn, symbol="BUYSTOCK.NS", signal="BUY", confidence=95.0)
        _insert_trade_signal(conn, symbol="SELLSTOCK.NS", signal="SELL", confidence=90.0)
    finally:
        release_connection(conn)

    resp = api_client.get("/api/signals/top-sells")
    assert resp.status_code == 200
    symbols = [s["symbol"] for s in resp.json()["signals"]]
    assert "SELLSTOCK.NS" in symbols
    assert "BUYSTOCK.NS" not in symbols


def test_portfolio_sectors_route(api_client):
    conn = get_connection()
    try:
        _insert_trade_signal(conn, symbol="RELIANCE.NS", signal="BUY", confidence=88.0)
    finally:
        release_connection(conn)

    resp = api_client.get("/api/portfolio/sectors")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_sectors" in body and body["total_sectors"] > 0


def test_signals_all_route_file_backed(api_client, monkeypatch, tmp_path):
    """File-backed route — fixture substituted via the module's path constant."""
    import api.routes.signals as signals_module

    raw_fixture = load_fixture("trade_signals_latest_raw")
    raw_fixture.pop("_mirrors")
    fixture_path = tmp_path / "trade_signals_latest.json"
    fixture_path.write_text(json.dumps(raw_fixture))
    monkeypatch.setattr(signals_module, "_SIGNALS_FILE", str(fixture_path))

    resp = api_client.get("/api/signals/all")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["signals"][0]["symbol"] == "TESTSTOCK.NS"
    assert body["signals"][0]["signal"] == "BUY"


def test_backtest_summary_route_file_backed(api_client, monkeypatch, tmp_path):
    """File-backed route (retrain_results.csv) — point DATA_DIR at an empty tmp dir."""
    import api.routes.backtest as backtest_module
    monkeypatch.setattr(backtest_module, "DATA_DIR", tmp_path)

    resp = api_client.get("/api/backtest/summary")
    assert resp.status_code == 200
    body = resp.json()
    live_shape = load_fixture("api_backtest_summary")
    assert set(["model_stats", "signal_stats", "history"]) <= set(body.keys())
    assert set(["model_stats", "signal_stats", "history"]) <= set(live_shape.keys())
