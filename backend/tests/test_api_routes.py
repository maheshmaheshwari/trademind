"""
API route tests against the TEST database.

Pattern: seed the test DB the same way production data would arrive
(direct INSERT via the same helpers/SQL the app itself uses), then hit the
real route through TestClient and assert on the response — mirrors how
values actually enter the system (collectors/schedulers -> DB -> API).

All signal routes (signals/all, backtest/summary, stocks) are DB-backed.
backtest/summary reads trade_signals (signal stats) and model_training_stats
(per-symbol model metrics, written by scripts/retrain_walk_forward.py) — both
from the DB, no file inputs (retrain_results.csv is retired; see CLAUDE.md).
"""
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


def test_signals_all_route_db_backed(api_client, monkeypatch):
    """DB-backed route — seeds trade_signals and reads back through the API.

    The route keeps an in-process cache keyed on generated_date; reset it so
    a previous test's (or process's) payload can't be served here.
    """
    import api.routes.signals as signals_module
    monkeypatch.setattr(signals_module, "_cache", {"date": None, "payload": None, "ts": 0.0})

    conn = get_connection()
    try:
        _insert_trade_signal(conn, symbol="TESTSTOCK.NS", signal="BUY", confidence=85.0)
    finally:
        release_connection(conn)

    resp = api_client.get("/api/signals/all")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["signals"][0]["symbol"] == "TESTSTOCK.NS"
    assert body["signals"][0]["signal"] == "BUY"


def test_signals_all_current_price_comes_from_prices_not_the_signal(api_client, monkeypatch):
    """CMP on the signal lists is the latest `prices` bar, not the signal's own
    current_price snapshot — otherwise a day the generator did not run would
    render a stale figure under a "current price" heading."""
    import api.routes.signals as signals_module
    from database.db import insert_prices_batch
    monkeypatch.setattr(signals_module, "_cache", {"date": None, "payload": None, "ts": 0.0})

    conn = get_connection()
    try:
        # Signal was generated when the stock was at 100.
        _insert_trade_signal(conn, symbol="CMPSTOCK.NS", current_price=100.0)
    finally:
        release_connection(conn)

    # …the stock has since moved to 137.40, over two bars.
    insert_prices_batch([
        ("CMPSTOCK.NS", "NSE", "2026-08-11", None, 100, 101, 99, 100.0, 1000, "1d"),
        ("CMPSTOCK.NS", "NSE", "2026-08-12", None, 130, 138, 129, 137.40, 1200, "1d"),
    ])

    resp = api_client.get("/api/signals/all")
    assert resp.status_code == 200
    sig = next(s for s in resp.json()["signals"] if s["symbol"] == "CMPSTOCK.NS")
    assert sig["current_price"] == 137.40


def test_signals_all_current_price_is_null_without_price_bars(api_client, monkeypatch):
    """A signal for a symbol with no price rows reports null rather than 0 — the
    UI renders an em dash, which is honest; ₹0.00 would not be."""
    import api.routes.signals as signals_module
    monkeypatch.setattr(signals_module, "_cache", {"date": None, "payload": None, "ts": 0.0})

    conn = get_connection()
    try:
        _insert_trade_signal(conn, symbol="NOPRICE.NS")
    finally:
        release_connection(conn)

    resp = api_client.get("/api/signals/all")
    assert resp.status_code == 200
    sig = next(s for s in resp.json()["signals"] if s["symbol"] == "NOPRICE.NS")
    assert sig["current_price"] is None


def test_backtest_summary_route_db_backed(api_client):
    """Model stats come from model_training_stats (latest run_id only), signal
    stats from trade_signals — both DB-backed, no file inputs. Seed one run and
    assert the route surfaces exactly that run's rows."""
    from database.db import insert_model_training_stats
    insert_model_training_stats("test_run_1", [
        {"symbol": "AAA.NS", "status": "ok", "best_model": "XGBoost", "horizon": "1 Month",
         "accuracy": 0.82, "precision": 0.75, "recall": 0.6, "f1": 0.67, "quality_tier": "high"},
        {"symbol": "BBB.NS", "status": "ok", "best_model": "RandForest", "horizon": "1 Week",
         "accuracy": 0.55, "precision": 0.50, "recall": 0.4, "f1": 0.44, "quality_tier": "low"},
        {"symbol": "CCC.NS", "status": "no_data", "best_model": "", "horizon": "",
         "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "quality_tier": None},
    ])

    resp = api_client.get("/api/backtest/summary")
    assert resp.status_code == 200
    body = resp.json()
    live_shape = load_fixture("api_backtest_summary")
    assert set(["model_stats", "signal_stats", "history"]) <= set(body.keys())
    assert set(["model_stats", "signal_stats", "history"]) <= set(live_shape.keys())

    ms = body["model_stats"]
    assert ms["total_models"] == 3          # all rows in the run
    assert ms["successful_models"] == 2     # status == "ok"
    assert ms["high_quality_models"] == 1   # AAA (acc>=.70 & prec>=.70)
    models = {m["model"] for m in ms["by_model_type"]}
    assert models == {"XGBoost", "RandForest"}


# ---------------------------------------------------------------------------
# insert_trade_signals_batch — is_active bookkeeping
#
# The API serves on is_active (db.get_trade_signals_formatted, stocks.py), so a
# row the newest run did not write must not stay live. Two ways that used to
# leak, both observed in production on 2026-08-07/08-10 — see the comment in
# insert_trade_signals_batch.
# ---------------------------------------------------------------------------

def _trade(symbol="AAA.NS", signal="BUY", horizon="1 Month", confidence=80.0,
           recommended_volume=100):
    """Minimal trade dict in the shape generate_trades.py hands to the batch insert."""
    return {
        "symbol": symbol, "name": f"{symbol} Ltd.", "signal": signal,
        "confidence": confidence,
        "trade": {"type": "LONG", "buy_price": 100.0, "target_price": 110.0,
                  "stop_loss": 95.0, "risk_reward": 2.0, "expected_return_pct": 10.0},
        "price": {"current": 100.0, "atr_14": 2.5, "atr_pct": 2.5},
        "position": {"avg_daily_volume": 500000, "daily_turnover_cr": 5.0,
                     "liquidity": "HIGH", "max_safe_qty": 100, "max_qty_per_user": 10,
                     "max_investment_per_user": 10000.0, "min_qty": 1,
                     "recommended_volume": recommended_volume},
        "model": {"name": "XGBoost", "horizon": horizon,
                  "accuracy": 82.0, "precision": 74.0},
        "top_drivers": ["RSI"], "sentiment": {},
    }


def _active_rows(conn):
    cur = _execute(conn,
        "SELECT symbol, model_horizon, generated_date, is_active FROM trade_signals "
        "ORDER BY symbol, model_horizon")
    return {(r[0], r[1], str(r[2])): r[3] for r in cur.fetchall()}


def test_same_date_rerun_retires_horizons_it_no_longer_emits():
    """The 2026-08-07 weekly retrain case: a second run on the SAME date whose
    models pick different best horizons. The upsert key includes model_horizon,
    so dropped horizons are never overwritten — they must be retired instead."""
    from database.db import insert_trade_signals_batch
    conn = get_connection()
    try:
        insert_trade_signals_batch(
            [_trade(horizon="1 Month"), _trade(horizon="1 Week", signal="SELL")],
            _TODAY, f"{_TODAY} 10:00:00")
        # Retrained models now favour 1 Month only.
        insert_trade_signals_batch(
            [_trade(horizon="1 Month", signal="STRONG BUY", confidence=95.0)],
            _TODAY, f"{_TODAY} 22:30:00")

        rows = _active_rows(conn)
        assert rows[("AAA.NS", "1 Month", _TODAY)] is True
        assert rows[("AAA.NS", "1 Week", _TODAY)] is False, \
            "dropped horizon from the same date stayed live"

        # The surviving row carries the newer run's values.
        cur = _execute(conn,
            "SELECT signal, confidence FROM trade_signals "
            "WHERE symbol=? AND model_horizon=? AND generated_date=?",
            ("AAA.NS", "1 Month", _TODAY))
        assert cur.fetchone()[:2] == ("STRONG BUY", 95.0)
    finally:
        release_connection(conn)


def test_symbol_absent_from_batch_is_retired():
    """A symbol that drops out of a run (de-indexed, training failed) used to keep
    its last signals active forever — the UPDATE was scoped to batch symbols."""
    from database.db import insert_trade_signals_batch
    conn = get_connection()
    try:
        insert_trade_signals_batch(
            [_trade(symbol="AAA.NS"), _trade(symbol="GONE.NS")],
            "2026-08-04", "2026-08-04 16:00:00")
        insert_trade_signals_batch(
            [_trade(symbol="AAA.NS")], "2026-08-05", "2026-08-05 16:00:00")

        rows = _active_rows(conn)
        assert rows[("GONE.NS", "1 Month", "2026-08-04")] is False, \
            "symbol absent from the newer batch stayed live"
        assert rows[("AAA.NS", "1 Month", "2026-08-04")] is False
        assert rows[("AAA.NS", "1 Month", "2026-08-05")] is True
    finally:
        release_connection(conn)


def test_reemitted_horizon_is_reactivated():
    """Guards the upsert trap: run 3 re-emits a horizon that run 2 retired, so it
    collides with run 1's now-inactive row. Without is_active in the ON CONFLICT
    update list the fresh signal would be written but never served."""
    from database.db import insert_trade_signals_batch
    conn = get_connection()
    try:
        insert_trade_signals_batch(
            [_trade(horizon="1 Month"), _trade(horizon="1 Week")],
            _TODAY, f"{_TODAY} 10:00:00")
        insert_trade_signals_batch(
            [_trade(horizon="1 Month")], _TODAY, f"{_TODAY} 16:00:00")
        insert_trade_signals_batch(
            [_trade(horizon="1 Week", signal="STRONG SELL")],
            _TODAY, f"{_TODAY} 22:00:00")

        rows = _active_rows(conn)
        assert rows[("AAA.NS", "1 Week", _TODAY)] is True, \
            "re-emitted horizon stayed retired"
        assert rows[("AAA.NS", "1 Month", _TODAY)] is False
    finally:
        release_connection(conn)


def test_consumed_volume_survives_a_same_date_rerun():
    """The carry-forward reads is_active = TRUE, so retiring must happen after the
    inserts, not before — otherwise capacity tracking resets on every refresh."""
    from database.db import insert_trade_signals_batch
    conn = get_connection()
    try:
        insert_trade_signals_batch([_trade()], _TODAY, f"{_TODAY} 10:00:00")
        _execute(conn, "UPDATE trade_signals SET consumed_volume = 42 WHERE symbol = ?",
                 ("AAA.NS",))
        conn.commit()

        insert_trade_signals_batch([_trade()], _TODAY, f"{_TODAY} 16:00:00")

        cur = _execute(conn,
            "SELECT consumed_volume FROM trade_signals WHERE symbol=? AND is_active = TRUE",
            ("AAA.NS",))
        assert cur.fetchone()[0] == 42
    finally:
        release_connection(conn)


def test_signals_route_serves_only_the_newest_run(api_client):
    """End-to-end: the route reads is_active, so the orphan rows must not surface."""
    from database.db import insert_trade_signals_batch
    insert_trade_signals_batch(
        [_trade(horizon="1 Month"), _trade(horizon="1 Week", signal="SELL")],
        _TODAY, f"{_TODAY} 10:00:00")
    insert_trade_signals_batch(
        [_trade(horizon="1 Month", signal="STRONG BUY", confidence=95.0)],
        _TODAY, f"{_TODAY} 22:30:00")

    resp = api_client.get("/api/signals/all")
    assert resp.status_code == 200
    signals = resp.json().get("signals") or []
    horizons = [s.get("horizon_long") for s in signals]
    assert horizons == ["1 Month"], f"stale horizon served: {horizons}"
    assert signals[0].get("raw_signal") == "STRONG BUY"
