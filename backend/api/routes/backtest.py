"""
Backtest & Model Performance API
GET /api/backtest/summary — aggregated model stats, signal history, top signals
"""
import csv
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter
from database.db import get_connection, release_connection, _execute, _rows_to_dicts

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _load_csv():
    path = DATA_DIR / "retrain_results.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_latest_from_db() -> dict:
    """Load latest trade signals from DB (latest generated_date, active only)."""
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT MAX(generated_date) FROM trade_signals", ())
        row = cur.fetchone()
        latest_date = row[0] if row and row[0] else None
        if not latest_date:
            return {}

        cur = _execute(conn,
            "SELECT symbol, signal, confidence, model_horizon, expected_return_pct, "
            "buy_price, target_price, stop_loss, risk_reward, generated_at "
            "FROM trade_signals WHERE generated_date = ? AND is_active = TRUE ORDER BY confidence DESC",
            (latest_date,))
        rows = _rows_to_dicts(cur)

        cur2 = _execute(conn, "SELECT MAX(generated_at) FROM trade_signals WHERE generated_date = ?", (latest_date,))
        gen_at_row = cur2.fetchone()

        actionable = [r for r in rows if r.get("signal") in ("STRONG BUY", "BUY")]
        avoid      = [r for r in rows if r.get("signal") in ("SELL", "STRONG SELL")]
        hold       = [r for r in rows if r.get("signal") == "HOLD"]
        return {
            "generated_at":    str(gen_at_row[0]) if gen_at_row and gen_at_row[0] else "",
            "total_signals":   len(rows),
            "actionable_trades": actionable,
            "avoid_list":      avoid,
            "hold_list":       hold,
            "summary": {
                "STRONG_BUY": sum(1 for r in rows if r.get("signal") == "STRONG BUY"),
                "BUY":        sum(1 for r in rows if r.get("signal") == "BUY"),
                "HOLD":       sum(1 for r in rows if r.get("signal") == "HOLD"),
                "SELL":       sum(1 for r in rows if r.get("signal") == "SELL"),
                "STRONG_SELL":sum(1 for r in rows if r.get("signal") == "STRONG SELL"),
            },
        }
    except Exception:
        return {}
    finally:
        release_connection(conn)


def _load_history_from_db() -> list:
    """Load per-date signal run summaries from DB (one row per generated_date)."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT
                generated_date,
                MAX(generated_at) AS generated_at,
                COUNT(*)          AS total_signals,
                SUM(CASE WHEN signal IN ('STRONG BUY','BUY') THEN 1 ELSE 0 END)   AS buy_count,
                SUM(CASE WHEN signal IN ('SELL','STRONG SELL') THEN 1 ELSE 0 END) AS sell_count,
                AVG(expected_return_pct) AS avg_expected_return
            FROM trade_signals
            GROUP BY generated_date
            ORDER BY generated_date ASC
        """, ())
        rows = cur.fetchall()
        return [
            {
                "generated_at":        str(r[1]),
                "date":                str(r[0]),
                "total_signals":       r[2],
                "buy_signals":         r[3],
                "sell_signals":        r[4],
                "avg_expected_return": round(float(r[5] or 0), 1),
                "summary": {},
                "actionable_trades":   [],
            }
            for r in rows
        ]
    except Exception:
        return []
    finally:
        release_connection(conn)


@router.get("/summary")
async def get_backtest_summary():
    rows = _load_csv()
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    # ── Model stats ───────────────────────────────────────────────────────
    accs  = [_safe_float(r["accuracy"])  for r in ok_rows if _safe_float(r["accuracy"])  > 0]
    precs = [_safe_float(r["precision"]) for r in ok_rows if _safe_float(r["precision"]) > 0]
    avg_acc  = round(sum(accs)  / len(accs)  * 100, 1) if accs  else 0
    avg_prec = round(sum(precs) / len(precs) * 100, 1) if precs else 0
    high_quality = sum(
        1 for r in ok_rows
        if _safe_float(r["accuracy"]) >= 0.70 and _safe_float(r["precision"]) >= 0.70
    )

    horizon_order = ["1 Week", "2 Weeks", "1 Month", "2 Months", "3 Months", "6 Months"]
    horizon_data: dict = defaultdict(lambda: {"accs": [], "precs": [], "count": 0})
    for r in ok_rows:
        h = r.get("horizon", "")
        if not h:
            continue
        a = _safe_float(r["accuracy"])
        p = _safe_float(r["precision"])
        if a > 0:
            horizon_data[h]["accs"].append(a)
        if p > 0:
            horizon_data[h]["precs"].append(p)
        horizon_data[h]["count"] += 1

    by_horizon = []
    for h in horizon_order:
        if h in horizon_data:
            d = horizon_data[h]
            by_horizon.append({
                "horizon": h,
                "avg_accuracy":  round(sum(d["accs"])  / len(d["accs"])  * 100, 1) if d["accs"]  else 0,
                "avg_precision": round(sum(d["precs"]) / len(d["precs"]) * 100, 1) if d["precs"] else 0,
                "count": d["count"],
            })

    model_counts: dict = defaultdict(int)
    for r in ok_rows:
        m = r.get("best_model", "")
        if m:
            model_counts[m] += 1
    by_model_type = sorted(
        [{"model": k, "count": v} for k, v in model_counts.items()],
        key=lambda x: -x["count"],
    )

    # ── Signal stats (latest) ─────────────────────────────────────────────
    latest     = _load_latest_from_db()
    actionable = latest.get("actionable_trades", [])
    summary    = latest.get("summary", {})

    # DB rows are flat — expected_return_pct is a direct column, not nested under "trade"
    all_signals = latest.get("actionable_trades", []) + latest.get("avoid_list", []) + latest.get("hold_list", [])
    confs    = [t.get("confidence", 0)          for t in actionable]
    exp_rets = [t.get("expected_return_pct", 0) for t in actionable]
    avg_conf     = round(sum(confs)    / len(confs),    1) if confs    else 0
    avg_exp_ret  = round(sum(exp_rets) / len(exp_rets), 1) if exp_rets else 0

    by_signal: dict = defaultdict(lambda: {"count": 0, "confs": [], "rets": []})
    for t in all_signals:
        sig = t.get("signal", "HOLD")
        by_signal[sig]["count"] += 1
        by_signal[sig]["confs"].append(t.get("confidence", 0))
        by_signal[sig]["rets"].append(t.get("expected_return_pct", 0))

    signal_order = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    by_signal_list = []
    for sig in signal_order:
        d = by_signal[sig]
        cs, rs = d["confs"], d["rets"]
        by_signal_list.append({
            "signal": sig,
            "count": d["count"],
            "avg_confidence":      round(sum(cs) / len(cs), 1) if cs else 0,
            "avg_expected_return": round(sum(rs) / len(rs), 1) if rs else 0,
        })

    top_signals = sorted(actionable, key=lambda t: -t.get("confidence", 0))[:10]

    # ── History timeline ──────────────────────────────────────────────────
    history_raw = _load_history_from_db()
    timeline = [
        {
            "date":               run.get("date", run.get("generated_at", ""))[:10],
            "total_signals":      run.get("total_signals", 0),
            "buy_signals":        run.get("buy_signals", 0),
            "sell_signals":       run.get("sell_signals", 0),
            "avg_expected_return": run.get("avg_expected_return", 0),
        }
        for run in history_raw
    ]
    timeline = sorted({t["date"]: t for t in timeline if t["date"]}.values(), key=lambda x: x["date"])

    return {
        "model_stats": {
            "total_models":    len(rows),
            "successful_models": len(ok_rows),
            "avg_accuracy":    avg_acc,
            "avg_precision":   avg_prec,
            "high_quality_models": high_quality,
            "by_horizon":      by_horizon,
            "by_model_type":   by_model_type,
        },
        "signal_stats": {
            "generated_at":      latest.get("generated_at", ""),
            "total_signals":     latest.get("total_signals", 0),
            "distribution":      summary,
            "avg_confidence":    avg_conf,
            "avg_expected_return": avg_exp_ret,
            "by_signal_type":    by_signal_list,
            "top_signals":       top_signals,
        },
        "history": timeline,
    }
