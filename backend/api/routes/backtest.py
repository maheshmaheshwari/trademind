"""
Backtest & Model Performance API
GET /api/backtest/summary — aggregated model stats, signal history, top signals
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter

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


def _load_latest():
    path = DATA_DIR / "trade_signals_latest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_history():
    path = DATA_DIR / "trade_history.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


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
    latest     = _load_latest()
    actionable = latest.get("actionable_trades", [])
    summary    = latest.get("summary", {})

    confs    = [t.get("confidence", 0)                            for t in actionable]
    exp_rets = [t.get("trade", {}).get("expected_return_pct", 0)  for t in actionable]
    avg_conf     = round(sum(confs)    / len(confs),    1) if confs    else 0
    avg_exp_ret  = round(sum(exp_rets) / len(exp_rets), 1) if exp_rets else 0

    by_signal: dict = defaultdict(lambda: {"count": 0, "confs": [], "rets": []})
    for t in actionable:
        sig = t.get("signal", "HOLD")
        by_signal[sig]["count"] += 1
        by_signal[sig]["confs"].append(t.get("confidence", 0))
        by_signal[sig]["rets"].append(t.get("trade", {}).get("expected_return_pct", 0))

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
    history_raw = _load_history()
    timeline = []
    for run in history_raw:
        s = run.get("summary", {})
        trades = run.get("actionable_trades", [])
        rets = [t.get("trade", {}).get("expected_return_pct", 0) for t in trades]
        timeline.append({
            "date":               run.get("generated_at", "")[:10],
            "total_signals":      sum(s.values()),
            "buy_signals":        s.get("STRONG_BUY", 0) + s.get("BUY", 0),
            "sell_signals":       s.get("SELL", 0)       + s.get("STRONG_SELL", 0),
            "avg_expected_return": round(sum(rets) / len(rets), 1) if rets else 0,
        })

    # Add the latest run as the final data point
    timeline.append({
        "date":               (latest.get("generated_at") or "")[:10],
        "total_signals":      latest.get("total_signals", 0),
        "buy_signals":        summary.get("STRONG_BUY", 0) + summary.get("BUY", 0),
        "sell_signals":       summary.get("SELL", 0)       + summary.get("STRONG_SELL", 0),
        "avg_expected_return": avg_exp_ret,
    })
    timeline = sorted({t["date"]: t for t in timeline}.values(), key=lambda x: x["date"])

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
