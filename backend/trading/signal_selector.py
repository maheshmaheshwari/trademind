"""
Conviction-ranked trade selection (recommend-only).

Turns the day's signals into a *ranked, capacity-aware set of trades to take* —
the portfolio-construction logic the backtester proved generates the alpha:

  - only STRONG BUY signals (the only net-profitable tier after costs),
  - ranked by conviction (confidence = calibrated win probability),
  - capped at MAX_CONCURRENT total open positions (counting what's already held),
  - risk-sized (RISK_FRAC of equity ÷ stop distance, capped at MAX_POS_FRAC),
  - constrained by available cash, skipping symbols already held.

It RECOMMENDS only — it does not place or authorize any trade. The user reviews
the list and authorizes via the normal autopilot flow. PAPER context: equity and
cash come from the user's virtual balance + open positions.

Backtest basis: first-come selection lost money (~-5% CAGR); this conviction-
ranked + concurrency-capped construction produced +10-17% CAGR vs a flat market.
"""
from database.db import get_connection, release_connection, _execute, _rows_to_dicts

# Defaults mirror the best backtested config (Cap 8, conviction-ranked).
MAX_CONCURRENT = 8      # total open positions the strategy holds at once
RISK_FRAC      = 0.01   # risk this fraction of equity per trade (via stop distance)
MAX_POS_FRAC   = 0.10   # cap any single position at this fraction of equity
MIN_CONFIDENCE = 0.0    # optional floor on confidence % (0 = rank only, no floor)

# Model-quality floor (percentages). A STRONG BUY only needs precision>=55% at
# the signal gate, so a mediocre model with an overconfident buy_prob can still
# emit one — ~23% of STRONG BUYs come from models with acc<70 or prec<60. We do
# NOT recommend trades from those: a user shouldn't act on a signal whose model
# is barely better than a coin flip, however confident that model claims to be.
MIN_ACCURACY   = 70.0   # model test accuracy % required to be recommendable
MIN_PRECISION  = 60.0   # model test precision % required to be recommendable


def recommend_trades(user_id: int,
                     max_concurrent: int = MAX_CONCURRENT,
                     risk_frac: float = RISK_FRAC,
                     max_pos_frac: float = MAX_POS_FRAC,
                     min_confidence: float = MIN_CONFIDENCE,
                     min_accuracy: float = MIN_ACCURACY,
                     min_precision: float = MIN_PRECISION) -> dict:
    """Return the conviction-ranked set of STRONG BUY trades to take now (PAPER).

    Only models that clear the quality floor (accuracy >= min_accuracy AND
    precision >= min_precision) are recommendable — a confident signal from an
    unreliable model is filtered out, not surfaced. Does NOT place trades.
    Sizing uses the user's virtual balance (cash) and total equity (cash +
    currently-invested). Positions already held count against the concurrency
    cap and are never re-recommended.
    """
    conn = get_connection()
    try:
        urow = _execute(conn, "SELECT virtual_balance FROM users WHERE id = ?", (user_id,)).fetchone()
        if not urow:
            return {"recommendations": [], "reason": "user not found"}
        cash = float(urow[0] or 0)

        pos = _rows_to_dicts(_execute(conn,
            "SELECT symbol, quantity, avg_buy_price, invested_amount FROM positions WHERE user_id = ?",
            (user_id,)))
        held = {p["symbol"] for p in pos}
        invested = sum(float(p.get("invested_amount") or (p["quantity"] * p["avg_buy_price"])) for p in pos)
        equity = cash + invested
        slots = max_concurrent - len(held)

        if slots <= 0:
            return {"recommendations": [], "held_positions": len(held),
                    "slots_available": 0, "equity": round(equity, 2),
                    "reason": f"portfolio full ({len(held)}/{max_concurrent} positions)"}

        rows = _rows_to_dicts(_execute(conn, """
            SELECT symbol, name, confidence, buy_price, target_price, stop_loss,
                   risk_reward, model_horizon, expected_return_pct,
                   model_accuracy, model_precision
            FROM trade_signals
            WHERE generated_date = (SELECT MAX(generated_date) FROM trade_signals)
              AND is_active = TRUE AND signal = 'STRONG BUY'
            ORDER BY confidence DESC,
                     (COALESCE(model_accuracy, 0) + COALESCE(model_precision, 0)) DESC
        """, ()))

        recs = []
        avail = cash
        rank = 0
        filtered_low_quality = 0
        for r in rows:
            if slots <= 0 or avail < equity * 0.005:
                break
            sym = r["symbol"]
            buy = r.get("buy_price"); stop = r.get("stop_loss")
            conf = float(r.get("confidence") or 0)
            acc = r.get("model_accuracy"); prec = r.get("model_precision")
            if sym in held or conf < min_confidence:
                continue
            # Quality floor: never recommend a trade from an unreliable model,
            # no matter how confident it claims to be.
            if acc is None or prec is None or acc < min_accuracy or prec < min_precision:
                filtered_low_quality += 1
                continue
            if not buy or not stop or buy <= stop:
                continue
            stop_frac = (buy - stop) / buy
            pos_value = min(risk_frac * equity / stop_frac, max_pos_frac * equity, avail)
            qty = int(pos_value // buy)
            if qty < 1:
                continue
            cost = round(qty * buy, 2)
            if cost > avail:
                continue
            rank += 1
            recs.append({
                "rank": rank,
                "symbol": sym,
                "name": r.get("name"),
                "confidence": round(conf, 1),
                "horizon": r.get("model_horizon"),
                "qty": qty,
                "investment": cost,
                "buy_price": buy,
                "target_price": r.get("target_price"),
                "stop_loss": stop,
                "risk_reward": r.get("risk_reward"),
                "expected_return_pct": r.get("expected_return_pct"),
                "model_accuracy": r.get("model_accuracy"),
                "model_precision": r.get("model_precision"),
            })
            avail -= cost
            slots -= 1
            held.add(sym)

        return {
            "recommendations": recs,
            "count": len(recs),
            "held_positions": len(pos),
            "slots_available": max_concurrent - len(pos),
            "max_concurrent": max_concurrent,
            "filtered_low_quality": filtered_low_quality,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "cash_after": round(avail, 2),
            "params": {"risk_frac": risk_frac, "max_pos_frac": max_pos_frac,
                       "min_confidence": min_confidence,
                       "min_accuracy": min_accuracy, "min_precision": min_precision},
            "note": "Recommend-only (PAPER). Ranked by conviction, capped at "
                    f"{max_concurrent} concurrent positions, risk-sized, and limited to "
                    f"models with accuracy>={min_accuracy:.0f}% and precision>={min_precision:.0f}%. "
                    "No trades placed.",
        }
    finally:
        release_connection(conn)
