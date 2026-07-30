"""
Signal strategy backtester.

Replays the PRODUCTION signal logic over the models' out-of-sample window
(2025-01-01 onward — models are trained through 2024-12-31) and simulates the
resulting trades against ACTUAL forward prices, so signal parameters (MIN_RR,
the expected-value gate) can be chosen from realized performance instead of
eyeballing signal counts.

For every (symbol, horizon, date) in the test window it:
  1. runs the horizon model to get buy_prob (same inference as generate_trades),
  2. builds target/stop with the production trade-level formula (RR-enforced),
  3. decides the signal with generate_trades.decide_signal (the EV gate),
  4. if it's a BUY, walks forward up to `forward_days` of real OHLC and records
     whether the target or the stop was hit first (win/loss), or marks to market
     at the horizon end (timeout).

Then it reports realized win-rate, average P&L per trade, profit factor and a
sequential-equity max drawdown, and sweeps MIN_RR to compare settings.

Two phases: buy_prob (the expensive part) is independent of MIN_RR, so it is
computed ONCE; each MIN_RR setting only re-derives levels/decision/outcome.

No look-ahead: features at date T use only data up to T (rolling windows); the
model never saw 2025+ in training; outcomes use only prices strictly after T.

Usage (from backend/, venv active):
    python scripts/backtest_signals.py                 # 60-symbol sample, MIN_RR sweep
    python scripts/backtest_signals.py --symbols 120
    python scripts/backtest_signals.py --min-rr 1.8    # single setting
    python scripts/backtest_signals.py --every 3       # sample every 3rd trading day (faster)
"""
import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.model_training import load_data_for_symbol, engineer_features_and_target  # noqa: E402
import scripts.generate_trades as gt  # noqa: E402

FINAL_DIR = "final_models"
OUTPUT_DIR = "data"
TEST_START = pd.Timestamp("2025-01-01")   # first out-of-sample day (train ends 2024-12-31)

# Round-trip cost for one delivery trade (buy + sell), as % of position value.
# Indian equity delivery, discount-broker assumption:
#   STT           0.10% buy + 0.10% sell        = 0.20%
#   stamp duty    0.015% (buy only)             = 0.015%
#   exch txn+SEBI+GST (~0.01% each side)        ~ 0.02%
#   brokerage     ~0 (discount broker, delivery)= 0.00%
#   ----------------------------------------------------
#   statutory subtotal                          ~ 0.235%
#   slippage      (default ~0.15% round trip)   = tunable
# Default 0.40% ≈ statutory + modest slippage. Override with --cost-pct.
DEFAULT_COST_PCT = 0.40


def _is_tabnet(m):
    return "tabnet" in type(m).__module__.lower()


def _buy_prob_series(art_h, feat_df):
    """Vectorized buy_prob for every row of feat_df, mirroring generate_trades.

    TabNet horizons are SKIPPED (NaN) on purpose: in production, TabNet's
    predict_proba throws on a DataFrame and calculate_signal swallows it, so
    those horizons emit no live signal. Skipping them here matches production
    behavior and avoids TabNet's very slow per-call inference."""
    features = art_h["features"]
    X = feat_df.reindex(columns=features, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    model = art_h.get("model")
    if model is not None:
        if _is_tabnet(model):
            return np.full(len(X), np.nan)
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]
    subs = art_h.get("sub_models") or {}
    weights = art_h.get("sub_weights") or {}
    num = np.zeros(len(X)); wsum = 0.0
    for mn, sm in subs.items():
        if sm is None or _is_tabnet(sm):
            continue
        try:
            p = sm.predict_proba(X)
            pr = p[:, 1] if p.shape[1] > 1 else p[:, 0]
        except Exception:
            continue
        w = abs(weights.get(mn, 1.0)); num += pr * w; wsum += w
    return num / wsum if wsum else np.full(len(X), np.nan)


def _levels(close, atr, target_pct):
    """Production long trade-level formula — reads gt.MIN_RR / gt.ATR_FLOOR_MULT live."""
    target = round(close * (1 + target_pct / 100), 2)
    reward = target - close
    stop = round(close - max(reward / gt.MIN_RR, atr * gt.ATR_FLOOR_MULT), 2)
    rr = round(reward / (close - stop), 2) if (close - stop) > 0 else 0
    return target, stop, rr


def _simulate(entry, target, stop, hi, lo, cl):
    """Walk forward: which of target/stop hits first?

    Returns (outcome, exit_price, days_held) — days_held is the number of
    trading days the position is open (for the portfolio capital model)."""
    for j in range(len(hi)):
        if lo[j] <= stop:                  # same-day ambiguity → pessimistic (stop first)
            return "loss", stop, j + 1
        if hi[j] >= target:
            return "win", target, j + 1
    return "timeout", (cl[-1] if len(cl) else entry), len(hi)


def build_candidates(picks, every):
    """Phase 1 (expensive, once): buy_prob + context per (symbol, horizon, date).

    Loads price data per symbol (bounded latency); the batch prefetch JOIN was
    tried but hangs on the degraded Timescale Cloud instance."""
    raw_cache = {}   # sym -> (close, high, low) arrays for forward simulation
    cands = []       # dicts with everything needed to decide+simulate under any MIN_RR
    for idx, fn in enumerate(picks, 1):
        print(f"  [{idx}/{len(picks)}] {fn.replace('_final.pkl','')} ...", flush=True)
        sym = fn.replace("_final.pkl", "")
        try:
            art = joblib.load(os.path.join(FINAL_DIR, fn))
            raw = load_data_for_symbol(sym)
        except Exception:
            continue
        if raw is None or len(raw) < 250:
            continue
        raw = raw.sort_index()
        r_close = raw["close"].to_numpy(float)
        r_high = raw["high"].to_numpy(float)
        r_low = raw["low"].to_numpy(float)
        r_atr = (raw["atr_14"] if "atr_14" in raw else raw["close"] * 0.02).to_numpy(float)
        raw_cache[sym] = (r_close, r_high, r_low, list(raw.index))
        pos = {d: i for i, d in enumerate(raw.index)}
        # Features are identical across horizons (only the discarded target uses
        # forward_days/target_pct) — engineer ONCE, and slice to the test window
        # BEFORE inference so the model (esp. slow TabNet) only scores test bars.
        feats_all, _ = engineer_features_and_target(raw, forward_days=1, target_pct=1.0)
        feats = feats_all[feats_all.index >= TEST_START]
        if feats.empty:
            continue
        for hz in (art.get("horizons") or {}).values():
            fwd = int(hz.get("forward_days", 20)); tgt = float(hz.get("target_pct", 3.5))
            prec = float((hz.get("metrics") or {}).get("precision", 0.0))
            probs = _buy_prob_series(hz, feats)
            fdates = list(feats.index)
            for k in range(0, len(fdates), every):
                T = fdates[k]
                if T < TEST_START or not np.isfinite(probs[k]):
                    continue
                i = pos.get(T)
                if i is None or i + fwd >= len(raw):
                    continue
                cands.append({
                    "sym": sym, "i": i, "date": T, "horizon": hz.get("horizon"),
                    "buy_prob": float(probs[k]), "prec": prec,
                    "close": r_close[i], "atr": r_atr[i], "tgt": tgt, "fwd": fwd,
                })
        if idx % 20 == 0:
            print(f"  ... {idx}/{len(picks)} symbols scanned, {len(cands)} candidate bars")
    return cands, raw_cache


def evaluate(cands, raw_cache, min_rr, cost_pct):
    """Phase 2 (cheap): decide + simulate every candidate under one MIN_RR.

    ret_pct is NET of the round-trip cost; gross_pct is before costs.
    """
    gt.MIN_RR = min_rr
    trades = []
    for c in cands:
        target, stop, rr = _levels(c["close"], c["atr"], c["tgt"])
        sig = gt.decide_signal(c["buy_prob"], c["prec"], rr)
        if sig not in ("STRONG BUY", "BUY"):
            continue
        r_close, r_high, r_low, r_dates = raw_cache[c["sym"]]
        i, fwd = c["i"], c["fwd"]
        outcome, exit_px, held = _simulate(c["close"], target, stop,
                                           r_high[i+1:i+1+fwd], r_low[i+1:i+1+fwd], r_close[i+1:i+1+fwd])
        gross = (exit_px / c["close"] - 1) * 100
        exit_date = r_dates[min(i + held, len(r_dates) - 1)]
        stop_frac = (c["close"] - stop) / c["close"] if c["close"] else 0.0
        trades.append({"symbol": c["sym"], "date": c["date"], "exit_date": exit_date,
                       "signal": sig, "rr": rr, "outcome": outcome, "stop_frac": stop_frac,
                       "buy_prob": c["buy_prob"], "gross_pct": gross, "ret_pct": gross - cost_pct})
    return trades


def summarize(trades, label):
    if not trades:
        print(f"\n{label}: no trades"); return
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df.outcome == "win"]; losses = df[df.outcome == "loss"]; tmo = df[df.outcome == "timeout"]
    gross_win = df[df.ret_pct > 0].ret_pct.sum(); gross_loss = abs(df[df.ret_pct < 0].ret_pct.sum())
    pf = gross_win / gross_loss if gross_loss else float("inf")
    eq = (1 + df.sort_values("date").ret_pct / 100).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    realized_rr = (wins.ret_pct.mean() / abs(losses.ret_pct.mean())) if len(losses) else float("nan")
    print(f"\n{label}")
    print(f"  trades={n}  win={len(wins)} ({len(wins)/n*100:.1f}%)  loss={len(losses)}  timeout={len(tmo)}")
    print(f"  NET avg P&L/trade={df.ret_pct.mean():+.2f}%   median={df.ret_pct.median():+.2f}%   "
          f"profit_factor={pf:.2f}")
    print(f"  gross avg/trade={df.gross_pct.mean():+.2f}%   (cost drag {df.gross_pct.mean()-df.ret_pct.mean():.2f}%/trade)   "
          f"realized RR={realized_rr:.2f}")
    for s in ("STRONG BUY", "BUY"):
        sub = df[df.signal == s]
        if len(sub):
            print(f"    {s:11s}: {len(sub):5d} trades  win {(sub.outcome=='win').mean()*100:4.1f}%  "
                  f"avg {sub.ret_pct.mean():+.2f}%")


def cost_sensitivity(trades, costs, label):
    """Net win-rate / avg / profit-factor at several round-trip cost levels.

    Reported for all BUYs and for STRONG BUY only, since gross returns are the
    same regardless of cost (cost is a flat per-trade deduction)."""
    if not trades:
        print(f"\n{label}: no trades"); return
    df = pd.DataFrame(trades)
    print(f"\n{label}   (n={len(df)}, STRONG BUY={int((df.signal=='STRONG BUY').sum())}, "
          f"BUY={int((df.signal=='BUY').sum())})")
    for name, sub in [("all BUYs", df), ("STRONG BUY only", df[df.signal == "STRONG BUY"])]:
        if not len(sub):
            continue
        print(f"  {name}:")
        print(f"    {'cost%':>6} {'net avg':>9} {'net med':>9} {'profit_factor':>14} {'edge?':>7}")
        for c in costs:
            net = sub.gross_pct - c
            gw = net[net > 0].sum(); gl = abs(net[net < 0].sum())
            pf = gw / gl if gl else float("inf")
            edge = "yes" if net.mean() > 0 and pf > 1 else "no"
            print(f"    {c:>6.2f} {net.mean():>+8.2f}% {net.median():>+8.2f}% {pf:>14.2f} {edge:>7}")


def _curve_stats(eq, capital):
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / capital - 1
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1
    maxdd = float((eq / eq.cummax() - 1).min())
    return total, cagr, maxdd, years


def benchmark(raw_cache, capital):
    """Buy-and-hold baselines over the identical window — the alpha-vs-beta test.

    (a) Equal-weight buy-and-hold of the SAME symbols the strategy traded:
        each symbol bought with equal rupees at its first test-window bar and
        held; equity = mean of per-symbol normalised closes.
    (b) Nifty 500 index over the same window (from market_overview)."""
    print("\n═══ Buy-and-hold benchmarks (same window) ═══")
    out = {}

    # (a) equal-weight universe
    series = []
    for sym, (close, _hi, _lo, dates) in raw_cache.items():
        s = pd.Series(close, index=pd.to_datetime(dates))
        s = s[~s.index.duplicated(keep="last")].sort_index()   # some feeds have dup dates
        s = s[s.index >= TEST_START]
        if len(s) >= 20 and s.iloc[0] > 0:
            series.append(s / s.iloc[0])
    if series:
        ew = pd.concat(series, axis=1).sort_index().mean(axis=1)  # equal-weight, rebalanced
        eq = (ew / ew.iloc[0] * capital)
        total, cagr, maxdd, yrs = _curve_stats(eq, capital)
        print(f"  equal-weight universe ({len(series)} stocks): total {total:+.1%}  "
              f"CAGR {cagr:+.1%}  maxDD {maxdd:.1%}  over {yrs:.2f}y")
        eqm = eq.groupby(eq.index.normalize()).last()
        out["equal_weight"] = {"label": f"Equal-weight ({len(series)} stocks)",
                               "total_return_pct": round(total*100, 1), "cagr_pct": round(cagr*100, 1),
                               "max_drawdown_pct": round(maxdd*100, 1),
                               "curve": [[d.strftime("%Y-%m-%d"), round(float(e), 0)] for d, e in eqm.items()]}

    # (b) Nifty 500 index
    try:
        from database.db import get_connection, release_connection, _execute
        conn = get_connection()
        # Use TEST_START so the Nifty benchmark spans the SAME window as the
        # strategy/equal-weight (was hardcoded 2025-01-01 → mismatched window).
        rows = _execute(conn, "SELECT date, nifty500_close FROM market_overview "
                              "WHERE date >= ? AND nifty500_close IS NOT NULL "
                              "ORDER BY date", (str(TEST_START.date()),)).fetchall()
        release_connection(conn)
        if rows and len(rows) > 20:
            s = pd.Series([float(r[1]) for r in rows], index=pd.to_datetime([r[0] for r in rows]))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            eq = s / s.iloc[0] * capital
            total, cagr, maxdd, yrs = _curve_stats(eq, capital)
            print(f"  Nifty 500 index:                     total {total:+.1%}  "
                  f"CAGR {cagr:+.1%}  maxDD {maxdd:.1%}  over {yrs:.2f}y")
            eqm = eq.groupby(eq.index.normalize()).last()
            out["nifty500"] = {"label": "Nifty 500", "total_return_pct": round(total*100, 1),
                               "cagr_pct": round(cagr*100, 1), "max_drawdown_pct": round(maxdd*100, 1),
                               "curve": [[d.strftime("%Y-%m-%d"), round(float(e), 0)] for d, e in eqm.items()]}
    except Exception as e:
        print(f"  (Nifty benchmark unavailable: {e})")
    return out


def portfolio_sim(trades, capital, risk_frac, max_pos_frac, strong_only,
                  max_concurrent=None, min_conf=0.0, label=""):
    """Capital-constrained portfolio: risk-based sizing, non-overlapping capital.

    Selectivity levers (this is the test of whether being pickier rescues the
    strategy):
      - min_conf: only take signals with buy_prob >= this (make the call rarer).
      - max_concurrent: hold at most N positions at once; and when several
        signals fire the same day, prefer the HIGHEST-conviction ones (sorted
        by buy_prob) so the limited slots go to the best calls, not first-come.

    Each trade risks `risk_frac` of equity (position = risk / stop-distance),
    capped at `max_pos_frac` and by cash. Drawdown is on the realized-equity
    curve (open losers not marked), the standard trade-based approximation."""
    df = pd.DataFrame(trades)
    if strong_only:
        df = df[df.signal == "STRONG BUY"]
    if min_conf > 0:
        df = df[df.buy_prob >= min_conf]
    # date first (chronology), conviction second → best signals win limited slots
    df = df.sort_values(["date", "buy_prob"], ascending=[True, False]).reset_index(drop=True)
    if df.empty:
        print(f"\n{label}: no trades after filters"); return

    cash = equity = float(capital)
    open_pos = []        # (exit_date, capital_deployed, net_ret_pct)
    taken = skipped_cash = skipped_slot = 0
    peak_open = 0
    curve = [(df.iloc[0]["date"], equity)]
    cap_max = max_concurrent or 10**9

    def close_matured(upto):
        nonlocal cash, equity, open_pos
        keep = []
        for xd, cap, nr in open_pos:
            if xd <= upto:
                pnl = cap * nr / 100.0
                cash += cap + pnl; equity += pnl
                curve.append((xd, equity))
            else:
                keep.append((xd, cap, nr))
        open_pos = keep

    for _, t in df.iterrows():
        close_matured(t["date"])
        if len(open_pos) >= cap_max:
            skipped_slot += 1; continue
        sf = t["stop_frac"]
        if sf <= 0:
            continue
        size = min(risk_frac * equity / sf, max_pos_frac * equity, cash)
        if size < capital * 0.005:          # <0.5% of base capital → treat as no cash
            skipped_cash += 1; continue
        cash -= size
        open_pos.append((t["exit_date"], size, t["ret_pct"]))
        taken += 1
        peak_open = max(peak_open, len(open_pos))
    close_matured(pd.Timestamp.max)

    cur = pd.DataFrame(curve, columns=["date", "equity"]).sort_values("date")
    # collapse to one point per day (last equity of the day) for a clean curve
    cur = cur.groupby(cur["date"].dt.normalize())["equity"].last().reset_index()
    cur.columns = ["date", "equity"]
    eq = cur["equity"].to_numpy(float)
    years = max((cur["date"].iloc[-1] - cur["date"].iloc[0]).days / 365.25, 1e-9)
    total_ret = eq[-1] / capital - 1
    cagr = (eq[-1] / capital) ** (1 / years) - 1
    maxdd = float((cur["equity"] / cur["equity"].cummax() - 1).min())

    # per-calendar-half sub-period returns → is the edge consistent or one stretch?
    sub = {}
    cur2 = cur.set_index("date")["equity"]
    for period, grp in cur2.groupby([cur2.index.year, cur2.index.month.map(lambda m: "H1" if m <= 6 else "H2")]):
        if len(grp) >= 2:
            sub[f"{period[0]}-{period[1]}"] = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)

    conf = f"  min_conf={min_conf:.2f}" if min_conf else ""
    slots = f"  max_concurrent={max_concurrent}" if max_concurrent else "  no cap"
    print(f"\n{label}{slots}{conf}")
    print(f"  trades taken={taken}  skipped(cash)={skipped_cash}  skipped(slot)={skipped_slot}  "
          f"peak concurrent={peak_open}")
    print(f"  final equity=₹{eq[-1]:,.0f}   total return={total_ret:+.1%} over {years:.2f}y   "
          f"CAGR={cagr:+.1%}   maxDD={maxdd:.1%}")
    print(f"  sub-period returns: {sub}")
    return {
        "label": label, "total_return_pct": round(total_ret * 100, 1),
        "cagr_pct": round(cagr * 100, 1), "max_drawdown_pct": round(maxdd * 100, 1),
        "trades_taken": taken, "peak_concurrent": peak_open, "years": round(years, 2),
        "max_concurrent": max_concurrent, "min_conf": round(min_conf, 2),
        "subperiods": sub,
        "curve": [[d.strftime("%Y-%m-%d"), round(float(e), 0)]
                  for d, e in zip(cur["date"], cur["equity"])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=60)
    ap.add_argument("--every", type=int, default=1, help="sample every Nth trading day")
    ap.add_argument("--min-rr", type=float, default=None, help="single MIN_RR (default: sweep)")
    ap.add_argument("--cost-pct", type=float, default=DEFAULT_COST_PCT,
                    help="round-trip cost per trade, %% of position (default %(default)s)")
    ap.add_argument("--cost-sweep", action="store_true",
                    help="at MIN_RR=1.0, report net metrics across cost levels, all vs STRONG-only")
    ap.add_argument("--portfolio", action="store_true",
                    help="capital-constrained portfolio sim (equity curve, CAGR, drawdown)")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--risk-frac", type=float, default=0.01, help="risk per trade as frac of equity")
    ap.add_argument("--max-pos-frac", type=float, default=0.10, help="cap on any single position")
    ap.add_argument("--include-buy", action="store_true", help="portfolio: include BUY tier (default STRONG only)")
    ap.add_argument("--save", action="store_true", help="write data/strategy_backtest.json for the app")
    ap.add_argument("--model-dir", type=str, default=None, help="dir of *.NS_final.pkl (default final_models)")
    ap.add_argument("--test-start", type=str, default=None, help="out-of-sample start, e.g. 2026-01-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global FINAL_DIR, TEST_START
    if args.model_dir:
        FINAL_DIR = args.model_dir
    if args.test_start:
        TEST_START = pd.Timestamp(args.test_start)

    files = sorted(f for f in os.listdir(FINAL_DIR) if f.endswith(".NS_final.pkl"))
    rng = np.random.default_rng(args.seed)
    picks = list(rng.choice(files, size=min(args.symbols, len(files)), replace=False))
    print(f"Backtesting {len(picks)} symbols over {TEST_START.date()}→present (every={args.every})")

    cands, raw_cache = build_candidates(picks, args.every)
    print(f"\nBuilt {len(cands)} candidate bars from {len(raw_cache)} symbols.")

    if args.cost_sweep:
        mr = args.min_rr or 1.0
        trades = evaluate(cands, raw_cache, mr, 0.0)   # cost applied analytically below
        cost_sensitivity(trades, [0.20, 0.40, 0.60, 0.75], f"═══ MIN_RR = {mr} · cost sensitivity ═══")
        return

    if args.portfolio:
        mr = args.min_rr or 1.0
        trades = evaluate(cands, raw_cache, mr, args.cost_pct)
        # buy_prob distribution → pick sensible conviction floors
        bp = pd.Series([t["buy_prob"] for t in trades if t["signal"] == "STRONG BUY"])
        print(f"\nSTRONG BUY buy_prob: median={bp.median():.2f}  p75={bp.quantile(.75):.2f}  "
              f"p90={bp.quantile(.90):.2f}  max={bp.max():.2f}")
        base = dict(capital=args.capital, risk_frac=args.risk_frac,
                    max_pos_frac=args.max_pos_frac, strong_only=not args.include_buy)
        bench = benchmark(raw_cache, args.capital)
        print(f"\n═══ Portfolio · MIN_RR={mr} · cost {args.cost_pct:.2f}% — selectivity comparison ═══")
        configs = [
            portfolio_sim(trades, **base, max_concurrent=None, min_conf=0.0, label="Conviction-ranked, no cap"),
            portfolio_sim(trades, **base, max_concurrent=8, min_conf=0.0, label="Cap 8, conviction-ranked"),
            portfolio_sim(trades, **base, max_concurrent=8, min_conf=float(bp.quantile(.75)), label="Cap 8 + conf≥p75"),
            portfolio_sim(trades, **base, max_concurrent=5, min_conf=float(bp.quantile(.90)), label="Cap 5 + conf≥p90"),
        ]
        if args.save:
            payload = {
                "window_start": str(TEST_START.date()),
                "universe_symbols": len(raw_cache),
                "cost_pct": args.cost_pct, "min_rr": mr,
                "capital": args.capital, "risk_frac": args.risk_frac,
                "benchmarks": bench,
                "configs": [c for c in configs if c],
                # headline = best config by CAGR, for the product's summary card
                "headline": max((c for c in configs if c), key=lambda c: c["cagr_pct"]),
            }
            # Stored in the DB (strategy_backtest_results), never a JSON file —
            # the HF Space never receives data/*, so a file would be invisible
            # in production (see CLAUDE.md: all app data lives in the database).
            from database.db import insert_strategy_backtest
            insert_strategy_backtest(payload)
            print("\n💾 Saved backtest results → strategy_backtest_results (DB)")
        return

    print(f"Round-trip cost = {args.cost_pct:.2f}%/trade")
    settings = [args.min_rr] if args.min_rr else [1.0, 1.5, 1.8, 2.0, 2.5]
    for mr in settings:
        summarize(evaluate(cands, raw_cache, mr, args.cost_pct), f"═══ MIN_RR = {mr} ═══")


if __name__ == "__main__":
    main()
