"""
Rolling-TRAIN_END A/B test (Point 2, Phase A — sample first, no prod writes).

The full-universe backtest showed the strategy's alpha is front-loaded in 2025
and decays to flat by 2026 — the fingerprint of models frozen at 2024-12-31
going stale. This tests the fix on a SAMPLE: does retraining on more recent
data restore the 2026 edge?

Design (fair A/B on the same symbols, same 2026 out-of-sample window):
  - FROZEN vintage  : existing final_models/ (trained through 2024-12-31)
  - ROLLING vintage : retrain the sample through 2025-12-31  → final_models_rolling/
  - Evaluate BOTH on 2026-01-01 → present via the signal backtester.

If ROLLING's 2026 P&L clearly beats FROZEN's, rolling the window restores the
edge and it's worth implementing for all 480 models. If not, the 2025 edge was
regime-specific, not staleness — important to know before a full retrain.

Writes only to scratch dirs (final_models_rolling/, final_models_frozen_sample/)
— never touches the live final_models/ set.

Usage:  python scripts/test_rolling_trainend.py --symbols 15
"""
import argparse
import os
import shutil
import subprocess
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FROZEN_DIR = "final_models"
ROLLING_DIR = "final_models_rolling"
FROZEN_SAMPLE_DIR = "final_models_frozen_sample"

ROLLING_TRAIN_END = "2025-12-31"
ROLLING_TEST_START = "2026-01-01"   # models never see 2026 → genuine out-of-sample
EVAL_TEST_START = "2026-01-01"      # both vintages judged on 2026 only


def train_rolling_sample(symbols):
    from analysis.model_training import train_and_evaluate
    os.makedirs(ROLLING_DIR, exist_ok=True)
    os.makedirs(FROZEN_SAMPLE_DIR, exist_ok=True)
    trained = []
    for i, sym in enumerate(symbols, 1):
        frozen_path = os.path.join(FROZEN_DIR, f"{sym}_final.pkl")
        if not os.path.exists(frozen_path):
            continue
        shutil.copy(frozen_path, os.path.join(FROZEN_SAMPLE_DIR, f"{sym}_final.pkl"))
        out = os.path.join(ROLLING_DIR, f"{sym}_final.pkl")
        if os.path.exists(out):
            trained.append(sym); print(f"  [{i}/{len(symbols)}] {sym}: rolling model exists, skip"); continue
        try:
            # output_dir keeps train_and_evaluate's own saves off final_models/;
            # we re-dump the returned artifact under the .NS name the backtester globs.
            art = train_and_evaluate(sym, train_end_date=ROLLING_TRAIN_END,
                                     test_start_date=ROLLING_TEST_START, output_dir=ROLLING_DIR)
            if art is None:
                print(f"  [{i}/{len(symbols)}] {sym}: no model (insufficient data)"); continue
            joblib.dump(art, out)
            trained.append(sym)
            print(f"  [{i}/{len(symbols)}] {sym}: rolling model trained ✅")
        except Exception as e:
            print(f"  [{i}/{len(symbols)}] {sym}: FAILED {e}")
    return trained


def run_backtest(model_dir, label):
    print(f"\n########## {label}  ({model_dir}, 2026 out-of-sample) ##########")
    subprocess.run([
        sys.executable, "scripts/backtest_signals.py",
        "--model-dir", model_dir, "--test-start", EVAL_TEST_START,
        "--symbols", "999", "--every", "1",
        "--portfolio", "--min-rr", "1.0", "--cost-pct", "0.4",
    ], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-train", action="store_true", help="reuse existing rolling models")
    args = ap.parse_args()

    files = sorted(f.replace("_final.pkl", "") for f in os.listdir(FROZEN_DIR)
                   if f.endswith(".NS_final.pkl"))
    rng = np.random.default_rng(args.seed)
    sample = list(rng.choice(files, size=min(args.symbols, len(files)), replace=False))
    print(f"Sample ({len(sample)}): {', '.join(sample)}")

    if not args.skip_train:
        print(f"\nTraining ROLLING vintage (train_end={ROLLING_TRAIN_END}) — this is the slow part...")
        trained = train_rolling_sample(sample)
        print(f"\nTrained {len(trained)} rolling models.")
    else:
        trained = sample

    print("\n" + "=" * 78)
    print("A/B RESULT — same symbols, same 2026 window, two model vintages")
    print("=" * 78)
    run_backtest(FROZEN_SAMPLE_DIR, "FROZEN vintage (train through 2024-12-31)")
    run_backtest(ROLLING_DIR, "ROLLING vintage (train through 2025-12-31)")
    print("\nCompare the two headline CAGRs / win-rates above. Rolling >> Frozen ⇒ "
          "staleness confirmed, implement rolling for all.")


if __name__ == "__main__":
    main()
