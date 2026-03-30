"""
Retrain the 225 models that failed to load due to sklearn version mismatch.

Identifies pickled RandomForest/GradientBoosting models that cannot be loaded
with the current sklearn version, retrains them from local DB data, and saves
them to final_models/ in the same joblib format.

Usage:
    cd backend
    python retrain_failed_models.py              # retrain all 225 failed models
    python retrain_failed_models.py --dry-run    # just list which need retraining
    python retrain_failed_models.py --symbol TCS # retrain single symbol
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import joblib

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FINAL_MODELS_DIR = os.path.join(_BACKEND_DIR, "final_models")
MODELS_DIR = os.path.join(_BACKEND_DIR, "models")


def find_failed_models() -> list[str]:
    """Return list of symbol strings (e.g. 'TCS.NS') whose final model fails to load."""
    failed = []
    for path in sorted(glob.glob(os.path.join(FINAL_MODELS_DIR, "*_final.pkl"))):
        try:
            with open(path, "rb") as f:
                pickle.load(f)
        except Exception:
            symbol = os.path.basename(path).replace("_final.pkl", "")
            failed.append(symbol)
    return failed


def retrain_symbol(symbol_ns: str) -> dict | None:
    """
    Retrain the model for a single symbol using local DB data.

    Returns a result dict on success, None on failure.
    """
    from analysis.model_training import train_and_evaluate  # type: ignore

    # train_and_evaluate saves to models/best_<symbol>_v3.pkl
    try:
        train_and_evaluate(symbol_ns)
    except Exception as exc:
        logger.error(f"[{symbol_ns}] train_and_evaluate failed: {exc}")
        return None

    src_path = os.path.join(MODELS_DIR, f"best_{symbol_ns}_v3.pkl")
    if not os.path.exists(src_path):
        logger.warning(f"[{symbol_ns}] No model file found at {src_path}")
        return None

    try:
        artifact = joblib.load(src_path)
    except Exception as exc:
        logger.error(f"[{symbol_ns}] Could not load freshly trained model: {exc}")
        return None

    # Overwrite the broken final model
    dst_path = os.path.join(FINAL_MODELS_DIR, f"{symbol_ns}_final.pkl")
    joblib.dump(artifact, dst_path)

    return {
        "symbol": symbol_ns,
        "model_name": artifact.get("model_name", "?"),
        "horizon": artifact.get("horizon", "?"),
        "accuracy": artifact["metrics"]["accuracy"],
        "precision": artifact["metrics"]["precision"],
        "f1": artifact["metrics"]["f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain sklearn models broken by version mismatch")
    parser.add_argument("--dry-run", action="store_true", help="List failed models without retraining")
    parser.add_argument("--symbol", default=None, help="Retrain a single symbol (e.g. TCS.NS or TCS)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.symbol:
        sym = args.symbol if args.symbol.endswith(".NS") else f"{args.symbol}.NS"
        failed = [sym]
    else:
        print("Scanning final_models/ for unloadable models ...")
        failed = find_failed_models()

    print(f"\nFound {len(failed)} model(s) to retrain.\n")

    if args.dry_run:
        for s in failed:
            print(f"  {s}")
        return

    if not failed:
        print("Nothing to do.")
        return

    success, errors = [], []
    total = len(failed)

    for idx, symbol_ns in enumerate(failed, 1):
        print(f"\n[{idx}/{total}] Retraining {symbol_ns} ...")
        t0 = time.time()
        result = retrain_symbol(symbol_ns)
        elapsed = time.time() - t0

        if result:
            success.append(result)
            print(
                f"  ✓ {result['model_name']} @ {result['horizon']} | "
                f"Acc={result['accuracy']:.1%} Prec={result['precision']:.1%} "
                f"F1={result['f1']:.1%} ({elapsed:.0f}s)"
            )
        else:
            errors.append(symbol_ns)
            print(f"  ✗ Failed ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    print(f"Retraining complete: {len(success)}/{total} succeeded, {len(errors)} failed")

    if success:
        avg_acc = sum(r["accuracy"] for r in success) / len(success)
        avg_prec = sum(r["precision"] for r in success) / len(success)
        print(f"Average accuracy:  {avg_acc:.1%}")
        print(f"Average precision: {avg_prec:.1%}")

    if errors:
        print(f"\nFailed symbols ({len(errors)}):")
        for s in errors:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
