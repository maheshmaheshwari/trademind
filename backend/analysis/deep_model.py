"""
TradeMind AI — TabNet Deep Learning Model (Priority 10)

TabNet uses sequential attention to select which features matter for each prediction.
Effectively learns feature interactions automatically — often outperforms GBM on
financial tabular data with 100+ features.

Used as an additional model in train_and_evaluate() alongside XGBoost/LightGBM.

Requirements: pip install pytorch-tabnet torch
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if TabNet + torch are installed."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
        return True
    except ImportError:
        return False


def train_tabnet(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pos_weight: float = 1.0,
    max_epochs: int = 200,
    patience: int = 20,
):
    """
    Train a TabNet classifier on tabular financial data.

    Args:
        X_tr:       Training features DataFrame
        y_tr:       Training labels Series
        X_val:      Validation features DataFrame
        y_val:      Validation labels Series
        pos_weight: Class imbalance weight (neg/pos ratio)
        max_epochs: Maximum training epochs
        patience:   Early stopping patience

    Returns:
        Fitted TabNetClassifier, or None on failure.
    """
    if not is_available():
        logger.warning("TabNet not available — install pytorch-tabnet")
        return None

    if len(X_tr) < 100 or len(np.unique(y_tr)) < 2:
        logger.warning("Insufficient training data for TabNet")
        return None

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        # Auto-detect device: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        clf = TabNetClassifier(
            n_d=16,              # embedding dimension (width)
            n_a=16,              # attention embedding dimension
            n_steps=5,           # number of sequential attention steps
            gamma=1.3,           # coefficient for feature reusage
            n_independent=2,     # independent GLU layers per step
            n_shared=2,          # shared GLU layers per step
            lambda_sparse=1e-4,  # sparsity regularisation (feature selection)
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
            scheduler_params=dict(step_size=50, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            device_name=device,
            verbose=0,
        )

        # Class weights to handle imbalance
        class_weights = {0: 1.0, 1: float(pos_weight)}

        clf.fit(
            X_train=X_tr.values.astype(np.float32),
            y_train=y_tr.values,
            eval_set=[(X_val.values.astype(np.float32), y_val.values)],
            eval_metric=["logloss"],
            max_epochs=max_epochs,
            patience=patience,
            weights=class_weights,
            batch_size=min(256, len(X_tr)),
            virtual_batch_size=min(128, len(X_tr) // 2),
            drop_last=False,
        )

        logger.info(f"TabNet trained on {device} | best epoch: {clf.best_epoch}")
        return clf

    except Exception as e:
        logger.warning(f"TabNet training failed: {e}")
        return None


def tabnet_predict_proba(clf, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from a TabNet model."""
    try:
        proba = clf.predict_proba(X.values.astype(np.float32))
        return proba[:, 1]
    except Exception as e:
        logger.warning(f"TabNet predict failed: {e}")
        return np.full(len(X), 0.5)


def tabnet_feature_importance(clf, feature_names: list) -> dict:
    """Return feature importances from TabNet's attention weights."""
    try:
        importances = clf.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))
    except Exception:
        return {}
