"""
Model Metrics

Calculate and store model performance metrics.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dict with metric values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Map to signal types (0=AVOID, 1=HOLD, 2=BUY)
    if len(precision) >= 3:
        metrics["precision_avoid"] = float(precision[0])
        metrics["precision_hold"] = float(precision[1])
        metrics["precision_buy"] = float(precision[2])
        metrics["recall_avoid"] = float(recall[0])
        metrics["recall_hold"] = float(recall[1])
        metrics["recall_buy"] = float(recall[2])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def backtest_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lookahead_return: float = 5.0,  # Assumed return for correct BUY
    loss_return: float = -3.0,  # Assumed loss for wrong BUY
) -> dict:
    """
    Simple backtest simulation.
    
    Assumes:
    - A BUY signal that is correct yields +5% return
    - A BUY signal that is AVOID yields -3% return
    - HOLD signals are neutral (0%)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        lookahead_return: Return for correct BUY
        loss_return: Loss for incorrect BUY
        
    Returns:
        Dict with backtest metrics
    """
    returns = []
    
    for true, pred in zip(y_true, y_pred):
        if pred == 2:  # Predicted BUY
            if true == 2:  # Correct BUY
                returns.append(lookahead_return)
            elif true == 0:  # Actually AVOID
                returns.append(loss_return)
            else:  # Actually HOLD
                returns.append(0)
        else:
            returns.append(0)  # No position
    
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = np.sum(returns)
    avg_return = np.mean(returns)
    
    # Number of trades
    n_trades = np.sum(y_pred == 2)
    winning_trades = np.sum((y_pred == 2) & (y_true == 2))
    win_rate = winning_trades / n_trades if n_trades > 0 else 0
    
    # Sharpe ratio (simplified, assuming daily returns)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / len(returns))
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Profit factor
    profits = returns[returns > 0].sum() if any(returns > 0) else 0
    losses = abs(returns[returns < 0].sum()) if any(returns < 0) else 0
    profit_factor = profits / losses if losses > 0 else profits
    
    return {
        "total_return": float(total_return),
        "avg_return": float(avg_return),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
    }


def calculate_feature_importance_report(
    feature_importance: dict,
    top_n: int = 10,
) -> str:
    """
    Generate human-readable feature importance report.
    
    Args:
        feature_importance: Dict of feature -> importance
        top_n: Number of top features to show
        
    Returns:
        Formatted report string
    """
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    report = "Top Feature Importance:\n"
    report += "-" * 40 + "\n"
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        bar = "█" * int(importance * 50)
        report += f"{i:2}. {feature:<20} {importance:.4f} {bar}\n"
    
    return report
