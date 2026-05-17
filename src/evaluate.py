"""
PainPrint - Evaluation & Metrics
Author: Rajat Mishra
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all relevant regression metrics for pain prediction."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)

    # Clinically significant error (pain literature: 1.5 threshold)
    clinically_acceptable = (mae < 1.5)

    results = {
        'MAE':  round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2':   round(r2, 4),
        'Pearson_r': round(r, 4),
        'Pearson_p': round(p, 6),
        'Clinically_Acceptable (MAE<1.5)': clinically_acceptable,
    }

    print("\n" + "="*45)
    print("  PainPrint Evaluation Report | Rajat Mishra")
    print("="*45)
    for k, v in results.items():
        flag = "✅" if (k == 'Clinically_Acceptable (MAE<1.5)' and v) else ""
        print(f"  {k:<38}: {v} {flag}")
    print("="*45)
    return results


def pain_spike_detection(y_true, y_pred, threshold=7.0):
    """Binary classification metrics for flare/spike detection."""
    true_spikes = (y_true >= threshold).astype(int)
    pred_spikes = (y_pred >= threshold).astype(int)

    tp = np.sum((true_spikes == 1) & (pred_spikes == 1))
    fp = np.sum((true_spikes == 0) & (pred_spikes == 1))
    fn = np.sum((true_spikes == 1) & (pred_spikes == 0))
    tn = np.sum((true_spikes == 0) & (pred_spikes == 0))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(y_true)

    print(f"\n  Flare Detection (threshold={threshold})")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")

    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(0)
    y_true = np.random.uniform(2, 8, 500)
    y_pred = y_true + np.random.normal(0, 1.0, 500)
    y_pred = np.clip(y_pred, 0, 10)

    full_evaluation(y_true, y_pred)
    pain_spike_detection(y_true, y_pred)
