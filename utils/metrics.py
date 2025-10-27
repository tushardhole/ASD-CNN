from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute binary classification metrics.

    Args:
        y_true: list or np.array of ground truth labels (0 or 1)
        y_pred: list or np.array of predicted probabilities (0.0-1.0)
        threshold: cutoff to convert probabilities to 0/1

    Returns:
        dict with accuracy, precision, recall, f1
    """
    y_pred_bin = (y_pred >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_bin),
        'precision': precision_score(y_true, y_pred_bin, zero_division=0),
        'recall': recall_score(y_true, y_pred_bin, zero_division=0),
        'f1': f1_score(y_true, y_pred_bin, zero_division=0)
    }
    return metrics
