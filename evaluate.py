"""
evaluate.py

Standard MIA evaluation: AUROC and TPR at a fixed low FPR (5%), following
Carlini et al., 2022 ("Membership Inference Attacks From First Principles")
and used as the standard metric in Shi et al., 2024 and Zhang et al., 2024.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate(labels, scores, target_fpr: float = 0.05) -> dict:
    """
    Args:
        labels: list/array of 0/1, 1 = member
        scores: list/array of float scores, higher = more likely member
        target_fpr: the FPR at which to report TPR (default 5%, matches
                    Zhang et al., 2024's reported metric)

    Returns:
        dict with "AUROC" and f"TPR@{target_fpr*100:.0f}%FPR"
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # Take the best (max) TPR among all ROC points at or below the target FPR.
    # (Using nearest-point-by-absolute-difference is wrong when several ROC
    # points share the same FPR, e.g. with small/tied datasets -- it can
    # pick an unnecessarily low TPR among the tied points.)
    valid = fpr <= target_fpr
    if valid.any():
        tpr_at_target = tpr[valid].max()
    else:
        # target_fpr is below the smallest achievable FPR; fall back to
        # the lowest-FPR point available
        tpr_at_target = tpr[int(np.argmin(fpr))]

    return {
        "AUROC": round(float(auroc) * 100, 2),
        f"TPR@{int(target_fpr*100)}%FPR": round(float(tpr_at_target) * 100, 2),
    }


def evaluate_all_methods(labels, scores_by_method: dict, target_fpr: float = 0.05) -> dict:
    """
    Convenience wrapper: evaluate several methods at once.

    Args:
        labels: shared list of 0/1 labels
        scores_by_method: {"loss": [...], "zlib": [...], "mink": [...], "minkpp": [...]}

    Returns:
        {"loss": {"AUROC":.., "TPR@5%FPR":..}, "zlib": {...}, ...}
    """
    return {
        method: evaluate(labels, scores, target_fpr=target_fpr)
        for method, scores in scores_by_method.items()
    }
