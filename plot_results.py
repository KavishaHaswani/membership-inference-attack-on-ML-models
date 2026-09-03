"""
plot_results.py

Plots ROC curves for all 4 methods at a given sequence length, using the
raw per-example scores saved by run_experiment.py (--save_raw_scores).

Usage:
    python run_experiment.py --save_raw_scores --lengths 128
    python plot_results.py --length 128
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

METHOD_LABELS = {
    "loss": "LOSS",
    "zlib": "Zlib",
    "mink": "Min-K%",
    "minkpp": "Min-K%++",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--raw_csv", type=str, default=None,
                         help="defaults to results/raw_scores_length{length}.csv")
    parser.add_argument("--out_png", type=str, default=None,
                         help="defaults to results/roc_length{length}.png")
    args = parser.parse_args()

    raw_csv = args.raw_csv or f"results/raw_scores_length{args.length}.csv"
    out_png = args.out_png or f"results/roc_length{args.length}.png"

    df = pd.read_csv(raw_csv)
    labels = df["label"].values

    plt.figure(figsize=(6, 6))
    for method, display_name in METHOD_LABELS.items():
        scores = df[method].values
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, label=f"{display_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"MIA ROC Curves — WikiMIA length={args.length}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
