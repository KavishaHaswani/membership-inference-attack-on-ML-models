"""
run_experiment.py

Orchestrates the full experiment:
    for each sequence length in {32, 64, 128}:
        load WikiMIA split
        for each text:
            run one forward pass
            compute all 4 attack scores
        evaluate all 4 methods (AUROC, TPR@5%FPR)
    save results to CSV

Usage:
    python run_experiment.py --model_id EleutherAI/pythia-1.4b --lengths 32 64 128
"""

import argparse
import pandas as pd
from tqdm import tqdm

from data_loader import load_wikimia
from model_utils import PythiaWrapper
from attacks import ATTACK_REGISTRY
from evaluate import evaluate_all_methods


def run_for_length(wrapper: PythiaWrapper, length: int) -> pd.DataFrame:
    data = load_wikimia(length=length)

    labels = []
    scores_by_method = {name: [] for name in ATTACK_REGISTRY}

    skipped = 0
    for example in tqdm(data, desc=f"Scoring length={length}"):
        text, label = example["text"], example["label"]
        stats = wrapper.score_text(text)
        if stats is None:
            skipped += 1
            continue

        labels.append(label)
        for name, fn in ATTACK_REGISTRY.items():
            scores_by_method[name].append(fn(text, stats))

    if skipped:
        print(f"  (skipped {skipped} examples that were too short to score)")

    results = evaluate_all_methods(labels, scores_by_method)

    rows = []
    for method, metrics in results.items():
        row = {"length": length, "method": method}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows), labels, scores_by_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--out_csv", type=str, default="results/results.csv")
    parser.add_argument("--save_raw_scores", action="store_true",
                         help="also dump per-example raw scores for plotting")
    args = parser.parse_args()

    import os
    os.makedirs("results", exist_ok=True)

    print(f"Loading model: {args.model_id}")
    wrapper = PythiaWrapper(model_id=args.model_id)

    all_results = []
    for length in args.lengths:
        df, labels, scores_by_method = run_for_length(wrapper, length)
        all_results.append(df)

        if args.save_raw_scores:
            raw_df = pd.DataFrame({"label": labels, **scores_by_method})
            raw_df.to_csv(f"results/raw_scores_length{length}.csv", index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    print("\n=== Final Results ===")
    print(final_df.pivot(index="method", columns="length"))
    print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
