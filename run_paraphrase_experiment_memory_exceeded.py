"""
run_paraphrase_experiment.py

Tests whether the four MIA attacks still detect membership after member
texts have been paraphrased.

Procedure, per sequence length:
    1. Load WikiMIA, split into members / non-members.
    2. Score all texts NORMALLY (original wording) -- this reproduces your
       existing run_experiment.py numbers, kept here as the "before" baseline.
    3. Paraphrase every MEMBER text (non-members are left untouched).
    4. Re-score: paraphrased members + the SAME original non-members.
    5. Compare AUROC / TPR@5%FPR before vs. after paraphrasing, per method.
       A big drop = that method relies on exact wording and is NOT robust
       to paraphrasing. A small drop = that method still picks up on
       memorization even when the wording has changed.

Usage:
    python run_paraphrase_experiment.py --model_id EleutherAI/pythia-1.4b --lengths 128
"""

import argparse
import os

# MUST come before pandas/datasets -- loads the tokenizers native extension
# first, which avoids a silent access-violation crash on Windows. See the
# module docstring in import_order.py.
import import_order  # noqa: F401
import pandas as pd
from tqdm import tqdm

from model_utils import PythiaWrapper
from paraphrase import Paraphraser
from attacks import ATTACK_REGISTRY
from evaluate import evaluate_all_methods
from data_loader import load_wikimia, split_members


def score_examples(wrapper: PythiaWrapper, examples: list):
    """Runs all 4 attacks on a list of {"text","label"} dicts. Skips
    examples too short to score (same guard as run_experiment.py)."""
    labels, scores_by_method = [], {name: [] for name in ATTACK_REGISTRY}
    skipped = 0
    for ex in examples:
        stats = wrapper.score_text(ex["text"])
        if stats is None:
            skipped += 1
            continue
        labels.append(ex["label"])
        for name, fn in ATTACK_REGISTRY.items():
            scores_by_method[name].append(fn(ex["text"], stats))
    if skipped:
        print(f"  (skipped {skipped} examples, too short to score)")
    return labels, scores_by_method


def run_for_length(wrapper: PythiaWrapper, paraphraser: Paraphraser, length: int):
    data = load_wikimia(length=length)
    members, non_members = split_members(data)

    # --- BEFORE: original wording ---
    print(f"[length={length}] Scoring ORIGINAL text...")
    original_all = members + non_members
    labels_before, scores_before = score_examples(wrapper, original_all)
    results_before = evaluate_all_methods(labels_before, scores_before)

    # --- Paraphrase the members only ---
    print(f"[length={length}] Paraphrasing {len(members)} member texts...")
    paraphrased_members = []
    for ex in tqdm(members, desc="Paraphrasing"):
        new_text = paraphraser.paraphrase(ex["text"])
        paraphrased_members.append({"text": new_text, "label": 1})

    # --- AFTER: paraphrased members + original non-members ---
    print(f"[length={length}] Scoring PARAPHRASED text...")
    after_all = paraphrased_members + non_members
    labels_after, scores_after = score_examples(wrapper, after_all)
    results_after = evaluate_all_methods(labels_after, scores_after)

    # --- Combine into one comparison table ---
    rows = []
    for method in ATTACK_REGISTRY:
        before = results_before[method]
        after = results_after[method]
        rows.append({
            "length": length,
            "method": method,
            "AUROC_before": before["AUROC"],
            "AUROC_after": after["AUROC"],
            "AUROC_drop": round(before["AUROC"] - after["AUROC"], 2),
            "TPR@5%FPR_before": before["TPR@5%FPR"],
            "TPR@5%FPR_after": after["TPR@5%FPR"],
            "TPR@5%FPR_drop": round(before["TPR@5%FPR"] - after["TPR@5%FPR"], 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--paraphraser_id", type=str, default="humarin/chatgpt_paraphraser_on_T5_base")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--out_csv", type=str, default="results/paraphrase_results.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print(f"Loading target model: {args.model_id}")
    wrapper = PythiaWrapper(model_id=args.model_id)

    print(f"Loading paraphraser: {args.paraphraser_id}")
    paraphraser = Paraphraser(model_id=args.paraphraser_id)

    all_results = []
    for length in args.lengths:
        df = run_for_length(wrapper, paraphraser, length)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    print("\n=== Robustness to Paraphrasing ===")
    print(final_df.to_string(index=False))
    print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
