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

import os
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-410m",
                         help="Default lowered to 410m to fit small-VRAM GPUs (e.g. 4GB). "
                              "Use pythia-1.4b or larger only if you have more VRAM.")
    parser.add_argument("--paraphraser_id", type=str, default="humarin/chatgpt_paraphraser_on_T5_base")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--out_csv", type=str, default="results/paraphrase_results.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # --- Phase 1: paraphrase everything first, only the paraphraser on GPU ---
    print(f"Loading paraphraser: {args.paraphraser_id}")
    paraphraser = Paraphraser(model_id=args.paraphraser_id)

    paraphrased_by_length = {}
    members_by_length = {}
    non_members_by_length = {}
    for length in args.lengths:
        data = load_wikimia(length=length)
        members, non_members = split_members(data)
        members_by_length[length] = members
        non_members_by_length[length] = non_members

        print(f"[length={length}] Paraphrasing {len(members)} member texts...")
        paraphrased = []
        for ex in tqdm(members, desc="Paraphrasing"):
            new_text = paraphraser.paraphrase(ex["text"])
            paraphrased.append({"text": new_text, "label": 1})
        paraphrased_by_length[length] = paraphrased

    # Free the paraphraser's GPU memory before loading the target model.
    del paraphraser
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Phase 2: score everything with the target model ---
    print(f"Loading target model: {args.model_id}")
    wrapper = PythiaWrapper(model_id=args.model_id)

    all_results = []
    for length in args.lengths:
        members = members_by_length[length]
        non_members = non_members_by_length[length]
        paraphrased_members = paraphrased_by_length[length]

        print(f"[length={length}] Scoring ORIGINAL text...")
        labels_before, scores_before = score_examples(wrapper, members + non_members)
        results_before = evaluate_all_methods(labels_before, scores_before)

        print(f"[length={length}] Scoring PARAPHRASED text...")
        labels_after, scores_after = score_examples(wrapper, paraphrased_members + non_members)
        results_after = evaluate_all_methods(labels_after, scores_after)

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
        all_results.append(pd.DataFrame(rows))

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    print("\n=== Robustness to Paraphrasing ===")
    print(final_df.to_string(index=False))
    print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
