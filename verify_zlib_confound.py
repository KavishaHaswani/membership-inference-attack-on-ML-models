"""
verify_zlib_confound.py

Verifies whether the Zlib attack's AUROC collapse under paraphrasing (62.5 ->
2.9) is a LENGTH artifact rather than a real result about memorization.

Hypothesis: run_paraphrase_experiment.py paraphrases ONLY members, so any
class-wide change the paraphraser makes to text length becomes perfectly
confounded with the label. Zlib divides by compressed size, so it is maximally
exposed: score = -loss / compressed_len.

Four variants are evaluated on the SAME example set:
    A. original members vs non-members, raw zlib          -> expect ~60
    B. paraphrased members vs non-members, raw zlib        -> expect ~3
    B reproduces the collapse.
    C. paraphrased members vs non-members, zlib normalized by compressibility
       RATE (compressed_len / n_chars) instead of total size
       -> if the confound is the cause, this recovers most of the AUROC
    D. paraphrased members truncated back to the original character length
       before scoring, raw zlib -> strongest length-matched control

Loss-only scores are reported alongside so we can see what actually survives
the wording change once the length term is removed.

Paraphrases are read back out of results/zlib_diagnostic.csv (produced by
diagnose_zlib.py), so this costs no paraphraser time -- only Pythia forwards.
"""

import argparse
import sys
import zlib

# Windows console defaults to cp1252; stdout is ASCII-safe anyway but this
# keeps the report from dying on unusual text. MUST precede pandas/datasets
# to load the tokenizers native extension first -- see import_order.py.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import import_order  # noqa: F401
import pandas as pd
from sklearn.metrics import roc_auc_score

from model_utils import PythiaWrapper
from attacks import ATTACK_REGISTRY
from evaluate import evaluate_all_methods
from data_loader import load_wikimia, split_members


def clen(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def score_all(wrapper: PythiaWrapper, examples: list):
    labels, scores = [], {name: [] for name in ATTACK_REGISTRY}
    stats = []
    skipped = 0
    for ex in examples:
        s = wrapper.score_text(ex["text"])
        if s is None:
            skipped += 1
            continue
        labels.append(ex["label"])
        stats.append((ex["text"], s))
        for name, fn in ATTACK_REGISTRY.items():
            scores[name].append(fn(ex["text"], s))
    if skipped:
        print(f"  (skipped {skipped} examples, too short to score)")
    return labels, scores, stats


def rate_zlib_scores(stats_list):
    """Length-normalized zlib: divide by compression RATE (compressed_bytes per
    character) instead of total compressed_bytes. Removes the dependence on how
    long the text is while keeping zlib's original intent (calibrate loss by
    how redundant/compressible the text is).

    stats_list: [(text, stats), ...] -- the pairs score_all() already
    computed, so no extra model passes are needed."""
    out = []
    for text, stats in stats_list:
        rate = clen(text) / max(1, len(text))
        loss = -stats["token_log_probs"].mean().item()
        out.append(-loss / rate)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--diag_csv", type=str, default="results/zlib_diagnostic.csv")
    args = parser.parse_args()

    diag = pd.read_csv(args.diag_csv)
    print(f"Loaded {len(diag)} paraphrased member pairs from {args.diag_csv}")

    data = load_wikimia(length=args.length)
    _, non_members = split_members(data)

    pairs = diag.to_dict("records")

    orig_members = [{"text": r["original_text"], "label": 1} for r in pairs]
    para_members = [{"text": r["paraphrased_text"], "label": 1} for r in pairs]
    non_mem = [{"text": e["text"], "label": 0} for e in non_members]

    # --- Length-matched control -------------------------------------------
    # Every one of the 50 paraphrases came out SHORTER than its source text
    # (mean 811 -> 473 chars), so truncating the members is a no-op and the
    # obvious control silently does nothing. The confound is that members are
    # now the SHORT class. So match the other way round: cut the NON-members
    # down to the paraphrase length distribution. Both classes then have the
    # same length distribution by construction, and the zlib denominator
    # difference attributable to length alone disappears.
    para_lens = [len(r["paraphrased_text"]) for r in pairs]
    non_mem_matched = [
        {"text": non_members[i % len(non_members)]["text"][:L], "label": 0}
        for i, L in enumerate(para_lens)
    ]
    # verifiable: member and non-member lengths now line up one-for-one
    assert [len(e["text"]) for e in non_mem_matched] == para_lens

    print(f"  original members: {len(orig_members)}")
    print(f"  non-members:      {len(non_mem)}")
    print(f"  paraphrase length mean={sum(para_lens)/len(para_lens):.0f} "
          f"chars (all {len(para_lens)} are shorter than their source)")

    print(f"\nLoading target model: {args.model_id}")
    wrapper = PythiaWrapper(model_id=args.model_id)

    # Each group is scored exactly once, then combined as needed:
    #   originals 50 + paraphrased 50 + non-members 111 + matched non-members 111
    # Variants B and D differ ONLY in which non-members they are paired with,
    # so they must reuse the same member scores -- re-scoring them would give
    # an unequal comparison for no reason.
    print("\n[1/4] scoring non-members (original length) ...")
    nm_labels, nm_scores, nm_stats = score_all(wrapper, non_mem)
    print("[2/4] scoring non-members (length-matched to paraphrases) ...")
    nm_len_labels, nm_len_scores, nm_len_stats = score_all(wrapper, non_mem_matched)
    print("[3/4] scoring original members ...")
    orig_m = score_all(wrapper, orig_members)
    print("[4/4] scoring paraphrased members ...")
    para_m = score_all(wrapper, para_members)

    def combine(m, n):
        m_labels, m_scores, m_stats = m
        n_labels, n_scores, n_stats = n
        labels = m_labels + n_labels
        scores = {k: m_scores[k] + n_scores[k] for k in ATTACK_REGISTRY}
        return labels, scores, m_stats + n_stats

    # --- A: baseline, raw zlib on original wording ---
    labels_a, scores_a, _ = combine(orig_m, (nm_labels, nm_scores, nm_stats))
    a_raw = evaluate_all_methods(labels_a, scores_a)

    # --- B: the collapse ---
    labels_b, scores_b, stats_b = combine(para_m, (nm_labels, nm_scores, nm_stats))
    b_raw = evaluate_all_methods(labels_b, scores_b)

    # --- C: compressibility-rate normalized zlib (reuses B's stats) ---
    print("[C] variant B with zlib normalized by compressibility RATE (no extra passes) ...")
    c_scores = rate_zlib_scores(stats_b)
    c_auroc = roc_auc_score(labels_b, c_scores)

    # --- D: length-matched control ---
    labels_d, scores_d, _ = combine(para_m, (nm_len_labels, nm_len_scores, nm_len_stats))
    d_raw = evaluate_all_methods(labels_d, scores_d)

    def fmt(res, method):
        return (res[method]["AUROC"], res[method]["TPR@5%FPR"])

    print("\n" + "=" * 72)
    print("ZLIB LENGTH-CONFOUND CONTROL TEST")
    print("=" * 72)
    print(f"{'variant':<52}{'AUROC':>7}{'TPR@5%':>8}")
    print("-" * 72)
    print(f"{'A  original members, raw zlib (baseline)':<52}{fmt(a_raw,'zlib')[0]:>7}{fmt(a_raw,'zlib')[1]:>8}")
    print(f"{'B  paraphrased members, raw zlib':<52}{fmt(b_raw,'zlib')[0]:>7}{fmt(b_raw,'zlib')[1]:>8}")
    print(f"{'C  paraphrased, rate-normalized zlib':<52}{c_auroc:>7.2f}{'-':>8}")
    print(f"{'D  same as B, non-members length-matched to paraphrases':<52}{fmt(d_raw,'zlib')[0]:>7}{fmt(d_raw,'zlib')[1]:>8}")
    print("-" * 72)
    print(f"{'LOSS only (no length term), original':<52}{fmt(a_raw,'loss')[0]:>7}{fmt(a_raw,'loss')[1]:>8}")
    print(f"{'LOSS only (no length term), paraphrased':<52}{fmt(b_raw,'loss')[0]:>7}{fmt(b_raw,'loss')[1]:>8}")
    print("=" * 72)

    print("\nVerdict:")
    print("  B vs A is the reported collapse; B vs D isolates LENGTH (identical")
    print("  members, only the non-member length distribution changes).")
    print(f"  B - D = {b_raw['zlib']['AUROC'] - d_raw['zlib']['AUROC']:+.1f} AUROC points "
          f"attributable to the length confound alone.")
    print("  The LOSS rows have no length term at all, so that drop is the honest")
    print("  effect of changing the wording.")


if __name__ == "__main__":
    main()
