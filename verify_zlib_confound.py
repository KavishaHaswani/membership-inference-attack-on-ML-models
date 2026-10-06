"""
verify_zlib_confound.py

Verifies whether the Zlib attack's AUROC collapse under paraphrasing (62.5 ->
2.9) is a LENGTH artifact rather than a real result about memorization.

Hypothesis: run_paraphrase_experiment.py paraphrases ONLY members, so any
class-wide change the paraphraser makes to text length becomes perfectly
confounded with the label. Zlib divides by compressed size, so it is maximally
exposed: score = -loss / compressed_len.

Four variants are evaluated on the SAME example set:
    A. original members vs non-members, raw zlib          (baseline, ~58)
    B. paraphrased members vs non-members, raw zlib        (reproduces ~4)
    C. paraphrased members vs non-members, zlib normalized by compressibility
       RATE (compressed_len / n_chars) instead of total size
       -> MEASURED WORSE THAN B: paraphrases are less compressible per char
          (0.626 vs 0.556), so the rate swaps one confound for another.
          Not a valid fix -- kept as a negative result.
    D. paraphrased members vs non-members TRUNCATED to the paraphrase length
       distribution -> the length-matched control.

Note on D: every sampled paraphrase came out SHORTER than its source text
(811 -> 473 chars), so the intuitive control -- truncate members to their
original length -- is a no-op and silently reproduces B exactly. Members are
the SHORT class after paraphrasing, so the matching must shorten the
NON-members instead. D shares B's member scores byte-for-byte; only the
non-member length distribution differs, which isolates the length effect.

Loss-only scores are reported alongside so we can see what actually survives
the wording change once the length term is removed.

Paraphrases are read back out of results/zlib_diagnostic.csv (produced by
diagnose_zlib.py), so this costs no paraphraser time -- only Pythia forwards.

Outputs:
    results/zlib_confound_report.csv  -- compound report joining
        1_reported : the actual AUROC numbers from paraphrase_results.csv
        2_length   : length/compressibility diagnostics from zlib_diagnostic.csv
        3_control  : AUROC of every control variant measured here
        4_verdict  : how many points of the reported drop the confound explains

Usage:
    python -u verify_zlib_confound.py --length 128
"""

import argparse
import os
import sys
import zlib

# Windows console defaults to cp1252; stdout is ASCII-safe anyway but this
# keeps the report from dying on unusual text. MUST precede pandas/datasets
# to load the tokenizers native extension first -- see import_order.py.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import import_order  # noqa: F401
import numpy as np
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
    parser.add_argument("--report_csv", type=str,
                        default="results/zlib_confound_report.csv",
                        help="compound report: reported AUROC + length stats + control")
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

    print("\n" + "=" * 78)
    print("ZLIB LENGTH-CONFOUND CONTROL TEST")
    print("=" * 78)
    print(f"{'variant':<56}{'AUROC':>7}{'TPR@5%':>8}")
    print("-" * 78)
    print(f"{'A  original members, raw zlib (baseline)':<56}{fmt(a_raw,'zlib')[0]:>7}{fmt(a_raw,'zlib')[1]:>8}")
    print(f"{'B  paraphrased members, raw zlib':<56}{fmt(b_raw,'zlib')[0]:>7}{fmt(b_raw,'zlib')[1]:>8}")
    print(f"{'C  paraphrased, rate-normalized zlib':<56}{c_auroc:>7.2f}{'-':>8}")
    print(f"{'D  same as B, non-members length-matched':<56}{fmt(d_raw,'zlib')[0]:>7}{fmt(d_raw,'zlib')[1]:>8}")
    print("-" * 78)
    print(f"{'LOSS only (no length term), original':<56}{fmt(a_raw,'loss')[0]:>7}{fmt(a_raw,'loss')[1]:>8}")
    print(f"{'LOSS only (no length term), paraphrased':<56}{fmt(b_raw,'loss')[0]:>7}{fmt(b_raw,'loss')[1]:>8}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Compound report: ONE persisted artifact joining
    #   (a) the number actually reported in paraphrase_results.csv,
    #   (b) the length diagnostics from zlib_diagnostic.csv,
    #   (c) this control experiment's AUROCs,
    #   (d) the derived verdict.
    # Without this, diagnose_zlib.py only printed length stats and this
    # script only printed a table that was never written to results/.
    # ------------------------------------------------------------------
    nm_clen_mean = float(np.mean([clen(e["text"]) for e in non_members]))
    mem_orig_clen = float(diag["orig_compressed_len"].mean())
    para_clen = float(diag["para_compressed_len"].mean())
    orig_char = float(diag["orig_char_len"].mean())
    para_char = float(diag["para_char_len"].mean())

    ratio_baseline = mem_orig_clen / nm_clen_mean
    ratio_after = para_clen / nm_clen_mean
    confound_pts = d_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"]  # AUROC recovered by removing length
    controlled_drop = a_raw["zlib"]["AUROC"] - d_raw["zlib"]["AUROC"]
    loss_drop = a_raw["loss"]["AUROC"] - b_raw["loss"]["AUROC"]

    rows = []

    def add(section, metric, value, note=""):
        rows.append({"section": section, "metric": metric,
                     "value": round(float(value), 3), "note": note})

    rep_path = "results/paraphrase_results.csv"
    if os.path.exists(rep_path):
        rep = pd.read_csv(rep_path)
        rep = rep[(rep["length"] == args.length) & (rep["method"] == "zlib")]
        if len(rep):
            r = rep.iloc[0]
            add("1_reported", "zlib_AUROC_before", r["AUROC_before"],
                f"{rep_path}, {args.length=}, model={args.model_id}")
            add("1_reported", "zlib_AUROC_after", r["AUROC_after"],
                "raw zlib, paraphrased members vs original non-members")
            add("1_reported", "zlib_AUROC_drop", r["AUROC_drop"],
                "reported as a robustness result")
    else:
        add("1_reported", "zlib_AUROC_before", float("nan"), f"MISSING: {rep_path}")
        add("1_reported", "zlib_AUROC_after", float("nan"), "MISSING")

    add("2_length", "n_members_profiled", len(diag), "from results/zlib_diagnostic.csv")
    add("2_length", "orig_char_len_mean", orig_char, "member source text")
    add("2_length", "para_char_len_mean", para_char, "after paraphrasing")
    add("2_length", "para_char_len_ratio", para_char / orig_char,
        "paraphraser condenses; all sampled paraphrases are shorter")
    add("2_length", "nonmember_clen_mean", nm_clen_mean, "untouched control class")
    add("2_length", "member_clen_orig_mean", mem_orig_clen, "members before paraphrasing")
    add("2_length", "member_clen_para_mean", para_clen, "members after paraphrasing")
    add("2_length", "denominator_ratio_baseline", ratio_baseline,
        "clen_member_orig / clen_nonmember; ~1.0 means length carries no label signal")
    add("2_length", "denominator_ratio_after_paraphrase", ratio_after,
        "clen_member_para / clen_nonmember; deviation from 1.0 is a free label feature")
    add("2_length", "denominator_shift_pct", abs(ratio_after - 1.0) * 100,
        "confound magnitude, since zlib score = -loss / clen")

    add("3_control", "AUROC_A_original_raw_zlib", a_raw["zlib"]["AUROC"],
        "baseline reproduction")
    add("3_control", "AUROC_B_paraphrased_raw_zlib", b_raw["zlib"]["AUROC"],
        "reproduces the reported collapse on this subset")
    add("3_control", "AUROC_C_paraphrased_rate_normalized_zlib", c_auroc,
        "NOT a valid fix: paraphrases are less compressible per char, new confound")
    add("3_control", "AUROC_D_paraphrased_length_matched_zlib", d_raw["zlib"]["AUROC"],
        "identical members to B, only the non-member length distribution differs")
    add("3_control", "AUROC_loss_original", a_raw["loss"]["AUROC"],
        "no length term; uncontaminated by the confound")
    add("3_control", "AUROC_loss_paraphrased", b_raw["loss"]["AUROC"],
        "no length term; uncontaminated by the confound")

    add("4_verdict", "points_explained_by_length", d_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"],
        "D - B: AUROC recovered by removing the length confound; compare with zlib_drop_as_reported")
    add("4_verdict", "zlib_drop_length_controlled", controlled_drop,
        "A - D: zlib's real degradation under paraphrasing")
    add("4_verdict", "zlib_drop_as_reported", a_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"],
        "A - B: what the naive comparison claims")
    add("4_verdict", "pct_of_drop_explained_by_length",
        100.0 * (d_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"]) /
        max(1e-9, a_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"]),
        "length-confound points as a share of the naive drop")
    add("4_verdict", "loss_drop", loss_drop,
        "A_loss - B_loss: the honest effect of changing the wording")

    report = pd.DataFrame(rows, columns=["section", "metric", "value", "note"])
    out_csv = args.report_csv
    report.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("COMPOUND REPORT")
    print("=" * 78)
    for section in report["section"].unique():
        sub = report[report["section"] == section]
        print(f"\n[{section}]")
        for r in sub.itertuples():
            note = f"  <- {r.note}" if r.note else ""
            print(f"  {r.metric:<44}{r.value:>10}{note}")
    print("\n" + "=" * 78)
    print(f"Saved compound report to {out_csv}")

    print("\nVerdict:")
    naive_drop = a_raw["zlib"]["AUROC"] - b_raw["zlib"]["AUROC"]
    pct = 100.0 * confound_pts / max(1e-9, naive_drop)
    print(f"  {confound_pts:.1f} of the {naive_drop:.1f}-point reported drop "
          f"({pct:.1f}%) is the length confound.")
    print(f"  Length-controlled, zlib actually falls only {controlled_drop:.1f} points "
          f"({a_raw['zlib']['AUROC']:.1f} -> {d_raw['zlib']['AUROC']:.1f}).")
    print(f"  LOSS (no length term) falls {loss_drop:.1f} points: that is the real signal.")


if __name__ == "__main__":
    main()
