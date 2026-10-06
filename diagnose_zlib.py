"""
diagnose_zlib.py

Standalone diagnostic for the Zlib-attack anomaly under paraphrasing
(AUROC collapsing to ~3 instead of just weakening). Does NOT load Pythia --
only the paraphraser -- so it's fast and isolates exactly the piece that
might be causing the problem: how paraphrasing changes text length and
compressibility.

What it does, per member text:
    1. Paraphrase it.
    2. Compare original vs. paraphrased: character length, compressed
       size (zlib), and compression RATIO (compressed_len / original_len).
    3. Flag anything suspicious:
        - paraphrase came back empty / fell back to original
        - paraphrase is drastically shorter (possible degenerate output)
        - paraphrase is drastically more/less compressible than original
    4. Print the most extreme cases so you can read them yourself.

Usage:
    python diagnose_zlib.py --length 128 --n 50
    (n = how many member examples to check; use a small number first)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import zlib

# The Windows console defaults to cp1252, which cannot encode most
# non-ASCII characters and makes printing example text die mid-run with a
# UnicodeEncodeError AFTER the CSV has already been written. Force UTF-8
# for stdout so the report section always completes.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# MUST come before pandas/datasets -- loads the tokenizers native extension
# first, which avoids a silent access-violation crash on Windows. See the
# module docstring in import_order.py.
import import_order  # noqa: F401
import numpy as np
import pandas as pd

from paraphrase import Paraphraser
from data_loader import load_wikimia, split_members


def compressed_len(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--n", type=int, default=50, help="number of member examples to check")
    parser.add_argument("--out_csv", type=str, default="results/zlib_diagnostic.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    data = load_wikimia(length=args.length)
    members, non_members = split_members(data)
    members = members[: args.n]

    print(f"Loading paraphraser...")
    paraphraser = Paraphraser()

    rows = []
    for ex in members:
        orig = ex["text"]
        para = paraphraser.paraphrase(orig)

        orig_len = len(orig)
        para_len = len(para)
        orig_clen = compressed_len(orig)
        para_clen = compressed_len(para)

        rows.append({
            "original_text": orig,
            "paraphrased_text": para,
            "orig_char_len": orig_len,
            "para_char_len": para_len,
            "orig_compressed_len": orig_clen,
            "para_compressed_len": para_clen,
            "orig_compression_ratio": round(orig_clen / orig_len, 3) if orig_len else None,
            "para_compression_ratio": round(para_clen / para_len, 3) if para_len else None,
            "fell_back_to_original": para.strip() == orig.strip(),
            "para_much_shorter": para_len < orig_len * 0.5,
            "para_much_longer": para_len > orig_len * 1.5,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved full diagnostic to {args.out_csv}\n")

    # --- Summary flags ---
    n_fallback = df["fell_back_to_original"].sum()
    n_shorter = df["para_much_shorter"].sum()
    n_longer = df["para_much_longer"].sum()
    print(f"Examples where paraphrase fell back to original text: {n_fallback}/{len(df)}")
    print(f"Examples where paraphrase is <50% the length of original: {n_shorter}/{len(df)}")
    print(f"Examples where paraphrase is >150% the length of original: {n_longer}/{len(df)}")

    print(f"\nMean compression ratio -- original: {df['orig_compression_ratio'].mean():.3f}, "
          f"paraphrased: {df['para_compression_ratio'].mean():.3f}")

    # --- The denominator comparison that actually drives the inversion ---
    # In run_paraphrase_experiment.py only MEMBERS are paraphrased; non-members
    # are reused verbatim. So the zlib score's denominator for members (after
    # paraphrasing) is compared against the denominator for non-members
    # (unchanged). If those two aren't drawn from the same distribution, the
    # denominator alone can separate labels regardless of memorization.
    print("\n=== Zlib denominator: paraphrased members vs ORIGINAL non-members ===")
    nm_clen = [compressed_len(e["text"]) for e in non_members]
    para_clen = df["para_compressed_len"]
    orig_clen = df["orig_compressed_len"]

    print(f"  non-members (untouched)  n={len(nm_clen)}  mean clen = {np.mean(nm_clen):.0f}")
    print(f"  members (original)       n={len(orig_clen)}  mean clen = {orig_clen.mean():.0f}")
    print(f"  members (paraphrased)    n={len(para_clen)}  mean clen = {para_clen.mean():.0f}")

    k = para_clen.mean() / np.mean(nm_clen)
    print(f"\n  ratio paraphrased_member_clen / nonmember_clen = {k:.3f}")
    print(f"  (the baseline ratio member_orig/nonmember = {orig_clen.mean() / np.mean(nm_clen):.3f})")

    if abs(k - 1.0) > 0.15:
        print(f"\n  !! Zlib's denominator is off by {abs(k - 1.0) * 100:.0f}% between the two "
              f"classes.")
        print("     Because score = -loss / clen, a class-wide denominator shift this large")
        print("     dominates the loss term and can invert the ranking on its own -- the")
        print("     reported zlib AUROC is then measuring text length, not memorization.")
    else:
        print("\n  Denominators are comparable; zlib's collapse needs a loss-term explanation.")

    # --- Show the most extreme cases for manual inspection ---
    print("\n=== 3 examples with the BIGGEST drop in compression ratio (paraphrase vs original) ===")
    df["ratio_change"] = df["para_compression_ratio"] - df["orig_compression_ratio"]
    worst = df.sort_values("ratio_change").head(3)
    for _, row in worst.iterrows():
        print(f"\n--- ORIGINAL (ratio={row['orig_compression_ratio']}) ---")
        print(row["original_text"][:300])
        print(f"--- PARAPHRASED (ratio={row['para_compression_ratio']}) ---")
        print(row["paraphrased_text"][:300])

    print("\n=== 3 examples with the BIGGEST increase in compression ratio ===")
    best = df.sort_values("ratio_change", ascending=False).head(3)
    for _, row in best.iterrows():
        print(f"\n--- ORIGINAL (ratio={row['orig_compression_ratio']}) ---")
        print(row["original_text"][:300])
        print(f"--- PARAPHRASED (ratio={row['para_compression_ratio']}) ---")
        print(row["paraphrased_text"][:300])


if __name__ == "__main__":
    main()
