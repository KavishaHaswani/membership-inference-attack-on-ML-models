# Membership Inference Attack on Pythia — WikiMIA

Implements and compares four MIA scoring methods against Pythia language
models on the WikiMIA benchmark.

**Methods:** LOSS (Yeom et al., 2018) · Zlib (Carlini et al., 2021) ·
Min-K% (Shi et al., 2024) · Min-K%++ (Zhang et al., 2024)

See `project_plan.md` for the full technical plan and `code_structure.md`
for what each file does.

## Setup

```bash
python -m pip install -r requirements.txt --prefer-binary
```

Requires internet / HuggingFace access on first run to download:
- Target model: `EleutherAI/pythia-410m` (default; larger fits only if you have memory)
- Paraphraser: `humarin/chatgpt_paraphraser_on_T5_base`
- Dataset: `swj0419/WikiMIA`

All download automatically via `transformers` / `datasets`.

### Windows: import order is mandatory

Importing `pandas` / `datasets` (pyarrow) before `transformers` leaves the
process unable to load the `tokenizers` native extension. It dies with an
access violation and **no Python traceback** (exit code `-1073741819` /
`0xC0000005`).

> Symptom: a script prints its first line, then vanishes.

Every script that loads a HuggingFace tokenizer starts with
`import import_order` to prevent this. If you write a new script that touches
a tokenizer, put that import first — before `pandas` and `datasets`. See
`import_order.py`. Scripts that never touch a tokenizer (e.g. `plot_results.py`)
don't need it.

---

## 1. Main experiment (no paraphrasing)

Scores every WikiMIA example once with all four attacks and reports AUROC /
TPR@5%FPR per method per sequence length.

```bash
python run_experiment.py \
    --model_id EleutherAI/pythia-410m \
    --lengths 32 64 128 \
    --save_raw_scores
```

| output | contents |
|---|---|
| `results/results.csv` | AUROC and TPR@5%FPR per method per length |
| `results/raw_scores_length{32,64,128}.csv` | per-example scores (for plotting) |

Plot ROC curves (requires `--save_raw_scores`):

```bash
python plot_results.py --length 128     # -> results/roc_length128.png
```

**Provenance:** `results/run_log.txt` records the exact command and model that
produced `results/results.csv`. That file was generated with `pythia-70m`,
which is no longer the default — re-run if you need it to match current
defaults.

---

## 2. Experiment with paraphrasing

Tests whether membership is still detectable after member texts have been
reworded. Members are paraphrased and re-scored against the same untouched
non-members.

```bash
# quick pre-flight: 8 examples, tiny model, ~1 min
python smoke_test.py

# real run
python run_paraphrase_experiment.py \
    --model_id EleutherAI/pythia-410m \
    --lengths 128
```

| output | contents |
|---|---|
| `results/paraphrase_results.csv` | AUROC / TPR@5%FPR before vs. after paraphrasing, per method |

**Phases (to fit in memory):** phase 1 paraphrases everything with only the
paraphraser resident, frees it, then phase 2 loads Pythia. Never both at once.
`run_paraphrase_experiment_memory_exceeded.py` is the older variant that keeps
both loaded — only use it if you have memory to spare.

**Runtime:** paraphrasing dominates on CPU (~6–7 s per text, so ~15 min for
the 139 members in length 128). Scoring 250 examples is comparatively quick.

> **Known confound:** only members are paraphrased, so paraphrase status is
> perfectly correlated with the label. Because the paraphraser *condenses*
> text, the Zlib attack's denominator shifts class-wide and its reported
> AUROC collapse is mostly an artifact — see section 3.

---

## 3. Zlib length-confound analysis

Two steps. The first is diagnostics only and does not load Pythia; the second
runs the control test.

```bash
# step 1: length / compressibility diagnostics (~6 min, paraphraser only)
python -u diagnose_zlib.py --length 128 --n 50

# step 2: control test + compound report (~5 min, Pythia only)
python -u verify_zlib_confound.py --length 128
```

| output | contents |
|---|---|
| `results/zlib_diagnostic.csv` | per-example original vs. paraphrased length and compressed size |
| `results/zlib_confound_report.csv` | **compound report**: reported AUROC + length stats + controls + verdict |

The compound report has four sections:

| section | contents |
|---|---|
| `1_reported` | the actual AUROC from `paraphrase_results.csv` (62.52 → 2.88) |
| `2_length` | length/compressibility stats and the denominator shift |
| `3_control` | AUROC of every control variant |
| `4_verdict` | how much of the reported drop the confound explains |

Four variants are measured on the same examples:

| variant | AUROC | meaning |
|---|---|---|
| A — original members, raw zlib | 57.71 | baseline |
| B — paraphrased members, raw zlib | 3.71 | reproduces the collapse |
| C — rate-normalized zlib | 0.68 | **not a valid fix** — swaps one confound for another |
| D — same as B, non-members length-matched | 55.00 | isolates the length effect |

B vs D uses *identical* member texts; only the non-member length distribution
differs. **~95% of the reported drop is length.** Length-controlled, zlib falls
only ~2.7 points; LOSS (no length term) falls ~7.2 points, which is the honest
effect of changing the wording.

**Note on variant D:** every paraphrase comes out *shorter* than its source, so
the intuitive control (truncate the paraphrase back to original length) is a
no-op. D shortens the *non-members* to match instead; both classes then have
identical character lengths and compressed sizes within ~1%.

---

## Notes

- All four attacks share a single forward pass per text (`model_utils.py`),
  so runtime is one pass through Pythia per WikiMIA example (~250 per length).
- Default `k=0.2` for Min-K% / Min-K%++, matching the paper's typical setting.
  Sweep it in `attacks.py` to reproduce the ablation in Zhang et al., 2024,
  Figure 4.
- CPU-only builds are supported; `model_utils.py` loads float32 on CPU and
  float16 on CUDA.
