# Membership Inference Attack on Pythia — WikiMIA

Implements and compares four MIA scoring methods against Pythia language
models on the WikiMIA benchmark.

**Methods:** LOSS (Yeom et al., 2018) · Zlib (Carlini et al., 2021) ·
Min-K% (Shi et al., 2024) · Min-K%++ (Zhang et al., 2024)

See `mia_project_plan.md` (one directory up) for the full technical plan.

## Setup

```bash
python -m pip install -r requirements.txt   --prefer-binary
```

Requires a HuggingFace account/internet access to download:
- Model: `EleutherAI/pythia-1.4b` (or another Pythia size)
- Dataset: `swj0419/WikiMIA`

Both download automatically on first run via `transformers`/`datasets`.

## Run

Full experiment, all three sequence lengths:

```bash
python run_experiment.py --model_id EleutherAI/pythia-1.4b --lengths 32 64 128 --save_raw_scores
```

Paraphrasing robustness experiment (member texts are paraphrased, then
re-scored against the same untouched non-members):

```bash
# quick pre-flight check on a few examples / a tiny model (~1 min)
python smoke_test.py

# real run
python run_paraphrase_experiment.py --model_id EleutherAI/pythia-410m --lengths 128
```

This writes:
- `results/results.csv` — AUROC and TPR@5%FPR per method per length
- `results/raw_scores_length{32,64,128}.csv` — per-example scores (for plotting)
- `results/paraphrase_results.csv` — AUROC / TPR@5%FPR before vs. after paraphrasing

Generate ROC curves for one length:

```bash
python plot_results.py --length 128
```

Output: `results/roc_length128.png`

## Notes

- **Import order matters on Windows.** Importing `pandas` / `datasets`
  (pyarrow) before `transformers` leaves the process unable to load the
  `tokenizers` native extension: it dies with an access violation and *no*
  Python traceback (exit code `-1073741819` / `0xC0000005`). Every entry
  script imports `import_order` first to avoid this — see that module's
  docstring. The symptom is a script that prints its first line and then
  vanishes with no error.
- All four attacks share a single forward pass per text (see `model_utils.py`)
  — no extra model calls, so runtime is dominated by one pass through Pythia
  per WikiMIA example (~250 examples per length split).
- Default `k=0.2` for Min-K% / Min-K%++, matching the paper's typical setting.
  Sweep it in `attacks.py` if you want to reproduce the ablation in Zhang et
  al., 2024, Figure 4.
- `run_paraphrase_experiment.py` paraphrases first and then loads Pythia, so
  only one model is resident at a time. `run_paraphrase_experiment_memory_exceeded.py`
  is the older variant that keeps both models loaded simultaneously; use it
  only if you have VRAM to spare.
- On CPU, paraphrasing is the bottleneck (~6–7 s per text, so ~15 min for the
  139 members in WikiMIA length 128). On a GPU it is far quicker.
