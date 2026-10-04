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
python run_experiment.py \
    --model_id EleutherAI/pythia-1.4b \
    --lengths 32 64 128 \
    --save_raw_scores

python run_paraphrase_experiment.py --model_id EleutherAI/pythia-1.4b --lengths 128

python run_paraphrase_experiment.py --lengths 128

```

This writes:
- `results/results.csv` — AUROC and TPR@5%FPR per method per length
- `results/raw_scores_length{32,64,128}.csv` — per-example scores (for plotting)

Generate ROC curves for one length:

```bash
python plot_results.py --length 128
```

Output: `results/roc_length128.png`

If file crashes in the middle of downloading a model from hugging face:

```bash
# Stop the process in case it is still running
Get-Process python* | Stop-Process -Force
# Delete lock file for model
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--humarin--chatgpt_paraphraser_on_T5_base"
```

## Notes

- All four attacks share a single forward pass per text (see `model_utils.py`)
  — no extra model calls, so runtime is dominated by one pass through Pythia
  per WikiMIA example (~800 examples per length split).
- Default `k=0.2` for Min-K% / Min-K%++, matching the paper's typical setting.
  Sweep it in `attacks.py` if you want to reproduce the ablation in Zhang et
  al., 2024, Figure 4.
- This environment could not download the model/dataset to run an end-to-end
  test (no HuggingFace network access here). All code has been syntax-checked
  and the core scoring math (`attacks.py`) and evaluation logic
  (`evaluate.py`) have been unit-tested against synthetic data — see the
  inline sanity checks run during development. Run it in your own environment
  with HF access to get real numbers.
