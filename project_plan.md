# Membership Inference Attack on Pretrained LLMs — Technical Project Plan

## 1. Objective

Implement and evaluate four membership inference attack (MIA) scoring methods against open-weight Pythia language models, using the WikiMIA benchmark, and reproduce (approximately) the published comparison between these methods.

**Question being answered:** given a piece of text and grey-box access to a pretrained LLM (logits/probabilities only, no weights or gradients), determine whether that text was part of the model's pretraining data.

## 2. Reference Papers

| # | Paper | Role in project |
|---|---|---|
| 1 | Yeom et al., 2018 — *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting* ([arXiv:1709.01604](https://arxiv.org/abs/1709.01604)) | Source of the LOSS attack (baseline 1) |
| 2 | Carlini et al., 2021 — *Extracting Training Data from Large Language Models* ([arXiv:2012.07805](https://arxiv.org/abs/2012.07805)) | Source of the Zlib calibration attack (baseline 2) |
| 3 | Shi et al., 2024 — *Detecting Pretraining Data from Large Language Models* ([arXiv:2310.16789](https://arxiv.org/abs/2310.16789)) | Source of WikiMIA benchmark and Min-K% attack (baseline 3) |
| 4 | Zhang et al., 2024 — *Min-K%++: Improved Baseline for Detecting Pre-Training Data from LLMs* ([arXiv:2404.02936](https://arxiv.org/abs/2404.02936)) | Source of Min-K%++ (primary method) |
| 5 | Biderman et al., 2023 — *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling* ([arXiv:2304.01373](https://arxiv.org/abs/2304.01373)) | Source/justification for target model family |

## 3. Scope

**In scope:**
- 4 scoring methods: LOSS, Zlib, Min-K%, Min-K%++
- WikiMIA benchmark, *original* (verbatim) setting only
- 1–2 Pythia model sizes
- AUROC and TPR@5%FPR evaluation, reported per sequence length (32/64/128)

**Out of scope (explicitly, to bound the project):**
- Paraphrased WikiMIA setting
- MIMIR benchmark
- Reference-model-based attacks (Ref, Neighbor) — require extra LLM inference, not needed for the core comparison
- Shadow models, fine-tuning attacks, federated/multimodal settings
- Any new/novel method — this is a reproduction project

## 4. Environment Setup

### 4.1 Compute
- Single GPU sufficient (T4/A10, e.g. free-tier Colab) for Pythia up to ~2.8B
- Pythia-6.9B optional, needs a bigger GPU (A100) or 8-bit/4-bit quantized loading

### 4.2 Dependencies
```
transformers>=4.40
torch>=2.1
datasets
scikit-learn      # for roc_curve, roc_auc_score
zlib               # standard library, for Zlib baseline
numpy
pandas
matplotlib         # for ROC plots
```

### 4.3 Data
- HuggingFace dataset: `swj0419/WikiMIA`
- Configs available by length: `WikiMIA_length32`, `WikiMIA_length64`, `WikiMIA_length128`
- Each example has a `label` field: `1` = member (pre-training data), `0` = non-member

### 4.4 Model
- HuggingFace model IDs: `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-2.8b`, `EleutherAI/pythia-6.9b`
- Load with `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)`
- Use `AutoTokenizer.from_pretrained(model_id)`

## 5. System Architecture

```
mia_project/
├── data_loader.py       # loads WikiMIA splits into (text, label) pairs
├── model_utils.py       # loads Pythia, runs forward pass, extracts logits/log-probs
├── attacks.py           # scoring functions: loss_attack, zlib_attack, mink_attack, minkpp_attack
├── evaluate.py           # computes AUROC, TPR@5%FPR, builds results table
├── run_experiment.py     # orchestrates: load model → load data → score → evaluate → save
├── plot_results.py       # ROC curves, bar charts
└── results/               # output CSVs, plots
```

## 6. Implementation Detail Per Component

### 6.1 `model_utils.py` — single forward pass, shared across all methods

For each input text:
1. Tokenize to `input_ids` (length T)
2. Run `model(input_ids, labels=input_ids)` → get `logits` of shape `[T, V]`
3. Compute:
   - `log_probs = log_softmax(logits, dim=-1)` → `[T, V]`
   - `probs = softmax(logits, dim=-1)` → `[T, V]`
   - `token_log_probs = log_probs.gather(-1, input_ids)` → `[T]` (log-prob assigned to each actual next token)

This single pass produces everything all four methods need — do not re-run the model per method.

### 6.2 `attacks.py` — scoring functions

**LOSS attack** (Yeom et al., 2018)
```python
def loss_attack(token_log_probs):
    return token_log_probs.mean()  # higher (less negative) = more likely member
```

**Zlib attack** (Carlini et al., 2021)
```python
import zlib

def zlib_attack(text, token_log_probs):
    loss = -token_log_probs.mean()
    compressed_len = len(zlib.compress(text.encode('utf-8')))
    return -loss / compressed_len   # calibrated score
```

**Min-K% attack** (Shi et al., 2024)
```python
def mink_attack(token_log_probs, k=0.2):
    n = int(len(token_log_probs) * k)
    lowest_k = torch.sort(token_log_probs)[0][:n]
    return lowest_k.mean()
```

**Min-K%++ attack** (Zhang et al., 2024) — pseudocode taken directly from paper Appendix A
```python
def minkpp_attack(token_log_probs, log_probs, probs, k=0.2):
    mu = (probs * log_probs).sum(-1)                      # [T]
    sigma_sq = (probs * log_probs.square()).sum(-1) - mu.square()  # [T]
    token_scores = (token_log_probs - mu) / sigma_sq.sqrt()
    n = int(len(token_scores) * k)
    lowest_k = torch.sort(token_scores)[0][:n]
    return lowest_k.mean()
```

Note: for Min-K% and Min-K%++, use `k=0.2` as the default (matches paper's typical setting); can sweep `k` later if time allows.

### 6.3 `evaluate.py`

```python
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate(labels, scores):
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # TPR at FPR closest to 0.05
    idx = np.argmin(np.abs(fpr - 0.05))
    tpr_at_5fpr = tpr[idx]
    return {"AUROC": auroc, "TPR@5%FPR": tpr_at_5fpr}
```

Run this once per (method, sequence length) combination.

### 6.4 `run_experiment.py` — orchestration

```
for length in [32, 64, 128]:
    load WikiMIA_length{length}
    for text, label in dataset:
        run forward pass → token_log_probs, log_probs, probs
        compute scores for all 4 methods
        store (label, score) per method
    for method in [loss, zlib, mink, minkpp]:
        evaluate(labels, scores[method])
    save results table for this length
```

## 7. Evaluation Plan

**Primary table** (per sequence length):

| Method | AUROC | TPR@5%FPR |
|---|---|---|
| LOSS | | |
| Zlib | | |
| Min-K% | | |
| Min-K%++ | | |

Produce one such table for length=32, 64, 128 (or just 128 if time-constrained — it's the easiest split and gives a complete result on its own).

**Comparison to paper:** paper's Table 1/5 (length=128, e.g. Pythia-6.9B) reports approximately:
- LOSS: 65.1 AUROC
- Zlib: 67.6 AUROC
- Min-K%: 69.5 AUROC
- Min-K%++: 70.7 AUROC

Your numbers should land in a similar range/ordering (Min-K%++ ≥ Min-K% ≥ Zlib ≥ LOSS); document any deviation and a plausible reason (different model checkpoint, k value, sample size, seed).

**Secondary output:** ROC curve plot overlaying all 4 methods for one chosen length/model, using `matplotlib`.

**Total: roughly a weekend (6–9 hours of active work + compute time).**

## 8. Deliverables

1. Code (`mia_project/` directory as structured above)
2. Results CSV(s) — one row per (method, length) combination
3. ROC curve plot(s)
4. Short writeup (1–2 pages): methodology, results table, comparison to published numbers, discussion of any gaps

## 9. Known Caveats to Note in Writeup

- WikiMIA's member/non-member split is based on Wikipedia publication date, which is known to introduce some topical/style distribution shift between classes (flagged by Duan et al., 2024, *Do Membership Inference Attacks Work on Large Language Models?*, arXiv:2402.07841) — not something to fix, just worth one sentence acknowledging it as a benchmark limitation.
- TPR@5%FPR is a noisier metric than AUROC at small sample sizes (WikiMIA has ~400 examples per class) — results may vary somewhat run to run to run due to threshold selection.
