# Code structure

What each file does, what it reads, and what it writes.

## Core scoring pipeline

These four files implement the actual attack and are imported by every
entrypoint. They have no side effects and never write files.

| file | role |
|---|---|
| **`model_utils.py`** | `PythiaWrapper` — loads a Pythia model and runs exactly **one** forward pass per text. Returns `token_log_probs`, `minkpp_token_scores`, `n_tokens`. Every attack consumes this, so the model is only ever called once per example. Min-K%++'s calibration is computed here rather than in `attacks.py` so the full `[T, V]` distribution is discarded immediately (avoids OOM on long texts). |
| **`attacks.py`** | The four scoring functions, each returning a single float where **higher = more likely member**: `loss` (mean log-prob), `zlib` (`-loss / compressed_len`), `mink` (mean of the lowest-`k`% token log-probs), `minkpp` (same aggregation over calibrated scores). Exposed as `ATTACK_REGISTRY = {name: fn(text, stats)}`. `k=0.2` everywhere. |
| **`evaluate.py`** | AUROC and TPR@5%FPR. Takes the max TPR at any FPR ≤ 0.05 rather than the nearest point — the naive nearest-point lookup picks a needlessly low TPR when several ROC points tie in FPR. `evaluate_all_methods()` runs all methods on one label vector. |
| **`data_loader.py`** | `load_wikimia(length)` → `[{"text", "label"}, ...]`, and `split_members(data)` → `(members, non_members)`. WikiMIA exposes length as a *split* (`WikiMIA_length32/64/128/256`), not a config. |

> `data_loader.py` imports `transformers` before `datasets` deliberately (it is
> the only module that does both), so it is safe to import from anywhere. See
> **`import_order.py`** below.

## Entrypoints

| file | what it runs | writes |
|---|---|---|
| **`run_experiment.py`** | The main experiment, no paraphrasing. For each length: score every example with all four attacks and report AUROC / TPR@5%FPR. | `results/results.csv`, and with `--save_raw_scores` also `results/raw_scores_length{N}.csv` |
| **`run_paraphrase_experiment.py`** | The paraphrasing experiment. Two phases so only one model is resident at a time: **(1)** paraphrase every *member*, free the paraphraser; **(2)** load Pythia and score both the original and the paraphrased sets. Compares before vs. after per method. | `results/paraphrase_results.csv` |
| **`plot_results.py`** | ROC curves for one length, from raw scores saved by `run_experiment.py`. | `results/roc_length{N}.png` |
| **`smoke_test.py`** | Fast pre-flight. Patches the loader down to 8 examples and runs the full paraphrase pipeline on `pythia-70m` (~1 min) to catch import/runtime breakage before a long run. | `results/_smoke.csv` (delete after) |

## Paraphrasing

| file | role |
|---|---|
| **`paraphrase.py`** | `Paraphraser` wraps `humarin/chatgpt_paraphraser_on_T5_base`. Prompts with `paraphrase: {text}`, beam search (5 beams, `no_repeat_ngram_size=3`), and falls back to the original if generation is empty. Does **not** load Pythia. |

Important behaviour: the paraphraser **condenses**. In practice all sampled
paraphrases came back shorter than their source (811 → 473 chars). This matters
because it breaks the Zlib attack — see below.

## Zlib length-confound analysis

`run_paraphrase_experiment.py` paraphrases *only* members, so paraphrase status
is perfectly correlated with the label. Because paraphrasing changes text
length, and Zlib divides by compressed size, Zlib's reported collapse is
largely an artifact. Two scripts investigate it:

| file | what it does | loads Pythia? | writes |
|---|---|---|---|
| **`diagnose_zlib.py`** | Paraphrases `--n` members and compares original vs. paraphrased: character length, compressed size, compression ratio. Flags fall-backs, drastic shortenings, drastic compressibility changes, and prints the most extreme examples. Compares against the **non-member** baseline, because that is the denominator actually being compared against at scoring time. | No (paraphraser only) | `results/zlib_diagnostic.csv` |
| **`verify_zlib_confound.py`** | The control test. Runs four variants on the same examples to isolate the length effect, then writes one compound report joining the reported AUROC, the length stats, the controls, and the verdict. | Yes | `results/zlib_confound_report.csv` |

The four variants in `verify_zlib_confound.py`:

| variant | construction | purpose |
|---|---|---|
| **A** | original members vs original non-members, raw zlib | baseline (~58) |
| **B** | paraphrased members vs original non-members, raw zlib | reproduces the collapse (~4) |
| **C** | B, but dividing by compressibility *rate* (`clen/chars`) instead of total size | negative result — paraphrases are less compressible per char (0.626 vs 0.556), so this swaps one confound for another and scores *worse* than B |
| **D** | B's identical members, but non-members truncated to the paraphrase length distribution | isolates length: identical member scores, only the control class's length changes |

Variant D deserves a note. The intuitive control — truncate the paraphrase
back to its original length — is a **no-op**, because every paraphrase is
*shorter* than its source, so the slice never cuts anything and D degenerates
into B (identical AUROC to two decimals). The match must therefore go the other
way and shorten the **non-members**. Result: both classes have exactly equal
character lengths (472.6 each) and compressed sizes within 1% (295.9 vs 293.0).

**Verdict produced:** ~95% of the reported drop is the length confound.
Length-controlled, zlib falls only ~2.7 points; `loss` (no length term) falls
~7.2 points, which is the real effect.

## Infrastructure and docs

| file | role |
|---|---|
| **`import_order.py`** | One-liner that imports `transformers` for its side effect. **Must be imported before `pandas` / `datasets`.** Windows only failure mode: otherwise the `tokenizers` Rust extension fails to load and the process exits `-1073741819` (`0xC0000005`) with no traceback. |
| **`requirements.txt`** | Pinned dependencies. Includes `sentencepiece` (needed by the T5 paraphraser) and `pyarrow` (pulled in by `datasets`). |
| **`project_plan.md`** | Full technical plan: objective, reference papers, methods, evaluation protocol. |
| **`code_structure.md`** | This file. |

## Legacy / unused

Kept for reference; not part of any current workflow.

| file | why it is legacy |
|---|---|
| **`run_paraphrase_experiment_memory_exceeded.py`** | Older paraphrase runner that loads the paraphraser *and* Pythia simultaneously. Survives now that imports are ordered, but wastes memory — only use it if you have room for both. |
| **`data_loader_old.py`** | Previous WikiMIA loader using the config-based API (`load_dataset(name, config, split="WikiMIA")`). Superseded by the split-based layout in `data_loader.py`. |

## Output artifacts (`results/`)

| file | produced by |
|---|---|
| `results.csv` | `run_experiment.py` |
| `raw_scores_length{32,64,128}.csv` | `run_experiment.py --save_raw_scores` |
| `run_log.txt` | `run_experiment.py` — **records the command and model used**; useful for provenance |
| `paraphrase_results.csv` | `run_paraphrase_experiment.py` |
| `zlib_diagnostic.csv` | `diagnose_zlib.py` |
| `zlib_confound_report.csv` | `verify_zlib_confound.py` |
| `roc_length{N}.png` | `plot_results.py` |
