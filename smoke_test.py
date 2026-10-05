"""
smoke_test.py

Fast end-to-end check of the paraphrasing pipeline on a tiny subset, using a
small model, so failures surface in seconds instead of tens of minutes.

Not part of the experiment; delete or keep as a quick pre-flight check.
"""

import sys

# MUST come before pandas/datasets -- see import_order.py.
import import_order  # noqa: F401
import data_loader

N_EXAMPLES = 8
_orig_load = data_loader.load_wikimia


def _small_load(length: int = 128):
    return _orig_load(length=length)[:N_EXAMPLES]


if __name__ == "__main__":
    import run_paraphrase_experiment as r

    # Patch the name the experiment module actually uses.
    r.load_wikimia = _small_load

    sys.argv = [
        "smoke_test.py",
        "--lengths", "128",
        "--model_id", "EleutherAI/pythia-70m",
        "--paraphraser_id", "humarin/chatgpt_paraphraser_on_T5_base",
        "--out_csv", "results/_smoke.csv",
    ]
    r.main()
    print("SMOKE TEST PASSED")