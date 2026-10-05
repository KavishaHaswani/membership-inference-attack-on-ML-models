"""
import_order.py

MUST be imported before `pandas`, `datasets`, or `pyarrow` in any script that
also loads a HuggingFace tokenizer.

Why this is needed
------------------
`tokenizers` (used by `transformers`) ships a native Rust extension. On Windows,
if `pandas` / `pyarrow` DLLs are loaded first, the process is left in a state
where importing the tokenizers extension afterwards dies with an access
violation and **no Python traceback** -- the interpreter just exits with code
-1073741819 (0xC0000005).

Minimal reproductions that crash:

    import pandas
    from datasets import load_dataset
    from transformers import AutoTokenizer   # <- hard crash here

`import transformers` pulls in the tokenizers extension eagerly, so doing that
first makes every subsequent import order safe. This is a no-op on Linux/macOS
and costs nothing, so it is kept unconditionally.

Usage: put this at the very top of the import block, above pandas/datasets.
"""

import transformers  # noqa: F401  (imported for its side effect only)