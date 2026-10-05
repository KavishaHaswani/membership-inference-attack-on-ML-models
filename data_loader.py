"""
data_loader.py

Loads the WikiMIA benchmark (Shi et al., 2024) from HuggingFace datasets.
Each example has:
    - "input": the text
    - "label": 1 = member (seen during pretraining), 0 = non-member

WikiMIA exposes length variants as *splits* (not configs):
WikiMIA_length32 / 64 / 128 / 256. We use the verbatim setting only.
"""

# IMPORTANT: `transformers` must be imported BEFORE `datasets` (which pulls in
# pyarrow). On Windows, loading pyarrow's DLLs before the tokenizers native
# extension corrupts the process: the next AutoTokenizer.from_pretrained()
# dies with an access violation and NO Python traceback (exit code
# -1073741819 / 0xC0000005). Importing transformers first loads tokenizers
# safely and avoids it. Doing it here means every script is safe regardless of
# its own import order.
import transformers  # noqa: F401
from datasets import load_dataset


def load_wikimia(length: int = 128):
    """
    Load a WikiMIA split by sequence length.

    Args:
        length: one of {32, 64, 128, 256}

    Returns:
        List of dicts: [{"text": str, "label": int}, ...]
    """
    if length not in (32, 64, 128, 256):
        raise ValueError("length must be one of 32, 64, 128, 256")

    # HF dataset layout (current): single default config, length as split name.
    ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")

    examples = []
    for row in ds:
        examples.append({
            "text": row["input"],
            "label": int(row["label"]),
        })
    return examples


def split_members(data: list) -> tuple:
    """
    Splits a loaded WikiMIA list into (members, non_members).
    Used by run_paraphrase_experiment.py, which only paraphrases members
    -- non-members stay as-is, since we're testing whether paraphrasing
    lets a genuine member text "slip past" the attack.
    """
    members = [e for e in data if e["label"] == 1]
    non_members = [e for e in data if e["label"] == 0]
    return members, non_members


if __name__ == "__main__":
    # quick smoke test
    data = load_wikimia(length=128)
    n_members = sum(e["label"] == 1 for e in data)
    n_non_members = sum(e["label"] == 0 for e in data)
    print(f"Loaded {len(data)} examples ({n_members} members, {n_non_members} non-members)")
    print("Example:", data[0])
