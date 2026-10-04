"""
data_loader.py

Loads the WikiMIA benchmark (Shi et al., 2024) from HuggingFace datasets.
Each example has:
    - "input": the text
    - "label": 1 = member (seen during pretraining), 0 = non-member

WikiMIA has three length-based configs: 32, 64, 128 tokens.
We only use the "original" (verbatim) setting -- WikiMIA does not separate
this into a distinct config; the paraphrased version is a different dataset
release, so loading the base configs below already gives you the original
setting.
"""

from datasets import load_dataset


def load_wikimia(length: int = 128, split: str = "WikiMIA_length" ):
    """
    Load a WikiMIA split by sequence length.

    Args:
        length: one of {32, 64, 128}
        split: internal, do not change

    Returns:
        List of dicts: [{"text": str, "label": int}, ...]
    """
    if length not in (32, 64, 128):
        raise ValueError("length must be one of 32, 64, 128")

    config_name = f"WikiMIA_length{length}"
    ds = load_dataset("swj0419/WikiMIA", config_name, split="WikiMIA")

    examples = []
    for row in ds:
        examples.append({
            "text": row["input"],
            "label": int(row["label"]),
        })
    return examples


if __name__ == "__main__":
    # quick smoke test
    data = load_wikimia(length=128)
    n_members = sum(e["label"] == 1 for e in data)
    n_non_members = sum(e["label"] == 0 for e in data)
    print(f"Loaded {len(data)} examples ({n_members} members, {n_non_members} non-members)")
    print("Example:", data[0])
