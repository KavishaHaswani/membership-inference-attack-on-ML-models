"""
paraphrase.py

Generates paraphrases of WikiMIA "member" texts, to test whether the MIA
attacks still detect membership when the exact wording has changed.

This directly follows up on an open question raised in Shi et al., 2024
(the WikiMIA paper): "It remains an open question whether MIA methods
generalize to paraphrased versions of member text."

Uses a local, open-weight paraphrasing model (no external API calls needed,
so this stays fully reproducible/self-contained -- same spirit as the rest
of the project).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Paraphraser:
    def __init__(self, model_id: str = "humarin/chatgpt_paraphraser_on_T5_base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  -> downloading/loading tokenizer for {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"  -> downloading/loading model weights for {model_id} (this can take a while on first run) ...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        print(f"  -> {model_id} loaded on {self.device}")
        self.model.eval()

    @torch.no_grad()
    def paraphrase(self, text: str, num_beams: int = 5, max_length: int = 256) -> str:
        """
        Returns one paraphrase of `text`. Falls back to the original text
        if generation produces something empty/degenerate (rare, but can
        happen on very short or unusual inputs).
        """
        prompt = f"paraphrase: {text}"
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            num_beams=num_beams,
            num_return_sequences=1,
            max_length=max_length,
            no_repeat_ngram_size=3,
        )
        paraphrased = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if not paraphrased.strip():
            return text
        return paraphrased


if __name__ == "__main__":
    # quick smoke test (requires HF access + model download)
    p = Paraphraser()
    original = "The quick brown fox jumps over the lazy dog near the old bridge."
    print("Original:   ", original)
    print("Paraphrased:", p.paraphrase(original))
