"""
model_utils.py

Loads a Pythia model and runs a single forward pass per text, extracting
everything needed by all four attack methods:
    - token_log_probs: log-prob the model assigned to each actual next token
    - log_probs: full log-softmax distribution over the vocab, per position
    - probs: full softmax distribution over the vocab, per position

This is deliberately factored out so the forward pass is run exactly once
per example, shared across all attack methods (matches the computational
note in Zhang et al., 2024, Appendix A: all methods besides Ref/Neighbor
cost the same single forward pass).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PythiaWrapper:
    def __init__(self, model_id: str = "EleutherAI/pythia-1.4b", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_text(self, text: str):
        """
        Run one forward pass and return the raw statistics needed by all
        attack methods.

        Returns:
            token_log_probs: FloatTensor [T-1]
            log_probs:       FloatTensor [T-1, V]
            probs:           FloatTensor [T-1, V]
            n_tokens:        int, T-1 (number of scored token positions)
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        if input_ids.shape[1] < 2:
            # too short to score (need at least one prediction step)
            return None

        outputs = self.model(input_ids, labels=input_ids)
        logits = outputs.logits[0, :-1]          # [T-1, V], predicting tokens 1..T-1
        targets = input_ids[0, 1:]                 # [T-1]

        log_probs = torch.log_softmax(logits, dim=-1)     # [T-1, V]
        probs = torch.softmax(logits, dim=-1)               # [T-1, V]
        token_log_probs = log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)                                        # [T-1]

        return {
            "token_log_probs": token_log_probs.float().cpu(),
            "log_probs": log_probs.float().cpu(),
            "probs": probs.float().cpu(),
            "n_tokens": token_log_probs.shape[0],
        }


if __name__ == "__main__":
    # quick smoke test (requires HF access + model download)
    wrapper = PythiaWrapper(model_id="EleutherAI/pythia-1.4b")
    stats = wrapper.score_text("The quick brown fox jumps over the lazy dog.")
    print("n_tokens:", stats["n_tokens"])
    print("mean token log-prob (≈ negative loss):", stats["token_log_probs"].mean().item())
