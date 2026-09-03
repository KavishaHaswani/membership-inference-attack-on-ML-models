"""
attacks.py

Four membership inference scoring functions. All take the output of
model_utils.PythiaWrapper.score_text() and return a single float score,
where HIGHER score = more likely to be a training member.

References:
    LOSS      -- Yeom et al., 2018 (arXiv:1709.01604)
    Zlib      -- Carlini et al., 2021 (arXiv:2012.07805)
    Min-K%    -- Shi et al., 2024 (arXiv:2310.16789)
    Min-K%++  -- Zhang et al., 2024 (arXiv:2404.02936), formula + pseudocode
                 from Section 4.2 / Appendix A
"""

import zlib
import torch


def loss_attack(stats: dict) -> float:
    """Yeom et al., 2018. Higher mean log-prob (lower loss) => more likely member."""
    return stats["token_log_probs"].mean().item()


def zlib_attack(text: str, stats: dict) -> float:
    """
    Carlini et al., 2021.
    Calibrates loss by the zlib-compressed length of the text (a proxy for
    text complexity/redundancy). Score = -loss / compressed_length.
    Higher score => more likely member.
    """
    loss = -stats["token_log_probs"].mean().item()
    compressed_len = len(zlib.compress(text.encode("utf-8")))
    if compressed_len == 0:
        compressed_len = 1  # guard against empty text
    return -loss / compressed_len


def mink_attack(stats: dict, k: float = 0.2) -> float:
    """
    Shi et al., 2024.
    Average log-prob over the k% of tokens with the LOWEST log-prob
    (i.e. the "most surprising" tokens). Higher score => more likely member.
    """
    token_log_probs = stats["token_log_probs"]
    n = max(1, int(len(token_log_probs) * k))
    lowest_k, _ = torch.sort(token_log_probs)
    return lowest_k[:n].mean().item()


def minkpp_attack(stats: dict, k: float = 0.2) -> float:
    """
    Zhang et al., 2024 (Min-K%++).
    Calibrates each token's log-prob by the mean (mu) and std (sigma) of the
    model's own predicted next-token distribution at that position, then
    averages the k% lowest-scoring tokens (same aggregation as Min-K%).

    token_score_t = (log p(x_t | x_<t) - mu_t) / sigma_t
        mu_t    = E_{z ~ p(.|x_<t)} [log p(z|x_<t)]
        sigma_t = sqrt(E_{z ~ p(.|x_<t)} [(log p(z|x_<t) - mu_t)^2])
    """
    token_log_probs = stats["token_log_probs"]   # [T]
    log_probs = stats["log_probs"]                 # [T, V]
    probs = stats["probs"]                           # [T, V]

    mu = (probs * log_probs).sum(-1)                                   # [T]
    sigma_sq = (probs * log_probs.square()).sum(-1) - mu.square()  # [T]
    sigma_sq = sigma_sq.clamp(min=1e-8)  # numerical safety

    token_scores = (token_log_probs - mu) / sigma_sq.sqrt()       # [T]

    n = max(1, int(len(token_scores) * k))
    lowest_k, _ = torch.sort(token_scores)
    return lowest_k[:n].mean().item()


ATTACK_REGISTRY = {
    "loss": lambda text, stats: loss_attack(stats),
    "zlib": lambda text, stats: zlib_attack(text, stats),
    "mink": lambda text, stats: mink_attack(stats, k=0.2),
    "minkpp": lambda text, stats: minkpp_attack(stats, k=0.2),
}
