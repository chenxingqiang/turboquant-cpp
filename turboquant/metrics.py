"""
Numerical tools to compare full-precision attention logits vs TurboQuant asymmetric scores.
Use these in CI or notebooks before wiring codecs into a serving stack.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "attention_cosine_similarity",
    "attention_rmse",
    "attention_topk_match_rate",
    "flatten_bhsd_scores",
    "summarize_attention_drift",
]


def flatten_bhsd_scores(scores: torch.Tensor) -> torch.Tensor:
    """(B, H, Sq, Sk) -> (B * H * Sq, Sk)."""
    if scores.dim() != 4:
        raise ValueError(f"expected (B,H,Sq,Sk), got {tuple(scores.shape)}")
    B, H, Sq, Sk = scores.shape
    return scores.reshape(B * H * Sq, Sk)


def attention_cosine_similarity(
    scores_ref: torch.Tensor,
    scores_cmp: torch.Tensor,
    *,
    dim: int = -1,
    reduction: Literal["mean", "none"] = "mean",
    eps: float = 1e-8,
) -> Union[torch.Tensor, float]:
    """
    Cosine similarity between two score tensors of the same shape, computed along ``dim``
    (default: per query row over key positions).

    For (B,H,Sq,Sk), flattens batch/head/query into one dimension, then compares row-wise.
    """
    if scores_ref.shape != scores_cmp.shape:
        raise ValueError(f"shape mismatch {scores_ref.shape} vs {scores_cmp.shape}")

    if scores_ref.dim() == 4:
        a = flatten_bhsd_scores(scores_ref.float())
        b = flatten_bhsd_scores(scores_cmp.float())
    else:
        a = scores_ref.float()
        b = scores_cmp.float()

    if dim != -1 and scores_ref.dim() != 4:
        cos = F.cosine_similarity(a, b, dim=dim, eps=eps)
    else:
        cos = F.cosine_similarity(a, b, dim=-1, eps=eps)

    if reduction == "mean":
        return float(cos.mean().item())
    return cos


def attention_rmse(
    scores_ref: torch.Tensor,
    scores_cmp: torch.Tensor,
    *,
    reduction: Literal["mean", "none"] = "mean",
) -> Union[torch.Tensor, float]:
    """Element-wise RMSE between score tensors of the same shape."""
    if scores_ref.shape != scores_cmp.shape:
        raise ValueError(f"shape mismatch {scores_ref.shape} vs {scores_cmp.shape}")
    diff = (scores_ref.float() - scores_cmp.float()).reshape(-1)
    rmse = (diff * diff).mean().sqrt()
    if reduction == "mean":
        return float(rmse.item())
    return diff


def attention_topk_match_rate(
    scores_ref: torch.Tensor,
    scores_cmp: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
) -> float:
    """
    Fraction of rows (queries) for which the top-``k`` key indices agree between ref and cmp.

    Works on (B,H,Sq,Sk) by flattening to (B*H*Sq, Sk).
    """
    if scores_ref.shape != scores_cmp.shape:
        raise ValueError(f"shape mismatch {scores_ref.shape} vs {scores_cmp.shape}")

    if scores_ref.dim() == 4:
        a = flatten_bhsd_scores(scores_ref)
        b = flatten_bhsd_scores(scores_cmp)
    else:
        a = scores_ref
        b = scores_cmp

    top_a = a.topk(k, dim=dim).indices.sort(dim=dim).values
    top_b = b.topk(k, dim=dim).indices.sort(dim=dim).values
    match = (top_a == top_b).all(dim=dim).float().mean()
    return float(match.item())


def summarize_attention_drift(
    scores_ref: torch.Tensor,
    scores_cmp: torch.Tensor,
    *,
    topk: Tuple[int, ...] = (1, 5),
) -> dict:
    """Single dict for logging: cosine, RMSE, top-k agreement rates."""
    out = {
        "cosine_similarity": attention_cosine_similarity(
            scores_ref, scores_cmp, reduction="mean"
        ),
        "rmse": attention_rmse(scores_ref, scores_cmp, reduction="mean"),
    }
    for kk in topk:
        out[f"top{kk}_match_rate"] = attention_topk_match_rate(
            scores_ref, scores_cmp, kk, dim=-1
        )
    return out
