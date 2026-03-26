"""
Higher-level validation helpers built on the public :mod:`turboquant.api` and :mod:`turboquant.metrics`.
"""

from __future__ import annotations

from typing import Literal, cast

import torch

from .api import KeyCodec
from .metrics import (
    attention_cosine_similarity,
    attention_rmse,
    attention_topk_match_rate,
)
from .types import AttentionDiagnostic, CompressionConfig

__all__ = ["diagnose_attention_bhsd", "quick_smoke_key_codec"]


def diagnose_attention_bhsd(
    queries: torch.Tensor,
    keys: torch.Tensor,
    key_codec: KeyCodec,
) -> AttentionDiagnostic:
    """
    Compare full-precision attention scores to TurboQuant asymmetric scores per (batch, head).

    ``queries`` and ``keys`` must share the same ``(B, H, *, D)`` layout (e.g. HuggingFace-style
    KV heads). Typical smoke test: ``queries = keys[:, :, -1:, :]`` (last-token query).

    Uses ``key_codec``'s backend for compression/scoring (native or torch).
    """
    if queries.dim() != 4 or keys.dim() != 4:
        raise ValueError("queries and keys must be (B, H, S, D)")
    B, H, _, D = queries.shape
    if keys.shape[0] != B or keys.shape[1] != H or keys.shape[3] != D:
        raise ValueError("B, H, D must match between queries and keys")

    cosines = []
    rmses = []
    top1 = []
    top5 = []

    for b in range(B):
        for h in range(H):
            q = queries[b : b + 1, h : h + 1].float()
            k = keys[b : b + 1, h : h + 1].float()
            scores_fp = torch.matmul(q, k.transpose(-2, -1))

            ck = key_codec.compress_bhsd(keys[b : b + 1, h : h + 1])
            scores_tq = key_codec.attention_scores_bhsd(
                queries[b : b + 1, h : h + 1], ck
            )

            cosines.append(
                attention_cosine_similarity(scores_fp, scores_tq, reduction="mean")
            )
            rmses.append(attention_rmse(scores_fp, scores_tq, reduction="mean"))
            top1.append(attention_topk_match_rate(scores_fp, scores_tq, 1, dim=-1))
            top5.append(attention_topk_match_rate(scores_fp, scores_tq, 5, dim=-1))

    n = len(cosines)
    return AttentionDiagnostic(
        mean_cosine_similarity=sum(cosines) / n,
        mean_top1_match=sum(top1) / n,
        mean_top5_match=sum(top5) / n,
        mean_rmse=sum(rmses) / n,
        n_heads_checked=n,
        details={"per_head_cosine": cosines, "per_head_rmse": rmses},
    )


def quick_smoke_key_codec(
    head_dim: int = 128,
    bits: int = 3,
    seq_len: int = 64,
    *,
    device: str = "cpu",
    backend: str = "auto",
) -> AttentionDiagnostic:
    """Synthetic random tensors; useful in CI when GPU model weights are unavailable."""
    cfg = CompressionConfig(head_dim=head_dim, bits=bits, device=device)
    bk = cast(Literal["auto", "native", "torch"], backend)
    codec = KeyCodec(cfg, backend=bk)
    keys = torch.randn(1, 1, seq_len, head_dim, device=device)
    queries = torch.randn(1, 1, 1, head_dim, device=device)
    return diagnose_attention_bhsd(queries, keys, codec)
