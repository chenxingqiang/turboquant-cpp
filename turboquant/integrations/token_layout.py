"""
Shared **token-major** KV tensor helpers for inference integrations (no engine imports).

vLLM (paged CPU-style) and SGLang (radix / token pool) both ultimately deal with
``[num_tokens, num_kv_heads, head_dim]``-style key rows when feeding :class:`turboquant.api.KeyCodec`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "VllmPagedKVLayout",
    "expect_paged_kv_layout",
    "split_kv_cache",
    "tokens_thd_from_key_rows",
]


@dataclass(frozen=True)
class VllmPagedKVLayout:
    num_blocks: int
    num_kv_heads: int
    block_size: int
    head_size: int


def expect_paged_kv_layout(kv_cache: torch.Tensor) -> VllmPagedKVLayout:
    if kv_cache.dim() != 5:
        raise ValueError(
            "vLLM CPU-style KV cache must be 5D "
            "[2, num_blocks, num_kv_heads, block_size, head_size]; "
            f"got {tuple(kv_cache.shape)}"
        )
    two, nb, nkv, bs, hs = kv_cache.shape
    if two != 2:
        raise ValueError(f"leading dim must be 2 (K/V), got {two}")
    return VllmPagedKVLayout(nb, nkv, bs, hs)


def split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(key_cache, value_cache)`` views, each 4D (vLLM-style paged layout)."""
    expect_paged_kv_layout(kv_cache)
    k, v = kv_cache[0], kv_cache[1]
    return k, v


def tokens_thd_from_key_rows(
    key_rows: torch.Tensor,
) -> torch.Tensor:
    """
    Map ``[num_tokens, num_kv_heads, head_size]`` to
    ``(1, num_kv_heads, num_tokens, head_size)`` for :meth:`turboquant.api.KeyCodec.compress_bhsd`.
    """
    if key_rows.dim() != 3:
        raise ValueError(f"expected (T, H, D), got {tuple(key_rows.shape)}")
    return key_rows.transpose(0, 1).unsqueeze(0).contiguous()
