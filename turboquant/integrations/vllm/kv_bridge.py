"""
Helpers for vLLM-style **paged KV** tensors (no dependency on ``vllm``).

Layout matches ``CPUAttentionBackend.get_kv_cache_shape`` in vLLM v0.18.x:
``[2, num_blocks, num_kv_heads, block_size, head_size]`` (dim 0 = K vs V).

Shared row layout helpers live in :mod:`turboquant.integrations.token_layout`.
"""

from __future__ import annotations

from turboquant.integrations.token_layout import (
    VllmPagedKVLayout,
    expect_paged_kv_layout,
    split_kv_cache,
    tokens_thd_from_key_rows,
)

__all__ = [
    "VllmPagedKVLayout",
    "expect_paged_kv_layout",
    "split_kv_cache",
    "tokens_thd_from_key_rows",
]
