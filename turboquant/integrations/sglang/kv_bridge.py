"""
SGLang **token pool** KV layout notes (no ``sglang`` import required).

In SGLang v0.5.x, backends such as :class:`TorchNativeAttnBackend` read per-layer K/V from
``forward_batch.token_to_kv_pool`` as dense buffers shaped like
``[max_total_num_tokens, num_kv_heads, head_size]`` (see ``torch_native_backend.py`` docstrings).

For :class:`turboquant.api.KeyCodec`, flatten or slice to token-major rows ``[T, H, D]``, then use
:func:`turboquant.integrations.token_layout.tokens_thd_from_key_rows`.
"""

from __future__ import annotations

from turboquant.integrations.token_layout import tokens_thd_from_key_rows

__all__ = ["tokens_thd_from_key_rows"]
