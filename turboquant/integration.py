"""
Where TurboQuant sits in decoder inference, and how to budget KV memory.

This module is **documentation + planning helpers** — it does not patch vLLM, TensorRT-LLM,
or PyTorch internals. Use :class:`turboquant.api.KeyCodec` at the call sites described below.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from .types import KVStorageEstimate

__all__ = [
    "InferencePhase",
    "dependency_guide",
    "estimate_kv_storage",
    "integration_hooks",
    "integration_readme",
]


def dependency_guide() -> str:
    """Long-form notes on PyTorch vs native vs vLLM/SGLang (Markdown)."""
    from .dependencies import GUIDE_MARKDOWN

    return GUIDE_MARKDOWN


class InferencePhase(str, Enum):
    """Stages where KV quantization hooks apply in a typical autoregressive decoder."""

    PREFILL_APPEND_KV = "prefill_append_kv"
    """After computing K/V projections for prompt tokens, before storing in cache."""

    DECODE_APPEND_KV = "decode_append_kv"
    """Each new token: append that step's K/V to the running cache."""

    ATTENTION_SCORES = "attention_scores"
    """When forming QK^T (or fused attention): use asymmetric scores from compressed K."""


def estimate_kv_storage(
    *,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits_key: int,
    bits_value: int,
) -> KVStorageEstimate:
    """
    Logical bit counts aligned with ``validate.py`` accounting (indices + signs + norms).

    **Not** the same as on-the-wire packing if you bit-pack indices; this is a planning upper
    bound close to the reference script.
    """
    # One K row and one V row per (layer, kv_head, time_step).
    n_positions = seq_len * n_kv_heads * n_layers
    mse_bits_k = max(bits_key - 1, 1)

    # Per key vector (TurboQuant-style key path; matches validate.py-style accounting)
    key_bits_per_vec = head_dim * mse_bits_k + head_dim * 1 + 16 + 16
    # Per value vector (MSE only)
    val_bits_per_vec = head_dim * bits_value + 16

    fp16_bits_per_position = 2 * head_dim * 16

    turbo_k = n_positions * key_bits_per_vec
    turbo_v = n_positions * val_bits_per_vec
    turbo_total = turbo_k + turbo_v
    fp_total = n_positions * fp16_bits_per_position

    return KVStorageEstimate(
        fp16_kv_bits_total=fp_total,
        turboquant_key_bits_total=turbo_k,
        turboquant_value_bits_total=turbo_v,
        turboquant_kv_bits_total=turbo_total,
        compression_ratio_vs_fp16=fp_total / turbo_total if turbo_total else 0.0,
        n_layer_vecs=n_positions,
        head_dim=head_dim,
        bits_key=bits_key,
        bits_value=bits_value,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
    )


def integration_hooks() -> List[dict]:
    """
    Structured checklist for engine integrators. Each item is a dict:
    ``phase``, ``action``, ``api``.
    """
    return [
        {
            "phase": InferencePhase.PREFILL_APPEND_KV.value,
            "action": "Replace FP16/BF16 K write with KeyCodec.compress_bhsd (or compress_flat).",
            "api": "turboquant.api.KeyCodec.compress_bhsd",
        },
        {
            "phase": InferencePhase.DECODE_APPEND_KV.value,
            "action": "Same as prefill: append one (or more) compressed key rows per new token.",
            "api": "turboquant.api.KeyCodec.compress_flat / compress_bhsd",
        },
        {
            "phase": InferencePhase.PREFILL_APPEND_KV.value,
            "action": "Store values with ValueCodec (MSE); decompress when applying softmax @ V unless fused.",
            "api": "turboquant.api.ValueCodec.compress_bhsd",
        },
        {
            "phase": InferencePhase.ATTENTION_SCORES.value,
            "action": "Use asymmetric logits: KeyCodec.attention_scores_bhsd (or attention_scores_flat).",
            "api": "turboquant.api.KeyCodec.attention_scores_bhsd",
        },
        {
            "phase": "quality_gate",
            "action": "Before production: run turboquant.validation.diagnose_attention_bhsd on representative tensors.",
            "api": "turboquant.validation.diagnose_attention_bhsd",
        },
    ]


def integration_readme() -> str:
    """Plain-text overview suitable for pasting into an internal design doc."""
    lines = [
        "TurboQuant KV integration (summary)",
        "==================================",
        "",
        "1) Prefill: for each layer/head, when K and V are produced for prompt tokens,",
        "   compress K with KeyCodec and V with ValueCodec instead of storing dense FP16/BF16.",
        "",
        "2) Decode: each step append compressed K/V for the new token to the per-layer cache.",
        "",
        "3) Attention: for logits QK^T, call KeyCodec.attention_scores_* on compressed K",
        "   (do not round-trip K through full dequant first).",
        "",
        "4) Values: decompress MSE values for the softmax-weighted sum, or fuse dequant+gemm in a custom kernel.",
        "",
        "5) Validate: use turboquant.metrics and turboquant.validation on slices of real tensors",
        "   to track cosine similarity and top-token agreement vs FP attention.",
        "",
        "See turboquant.integration.integration_hooks() for a machine-readable checklist.",
    ]
    return "\n".join(lines)
