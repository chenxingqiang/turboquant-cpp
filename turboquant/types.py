"""Shared datatypes for the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

__all__ = [
    "AttentionDiagnostic",
    "CompressedKeys",
    "CompressedValues",
    "CompressionConfig",
    "KVStorageEstimate",
]


@dataclass(frozen=True)
class CompressionConfig:
    """Parameters shared by key/value codecs."""

    head_dim: int
    bits: int = 3
    """Total bit budget per coordinate for the key path (MSE uses max(bits-1,1), QJL uses 1)."""
    seed: int = 42
    """Base seed. Torch ``TurboQuantCompressorV2`` uses ``seed`` for ``Pi`` and ``seed+10000`` for ``S``."""
    device: str = "cpu"
    """Device for the PyTorch reference backend only."""


BackendName = Literal["native", "torch"]


@dataclass
class CompressedKeys:
    """Flat storage for compressed key rows (one row per KV position × head)."""

    k_mse: "object"  # np.ndarray float32 (N, D) — quoted for lazy numpy import in hints
    signs: "object"  # np.ndarray int8 (N, D)
    residual_norm: "object"  # np.ndarray float32 (N,)
    head_dim: int
    bits: int
    backend: BackendName
    original_shape_bhsd: Optional[Tuple[int, int, int, int]] = None
    """If set, ``(B, H, S, D)`` before flatten; used to rebuild BHSD torch dicts."""

    @property
    def n_keys(self) -> int:
        import numpy as np

        km = np.asarray(self.k_mse)
        return int(km.shape[0])


@dataclass
class CompressedValues:
    """MSE-only value quantization (indices + per-vector norm)."""

    indices: "object"  # np.ndarray uint8 (N, D)
    vec_norm: "object"  # np.ndarray float32 (N,)
    head_dim: int
    bits: int
    backend: BackendName
    original_shape_bhsd: Optional[Tuple[int, int, int, int]] = None

    @property
    def n_vectors(self) -> int:
        import numpy as np

        ix = np.asarray(self.indices)
        return int(ix.shape[0])


@dataclass
class KVStorageEstimate:
    """Rough KV budget comparing FP16 baseline vs TurboQuant-style logical bit counts."""

    fp16_kv_bits_total: int
    turboquant_key_bits_total: int
    turboquant_value_bits_total: int
    turboquant_kv_bits_total: int
    compression_ratio_vs_fp16: float
    n_layer_vecs: int
    head_dim: int
    bits_key: int
    bits_value: int
    n_layers: int
    n_kv_heads: int
    seq_len: int


@dataclass
class AttentionDiagnostic:
    """Output of :func:`turboquant.validation.diagnose_attention_bhsd`."""

    mean_cosine_similarity: float
    mean_top1_match: float
    mean_top5_match: float
    mean_rmse: float
    n_heads_checked: int
    details: dict = field(default_factory=dict)
