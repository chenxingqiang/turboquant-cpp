"""
TurboQuant-style KV cache quantization (PyTorch reference + optional native core).

PyPI distribution name: ``turboquant-kv``. Import package: ``turboquant``.

**Core install** is ``numpy`` only; PyTorch/SciPy load when you use ``[torch]`` or lazy-import
reference symbols (e.g. ``TurboQuantMSE``). See ``turboquant.dependencies.GUIDE_MARKDOWN``.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    from . import _native
except ImportError:
    _native = None

NATIVE_AVAILABLE = _native is not None
native = _native

from .api import (
    KeyCodec,
    ValueCodec,
    compressed_keys_to_torch_dict,
    default_backend,
    resolve_backend,
)
from .dependencies import GUIDE_MARKDOWN, print_guide
from .integration import (
    InferencePhase,
    dependency_guide,
    estimate_kv_storage,
    integration_hooks,
    integration_readme,
)
from .types import (
    AttentionDiagnostic,
    CompressedKeys,
    CompressedValues,
    CompressionConfig,
    KVStorageEstimate,
)

_LAZY = {
    "LloydMaxCodebook": ("turboquant.lloyd_max", "LloydMaxCodebook"),
    "solve_lloyd_max": ("turboquant.lloyd_max", "solve_lloyd_max"),
    "TurboQuantCompressorMSE": ("turboquant.compressors", "TurboQuantCompressorMSE"),
    "TurboQuantCompressorV2": ("turboquant.compressors", "TurboQuantCompressorV2"),
    "TurboQuantKVCache": ("turboquant.torch_impl", "TurboQuantKVCache"),
    "TurboQuantMSE": ("turboquant.torch_impl", "TurboQuantMSE"),
    "TurboQuantProd": ("turboquant.torch_impl", "TurboQuantProd"),
    "attention_cosine_similarity": ("turboquant.metrics", "attention_cosine_similarity"),
    "attention_rmse": ("turboquant.metrics", "attention_rmse"),
    "attention_topk_match_rate": ("turboquant.metrics", "attention_topk_match_rate"),
    "flatten_bhsd_scores": ("turboquant.metrics", "flatten_bhsd_scores"),
    "summarize_attention_drift": ("turboquant.metrics", "summarize_attention_drift"),
    "diagnose_attention_bhsd": ("turboquant.validation", "diagnose_attention_bhsd"),
    "quick_smoke_key_codec": ("turboquant.validation", "quick_smoke_key_codec"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(__all__))


__all__ = [
    "AttentionDiagnostic",
    "CompressedKeys",
    "CompressedValues",
    "CompressionConfig",
    "GUIDE_MARKDOWN",
    "InferencePhase",
    "dependency_guide",
    "KeyCodec",
    "KVStorageEstimate",
    "LloydMaxCodebook",
    "NATIVE_AVAILABLE",
    "TurboQuantCompressorMSE",
    "TurboQuantCompressorV2",
    "TurboQuantKVCache",
    "TurboQuantMSE",
    "TurboQuantProd",
    "ValueCodec",
    "attention_cosine_similarity",
    "attention_rmse",
    "attention_topk_match_rate",
    "compressed_keys_to_torch_dict",
    "default_backend",
    "diagnose_attention_bhsd",
    "estimate_kv_storage",
    "flatten_bhsd_scores",
    "integration_hooks",
    "integration_readme",
    "native",
    "print_guide",
    "quick_smoke_key_codec",
    "resolve_backend",
    "solve_lloyd_max",
    "summarize_attention_drift",
]

try:
    __version__ = version("turboquant-kv")
except PackageNotFoundError:
    __version__ = "0.1.0"
