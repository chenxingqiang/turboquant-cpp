"""
Register a **development** SGLang attention backend name that delegates to ``TorchNativeAttnBackend``.

SGLang selects backends via ``ATTENTION_BACKENDS`` in
``sglang.srt.layers.attention.attention_registry`` (v0.5.9+). This module adds a factory under a
new string key so you can run ``python -m sglang.launch_server --attention-backend <name>`` after
calling :func:`register_turboquant_dev_sglang_backend` in the **same process** before server start
(typically via a small launcher script that imports this module first).

The implementation is **not** TurboQuant yet — it exists to validate registry wiring before you
replace ``forward_*`` with compressed-K + asymmetric scores inside a fork of SGLang.
"""

from __future__ import annotations

import importlib.util

__all__ = [
    "register_turboquant_dev_sglang_backend",
    "turboquant_dev_sglang_backend_available",
]

_DEFAULT_NAME = "turboquant_dev"


def turboquant_dev_sglang_backend_available() -> bool:
    return importlib.util.find_spec("sglang") is not None


def register_turboquant_dev_sglang_backend(*, name: str = _DEFAULT_NAME) -> str:
    """
    Register ``name`` in SGLang's attention registry, pointing at ``TorchNativeAttnBackend``.

    Returns the backend string to pass as ``--attention-backend``.
    """
    if not turboquant_dev_sglang_backend_available():
        raise RuntimeError(
            "SGLang is not installed. Install: pip install 'turboquant-kv[sglang]' "
            "(see SGLang docs for platform/GPU requirements)."
        )

    from sglang.srt.layers.attention import attention_registry as reg

    if name in reg.ATTENTION_BACKENDS:
        return name

    @reg.register_attention_backend(name)
    def _create_turboquant_dev_backend(runner):
        from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

        return TorchNativeAttnBackend(runner)

    return name
