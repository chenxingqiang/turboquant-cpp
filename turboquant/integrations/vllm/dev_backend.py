"""
Development attention backend for vLLM **v0.18.x**: registers ``AttentionBackendEnum.CUSTOM``.

Currently **delegates entirely** to the stock CPU attention implementation (dense KV) so you can
verify ``--attention-backend CUSTOM`` wiring before replacing ``forward`` with TurboQuant math.

Call :func:`register_turboquant_dev_backend` **before** constructing the vLLM engine / worker.
"""

from __future__ import annotations

import importlib.util

__all__ = [
    "register_turboquant_dev_backend",
    "turboquant_dev_backend_available",
]


def turboquant_dev_backend_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def register_turboquant_dev_backend() -> str:
    """
    Register :class:`TurboQuantDevAttentionBackend` as ``AttentionBackendEnum.CUSTOM``.

    Returns the fully-qualified class path passed to vLLM's registry.
    """
    if not turboquant_dev_backend_available():
        raise RuntimeError(
            "vLLM is not installed. Install: pip install 'turboquant-kv[vllm]' "
            "(Linux + CUDA recommended; macOS may be unsupported by vLLM wheels)."
        )

    from vllm.v1.attention.backends.cpu_attn import (
        CPUAttentionBackend,
        CPUAttentionBackendImpl,
    )
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    mod = __name__

    TurboQuantDevAttentionBackendImpl = type(
        "TurboQuantDevAttentionBackendImpl",
        (CPUAttentionBackendImpl,),
        {"__module__": mod},
    )

    def _get_impl_cls():
        return TurboQuantDevAttentionBackendImpl

    TurboQuantDevAttentionBackend = type(
        "TurboQuantDevAttentionBackend",
        (CPUAttentionBackend,),
        {
            "__module__": mod,
            "get_name": staticmethod(lambda: "TURBOQUANT_DEV"),
            "get_impl_cls": staticmethod(_get_impl_cls),
        },
    )

    g = globals()
    g["TurboQuantDevAttentionBackend"] = TurboQuantDevAttentionBackend
    g["TurboQuantDevAttentionBackendImpl"] = TurboQuantDevAttentionBackendImpl

    path = f"{mod}.TurboQuantDevAttentionBackend"
    register_backend(AttentionBackendEnum.CUSTOM, path)
    return path
