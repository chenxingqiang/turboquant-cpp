"""
vLLM integration helpers (optional ``vllm`` dependency).

- :mod:`turboquant.integrations.vllm.kv_bridge` — tensor layout helpers (always importable).
- :func:`register_turboquant_dev_backend` — register a **CUSTOM** attention backend (v0.18.x),
  currently aliasing CPU attention for plumbing tests.

**Usage (before starting vLLM)**

.. code-block:: python

    from turboquant.integrations.vllm import register_turboquant_dev_backend
    register_turboquant_dev_backend()
    # Then: vllm serve ... --attention-backend CUSTOM
"""

from __future__ import annotations

from .dev_backend import register_turboquant_dev_backend, turboquant_dev_backend_available
from .kv_bridge import (
    VllmPagedKVLayout,
    expect_paged_kv_layout,
    split_kv_cache,
    tokens_thd_from_key_rows,
)

__all__ = [
    "VllmPagedKVLayout",
    "expect_paged_kv_layout",
    "register_turboquant_dev_backend",
    "split_kv_cache",
    "tokens_thd_from_key_rows",
    "turboquant_dev_backend_available",
]
