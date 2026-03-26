"""
SGLang integration helpers (optional ``sglang`` dependency).

- :mod:`turboquant.integrations.sglang.kv_bridge` — token-pool KV layout notes + row helpers.
- :func:`register_turboquant_dev_sglang_backend` — register ``--attention-backend turboquant_dev``
  (delegates to ``TorchNativeAttnBackend`` for wiring tests).

**Usage (before starting the SGLang server in the same Python process)**

.. code-block:: python

    from turboquant.integrations.sglang import register_turboquant_dev_sglang_backend
    register_turboquant_dev_sglang_backend()
    # Then launch with: --attention-backend turboquant_dev
"""

from __future__ import annotations

from .dev_backend import (
    register_turboquant_dev_sglang_backend,
    turboquant_dev_sglang_backend_available,
)
from .kv_bridge import tokens_thd_from_key_rows

__all__ = [
    "register_turboquant_dev_sglang_backend",
    "tokens_thd_from_key_rows",
    "turboquant_dev_sglang_backend_available",
]
