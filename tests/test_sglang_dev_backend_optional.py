"""Optional: requires a working ``sglang`` install."""

from __future__ import annotations

import pytest

pytest.importorskip("sglang")

from turboquant.integrations.sglang import register_turboquant_dev_sglang_backend
from sglang.srt.layers.attention import attention_registry as reg


def test_register_turboquant_dev_factory() -> None:
    name = register_turboquant_dev_sglang_backend()
    assert name == "turboquant_dev"
    assert name in reg.ATTENTION_BACKENDS
    factory = reg.ATTENTION_BACKENDS[name]
    assert callable(factory)
