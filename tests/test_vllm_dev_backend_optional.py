"""Optional: requires a working ``vllm`` install (typically Linux + CUDA)."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from turboquant.integrations.vllm import register_turboquant_dev_backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_register_custom_reports_turboquant_dev_name() -> None:
    register_turboquant_dev_backend()
    cls = AttentionBackendEnum.CUSTOM.get_class()
    assert cls.get_name() == "TURBOQUANT_DEV"
    impl = cls.get_impl_cls()
    assert impl.__name__ == "TurboQuantDevAttentionBackendImpl"
