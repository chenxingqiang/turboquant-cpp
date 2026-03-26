"""Shared token / vLLM paged layout helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from turboquant.integrations.token_layout import (
    expect_paged_kv_layout,
    split_kv_cache,
    tokens_thd_from_key_rows,
)


def test_split_kv_cache_shape() -> None:
    kv = torch.zeros(2, 10, 8, 16, 128)
    k, v = split_kv_cache(kv)
    assert k.shape == (10, 8, 16, 128)
    assert v.shape == (10, 8, 16, 128)


def test_tokens_thd_from_key_rows() -> None:
    t, h, d = 5, 4, 64
    rows = torch.randn(t, h, d)
    bhsd = tokens_thd_from_key_rows(rows)
    assert bhsd.shape == (1, h, t, d)
