"""Re-export tests: vLLM ``kv_bridge`` mirrors :mod:`turboquant.integrations.token_layout`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from turboquant.integrations.vllm.kv_bridge import (
    expect_paged_kv_layout,
    split_kv_cache,
    tokens_thd_from_key_rows,
)


def test_vllm_kv_bridge_matches_token_layout_import() -> None:
    from turboquant.integrations import token_layout as tl

    kv = torch.zeros(2, 3, 4, 5, 6)
    assert split_kv_cache(kv)[0].shape == tl.split_kv_cache(kv)[0].shape


def test_expect_paged_kv_layout_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="5D"):
        expect_paged_kv_layout(torch.zeros(2, 3))
