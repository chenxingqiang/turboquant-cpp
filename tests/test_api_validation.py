from __future__ import annotations

import numpy as np
import torch

from turboquant.api import KeyCodec, ValueCodec, compressed_keys_to_torch_dict, resolve_backend
from turboquant.integration import estimate_kv_storage, integration_hooks
from turboquant.metrics import summarize_attention_drift
from turboquant.types import CompressedKeys, CompressionConfig
from turboquant.validation import diagnose_attention_bhsd, quick_smoke_key_codec


def test_estimate_kv_storage_positive_ratio() -> None:
    est = estimate_kv_storage(
        seq_len=1024,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        bits_key=3,
        bits_value=3,
    )
    assert est.compression_ratio_vs_fp16 > 1.0
    assert est.turboquant_kv_bits_total < est.fp16_kv_bits_total


def test_key_value_codec_roundtrip_torch() -> None:
    cfg = CompressionConfig(head_dim=16, bits=3, device="cpu")
    kc = KeyCodec(cfg, backend="torch")
    vc = ValueCodec(cfg, backend="torch")
    keys = torch.randn(1, 2, 5, 16)
    vals = torch.randn(1, 2, 5, 16)
    ck = kc.compress_bhsd(keys)
    cv = vc.compress_bhsd(vals)
    assert ck.n_keys == 10
    q = torch.randn(1, 2, 1, 16)
    s = kc.attention_scores_bhsd(q, ck)
    assert s.shape == (1, 2, 1, 5)
    v_back = vc.decompress_bhsd(cv)
    assert v_back.shape == vals.shape


def test_compressed_keys_to_torch_dict() -> None:
    cfg = CompressionConfig(head_dim=8, bits=3, device="cpu")
    kc = KeyCodec(cfg, backend="torch")
    x = np.random.randn(3, 8).astype(np.float32)
    ck = kc.compress_flat(x)
    d = compressed_keys_to_torch_dict(
        CompressedKeys(
            k_mse=ck.k_mse,
            signs=ck.signs,
            residual_norm=ck.residual_norm,
            head_dim=ck.head_dim,
            bits=ck.bits,
            backend=ck.backend,
            original_shape_bhsd=(1, 1, 3, 8),
        ),
        "cpu",
    )
    assert d["k_mse"].shape == (1, 1, 3, 8)


def test_diagnose_and_smoke() -> None:
    rep = quick_smoke_key_codec(head_dim=32, bits=3, seq_len=16, backend="torch")
    assert 0.0 <= rep.mean_cosine_similarity <= 1.0
    assert rep.n_heads_checked == 1

    cfg = CompressionConfig(head_dim=32, bits=3, device="cpu")
    codec = KeyCodec(cfg, backend="torch")
    keys = torch.randn(1, 1, 8, 32)
    q = keys[:, :, -1:, :]
    diag = diagnose_attention_bhsd(q, keys, codec)
    assert diag.n_heads_checked == 1

    scores_fp = torch.matmul(q, keys.transpose(-2, -1))
    scores_tq = codec.attention_scores_bhsd(q, codec.compress_bhsd(keys))
    drift = summarize_attention_drift(scores_fp, scores_tq)
    assert "cosine_similarity" in drift


def test_integration_hooks_nonempty() -> None:
    assert len(integration_hooks()) >= 4


def test_resolve_backend_torch_explicit() -> None:
    assert resolve_backend("torch") == "torch"
