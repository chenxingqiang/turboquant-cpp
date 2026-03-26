"""
Synthetic verification (paper-style checks). Invoked by ``turboquant-verify`` CLI.
Requires optional dependency: ``pip install 'turboquant-kv[torch]'``.
"""

from __future__ import annotations

import math
import time

import torch

from .lloyd_max import LloydMaxCodebook
from .torch_impl import TurboQuantKVCache, TurboQuantMSE, TurboQuantProd


def test_lloyd_max_codebook() -> None:
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d, bits)
            print(
                f"  d={d:>4d}, bits={bits}: {cb.n_levels} levels, "
                f"distortion/coord={cb.distortion:.6f}, "
                f"centroids range=[{cb.centroids.min():.4f}, {cb.centroids.max():.4f}]"
            )

    cb = LloydMaxCodebook(128, 3)
    centroid_sum = cb.centroids.sum().abs().item()
    print(
        f"\n  Symmetry check (d=128, b=3): sum of centroids = {centroid_sum:.6f} (should be ~0)"
    )
    assert centroid_sum < 0.01, "Centroids should be symmetric!"
    print("  PASSED\n")


def test_mse_quantizer() -> None:
    print("=" * 60)
    print("TEST 2: MSE Quantizer Distortion")
    print("=" * 60)

    d = 128
    n_vectors = 1000
    device = "cpu"

    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        x = torch.randn(n_vectors, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        x_hat, _indices = quantizer(x)

        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))

        ratio = mse / theoretical_bound
        status = "OK" if ratio <= 1.5 else "WARN"

        print(
            f"  bits={bits}: MSE={mse:.6f}, theory_bound={theoretical_bound:.6f}, "
            f"ratio={ratio:.3f} [{status}]"
        )

    print()


def test_inner_product_unbiasedness() -> None:
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness (QJL Correction)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(d, bits, seed=42, device=device)

        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)

        compressed = quantizer.quantize(x)
        estimated_ip = quantizer.inner_product(y, compressed)

        bias = (estimated_ip - true_ip).mean().item()
        rmse = ((estimated_ip - true_ip) ** 2).mean().sqrt().item()
        correlation = torch.corrcoef(torch.stack([true_ip, estimated_ip]))[0, 1].item()

        theoretical_distortion = math.sqrt(3) * math.pi**2 / d * (1 / (4**bits))

        print(
            f"  bits={bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, "
            f"corr={correlation:.4f}, theory_D={theoretical_distortion:.6f}"
        )

    print()


def test_mse_only_inner_product_bias() -> None:
    print("=" * 60)
    print("TEST 4: MSE-Only Inner Product Bias (motivation for QJL)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)

        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)
        x_hat, _ = quantizer(x)
        mse_ip = (x_hat * y).sum(dim=-1)

        bias = (mse_ip - true_ip).mean().item()

        print(f"  bits={bits}: bias={bias:+.6f} (MSE-only is biased, QJL fixes this)")

    print()


def test_kv_cache() -> None:
    print("=" * 60)
    print("TEST 5: KV Cache Compression Ratios")
    print("=" * 60)

    d_key = 128
    d_value = 128
    seq_len = 1024
    device = "cpu"

    for bits in [2, 3, 4]:
        cache = TurboQuantKVCache(d_key, d_value, bits=bits, seed=42, device=device)

        keys = torch.randn(seq_len, d_key, device=device)
        values = torch.randn(seq_len, d_value, device=device)

        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        print(
            f"  bits={bits}: compression={usage['compression_ratio']:.2f}x "
            f"({usage['total_bits'] / 8 / 1024:.1f} KB vs "
            f"{usage['fp16_bits'] / 8 / 1024:.1f} KB fp16)"
        )

        query = torch.randn(1, d_key, device=device)
        scores = cache.attention_scores(query)
        print(
            f"           attention scores shape: {scores.shape}, "
            f"range=[{scores.min():.3f}, {scores.max():.3f}]"
        )

    print()


def test_needle_in_haystack() -> None:
    print("=" * 60)
    print("TEST 6: Needle-in-Haystack Retrieval")
    print("=" * 60)

    d = 128
    device = "cpu"

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)

            quantizer = TurboQuantProd(d, bits, seed=42, device=device)
            compressed = quantizer.quantize(keys)

            estimated_ips = quantizer.inner_product(query.expand(seq_len, -1), compressed)

            top_idx = estimated_ips.argmax().item()
            found = top_idx == needle_pos

            top5 = estimated_ips.topk(5).indices.tolist()
            in_top5 = needle_pos in top5

            status = "EXACT" if found else ("TOP-5" if in_top5 else "MISS")
            print(
                f"  bits={bits}, seq={seq_len:>5d}: top1={top_idx:>5d} "
                f"(needle={needle_pos:>5d}) [{status}]"
            )

    print()


def test_gpu_if_available() -> None:
    print("=" * 60)
    print("TEST 7: GPU Benchmark (if CUDA available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        print()
        return

    device = "cuda"
    d = 128
    bits = 3
    seq_len = 8192
    n_queries = 64

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Config: d={d}, bits={bits}, seq_len={seq_len}, n_queries={n_queries}")

    quantizer = TurboQuantProd(d, bits, seed=42, device=device)

    keys = torch.randn(seq_len, d, device=device)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    queries = torch.randn(n_queries, d, device=device)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    quant_time = (time.perf_counter() - t0) / 10
    print(f"  Quantize {seq_len} keys: {quant_time * 1000:.2f} ms")

    compressed = quantizer.quantize(keys)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        for i in range(n_queries):
            quantizer.inner_product(queries[i : i + 1].expand(seq_len, -1), compressed)
    torch.cuda.synchronize()
    ip_time = (time.perf_counter() - t0) / 100
    print(
        f"  Inner product ({n_queries} queries x {seq_len} keys): {ip_time * 1000:.2f} ms"
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        queries @ keys.T
    torch.cuda.synchronize()
    fp_time = (time.perf_counter() - t0) / 100
    print(f"  Full-precision matmul: {fp_time * 1000:.2f} ms")

    fp16_bytes = seq_len * d * 2
    quant_bytes = seq_len * d * bits / 8
    print(f"  Memory: {fp16_bytes / 1024:.1f} KB (fp16) vs {quant_bytes / 1024:.1f} KB (TQ-{bits}bit)")
    print(f"  Compression: {fp16_bytes / quant_bytes:.1f}x")
    print()


def run_all() -> None:
    print()
    print("TurboQuant Implementation Verification")
    print("Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)")
    print()

    test_lloyd_max_codebook()
    test_mse_quantizer()
    test_inner_product_unbiasedness()
    test_mse_only_inner_product_bias()
    test_kv_cache()
    test_needle_in_haystack()
    test_gpu_if_available()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
