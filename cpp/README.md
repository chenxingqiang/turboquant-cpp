# TurboQuant C++ core

Header-only **Eigen** is fetched at configure time (no system Eigen required). Optional **OpenMP** accelerates batch compression when available.

## Public headers

| Header | Purpose |
|--------|---------|
| `include/turboquant/lloyd_max.hpp` | Gaussian-approx Lloyd–Max codebook (`build_lloyd_max_codebook`, scalar quant helpers) |
| `include/turboquant/turboquant.hpp` | `TurboQuantKeyCompressor` (normalize → rotate → quantize → QJL signs; packed asymmetric scores), `TurboQuantMSECompressor` (values) |

## Symbols (summary)

- **`TurboQuantKeyCompressor`**: `compress_row` / `compress_batch`; `asymmetric_attention_scores` / `asymmetric_attention_scores_packed` for \(\langle Q,K\rangle\) without full-precision \(K\).
- **`TurboQuantMSECompressor`**: `compress_row`, `decompress_row` for value vectors.
- **`random_orthogonal` / `gaussian_fill`**: matrix initialization (not PyTorch RNG-compatible).

## Python extension

With `pip install turboquant-kv` (sdist/wheel build), CMake builds `turboquant._native`:

- `KeyCompressor(head_dim, bits, seed_pi=42, seed_s=10042).compress(x)` → `k_mse`, `signs`, `residual_norm`
- `KeyCompressor.attention_scores(queries, k_mse, signs, residual_norm)`
- `ValueCompressor(head_dim, bits, seed_pi=42).compress` / `decompress`
- `lloyd_max_centroids(d, bits)` → NumPy arrays

## Standalone build (tests only)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBOQUANT_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build
```
