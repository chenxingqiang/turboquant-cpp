# TurboQuant-KV (`turboquant-kv`)

Implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026): two-stage vector quantization for **LLM KV caches** (random rotation + Lloyd–Max, plus **QJL** for unbiased attention inner products). This repo ships a **pip/uv package**, a **PyTorch reference**, an optional **C++/Eigen extension** (`turboquant._native`), and CLI tools for synthetic and real-model checks.

**PyPI name:** `turboquant-kv` · **Import:** `import turboquant`

**Canonical upstream:** development and releases are tracked at [github.com/chenxingqiang/turboquant-cpp](https://github.com/chenxingqiang/turboquant-cpp) (default branch **`main`**).

**Git default branch:** local and remote use `main` (not `master`). If the GitHub UI still shows `master` as default, set **Settings → General → Default branch** to `main`, or run `gh repo edit --default-branch main`. After that: `git fetch origin && git remote set-head origin -a`. Optional cleanup: `git push origin --delete master` once nothing depends on it.

---

## Contents

- [What you get](#what-you-get)
- [Install](#install)
- [Dependencies: PyTorch, Transformers, vLLM/SGLang](#dependencies-pytorch-transformers-vllmsglang)
- [vLLM 完整适配（单独文档）](docs/vllm-integration.md)
- [SGLang 适配（单独文档）](docs/sglang-integration.md)
- [vLLM / SGLang 实操教程（英文）](docs/tutorial-inference-integrations.md)
- [NumPy, JAX, and this package](#numpy-jax-and-this-package)
- [Public API](#public-api)
- [Where it fits in inference](#where-it-fits-in-inference)
- [C++ library](#c-library)
- [Algorithm (short)](#algorithm-short)
- [Reported results](#reported-results)
- [CLI](#cli)
- [Repository layout](#repository-layout)
- [Requirements & CUDA PyTorch](#requirements--cuda-pytorch)
- [References & license](#references--license)

---

## What you get

| Layer | Purpose |
|--------|---------|
| **`KeyCodec` / `ValueCodec`** | Stable service API: compress K/V, asymmetric QK scores, MSE value path. |
| **`turboquant._native`** | Fast C++ path (Eigen, optional OpenMP); **NumPy in/out**, no PyTorch required. |
| **PyTorch modules** (`[torch]` extra) | `TurboQuantMSE`, `TurboQuantProd`, `TurboQuantKVCache`, `compressors` — paper-aligned reference. |
| **`turboquant-verify`** | Synthetic validation (needs `[torch]`). |
| **`turboquant-validate`** | Qwen2.5 KV experiment (needs `[validate]` + CUDA). |
| **`cpp/`** | Static library + tests; same math as native wheel. |

---

## Install

### pip

```bash
pip install turboquant-kv                 # core: NumPy only; builds/links _native when possible
pip install turboquant-kv[torch]         # + PyTorch, SciPy (reference, metrics, turboquant-verify)
pip install turboquant-kv[validate]     # + Transformers stack (turboquant-validate)
pip install turboquant-kv[vllm]        # + vLLM 0.18.x (integrations.dev_backend; Linux/CUDA typical)
pip install turboquant-kv[sglang]      # + SGLang (integrations.sglang.dev_backend; see SGLang docs)
pip install -e .                         # from source: CMake + Eigen fetch for _native
```

### uv

```bash
cd /path/to/turboquant-cpp
uv sync --group dev --extra torch       # pytest + torch + scipy (for tests)
uv run pytest tests/
uv run turboquant-verify                # needs [torch]
```

Minimal resolver check (NumPy-only tree): `uv sync` without `--extra torch`.

### macOS (Xcode license)

If CMake says the C++ compiler is “broken” with *“You have not agreed to the Xcode license agreements”*, run `sudo xcodebuild -license`, then remove `cpp/build` and reconfigure.

---

## Dependencies: PyTorch, Transformers, vLLM/SGLang

- **PyTorch is optional.** Default install is **NumPy only**. With `_native` built, `KeyCodec` / `ValueCodec` with `backend="native"` run on **NumPy `float32`** buffers without importing PyTorch.
- **`[torch]`** adds PyTorch + SciPy for the reference stack, `metrics`, `validation`, and **`turboquant-verify`**.
- **Transformers** does not replace PyTorch; it typically sits on PyTorch. Use **`[validate]`** for the Qwen validation script.
- **vLLM / SGLang** require **per-engine integration** (KV write + attention path). This repo stays **framework-agnostic** at the API level. Longer text: `turboquant.dependency_guide()` or `turboquant.GUIDE_MARKDOWN`.

**vLLM 专项：** TurboQuant 不是 FP8 KV 那种「反量化再 matmul」；要在 vLLM 里 **省显存且真加速**，需要新 attention/KV 分支 + **GPU kernel** 与 **Paged KV** 对齐。完整分期、改哪些层、和 vLLM 官方 [Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 的差异，见 **[docs/vllm-integration.md](docs/vllm-integration.md)**。

**本仓库开发接口：** `pip install 'turboquant-kv[vllm]'` 后可用 `turboquant.integrations.vllm.register_turboquant_dev_backend()` 向 vLLM 0.18 注册 **CUSTOM** 开发 backend（当前委托 CPU 注意力，用于接线验证）；细节见 [docs/vllm-integration.md](docs/vllm-integration.md) §8。**SGLang**：`pip install 'turboquant-kv[sglang]'` 与 `turboquant.integrations.sglang.register_turboquant_dev_sglang_backend()`，启动时 `--attention-backend turboquant_dev`，见 [docs/sglang-integration.md](docs/sglang-integration.md)。

---

## NumPy, JAX, and this package

**There is no JAX implementation in this repository.** The C++ codec returns **NumPy arrays**; you can use it from a JAX workflow by moving data to the host and wrapping back.

**Feeding `KeyCodec` from JAX**

- Convert to host **NumPy**, **C-contiguous `float32`**, shape `(N, head_dim)`:

  ```python
  import numpy as np
  import jax

  x_np = np.asarray(jax.device_get(x_jax), dtype=np.float32, order="C")
  ```

- Then `codec.compress_flat(x_np)` and `codec.attention_scores_flat(q_np, compressed)` as usual.

**NumPy version vs JAX**

- This package declares **`numpy>=1.22`** in `pyproject.toml`. In an environment that also installs **JAX**, the resolver will satisfy **both** packages’ constraints.
- **Current `jax` on PyPI** (check [jax on PyPI](https://pypi.org/project/jax/)) may require **`numpy>=2.0`** — in that case your env will get NumPy 2.x automatically.
- **Older JAX** (e.g. **0.4.35**) used **`numpy>=1.24`**, and **`numpy>=1.26`** on **Python ≥ 3.12** (see that release’s metadata on PyPI).
- JAX’s support policy (Python / NumPy ranges) is documented here: [Python and NumPy version support policy](https://docs.jax.dev/en/latest/deprecation.html).

**Practical rule:** pick your **JAX (and jaxlib) version first**, then install `turboquant-kv`; let pip/uv resolve **NumPy** to a version compatible with both. If you need a specific NumPy major line, pin **`jax`** / **`jaxlib`** accordingly.

---

## Public API

| Symbol | Role |
|--------|------|
| **`CompressionConfig`** | `head_dim`, `bits`, `seed`, `device` (device used by PyTorch backend only). |
| **`KeyCodec`** | `compress_flat` / `compress_bhsd` → **`CompressedKeys`**; `attention_scores_flat` / `attention_scores_bhsd`. `backend="auto"`: **native** if `_native` exists, else **torch**. |
| **`ValueCodec`** | MSE path: `compress_*` / `decompress_*`. |
| **`compressed_keys_to_torch_dict`** | Bridge to `TurboQuantCompressorV2.asymmetric_attention_scores`. |
| **`estimate_kv_storage`** | KV bit budget vs FP16 (same style as `validate.py`). |
| **`integration_hooks`**, **`integration_readme`**, **`dependency_guide`** | Integration checklist and prose. |
| **`metrics.*`** | Cosine / RMSE / top-k agreement on score tensors. |
| **`diagnose_attention_bhsd`**, **`quick_smoke_key_codec`** | Quality checks vs full-precision attention. |

Reference classes (`TurboQuantMSE`, …) are **lazy-imported** on first access so `import turboquant` does not load PyTorch until needed.

**Example (PyTorch backend, GPU)**

```python
import torch
from turboquant import CompressionConfig, KeyCodec, diagnose_attention_bhsd

cfg = CompressionConfig(head_dim=128, bits=3, device="cuda")
codec = KeyCodec(cfg)
keys = torch.randn(1, 8, 1024, 128, device="cuda")
q = keys[:, :, -1:, :]
ck = codec.compress_bhsd(keys)
scores = codec.attention_scores_bhsd(q, ck)
diag = diagnose_attention_bhsd(q, keys, codec)
print(diag.mean_cosine_similarity, diag.mean_top1_match)
```

**Example (native + NumPy only)**

```python
import numpy as np
from turboquant import CompressionConfig, KeyCodec

cfg = CompressionConfig(head_dim=128, bits=3)
codec = KeyCodec(cfg, backend="native")
keys = np.random.randn(256, 128).astype(np.float32)
ck = codec.compress_flat(keys)
q = np.random.randn(1, 128).astype(np.float32)
scores = codec.attention_scores_flat(q, ck)  # (1, 256)
```

---

## Where it fits in inference

1. **Prefill / decode:** when K and V are produced, store **compressed** representations instead of dense FP16/BF16 (see `KeyCodec` / `ValueCodec`).
2. **Attention:** form QK logits with **`attention_scores_*`** on compressed K; avoid full dequant + matmul unless you choose to.
3. **Values:** decompress MSE values for `softmax @ V`, or fuse in a custom kernel later.

Structured checklist: `turboquant.integration.integration_hooks()`.

---

## C++ library

Location: **`cpp/`**. Eigen is fetched at configure time (no system Eigen required). Optional **OpenMP** for batch compression.

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBOQUANT_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build
```

API overview: **`cpp/README.md`**. Pybind module **`turboquant._native`** is built when installing the Python package (scikit-build-core + CMake).

**Note:** C++ RNG is **not** bit-identical to PyTorch for the same integer seed.

---

## Algorithm (short)

1. **Stage 1 (MSE):** random orthogonal rotation, then per-coordinate **Lloyd–Max** quantization (coordinate distribution ~ Gaussian in high `d`).
2. **Stage 2 (QJL):** store **signs** of random Gaussian projections of the **residual** so estimated **inner products** (attention scores) are **unbiased** despite aggressive quantization.

Estimator (conceptually):

```text
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>
```

High per-vector error is acceptable if **score distribution** over keys is preserved — that is what QJL targets.

---

## Reported results

### Synthetic (`turboquant-verify`)

**MSE** (d=128, 1000 random unit vectors):

| Bits | Measured MSE | Paper bound | Ratio |
|------|-------------|-------------|-------|
| 1 | 0.362 | 0.680 | 0.53x |
| 2 | 0.116 | 0.170 | 0.68x |
| 3 | 0.034 | 0.043 | 0.81x |
| 4 | 0.009 | 0.011 | 0.87x |

**Inner products** (d=128, 2000 pairs): near-zero bias; correlation with true IP up to **~0.98** at 4-bit. **Needle-in-haystack:** 9/9 exact in tested settings.

### Real model (`turboquant-validate`, Qwen2.5-3B-Instruct, RTX 3060)

**Compression (8K context, illustrative):**

| Config | KV size | vs FP16 |
|--------|---------|---------|
| FP16 | ~289 MB | 1.0x |
| TQ 4-bit | ~76 MB | ~3.8x |
| TQ 3-bit | ~58 MB | ~5.0x |
| TQ 2-bit | ~40 MB | ~7.3x |

**Attention fidelity (excerpt):** cosine similarity of score vectors often **>0.99** at 3–4 bit; top-1 token match per head varies (see full tables in earlier project notes / paper comparison).

---

## CLI

| Command | Needs | Description |
|---------|--------|-------------|
| `turboquant-verify` | `[torch]` | Synthetic checks (Lloyd–Max, MSE, QJL, KV ratios, needle, optional GPU timing). |
| `turboquant-validate` | `[validate]`, CUDA | Qwen KV capture + score comparison. |

Also: `python -m turboquant.cli`, `python -m turboquant.validate`.

---

## Repository layout

```text
pyproject.toml          # scikit-build-core, optional extras, wheel.packages = turboquant
uv.lock                 # uv lock (uv lock)
turboquant/
  __init__.py           # Lazy imports for torch stack; NATIVE_AVAILABLE
  api.py                # KeyCodec, ValueCodec (numpy-first, lazy torch)
  types.py              # CompressionConfig, CompressedKeys, …
  metrics.py            # Score metrics ([torch])
  validation.py         # diagnose_attention_bhsd, … ([torch])
  integration.py        # Hooks, KV estimates, dependency_guide
  dependencies.py       # GUIDE_MARKDOWN (framework / PyTorch notes)
  torch_impl.py         # Reference modules ([torch])
  compressors.py
  lloyd_max.py
  synthetic_checks.py
  validate.py
  cli.py
tests/                  # pytest
docs/                   # vLLM / SGLang integration + tutorial
  diagrams/             # draw.io 架构图（.drawio）
examples/               # vLLM / SGLang dev launcher scripts
cpp/                    # Eigen, libturboquant, pybind _native, tests
requirements.txt        # Notes; prefer pyproject/uv
```

---

## Requirements & CUDA PyTorch

- **Python:** 3.10+
- **Default:** `numpy>=1.22`
- **Optional `[torch]`:** `torch`, `scipy`
- **Optional `[validate]`:** `transformers`, `accelerate`, `bitsandbytes` (+ torch/scipy as listed)

Install CUDA-enabled PyTorch from the official index, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## References & license

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [QJL](https://arxiv.org/abs/2406.03482)
- [PolarQuant](https://arxiv.org/abs/2502.02617)
- [QJL reference (CUDA)](https://github.com/amirzandieh/QJL)
- [PolarQuant reference](https://github.com/ericshwu/PolarQuant)

**License:** MIT
