# Tutorial: vLLM & SGLang dev integration (`turboquant-kv`)

This walkthrough covers the **Phase B wiring** shipped in this repo: registering a named attention backend and aligning KV tensor layouts with `KeyCodec`. It does **not** turn on production TurboQuant KV (that needs GPU kernels + engine fork work — see [vLLM integration](./vllm-integration.md) and [SGLang integration](./sglang-integration.md)).

---

## What you will do

1. Install optional extras `[vllm]` and/or `[sglang]`.
2. Run **launcher scripts** that register the dev backend **in-process**, then start the engine CLI.
3. (Optional) Use **shared layout helpers** with `KeyCodec` on synthetic tensors.

---

## Prerequisites

| Piece | vLLM | SGLang |
|--------|------|--------|
| Python | 3.10+ (match engine) | 3.10+ (match engine) |
| Hardware | Linux + NVIDIA CUDA typical for `vllm` wheels | Per [SGLang docs](https://github.com/sgl-project/sglang) |
| This package | From PyPI or `pip install -e .` | Same |

Install:

```bash
pip install 'turboquant-kv[torch]'    # KeyCodec torch path (SciPy Lloyd–Max)
pip install 'turboquant-kv[vllm]'     # only if you run vLLM
pip install 'turboquant-kv[sglang]'   # only if you run SGLang
```

---

## Part 1 — Shared: `KeyCodec` + token layout

`RadixAttention` / vLLM token paths ultimately give you key rows shaped like **`[num_tokens, num_kv_heads, head_dim]`** (float). To call `KeyCodec.compress_bhsd`, reshape to **`(1, num_kv_heads, num_tokens, head_dim)`**:

```python
import torch
from turboquant import CompressionConfig, KeyCodec
from turboquant.integrations.token_layout import tokens_thd_from_key_rows

cfg = CompressionConfig(head_dim=64, bits=3, device="cpu")
codec = KeyCodec(cfg, backend="torch")

# Synthetic [T, H, D] as if one layer's K rows for T new tokens
T, H, D = 8, 4, 64
k_rows = torch.randn(T, H, D, dtype=torch.float32)

bhsd = tokens_thd_from_key_rows(k_rows)
ck = codec.compress_bhsd(bhsd)
print("compressed keys:", ck.n_keys, "head_dim=", ck.head_dim)
```

**vLLM paged CPU layout** (5D) is documented in `turboquant.integrations.vllm.kv_bridge` — use `split_kv_cache` when you have `[2, blocks, kv_heads, block_size, head_size]`.

---

## Part 2 — vLLM: register `CUSTOM` then serve

### Important

`register_turboquant_dev_backend()` mutates vLLM’s **in-process** registry. It must run in the **same Python process** that later builds the engine. Spawning `subprocess.run(["vllm", "serve", ...])` after registering in the parent **does not** work.

### Option A — Launcher script (recommended)

From the repo root (or anywhere on `PYTHONPATH` with `turboquant` installed):

```bash
python examples/vllm_dev_launcher.py serve YOUR_MODEL \
  --attention-backend CUSTOM \
  # ... other vllm serve flags
```

The script calls `register_turboquant_dev_backend()` then `vllm.entrypoints.cli.main.main()`.

### Option B — One-liner check (no model)

Verify registration without loading weights:

```bash
python -m turboquant.integrations.vllm
# Prints the fully-qualified backend class path
```

### Expected behavior

- Backend name reported by vLLM should correspond to **`TURBOQUANT_DEV`** (delegates to CPU attention implementation).
- You still need a valid model and CUDA (or CPU) setup per vLLM; this repo does not bundle models.

### Deep dive

File-level map and version pin: [vllm-integration.md](./vllm-integration.md) (§7–§8).

---

## Part 3 — SGLang: register `turboquant_dev` then launch

### Important

Same rule: register **before** `ModelRunner` / server startup in the **same process**.

### Option A — Launcher script

```bash
python examples/sglang_dev_launcher.py \
  --model-path YOUR_MODEL \
  --attention-backend turboquant_dev \
  # ... other sglang flags (see sglang launch_server / docs)
```

### Option B — Print backend name

```bash
python -m turboquant.integrations.sglang
```

### Deep dive

[sglang-integration.md](./sglang-integration.md).

---

## Part 4 — Tests you can run locally

Without installing engines:

```bash
pytest tests/test_token_layout.py tests/test_kv_bridge.py -q
```

With `vllm` installed:

```bash
pytest tests/test_vllm_dev_backend_optional.py -q
```

With `sglang` installed:

```bash
pytest tests/test_sglang_dev_backend_optional.py -q
```

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `Unknown attention backend` / invalid backend string | Registration never ran in-process, or wrong flag spelling (`CUSTOM` vs `turboquant_dev`). |
| `vllm` / `sglang` import error | Extra not installed or unsupported platform (e.g. macOS for many `vllm` wheels). |
| `KeyCodec` / SciPy errors | Install `[torch]` so Lloyd–Max codebook setup works. |

---

## Next steps (outside this tutorial)

1. **Fork** vLLM or SGLang and replace the dev backend’s `forward` / `token_to_kv_pool` path with TurboQuant compress + asymmetric scores + V decode/fuse.
2. Use `turboquant.metrics` / `turboquant.validation` on captured Q/K tensors (Phase A).
3. Target **GPU kernels** for real speed (Phase C).

---

## Architecture diagram (draw.io)

Editable diagram (open in [diagrams.net](https://app.diagrams.net) or VS Code Draw.io extension):

- [docs/diagrams/turboquant-kv-architecture.drawio](./diagrams/turboquant-kv-architecture.drawio)

---

## Reference links

- [vLLM v0.18.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.18.0)
- [SGLang releases](https://github.com/sgl-project/sglang/releases)
- TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
