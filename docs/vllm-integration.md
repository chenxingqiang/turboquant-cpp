# TurboQuant × vLLM：完整适配与加速路线

姊妹篇：**[SGLang](./sglang-integration.md)**（radix / ``attention_registry``）· **Step-by-step：[英文教程](./tutorial-inference-integrations.md)**（含 `examples/` 启动脚本）。

本文说明：**为什么不能**把 TurboQuant 当成「另一种 `kv_cache_dtype`」直接打开，以及要在 vLLM 里做到**生产级加速**需要改哪些层、分几期完成。

**版本锚点：** 下文「源码路径」以 **[v0.18.0 发行说明](https://github.com/vllm-project/vllm/releases/tag/v0.18.0)** 对应的源码树为准（tag `v0.18.0`，commit `bcf2be9` 起）；与当前 `main` 上 **V1 engine**（`vllm/v1/`）结构一致，后续 commit 可能增删文件。文档内 GitHub 链接默认指向 `main`；要严格可复现时把 URL 中的 `main` 换成 `v0.18.0`。

与 **KV / 注意力集成**相关的 v0.18.0 背景（摘自该 release，便于你对照官方演进）：

- **FlashInfer 0.6.6**（依赖升级，多数 CUDA 路径的注意力核与此相关）。
- **`--attention-backend auto`**：backend 选择逻辑在集成自定义 TurboQuant backend 时需要一并理解。
- **KV cache offloading / FlexKV / 多 KV group**：改的是 **块复用与卸载**，不是 TurboQuant 数学，但和 **§7.4** 的 KV 管理代码同一圈层，联调时注意别混用假设。
- **Known issues**：Qwen3.5 + **FP8 KV cache** 在 B200 上精度退化（[#37618](https://github.com/vllm-project/vllm/issues/37618)，见 [v0.18.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.18.0)）；调试「量化 KV」类路径时建议读官方说明，避免与 TurboQuant 实验结论混淆。

---

## 1. 先对齐预期：vLLM 现有 KV 路径是什么

vLLM 当前成熟的 KV 优化主要是：

- **PagedAttention / FlashAttention / FlashInfer** 等：**整块连续或分页的 K/V 张量**，在 GPU 上以固定 dtype（FP16/BF16/FP8 等）参与 **融合注意力核**。
- **量化 KV**（见 [Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)）：本质是 **可反量化的 K/V**（如 FP8 + scale），注意力仍在「量化域或反量化后」与 **标准 QKᵀ** 语义对齐。
- **`BaseKVCacheMethod`**（[`kv_cache.py`](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/kv_cache/)）：在 **写入/读出 cache 时**做 quant/dequant，仍服务于上述注意力形式。

**TurboQuant 的关键差异**：

- **K 的目标不是「存成可解压的稠密 K」**，而是用 **压缩表示 + 非对称估计**直接近似 **⟨Q, K⟩**（再加 QJL 项）。
- 即：注意力 logits **不是**「先还原 K 再 `Q @ Kᵀ`」这一条路（论文也强调全解压再算会坏掉分布）。

因此：**无法在不改注意力数学路径的前提下，仅靠换一个 `kv_cache_dtype` 完成 TurboQuant**；必须在 vLLM 的 **Attention 前向**里增加 **TurboQuant 分支**（或新 backend）。

---

## 2. 在 vLLM 里「完整适配」要接的三件事

| 环节 | vLLM 里大致位置 | TurboQuant 要做什么 |
|------|-----------------|---------------------|
| **KV 写入** | Prefill/Decode 产生 K、V 之后，写入 KV cache 前 | 对 **K** 做 TurboQuant 压缩（本仓库 `KeyCodec` / `_native`）；对 **V** 做 MSE 路径（`ValueCodec`）或后续再融合 |
| **QK  logits** | `Attention.forward` / 具体 backend（FlashInfer 等） | 用 **非对称分数** `attention_scores_*` 替代「稠密 K 上的 matmul」；形状需与 **分页/块表**一致 |
| **Softmax @ V** | 同一注意力内核或后续 GEMM | **V** 需 **解压**（或未来做「量化 V + 融合 GEMM」）；与现有 **paged V** 读路径对接 |

**GQA / MQA**：TurboQuant 按 **head_dim × num_kv_heads** 与论文一致；vLLM 里 K/V 的 **物理布局**（按 head 交错、按页对齐）必须与压缩批次的 **reshape 规则**一一对应，否则要在层里做 **permute/视图** 约定。

---

## 3. 推荐分期（从可验证到真加速）

### Phase A — 正确性 / 集成验证（几乎不加速）

- **在 vLLM 外**或对 **单 layer hook**：用本仓库 **`diagnose_attention_bhsd`**、**`metrics`** 在 **真实 Q/K 张量**上与 FP 对齐。
- 目的：确认 **head 布局、RoPE 后 K、dtype** 与你们假设一致（很多坑在 **RoPE 作用在 K 之后** 才存 cache）。

### Phase B — Python/C++ 扩展接入 vLLM（CPU 或小 kernel）

- 在 **Attention 子类**或 **自定义 backend** 分支中：
  - 写 K 时调用 **`turboquant._native`** 或 **Torch 参考**（仅调试）。
  - 算分时调用 **`attention_scores_bhsd`** 等价逻辑（可先 PyTorch 实现再对拍）。
- **瓶颈**：Python + CPU 拷贝会极慢，仅用于 **功能打通**。

### Phase C — GPU 上可实用的 TurboQuant（真正「加速」）

要做的事通常包括：

1. **CUDA / Triton kernel**（或 CUTLASS）实现：
   - 批量 **compress K**（旋转 + Lloyd–Max + QJL sign）；
   - 批量 **asymmetric QK**（与分页索引配合：只对「当前 query × 本 sequence 有效 key slots」算分）。
2. **与 PagedAttention 对齐**：
   - 每个 **block/page** 存 **压缩槽位**（固定 `page_size` 个 token 的压缩 payload），**block table** 逻辑与现有一致，只是 **slot 内容从 FP16 K 变为 TurboQuant blob**。
3. **V 路径**：MSE 解压 kernel 或 **融合 `softmax(scores) @ V_dequant`**。

没有 Phase C，**显存可能省**，但 **吞吐往往不如** 现网 FlashAttention；「完整适配加速」一般指 **做到 Phase C**。

### Phase D — 与 vLLM 上游合流（可选）

- 以 **`BaseKVCacheMethod` 扩展**或 **`kv_cache_dtype="turboquant"`** 等形式提交设计文档 + PR；
- 需满足 vLLM 的 **多后端**（CUDA/ROCm）、**分布式**、**speculative decoding** 等测试矩阵。

---

## 4. 与本仓库的衔接方式

| 资产 | 在 vLLM 中的用途 |
|------|------------------|
| **`cpp/` + `_native`** | 作为 **参考实现 / CPU 金标准**；生产 GPU 路径建议 **重写 kernel**，接口可对齐 `KeyCompressor::compress` / `attention_scores_packed` |
| **`turboquant.api.KeyCodec`** | 原型与 **单测对拍**；线上应替换为 **设备端实现** |
| **`integration_hooks()`** | 与 vLLM 的 phase 映射：prefill/decode 写 KV、attention 读 K |

**随机矩阵**：C++ 与 PyTorch **同 seed 不对齐**；vLLM 集成时应 **统一** `Pi`、`S` 的生成与持久化（按 layer/按 rank），与 **模型 checkpoint 或 config** 绑定，避免多进程不一致。

---

## 5. 风险与测试清单

- **RoPE / 位置编码**：K 必须是 **进入 cache 前、与框架一致的最终 K**。
- **FP8 主模型权重 + TurboQuant KV**：互不冲突，但调试时注意 **注意力里 dtype 提升**（scores 常 FP32）。
- **长上下文**：验证 **分页边界**（page 内 partial slot）不写越界、不混 batch。
- **回归**：同一 prompt 下 **FP16 KV vs TurboQuant KV** 的 **per-layer cosine / top-1**（可用本仓库 `diagnose_attention_bhsd` 思路在 CI 子集跑）。

---

## 6. 小结

- **完整适配 vLLM** = 在 **vLLM 的 Attention + KV 分页**里实现 **TurboQuant 写路径 + 非对称 QK + V 解压/融合**，而不是只改配置项。
- **真正加速**依赖 **GPU 核** 与 **Paged KV 布局** 的深度结合；本仓库提供 **算法参考、Python API、C++ 原型**，**不能**单独替代 vLLM 内部 kernel。
- 建议路线：**Phase A 对拍 → Phase B 分支接入 → Phase C CUDA/Triton → Phase D 上游化**。

---

## 7. 最新 vLLM（v0.18.x / `main`）文件级锚点（V1）

集成时优先读 **V1** 树；标准 decoder MHA/GQA 与 **MLA**（DeepSeek 类）是不同 KV 形态，TurboQuant 若只覆盖前者，需在 selector/backend 层显式区分。

### 7.1 模型侧 Attention 层

- [vllm/model_executor/layers/attention/attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/attention.py) — 常见 MHA/GQA 注意力入口，向下挂 V1 backend。
- [vllm/model_executor/layers/attention_layer_base.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention_layer_base.py) — 共享基类逻辑。
- [vllm/model_executor/layers/attention/mla_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py) — MLA 专用大块实现；**非**标准 K/V TurboQuant 默认可插入点。

### 7.2 V1 backend 编排与注册

- [vllm/v1/attention/backend.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py) — V1 attention backend 主逻辑。
- [vllm/v1/attention/selector.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py) — 按环境选择 backend。
- [vllm/v1/attention/backends/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) — backend 注册表。
- 常见实现（按你目标硬件只深挖一条即可）：[flashinfer.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)、[flash_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)、[triton_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py)、[cpu_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/cpu_attn.py)。
- **参考（非 TurboQuant）**：[flash_attn_diffkv.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn_diffkv.py) — Q 与 K/V dtype 不一致路径，可借鉴「非对称存储」工程接法，数学仍为标准 QK。

### 7.3 V1 ops：分页、写 cache、prefill / decode

Phase C 通常要对齐或替换这些 **kernel 边界**（Triton/CUDA/FlashInfer 调用处）：

- [vllm/v1/attention/ops/paged_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/paged_attn.py)
- [vllm/v1/attention/ops/chunked_prefill_paged_decode.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py)
- [vllm/v1/attention/ops/triton_reshape_and_cache_flash.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py) — **reshape 并写入分页 KV** 的关键链路之一。
- [vllm/v1/attention/ops/triton_decode_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py)
- [vllm/v1/attention/ops/triton_prefill_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py)
- [vllm/v1/attention/ops/prefix_prefill.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/prefix_prefill.py)
- [vllm/v1/attention/ops/triton_unified_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py)

### 7.4 V1 KV 核心：块池、槽位、dtype 与预算

把「每 page 存稠密 K」改成「存 TurboQuant blob」时，除 ops 外要对齐：

- [vllm/v1/core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)
- [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [vllm/v1/core/single_type_kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py)
- [vllm/v1/core/kv_cache_coordinator.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py)
- [vllm/v1/core/block_pool.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)

### 7.5 官方 Quantized KV（对标生命周期，不是数学捷径）

- [vllm/model_executor/layers/quantization/kv_cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py) — `BaseKVCacheMethod` 等；适合学 **何时 quant/dequant**，TurboQuant 仍需独立 attention 分支。

### 7.6 分布式 / KV 传输（多卡、P/D 等）

- [vllm/v1/worker/kv_connector_model_runner_mixin.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/kv_connector_model_runner_mixin.py)
- [vllm/model_executor/layers/attention/kv_transfer_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/kv_transfer_utils.py)

---

## 8. 本仓库已落地的开发接口（Phase B 接线）

以下 **不替代** Phase C 的 GPU kernel，但可在 vLLM 进程里先做 **CUSTOM backend 注册** 与 **KV 张量形状对齐**。

### 8.1 依赖

```bash
pip install 'turboquant-kv[vllm]'   # 约束 vllm>=0.18,<0.19；多数环境为 Linux + CUDA
```

### 8.2 注册开发用 CUSTOM backend（当前 = CPU 注意力语义）

在 **创建 vLLM engine / worker 之前** 调用：

```python
from turboquant.integrations.vllm import register_turboquant_dev_backend
register_turboquant_dev_backend()
```

启动推理时使用 **`--attention-backend CUSTOM`**。注册的类名为 **`TURBOQUANT_DEV`**，实现上 **委托** v0.18 的 `CPUAttentionBackendImpl`（稠密 KV），用于验证注册链路与 CI；下一步应把 `TurboQuantDevAttentionBackendImpl.forward` 换成 TurboQuant 写 cache + 非对称 QK（需 fork vLLM 或上游接受补丁）。

也可在命令行侧先执行一次注册（同一进程内需再启动 vLLM）：

```bash
python -m turboquant.integrations.vllm
```

### 8.3 KV 布局桥接（不依赖 vLLM import）

`turboquant.integrations.vllm.kv_bridge` 提供与 **CPU backend** 一致的 5D cache 校验与 `K/V` 拆分，以及把 token 主序的 `[T, H, D]` 转成 `KeyCodec.compress_bhsd` 所需的 `(1, H, T, D)`。
