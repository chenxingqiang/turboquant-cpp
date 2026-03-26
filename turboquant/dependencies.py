"""
Dependency and inference-framework positioning (documentation for integrators).

This module has **no heavy imports**; safe to read in constrained environments.
"""

from __future__ import annotations

__all__ = ["GUIDE_MARKDOWN", "print_guide"]


GUIDE_MARKDOWN = """
## 是否必须依赖 PyTorch？

**不必。** 功能上分三层：

1. **Codec 服务（推荐给推理集成）**  
   - **C++ 扩展** `turboquant._native` + **NumPy**：压缩 K/V、`attention_scores` 等可在 **无 PyTorch** 下工作（需从源码构建出 `_native`）。  
   - Python 包默认只硬依赖 **NumPy**；安装 **`[torch]`** 额外装 PyTorch + SciPy，用于参考实现与脚本。

2. **参考实现（论文对齐 / 实验）**  
   - `torch_impl`、`compressors`、`lloyd_max`（SciPy 积分 + PyTorch 张量）等依赖 **PyTorch + SciPy**。  
   - 与 C++ 路径 **数值/随机数不保证逐位一致**，对齐实验需约定同一套矩阵或 RNG。

3. **Transformers / 验证脚本**  
   - `transformers`、`accelerate`、`bitsandbytes` 等仅在 **`[validate]`** 中；它们 **不能替代** PyTorch，而是在 PyTorch 上加载模型做 `turboquant-validate`。

## Transformers 能替代 PyTorch 吗？

**不能。** Hugging Face Transformers 的 PyTorch 后端仍以 **torch.Tensor** 与 CUDA 生态为主；它只是高层 API，不是更轻的张量库。

## vLLM / SGLang 能否「直接依赖一套版本」？

**不能靠换一个 pip 依赖自动完成。** 原因简述：

- vLLM、SGLang 是 **完整推理服务栈**（调度、PagedAttention、自定义 CUDA/Triton 等），内部仍以 **PyTorch 生态**为主。  
- TurboQuant 要生效，必须在它们的 **KV 写入**与 **QK 打分**路径上 **显式接入**（类似自定义 attention 或 KV cache 插件），属于 **按框架 PR / 插件** 的集成，而不是本仓库单独再发一个「vLLM 专用 wheel」就能全局生效。

**可行产品形态：**

- **本仓库**：`numpy` + 可选 `[torch]` + 可选 `[validate]`；C++ `_native` 供 **无 torch 服务**或 **被 vLLM/SGLang 的 C++/Python 扩展调用**。  
- **下游**：单独的集成包或 fork（例如 `turboquant-kv` + vLLM patch），在本库 API 之上对接具体引擎的 KV 布局与 kernel。

## 安装建议

- 仅服务 / 仅 C++ 绑定：**`pip install turboquant-kv`**（需构建带 `_native` 的 wheel/sdist）。  
- 论文复现、PyTorch 实验：**`pip install turboquant-kv[torch]`**  
- Qwen 等实机校验：**`pip install turboquant-kv[validate]`**（已包含 torch 依赖链）
""".strip()


def print_guide() -> None:
    print(GUIDE_MARKDOWN)
