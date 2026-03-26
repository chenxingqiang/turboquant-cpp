# TurboQuant × SGLang：适配与加速路线

本文与 [vLLM 版](./vllm-integration.md) 并列。**实操教程（英文）+ 启动脚本：** [tutorial-inference-integrations.md](./tutorial-inference-integrations.md)、仓库 `examples/`。

下文说明 TurboQuant **不是**换一项 ``kv_cache_dtype`` 就能完成，以及在 **SGLang**（[latest 发行版](https://github.com/sgl-project/sglang/releases/latest) 当前为 **v0.5.9**）里对接时要动哪些层次。

---

## 1. SGLang 里注意力 / KV 怎么走

- **入口层**：[`RadixAttention.forward`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py) 在非 compile 路径下调用 ``forward_batch.attn_backend.forward(...)``。
- **Backend 注册**：[`attention_registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py) 用 ``@register_attention_backend("flashinfer")`` 等形式把 **字符串名字** 映射到 ``create_*(runner)`` 工厂；``ModelRunner`` 按 ``--attention-backend`` 选用。
- **基类**：[`base_attn_backend.AttentionBackend`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py) 约定 ``init_forward_metadata``、``forward_extend``、``forward_decode`` 等。
- **KV 存储**：通过 ``forward_batch.token_to_kv_pool``（radix / token 池）与 ``req_to_token`` 等索引；例如 [`TorchNativeAttnBackend`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/torch_native_backend.py) 使用形状接近 ``[max_tokens, num_kv_heads, head_size]`` 的 K/V buffer。

**TurboQuant 差异**（与 vLLM 小结相同）：K 侧要 **压缩 + 非对称 QK**，不能等价于「存 FP8/FP16 再标准 matmul」；必须在 **backend 的 forward / token_to_kv_pool 布局** 上与 radix 调度一致。

**Radix 缓存**：多请求共享前缀时，KV 在 radix 树中复用。把 K 换成 TurboQuant blob 时，除算子外还要保证 **节点分裂、引用计数、与索引表一致**，工程难度通常 **不低于** vLLM 分页路径。

---

## 2. 本仓库已提供的开发接口（Phase B）

### 2.1 依赖

```bash
pip install 'turboquant-kv[sglang]'   # 见 pyproject 版本下界；需满足 SGLang 官方环境说明
```

### 2.2 注册 ``turboquant_dev`` backend（当前 = TorchNative）

在 **启动 SGLang server 的同一 Python 进程内、创建 ModelRunner 之前**：

```python
from turboquant.integrations.sglang import register_turboquant_dev_sglang_backend
register_turboquant_dev_sglang_backend()
```

启动时使用：

```text
--attention-backend turboquant_dev
```

实现上 **委托** ``TorchNativeAttnBackend``（稠密 KV + SDPA 路径），用于验证 **registry 字符串 → 工厂 → runner** 全链路。真 TurboQuant 需在 **SGLang fork** 里新 backend 类或改 ``token_to_kv_pool`` 与 ``forward_decode``。

命令行侧可先打印注册名：

```bash
python -m turboquant.integrations.sglang
```

### 2.3 Token 行布局与 ``KeyCodec``

``[num_tokens, num_kv_heads, head_dim]`` → ``(1, H, T, D)`` 使用 ``turboquant.integrations.sglang.kv_bridge`` / ``turboquant.integrations.token_layout.tokens_thd_from_key_rows``。

---

## 3. 推荐分期（与 vLLM 文档 Phase A–D 对齐）

| 阶段 | SGLang 侧重 |
|------|-------------|
| **A** | 用本仓库 ``metrics`` / ``validation`` 在真实 Q/K 张量上对拍；确认 RoPE 后 K、GQA 布局与 ``RadixAttention`` 一致。 |
| **B** | 注册自定义 ``attention_backend`` 名；在 ``forward_decode`` 中单层 hook ``KeyCodec``（可能极慢）。 |
| **C** | GPU kernel + 与 **token pool / radix** 一致的压缩槽位；V 路径 MSE 解压或融合。 |
| **D** | 上游化：设计文档 + CI（多硬件、spec decode、disaggregation 等）。 |

---

## 4. 源码锚点（v0.5.9 / ``main``）

- [python/sglang/srt/layers/radix_attention.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py)
- [python/sglang/srt/layers/attention/attention_registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py)
- [python/sglang/srt/layers/attention/base_attn_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py)
- [python/sglang/srt/layers/attention/torch_native_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/torch_native_backend.py)
- [python/sglang/srt/layers/attention/flashinfer_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py)（多数 GPU 默认路径之一）

锁定版本时请把链接中的 ``main`` 换成你的 tag（如 ``v0.5.9``）。
