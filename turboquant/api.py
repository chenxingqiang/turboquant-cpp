"""
Stable, service-oriented API for TurboQuant KV compression.

Use :class:`KeyCodec` / :class:`ValueCodec` in inference integrations; use
:mod:`turboquant.metrics` and :mod:`turboquant.validation` for quality checks.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Literal, Optional, Union, cast

import numpy as np

from . import NATIVE_AVAILABLE, native as _native_mod
from .types import BackendName, CompressedKeys, CompressedValues, CompressionConfig

if TYPE_CHECKING:
    import torch

    from .compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2

__all__ = [
    "KeyCodec",
    "ValueCodec",
    "compressed_keys_to_torch_dict",
    "default_backend",
    "resolve_backend",
]


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _import_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required. Install: pip install 'turboquant-kv[torch]'"
        ) from e
    return torch


def default_backend() -> BackendName:
    if NATIVE_AVAILABLE:
        return "native"
    if _torch_installed():
        return "torch"
    raise RuntimeError(
        "Neither turboquant._native nor PyTorch is available. "
        "Install: pip install 'turboquant-kv[torch]' or build the C++ extension."
    )


def resolve_backend(requested: Literal["auto", "native", "torch"]) -> BackendName:
    if requested == "auto":
        return default_backend()
    if requested == "native" and not NATIVE_AVAILABLE:
        raise RuntimeError(
            "backend='native' requested but turboquant._native is not built. "
            "Install from source with a C++ toolchain or use backend='torch'."
        )
    if requested == "torch" and not _torch_installed():
        raise RuntimeError(
            "backend='torch' requires PyTorch. Install: pip install 'turboquant-kv[torch]'"
        )
    return cast(BackendName, requested)


def compressed_keys_to_torch_dict(ck: CompressedKeys, device: str) -> dict:
    """Rebuild the dict expected by :meth:`TurboQuantCompressorV2.asymmetric_attention_scores`."""
    torch = _import_torch()
    D = ck.head_dim
    if ck.original_shape_bhsd is None:
        N = ck.n_keys
        B, H, S = 1, 1, N
    else:
        B, H, S, D_ = ck.original_shape_bhsd
        if D_ != D:
            raise ValueError("head_dim mismatch vs original_shape_bhsd")
        N = B * H * S
    if ck.n_keys != N:
        raise ValueError("n_keys inconsistent with original_shape_bhsd")

    km = np.asarray(ck.k_mse, dtype=np.float32).reshape(N, D)
    sg = np.asarray(ck.signs, dtype=np.int8).reshape(N, D)
    rn = np.asarray(ck.residual_norm, dtype=np.float32).reshape(N)

    return {
        "k_mse": torch.from_numpy(km).half().reshape(B, H, S, D).to(device),
        "qjl_signs": torch.from_numpy(sg).reshape(B, H, S, D).float().to(device),
        "residual_norm": torch.from_numpy(rn).half().reshape(B, H, S).to(device),
        "shape": (B, H, S, D),
    }


class KeyCodec:
    """
    Compress decoder key vectors and estimate attention logits without decompressing full keys.

    **Backends**

    - ``native``: C++/Eigen implementation (``turboquant._native.KeyCompressor``). Fast for batch CPU.
    - ``torch``: :class:`TurboQuantCompressorV2` (same math family as ``validate.py``).

    Native and PyTorch RNGs differ; do not expect bitwise-identical matrices for the same ``seed``.
    """

    def __init__(
        self,
        config: CompressionConfig,
        *,
        backend: Literal["auto", "native", "torch"] = "auto",
    ) -> None:
        self.config = config
        self.backend: BackendName = resolve_backend(backend)
        self._device = config.device

        if self.backend == "native":
            assert _native_mod is not None
            self._native = _native_mod.KeyCompressor(
                config.head_dim,
                config.bits,
                int(config.seed) & 0xFFFFFFFFFFFFFFFF,
                (int(config.seed) + 10000) & 0xFFFFFFFFFFFFFFFF,
            )
            self._torch_v2: Optional["TurboQuantCompressorV2"] = None
        else:
            self._native = None
            from .compressors import TurboQuantCompressorV2

            self._torch_v2 = TurboQuantCompressorV2(
                config.head_dim,
                config.bits,
                seed=config.seed,
                device=config.device,
            )

    def compress_flat(
        self,
        keys: Union[np.ndarray, "torch.Tensor"],
        *,
        dtype_out: np.dtype = np.float32,
    ) -> CompressedKeys:
        """
        Compress key rows of shape ``(N, head_dim)``.

        Accepts ``float32`` NumPy or PyTorch tensor (any device; copied to CPU for native).
        """
        D = self.config.head_dim
        if hasattr(keys, "detach") and type(keys).__module__.startswith("torch"):
            torch = _import_torch()
            if isinstance(keys, torch.Tensor):
                x = keys.detach().float().cpu().numpy()
            else:
                x = np.asarray(keys, dtype=np.float32)
        else:
            x = np.asarray(keys, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != D:
            raise ValueError(f"keys must be (N, {D}), got {x.shape}")
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        if self.backend == "native":
            assert self._native is not None
            out = self._native.compress(x)
            km = np.asarray(out["k_mse"], dtype=np.float32)
            sg = np.asarray(out["signs"], dtype=np.int8)
            rn = np.asarray(out["residual_norm"], dtype=np.float32)
        else:
            assert self._torch_v2 is not None
            torch = _import_torch()
            t = torch.from_numpy(x).to(self._device)
            t4 = t.view(1, 1, -1, D)
            d = self._torch_v2.compress(t4)
            km = d["k_mse"].float().cpu().numpy().reshape(-1, D).astype(dtype_out)
            sg = d["qjl_signs"].cpu().numpy().reshape(-1, D).astype(np.int8)
            rn = d["residual_norm"].float().cpu().numpy().reshape(-1).astype(np.float32)

        return CompressedKeys(
            k_mse=km,
            signs=sg,
            residual_norm=rn,
            head_dim=D,
            bits=self.config.bits,
            backend=self.backend,
            original_shape_bhsd=None,
        )

    def compress_bhsd(self, keys: "torch.Tensor") -> CompressedKeys:
        """
        Compress ``keys`` of shape ``(B, H, S, head_dim)`` (typical KV layout).
        """
        _import_torch()
        if keys.dim() != 4:
            raise ValueError(f"expected (B,H,S,D), got {tuple(keys.shape)}")
        B, H, S, D = keys.shape
        if D != self.config.head_dim:
            raise ValueError(f"head_dim mismatch: {D} vs config {self.config.head_dim}")
        flat = self.compress_flat(keys.reshape(-1, D))
        return CompressedKeys(
            k_mse=flat.k_mse,
            signs=flat.signs,
            residual_norm=flat.residual_norm,
            head_dim=flat.head_dim,
            bits=flat.bits,
            backend=flat.backend,
            original_shape_bhsd=(B, H, S, D),
        )

    def attention_scores_flat(
        self,
        queries: np.ndarray,
        compressed: CompressedKeys,
    ) -> np.ndarray:
        """
        ``queries`` of shape ``(n_queries, head_dim)`` vs flat compressed keys ``(N, D)``.

        Returns float32 ``(n_queries, N)``.
        """
        D = self.config.head_dim
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != D:
            raise ValueError(f"queries must be (Q, {D}), got {q.shape}")
        if compressed.backend != self.backend:
            raise ValueError("compressed.backend does not match this codec")
        if not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q)

        if self.backend == "native":
            assert self._native is not None
            km = np.asarray(compressed.k_mse, dtype=np.float32)
            sg = np.asarray(compressed.signs, dtype=np.int8)
            rn = np.asarray(compressed.residual_norm, dtype=np.float32)
            scores = self._native.attention_scores(q, km, sg, rn)
            return np.asarray(scores, dtype=np.float32)

        assert self._torch_v2 is not None
        torch = _import_torch()
        Q = q.shape[0]
        N = compressed.n_keys
        qt = torch.from_numpy(q).to(self._device).view(1, 1, Q, D)
        ck_wrapped = CompressedKeys(
            k_mse=compressed.k_mse,
            signs=compressed.signs,
            residual_norm=compressed.residual_norm,
            head_dim=D,
            bits=compressed.bits,
            backend="torch",
            original_shape_bhsd=(1, 1, N, D),
        )
        td = compressed_keys_to_torch_dict(ck_wrapped, self._device)
        s = self._torch_v2.asymmetric_attention_scores(qt, td)
        return s.squeeze(0).squeeze(0).float().cpu().numpy()

    def attention_scores_bhsd(
        self,
        queries: "torch.Tensor",
        compressed: CompressedKeys,
    ) -> "torch.Tensor":
        """
        ``queries`` shape ``(B, H, n_queries, head_dim)`` vs compressed keys whose
        :attr:`CompressedKeys.original_shape_bhsd` matches the same ``B, H`` as key storage.
        """
        if queries.dim() != 4:
            raise ValueError(f"expected (B,H,Sq,D), got {tuple(queries.shape)}")
        B, H, Sq, D = queries.shape
        if D != self.config.head_dim:
            raise ValueError("head_dim mismatch")
        if compressed.original_shape_bhsd is None:
            raise ValueError("compressed keys missing original_shape_bhsd; use compress_bhsd()")
        Bk, Hk, Sk, Dk = compressed.original_shape_bhsd
        if Dk != D or Bk != B or Hk != H:
            raise ValueError(
                f"query layout (B,H)=({B},{H}) incompatible with keys {compressed.original_shape_bhsd}"
            )

        if self.backend == "torch":
            assert self._torch_v2 is not None
            td = compressed_keys_to_torch_dict(compressed, self._device)
            return self._torch_v2.asymmetric_attention_scores(queries.float(), td)

        # Native: flatten queries and keys, then reshape scores
        assert self._native is not None
        torch = _import_torch()
        q_flat = queries.reshape(-1, D).detach().cpu().numpy().astype(np.float32)
        scores_flat = self.attention_scores_flat(q_flat, compressed)
        return torch.from_numpy(scores_flat).to(queries.device).view(B, H, Sq, Sk)


class ValueCodec:
    """MSE-only value path (compress indices + norm, decompress to FP for ``softmax @ V``)."""

    def __init__(
        self,
        config: CompressionConfig,
        *,
        backend: Literal["auto", "native", "torch"] = "auto",
        value_seed_offset: int = 500,
    ) -> None:
        self.config = config
        self.backend: BackendName = resolve_backend(backend)
        self._device = config.device
        self._value_seed = config.seed + value_seed_offset

        if self.backend == "native":
            assert _native_mod is not None
            self._native = _native_mod.ValueCompressor(
                config.head_dim,
                config.bits,
                int(self._value_seed) & 0xFFFFFFFFFFFFFFFF,
            )
            self._torch_mse: Optional["TurboQuantCompressorMSE"] = None
        else:
            self._native = None
            from .compressors import TurboQuantCompressorMSE

            self._torch_mse = TurboQuantCompressorMSE(
                config.head_dim,
                config.bits,
                seed=self._value_seed,
                device=config.device,
            )

    def compress_flat(self, values: Union[np.ndarray, "torch.Tensor"]) -> CompressedValues:
        D = self.config.head_dim
        if hasattr(values, "detach") and type(values).__module__.startswith("torch"):
            torch = _import_torch()
            if isinstance(values, torch.Tensor):
                x = values.detach().float().cpu().numpy()
            else:
                x = np.asarray(values, dtype=np.float32)
        else:
            x = np.asarray(values, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != D:
            raise ValueError(f"values must be (N, {D}), got {x.shape}")
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        N = x.shape[0]

        if self.backend == "native":
            assert self._native is not None
            batch = self._native.compress(x)
            ix = np.asarray(batch["indices"], dtype=np.uint8)
            norms = np.asarray(batch["vec_norm"], dtype=np.float32)
        else:
            assert self._torch_mse is not None
            torch = _import_torch()
            t = torch.from_numpy(x).to(self._device).view(1, 1, N, D)
            d = self._torch_mse.compress(t)
            ix = d["indices"].cpu().numpy().reshape(N, D).astype(np.uint8)
            norms = d["vec_norms"].float().cpu().numpy().reshape(N).astype(np.float32)

        return CompressedValues(
            indices=ix,
            vec_norm=norms,
            head_dim=D,
            bits=self.config.bits,
            backend=self.backend,
            original_shape_bhsd=None,
        )

    def compress_bhsd(self, values: "torch.Tensor") -> CompressedValues:
        _import_torch()
        if values.dim() != 4:
            raise ValueError(f"expected (B,H,S,D), got {tuple(values.shape)}")
        B, H, S, D = values.shape
        if D != self.config.head_dim:
            raise ValueError("head_dim mismatch")
        flat = self.compress_flat(values.reshape(-1, D))
        return CompressedValues(
            indices=flat.indices,
            vec_norm=flat.vec_norm,
            head_dim=D,
            bits=flat.bits,
            backend=flat.backend,
            original_shape_bhsd=(B, H, S, D),
        )

    def decompress_flat(self, compressed: CompressedValues) -> np.ndarray:
        D = self.config.head_dim
        ix = np.asarray(compressed.indices, dtype=np.uint8)
        norms = np.asarray(compressed.vec_norm, dtype=np.float32)
        N = ix.shape[0]
        if ix.shape[1] != D:
            raise ValueError("indices shape mismatch")

        if self.backend == "native":
            assert self._native is not None
            out = self._native.decompress(ix, norms)
            return np.asarray(out, dtype=np.float32)

        assert self._torch_mse is not None
        torch = _import_torch()
        if compressed.original_shape_bhsd is None:
            B, H, S = 1, 1, N
        else:
            B, H, S, _ = compressed.original_shape_bhsd
        td = {
            "indices": torch.from_numpy(ix).to(self._device).reshape(B, H, S, D),
            "vec_norms": torch.from_numpy(norms).half().to(self._device).reshape(B, H, S),
            "shape": (B, H, S, D),
        }
        t = self._torch_mse.decompress(td)
        return t.float().cpu().numpy().reshape(N, D)

    def decompress_bhsd(self, compressed: CompressedValues) -> "torch.Tensor":
        torch = _import_torch()
        arr = self.decompress_flat(compressed)
        if compressed.original_shape_bhsd is None:
            raise ValueError("original_shape_bhsd required")
        B, H, S, D = compressed.original_shape_bhsd
        return torch.from_numpy(arr).to(self._device).reshape(B, H, S, D)
