from __future__ import annotations


def test_import_package_lightweight() -> None:
    """Top-level import must not require PyTorch (codec API is numpy + optional native)."""
    import turboquant

    assert turboquant.__version__
    assert turboquant.GUIDE_MARKDOWN
    assert turboquant.KeyCodec is not None


def test_torch_reference_lazy() -> None:
    import turboquant

    assert turboquant.TurboQuantMSE is not None


def test_native_optional() -> None:
    import turboquant

    assert isinstance(turboquant.NATIVE_AVAILABLE, bool)
    if turboquant.NATIVE_AVAILABLE:
        kc = turboquant.native.KeyCompressor(8, 3, 1, 2)
        assert kc.head_dim == 8
