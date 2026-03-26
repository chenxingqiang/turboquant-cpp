"""Console entry points for ``turboquant-verify``."""

from __future__ import annotations


def verify_cli() -> None:
    """Run synthetic checks (CPU; optional CUDA section if PyTorch sees a GPU)."""
    try:
        from .synthetic_checks import run_all
    except ImportError as e:
        raise SystemExit(
            "turboquant-verify requires the [torch] extra (PyTorch + SciPy). "
            "Install: pip install 'turboquant-kv[torch]'"
        ) from e

    run_all()


if __name__ == "__main__":
    verify_cli()
