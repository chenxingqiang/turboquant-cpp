"""Register the dev attention backend name (requires SGLang)."""

from __future__ import annotations

from .dev_backend import register_turboquant_dev_sglang_backend

if __name__ == "__main__":
    n = register_turboquant_dev_sglang_backend()
    print(n)
    print("Use: --attention-backend", n)
    print("Call this before starting the SGLang server in the same process, or use a launcher script.")
