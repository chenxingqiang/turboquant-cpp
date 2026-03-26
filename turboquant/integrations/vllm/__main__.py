"""Register the dev backend and print the CUSTOM class path (requires vLLM)."""

from __future__ import annotations

from .dev_backend import register_turboquant_dev_backend

if __name__ == "__main__":
    path = register_turboquant_dev_backend()
    print(path)
    print("Start vLLM with attention backend CUSTOM after this module runs in the same process.")
