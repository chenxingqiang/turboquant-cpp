#!/usr/bin/env python3
"""
Register TurboQuant's vLLM CUSTOM dev backend, then invoke the vLLM CLI in the same process.

Usage (same argv shape as ``vllm`` after the script name)::

    python examples/vllm_dev_launcher.py serve <model> --attention-backend CUSTOM [more flags]

Requires: ``pip install 'turboquant-kv[vllm]'`` and a vLLM-supported environment.
"""

from __future__ import annotations

from turboquant.integrations.vllm import register_turboquant_dev_backend

register_turboquant_dev_backend()


def main() -> None:
    from vllm.entrypoints.cli.main import main as vllm_main

    raise SystemExit(vllm_main())


if __name__ == "__main__":
    main()
