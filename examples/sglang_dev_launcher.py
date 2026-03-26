#!/usr/bin/env python3
"""
Register ``turboquant_dev`` in SGLang's attention registry, then start the HTTP server.

Pass the same arguments you would pass to ``python -m sglang.launch_server`` (everything
after the module name), e.g.::

    python examples/sglang_dev_launcher.py --model-path /path/to/model --attention-backend turboquant_dev

Requires: ``pip install 'turboquant-kv[sglang]'`` and a SGLang-supported environment.
"""

from __future__ import annotations

import os
import sys

from turboquant.integrations.sglang import register_turboquant_dev_sglang_backend

register_turboquant_dev_sglang_backend()


def main() -> None:
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
