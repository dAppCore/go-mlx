#!/usr/bin/env python3
"""
mock_bridge.py — Mock bridge for testing the mlxlm Go backend.

Implements the same JSON Lines protocol as bridge.py but without mlx_lm.
Returns deterministic fake responses for testing.

SPDX-Licence-Identifier: EUPL-1.2
"""

import json
import sys
import os

_loaded = False
_model_path = ""


def _write(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _error(msg):
    _write({"error": str(msg)})


def main():
    global _loaded, _model_path

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _error(f"parse error: {e}")
            continue

        cmd = req.get("cmd", "")

        if cmd == "quit":
            break

        elif cmd == "load":
            path = req.get("path", "")
            if not path:
                _error("load: missing 'path'")
                continue
            # Simulate failure for paths containing "FAIL".
            if "FAIL" in path:
                _error(f"load: cannot open model at {path}")
                continue
            _loaded = True
            _model_path = path
            _write({
                "ok": True,
                "model_type": "mock_model",
                "vocab_size": 32000,
            })

        elif cmd == "generate":
            if not _loaded:
                _error("generate: no model loaded")
                continue

            max_tokens = req.get("max_tokens", 5)
            # Check for error trigger.
            prompt = req.get("prompt", "")
            if "ERROR" in prompt:
                _error("generate: simulated model error")
                continue

            # Emit fixed tokens.
            tokens = ["Hello", " ", "world", "!", "\n"]
            count = min(max_tokens, len(tokens))
            for i in range(count):
                _write({"token": tokens[i], "token_id": 100 + i})
            _write({"done": True, "tokens_generated": count})

        elif cmd == "chat":
            if not _loaded:
                _error("chat: no model loaded")
                continue

            messages = req.get("messages", [])
            max_tokens = req.get("max_tokens", 5)

            # Emit tokens reflecting the last user message.
            tokens = ["I", " ", "heard", " ", "you"]
            count = min(max_tokens, len(tokens))
            for i in range(count):
                _write({"token": tokens[i], "token_id": 200 + i})
            _write({"done": True, "tokens_generated": count})

        elif cmd == "info":
            if not _loaded:
                _error("info: no model loaded")
                continue
            _write({
                "model_type": "mock_model",
                "vocab_size": 32000,
                "layers": 24,
                "hidden_size": 2048,
            })

        elif cmd == "cancel":
            # No-op in mock — real bridge sets a flag.
            pass

        else:
            _error(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
