#!/usr/bin/env python3
"""
bridge.py — JSON Lines bridge between Go subprocess and mlx_lm.

Reads JSON commands from stdin, writes JSON responses to stdout.
Each line is one JSON object. Flushes after every write (critical for streaming).

Commands:
    load     — Load model + tokeniser from path
    generate — Stream tokens for a prompt
    chat     — Stream tokens for a multi-turn conversation
    info     — Return model metadata
    cancel   — Interrupt current generation (no-op outside generation)
    quit     — Exit cleanly

Requires: mlx-lm (pip install mlx-lm)

SPDX-Licence-Identifier: EUPL-1.2
"""

import json
import sys

_model = None
_tokeniser = None
_model_type = None
_vocab_size = 0
_cancelled = False


def _write(obj):
    """Write a JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _error(msg):
    """Write an error response."""
    _write({"error": str(msg)})


def _build_gen_kwargs(req):
    """Build sampler and logits_processors kwargs for stream_generate."""
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    temperature = req.get("temperature", 0.0)
    top_p = req.get("top_p", 0.0)
    top_k = req.get("top_k", 0)
    repeat_penalty = req.get("repeat_penalty", 0.0)

    kwargs = {
        "max_tokens": req.get("max_tokens", 256),
        "sampler": make_sampler(temp=temperature, top_p=top_p, top_k=top_k),
    }

    if repeat_penalty > 1.0:
        kwargs["logits_processors"] = make_logits_processors(
            repetition_penalty=repeat_penalty,
        )

    return kwargs


def handle_load(req):
    global _model, _tokeniser, _model_type, _vocab_size

    path = req.get("path", "")
    if not path:
        _error("load: missing 'path'")
        return

    try:
        import mlx_lm
        _model, _tokeniser = mlx_lm.load(path)
    except Exception as e:
        _error(f"load: {e}")
        return

    # Detect model type from config if available.
    _model_type = getattr(_model, "model_type", "unknown")
    _vocab_size = getattr(_tokeniser, "vocab_size", 0)

    _write({
        "ok": True,
        "model_type": _model_type,
        "vocab_size": _vocab_size,
    })


def handle_generate(req):
    global _cancelled

    if _model is None or _tokeniser is None:
        _error("generate: no model loaded")
        return

    prompt = req.get("prompt", "")
    _cancelled = False

    try:
        import mlx_lm

        kwargs = _build_gen_kwargs(req)

        count = 0
        for response in mlx_lm.stream_generate(
            _model, _tokeniser, prompt=prompt, **kwargs
        ):
            if _cancelled:
                break
            text = response.text if hasattr(response, "text") else str(response)
            token_id = response.token if hasattr(response, "token") else 0
            _write({"token": text, "token_id": int(token_id)})
            count += 1

        _write({"done": True, "tokens_generated": count})

    except Exception as e:
        _error(f"generate: {e}")


def handle_chat(req):
    global _cancelled

    if _model is None or _tokeniser is None:
        _error("chat: no model loaded")
        return

    messages = req.get("messages", [])
    _cancelled = False

    try:
        import mlx_lm

        # Apply chat template via tokeniser.
        if hasattr(_tokeniser, "apply_chat_template"):
            prompt = _tokeniser.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: concatenate messages.
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
            )

        kwargs = _build_gen_kwargs(req)

        count = 0
        for response in mlx_lm.stream_generate(
            _model, _tokeniser, prompt=prompt, **kwargs
        ):
            if _cancelled:
                break
            text = response.text if hasattr(response, "text") else str(response)
            token_id = response.token if hasattr(response, "token") else 0
            _write({"token": text, "token_id": int(token_id)})
            count += 1

        _write({"done": True, "tokens_generated": count})

    except Exception as e:
        _error(f"chat: {e}")


def handle_info(_req):
    if _model is None:
        _error("info: no model loaded")
        return

    num_layers = 0
    hidden_size = 0
    if hasattr(_model, "config"):
        cfg = _model.config
        num_layers = getattr(cfg, "num_hidden_layers", 0)
        hidden_size = getattr(cfg, "hidden_size", 0)

    _write({
        "model_type": _model_type or "unknown",
        "vocab_size": _vocab_size,
        "layers": num_layers,
        "hidden_size": hidden_size,
    })


def handle_cancel(_req):
    global _cancelled
    _cancelled = True


def main():
    handlers = {
        "load": handle_load,
        "generate": handle_generate,
        "chat": handle_chat,
        "info": handle_info,
        "cancel": handle_cancel,
        "quit": None,
    }

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

        handler = handlers.get(cmd)
        if handler is None:
            _error(f"unknown command: {cmd}")
            continue

        handler(req)


if __name__ == "__main__":
    main()
