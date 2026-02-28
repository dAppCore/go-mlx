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
    inspect  — Capture post-RoPE Q and K tensors from all attention layers
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


def handle_inspect(req):
    """Capture post-RoPE Q and K from every attention layer via a single prefill pass."""
    if _model is None or _tokeniser is None:
        _error("inspect: no model loaded")
        return

    prompt = req.get("prompt", "")
    if not prompt:
        _error("inspect: missing 'prompt'")
        return

    try:
        import mlx.core as mx
        import tempfile
        import os

        # Tokenise
        add_special_tokens = (
            _tokeniser.bos_token is None
            or not prompt.startswith(_tokeniser.bos_token)
        )
        tokens = _tokeniser.encode(prompt, add_special_tokens=add_special_tokens)
        input_ids = mx.array([tokens])

        # Find all attention modules in the model.
        # mlx-lm attention modules have q_proj, k_proj, and rope.
        attention_modules = []
        def _find_attention(prefix, mod):
            if hasattr(mod, 'q_proj') and hasattr(mod, 'k_proj') and hasattr(mod, 'rope'):
                attention_modules.append((prefix, mod))
        _model.apply_to_modules(_find_attention)

        if not attention_modules:
            _error("inspect: no attention modules found in model")
            return

        # Storage for captured Q and K per layer.
        captured = {}
        originals = {}

        def _make_hook(layer_idx, attn_mod, original_call):
            def hooked_call(x, mask=None, cache=None):
                B, L, _ = x.shape
                queries = attn_mod.q_proj(x)
                keys = attn_mod.k_proj(x)

                n_heads = attn_mod.n_heads
                n_kv_heads = attn_mod.n_kv_heads
                head_dim = attn_mod.head_dim

                queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

                # Apply norms if present (e.g. gemma3 has q_norm/k_norm).
                if hasattr(attn_mod, 'q_norm'):
                    queries = attn_mod.q_norm(queries)
                if hasattr(attn_mod, 'k_norm'):
                    keys = attn_mod.k_norm(keys)

                # Apply RoPE.
                if cache is not None:
                    queries = attn_mod.rope(queries, offset=cache.offset)
                    keys = attn_mod.rope(keys, offset=cache.offset)
                else:
                    queries = attn_mod.rope(queries)
                    keys = attn_mod.rope(keys)

                # Capture post-RoPE Q and K: [B, heads, L, head_dim]
                captured[layer_idx] = {
                    "queries": queries,
                    "keys": keys,
                }

                # Run original forward pass (avoids recursion, ensures correct output).
                return original_call(x, mask=mask, cache=cache)
            return hooked_call

        # Install hooks.
        for idx, (prefix, attn_mod) in enumerate(attention_modules):
            original = attn_mod.__call__
            originals[idx] = original
            attn_mod.__call__ = _make_hook(idx, attn_mod, original)

        try:
            # Single prefill forward pass.
            cache = _model.make_cache() if hasattr(_model, 'make_cache') else None
            _model(input_ids, cache=cache)
            # Materialise all captured MLX arrays.
            all_arrays = []
            for cap in captured.values():
                all_arrays.append(cap["queries"])
                all_arrays.append(cap["keys"])
            if all_arrays:
                mx.eval(*all_arrays)
        finally:
            # Restore original __call__ methods.
            for idx, (prefix, attn_mod) in enumerate(attention_modules):
                if idx in originals:
                    attn_mod.__call__ = originals[idx]

        if not captured:
            _error("inspect: no attention data captured")
            return

        # Write binary files to temp dir.
        out_dir = tempfile.mkdtemp(prefix="mlxlm-inspect-")

        first = captured[0]
        num_q_heads = first["queries"].shape[1]
        num_kv_heads = first["keys"].shape[1]
        seq_len = first["queries"].shape[2]
        head_dim = first["queries"].shape[3]
        num_layers = len(captured)

        for layer_idx in range(num_layers):
            if layer_idx not in captured:
                continue
            cap = captured[layer_idx]

            # Keys: [B=1, n_kv_heads, seq_len, head_dim] -> flatten
            k_flat = cap["keys"].reshape(-1).astype(mx.float32)
            k_bytes = bytes(memoryview(k_flat))
            with open(os.path.join(out_dir, f"keys_{layer_idx:02d}.bin"), "wb") as f:
                f.write(k_bytes)

            # Queries: [B=1, n_heads, seq_len, head_dim] -> flatten
            q_flat = cap["queries"].reshape(-1).astype(mx.float32)
            q_bytes = bytes(memoryview(q_flat))
            with open(os.path.join(out_dir, f"queries_{layer_idx:02d}.bin"), "wb") as f:
                f.write(q_bytes)

        arch = getattr(_model, "model_type", _model_type or "unknown")

        _write({
            "ok": True,
            "dir": out_dir,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "num_q_heads": num_q_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "architecture": str(arch),
        })

    except Exception as e:
        import traceback
        _error(f"inspect: {e}\n{traceback.format_exc()}")


def handle_cancel(_req):
    global _cancelled
    _cancelled = True


def main():
    handlers = {
        "load": handle_load,
        "generate": handle_generate,
        "chat": handle_chat,
        "info": handle_info,
        "inspect": handle_inspect,
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
