#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

import argparse
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load_model, load_tokenizer

from state_ramp_prompts import (
    gemma4_initial_prompt,
    gemma4_turn_prompt,
    visible_text,
)


def encode(tokenizer, text):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def decode(tokenizer, tokens):
    return tokenizer.decode(tokens)


def token_id(tokenizer, text):
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict) and text in vocab:
        return int(vocab[text])
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        value = convert(text)
        if isinstance(value, int) and value >= 0:
            return value
    ids = encode(tokenizer, text)
    if len(ids) == 1:
        return int(ids[0])
    return None


def initial_seed_tokens(tokenizer, source_tokens, start_tokens, enable_thinking):
    context_budget = min(start_tokens, len(source_tokens))
    while context_budget >= 0:
        context_text = decode(tokenizer, source_tokens[:context_budget])
        tokens = encode(
            tokenizer,
            gemma4_initial_prompt(context_text, enable_thinking),
        )
        if len(tokens) <= start_tokens or context_budget == 0:
            return tokens
        overage = max(1, len(tokens) - start_tokens)
        context_budget -= overage
    raise RuntimeError("could not fit chat-wrapped seed prompt")


def append_sections(tokenizer, append_text, delimiter, enable_thinking):
    sections = []
    for raw in append_text.split(delimiter):
        section = raw.strip()
        if not section:
            continue
        tokens = encode(tokenizer, gemma4_turn_prompt(section, enable_thinking))
        if tokens:
            sections.append(tokens)
    if not sections:
        raise RuntimeError("append delimiter produced no token sections")
    return sections


def prefill_tokens(model, cache, tokens, step_size):
    if not tokens:
        return 0.0
    start = time.perf_counter()
    for _ in generate_step(
        mx.array(tokens),
        model,
        max_tokens=0,
        prompt_cache=cache,
        prefill_step_size=step_size,
    ):
        pass
    mx.eval([c.state for c in cache])
    return time.perf_counter() - start


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value < 1024 * 1024:
        return int(value * 1024)
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--append-file", required=True)
    parser.add_argument("--report-file", default="")
    parser.add_argument("--append-turn-delimiter", default="---TURN---")
    parser.add_argument("--start-tokens", type=int, default=30000)
    parser.add_argument("--target-tokens", type=int, default=70000)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--turn-min-tokens", type=int, default=0)
    parser.add_argument("--turn-min-tokens-policy", choices=["fail", "mark"], default="mark")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--power-watts", type=float, default=100.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--ignore-extra-weights", action="store_true")
    parser.add_argument("--include-output", action="store_true")
    args = parser.parse_args()

    load_start = time.perf_counter()
    model, config = load_model(Path(args.model), strict=not args.ignore_extra_weights)
    tokenizer = load_tokenizer(Path(args.model), eos_token_ids=config.get("eos_token_id", None))
    load_seconds = time.perf_counter() - load_start

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    append_text = Path(args.append_file).read_text(encoding="utf-8")
    source_tokens = encode(tokenizer, prompt_text.strip())
    seed_tokens = initial_seed_tokens(tokenizer, source_tokens, args.start_tokens, args.enable_thinking)
    sections = append_sections(
        tokenizer,
        append_text,
        args.append_turn_delimiter,
        args.enable_thinking,
    )

    cache = make_prompt_cache(model, args.max_kv_size)
    prefill_seconds = prefill_tokens(model, cache, seed_tokens, args.prefill_step_size)

    suppress_ids = []
    for text in (
        "<pad>",
        "<bos>",
        "<unk>",
        "<mask>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        '<|"|>',
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<|image>",
        "<|audio>",
        "<|image|>",
        "<|audio|>",
        "<image|>",
        "<audio|>",
        "<|video|>",
    ):
        ident = token_id(tokenizer, text)
        if ident is not None:
            suppress_ids.append(ident)
    logit_bias = {ident: -1e9 for ident in suppress_ids}
    processors = make_logits_processors(logit_bias=logit_bias) if logit_bias else None
    sampler = make_sampler(args.temperature, args.top_p, 0.0, top_k=args.top_k)
    turn_stop_id = token_id(tokenizer, "<turn|>")
    close_tokens = encode(tokenizer, "<turn|>\n")

    turns = []
    current_tokens = len(seed_tokens)
    generation_start = time.perf_counter()
    first_error = None
    for index in range(1, args.turns + 1):
        if current_tokens >= args.target_tokens:
            break
        turn_tokens = sections[(index - 1) % len(sections)]
        turn_start = time.perf_counter()
        first_token_seconds = None
        last = None
        output_parts = []
        sampled_ids = []
        sampled_texts = []
        stop_reason = None
        for response in stream_generate(
            model,
            tokenizer,
            turn_tokens,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=processors,
            max_kv_size=args.max_kv_size,
            prompt_cache=cache,
            prefill_step_size=args.prefill_step_size,
        ):
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - turn_start
            last = response
            output_parts.append(response.text)
            if len(sampled_ids) < 32:
                sampled_ids.append(int(response.token))
                sampled_texts.append(response.text)
            if turn_stop_id is not None and int(response.token) == turn_stop_id:
                stop_reason = "turn"
                break
        duration = time.perf_counter() - turn_start
        generated_tokens = int(last.generation_tokens) if last is not None else 0
        prompt_tps = float(last.prompt_tps) if last is not None else 0.0
        prompt_seconds = len(turn_tokens) / prompt_tps if prompt_tps > 0 else 0.0
        generation_tps = float(last.generation_tps) if last is not None else 0.0
        if stop_reason is None and last is not None:
            stop_reason = last.finish_reason
        close_seconds = prefill_tokens(model, cache, close_tokens, args.prefill_step_size)
        current_tokens += len(turn_tokens) + generated_tokens + len(close_tokens)
        text = "".join(output_parts)
        visible = visible_text(text)
        visible_tokens = generated_tokens
        below_min = bool(args.turn_min_tokens and visible_tokens < args.turn_min_tokens)
        output_issues = []
        error = ""
        if below_min:
            output_issues.append(f"below_debug_visible_token_floor:{visible_tokens}/{args.turn_min_tokens}")
            if args.turn_min_tokens_policy == "fail":
                error = (
                    f"mlx_lm opencode workflow: turn {index} produced {visible_tokens} "
                    f"visible tokens, below requested visible-token debug floor {args.turn_min_tokens}"
                )
            if error and first_error is None:
                first_error = error
        turns.append(
            {
                "index": index,
                "tokens_before_append": current_tokens - len(turn_tokens) - generated_tokens - len(close_tokens),
                "appended_tokens": len(turn_tokens),
                "tokens_after_append": current_tokens - generated_tokens - len(close_tokens),
                "tokens_after_generate": current_tokens,
                "turn_close_tokens": len(close_tokens),
                "duration_seconds": duration,
                "append_prompt_seconds": prompt_seconds,
                "close_seconds": close_seconds,
                "first_token_seconds": first_token_seconds or 0.0,
                "generated_tokens": generated_tokens,
                "visible_tokens": visible_tokens,
                "generation_tokens_per_sec": generation_tps,
                "prompt_tokens_per_sec": prompt_tps,
                "peak_memory_gb": float(last.peak_memory) if last is not None else mx.get_peak_memory() / 1e9,
                "finish_reason": stop_reason,
                "below_min_tokens": below_min,
                "output_issues": output_issues,
                "error": error,
                "sampled_token_ids": sampled_ids,
                "sampled_token_texts": sampled_texts,
                "output": visible if args.include_output else "",
            }
        )
        mx.clear_cache()
        if first_error is not None:
            break
    generation_seconds = time.perf_counter() - generation_start

    generated = sum(turn["generated_tokens"] for turn in turns)
    visible = sum(turn["visible_tokens"] for turn in turns)
    append_seconds = sum(turn["append_prompt_seconds"] + turn["close_seconds"] for turn in turns)
    turn_wall_seconds = sum(turn["duration_seconds"] + turn["close_seconds"] for turn in turns)
    decode_tps_values = [turn["generation_tokens_per_sec"] for turn in turns if turn["generation_tokens_per_sec"] > 0]
    total_seconds = load_seconds + prefill_seconds + generation_seconds
    report = {
        "runner": "mlx_lm",
        "versions": {
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "model": args.model,
        "strict_load": not args.ignore_extra_weights,
        "ignored_extra_weights": args.ignore_extra_weights,
        "prompt_file": args.prompt_file,
        "append_file": args.append_file,
        "append_turn_delimiter": args.append_turn_delimiter,
        "prompt_bytes": len(prompt_text.encode("utf-8")),
        "append_prompt_bytes": len(append_text.encode("utf-8")),
        "source_tokens": len(source_tokens),
        "initial_prefill_tokens": len(seed_tokens),
        "append_turn_sections": len(sections),
        "append_source_tokens": sum(len(section) for section in sections),
        "start_tokens": args.start_tokens,
        "target_tokens": args.target_tokens,
        "runs_requested": args.turns,
        "max_tokens": args.max_tokens,
        "turn_min_tokens": args.turn_min_tokens,
        "turn_min_tokens_policy": args.turn_min_tokens_policy,
        "prefill_step_size": args.prefill_step_size,
        "max_kv_size": args.max_kv_size,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "load_seconds": load_seconds,
        "initial_prefill_seconds": prefill_seconds,
        "initial_prefill_tokens_per_sec": len(seed_tokens) / prefill_seconds if prefill_seconds > 0 else 0.0,
        "generation_wall_seconds": generation_seconds,
        "total_wall_seconds_including_load_and_prefill": total_seconds,
        "summary": {
            "successful_turns": sum(1 for turn in turns if not turn["error"]),
            "failed_turns": sum(1 for turn in turns if turn["error"]),
            "final_state_tokens": current_tokens,
            "appended_tokens": sum(turn["appended_tokens"] for turn in turns),
            "generated_tokens": generated,
            "visible_tokens": visible,
            "append_seconds_estimated": append_seconds,
            "decode_tokens_per_sec_average": sum(decode_tps_values) / len(decode_tps_values) if decode_tps_values else 0.0,
            "effective_turn_tokens_per_sec": generated / turn_wall_seconds if turn_wall_seconds > 0 else 0.0,
            "peak_memory_gb": max((turn["peak_memory_gb"] for turn in turns), default=mx.get_peak_memory() / 1e9),
            "peak_process_rss_bytes": peak_rss_bytes(),
        },
        "estimated_energy": {
            "method": "estimated_wall_clock_seconds_times_average_active_watts",
            "power_watts": args.power_watts,
            "total_joules": total_seconds * args.power_watts,
            "generation_joules": generation_seconds * args.power_watts,
            "initial_prefill_joules": prefill_seconds * args.power_watts,
            "joules_per_visible_token": (total_seconds * args.power_watts / visible) if visible > 0 else 0.0,
        },
        "error": first_error or "",
        "turns": turns,
    }
    data = json.dumps(report, indent=2)
    if args.report_file:
        path = Path(args.report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    if first_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
