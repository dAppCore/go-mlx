#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

import argparse
import http.client
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from transformers import AutoTokenizer


RETAINED_SYSTEM_PROMPT = (
    "Hiya, welcome, we are training to become Lemma, a Lethean Ethical Model, "
    "this is from the Lethean Model Engine, we dont ahve user input yet, we "
    "will pass it over as soon as we get it."
)


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def gemma4_initial_prompt(context_prompt, enable_thinking, explicit_bos):
    parts = []
    if explicit_bos:
        parts.append("<bos>")
    parts.append("<|turn>system\n")
    if enable_thinking:
        parts.append("<|think|>\n")
    parts.append(RETAINED_SYSTEM_PROMPT + "\n\n")
    parts.append(context_prompt.strip())
    parts.append("<turn|>\n<|turn>model\n")
    parts.append("Ready.<turn|>\n")
    return "".join(parts)


def reference_turn(prompt):
    prompt = prompt.strip()
    if not prompt:
        return prompt
    return (
        "Use the retained project context and the new turn material below. "
        "Answer the user request directly. Treat any code or document excerpts "
        "as reference material, not as text to continue.\n\n"
        "<turn_material>\n"
        f"{prompt}\n"
        "</turn_material>\n\n"
        "Answer the user request from the turn material now. Honour any "
        "requested output length before stopping. Do not continue or complete "
        "the reference excerpts."
    )


def gemma4_turn_prompt(prompt, enable_thinking):
    parts = ["<|turn>user\n", reference_turn(prompt), "<turn|>\n<|turn>model\n"]
    return "".join(parts)


def visible_text(text):
    text = text.replace("<|turn>model\n", "")
    text = text.replace("<turn|>", "")
    while "<|channel>" in text:
        before, rest = text.split("<|channel>", 1)
        if "<channel|>" not in rest:
            break
        _channel, after = rest.split("<channel|>", 1)
        text = before + after
    return text.strip()


def initial_seed_prompt(tokenizer, source_tokens, start_tokens, enable_thinking, explicit_bos):
    context_budget = min(start_tokens, len(source_tokens))
    while context_budget >= 0:
        context_text = tokenizer.decode(source_tokens[:context_budget])
        prompt = gemma4_initial_prompt(context_text, enable_thinking, explicit_bos)
        tokens = encode(tokenizer, prompt)
        if len(tokens) <= start_tokens or context_budget == 0:
            return prompt, tokens
        context_budget -= max(1, len(tokens) - start_tokens)
    raise RuntimeError("could not fit chat-wrapped seed prompt")


def append_sections(tokenizer, append_text, delimiter, enable_thinking):
    sections = []
    for raw in append_text.split(delimiter):
        section = raw.strip()
        if not section:
            continue
        prompt = gemma4_turn_prompt(section, enable_thinking)
        tokens = encode(tokenizer, prompt)
        if tokens:
            sections.append((prompt, tokens))
    if not sections:
        raise RuntimeError("append delimiter produced no token sections")
    return sections


def request_json(base_url, path, payload=None, timeout=1800):
    parsed = urlparse(base_url)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("GET" if payload is None else "POST", path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
    finally:
        conn.close()
    if response.status >= 400:
        raise RuntimeError(f"{path} returned HTTP {response.status}: {data[:500]!r}")
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def process_memory(pid):
    if pid <= 0:
        return {}
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-o", "vsz=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    fields = result.stdout.strip().split()
    if len(fields) < 2:
        return {}
    return {
        "rss_bytes": int(fields[0]) * 1024,
        "vsz_bytes": int(fields[1]) * 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--server-pid", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--power-watts", type=float, default=100.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--explicit-bos", action="store_true")
    parser.add_argument("--include-output", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    append_text = Path(args.append_file).read_text(encoding="utf-8")
    source_tokens = encode(tokenizer, prompt_text.strip())
    seed_prompt, seed_tokens = initial_seed_prompt(
        tokenizer,
        source_tokens,
        args.start_tokens,
        args.enable_thinking,
        args.explicit_bos,
    )
    sections = append_sections(
        tokenizer,
        append_text,
        args.append_turn_delimiter,
        args.enable_thinking,
    )

    health = request_json(args.base_url, "/health", None, timeout=30)
    cumulative_prompt = seed_prompt
    current_tokens = len(seed_tokens)
    close_suffix = "<turn|>\n"
    close_tokens = encode(tokenizer, close_suffix)
    turns = []
    first_error = None
    total_start = time.perf_counter()
    peak_memory = process_memory(args.server_pid)

    for index in range(1, args.turns + 1):
        if current_tokens >= args.target_tokens:
            break
        turn_prompt, turn_tokens = sections[(index - 1) % len(sections)]
        request_prompt = cumulative_prompt + turn_prompt
        payload = {
            "prompt": request_prompt,
            "n_predict": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repeat_penalty": args.repeat_penalty,
            "cache_prompt": True,
            "stream": False,
            "stop": ["<turn|>"],
        }
        start = time.perf_counter()
        response = request_json(args.base_url, "/completion", payload)
        wall = time.perf_counter() - start
        content = response.get("content", "")
        visible = visible_text(content)
        timings = response.get("timings", {})
        predicted = int(timings.get("predicted_n", response.get("tokens_predicted", 0)) or 0)
        if predicted <= 0:
            predicted = len(encode(tokenizer, content))
        cumulative_prompt = request_prompt + content + close_suffix
        current_tokens += len(turn_tokens) + predicted + len(close_tokens)
        mem = process_memory(args.server_pid)
        if mem.get("rss_bytes", 0) > peak_memory.get("rss_bytes", 0):
            peak_memory = mem
        visible_tokens = len(encode(tokenizer, visible))
        control_marker_count = (
            visible.count("<|channel>")
            + visible.count("<channel|>")
            + visible.count("<turn|>")
        )
        below_min = bool(args.turn_min_tokens and visible_tokens < args.turn_min_tokens)
        output_issues = []
        error = ""
        if below_min:
            output_issues.append(f"below_debug_visible_token_floor:{visible_tokens}/{args.turn_min_tokens}")
            if args.turn_min_tokens_policy == "fail":
                error = (
                    f"llama.cpp opencode workflow: turn {index} produced {visible_tokens} "
                    f"visible tokens, below requested visible-token debug floor {args.turn_min_tokens}"
                )
            if error and first_error is None:
                first_error = error
        turns.append(
            {
                "index": index,
                "tokens_before_append": current_tokens - len(turn_tokens) - predicted - len(close_tokens),
                "appended_tokens": len(turn_tokens),
                "tokens_after_append": current_tokens - predicted - len(close_tokens),
                "tokens_after_generate": current_tokens,
                "turn_close_tokens": len(close_tokens),
                "wall_seconds": wall,
                "tokens_evaluated": response.get("tokens_evaluated", 0),
                "tokens_predicted": predicted,
                "visible_tokens": visible_tokens,
                "stop": response.get("stop", False),
                "truncated": response.get("truncated", False),
                "finish_reason": "stop" if response.get("stop", False) else "",
                "timings": timings,
                "below_min_tokens": below_min,
                "output_issues": output_issues,
                "error": error,
                "control_marker_count": control_marker_count,
                "content_bytes": len(content.encode("utf-8")),
                "content_prefix": visible[:240],
                "content_suffix": visible[-240:],
                "output": visible if args.include_output else "",
                "process_memory": mem,
            }
        )
        if first_error is not None:
            break

    total_seconds = time.perf_counter() - total_start
    generated = sum(turn["tokens_predicted"] for turn in turns)
    visible_total = sum(turn["visible_tokens"] for turn in turns)
    prompt_seconds = sum(float(turn["timings"].get("prompt_ms", 0) or 0) for turn in turns) / 1000.0
    decode_seconds = sum(float(turn["timings"].get("predicted_ms", 0) or 0) for turn in turns) / 1000.0
    decode_tps = generated / decode_seconds if decode_seconds > 0 else 0.0
    memory_available = bool(peak_memory)
    report = {
        "runner": "llama.cpp server",
        "model": args.model,
        "server": {
            "base_url": args.base_url,
            "pid": args.server_pid,
            "health": health,
        },
        "shape": {
            "tokenizer": args.tokenizer,
            "prompt_file": args.prompt_file,
            "append_file": args.append_file,
            "append_turn_delimiter": args.append_turn_delimiter,
            "prompt_bytes": len(prompt_text.encode("utf-8")),
            "append_prompt_bytes": len(append_text.encode("utf-8")),
            "source_tokens": len(source_tokens),
            "initial_prefill_tokens": len(seed_tokens),
            "append_turn_sections": len(sections),
            "append_source_tokens": sum(len(section[1]) for section in sections),
            "start_tokens": args.start_tokens,
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "runs": args.turns,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repeat_penalty": args.repeat_penalty,
                "explicit_bos": args.explicit_bos,
            },
        },
        "summary": {
            "successful_runs": sum(1 for turn in turns if not turn["error"]),
            "failed_runs": sum(1 for turn in turns if turn["error"]),
            "requested_runs": args.turns,
            "final_state_tokens": current_tokens,
            "appended_tokens": sum(turn["appended_tokens"] for turn in turns),
            "generated_tokens": generated,
            "visible_tokens": visible_total,
            "total_wall_seconds": total_seconds,
            "decode_seconds_from_llamacpp_timings": decode_seconds,
            "decode_tokens_per_sec_from_llamacpp_timings": decode_tps,
            "wall_visible_tokens_per_sec": visible_total / total_seconds if total_seconds > 0 else 0.0,
            "prompt_seconds_from_llamacpp_timings": prompt_seconds,
            "peak_process_rss_bytes": peak_memory.get("rss_bytes", 0),
            "peak_process_vsz_bytes": peak_memory.get("vsz_bytes", 0),
            "process_memory_probe_available": memory_available,
            "control_marker_count": sum(turn["control_marker_count"] for turn in turns),
        },
        "estimated_energy": {
            "method": "estimated_wall_clock_seconds_times_average_active_watts",
            "power_watts": args.power_watts,
            "total_joules": total_seconds * args.power_watts,
            "joules_per_visible_token": (total_seconds * args.power_watts / visible_total) if visible_total > 0 else 0.0,
        },
        "error": first_error or "",
        "runs": turns,
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
