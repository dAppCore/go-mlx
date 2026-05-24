#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

"""Check retained Gemma 4 prompt helpers against a local HF chat template.

This is a prompt-shape contract probe, not a content-quality metric. It compares
the retained seed plus one append turn with the model tokenizer's
apply_chat_template rendering for the same message history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from state_ramp_prompts import (
    RETAINED_SYSTEM_PROMPT,
    gemma4_initial_prompt,
    gemma4_turn_prompt,
    reference_turn,
)


def first_diff(left: str, right: str) -> dict[str, object]:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "index": index,
                "left": left[max(0, index - 80) : index + 80],
                "right": right[max(0, index - 80) : index + 80],
            }
    if len(left) != len(right):
        return {
            "index": limit,
            "left": left[max(0, limit - 80) : limit + 80],
            "right": right[max(0, limit - 80) : limit + 80],
        }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--context", default="Seed arc")
    parser.add_argument("--turn", default="Write the next chapter.")
    parser.add_argument("--turn-prompt-mode", choices=("reference", "direct"), default="reference")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    context = args.context.strip()
    turn = args.turn.strip()
    turn_text = turn if args.turn_prompt_mode == "direct" else reference_turn(turn)
    expected = gemma4_initial_prompt(context, args.enable_thinking, explicit_bos=True)
    expected += gemma4_turn_prompt(turn, args.enable_thinking, args.turn_prompt_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = [
        {"role": "system", "content": RETAINED_SYSTEM_PROMPT + "\n\n" + context},
        {"role": "assistant", "content": "Ready."},
        {"role": "user", "content": turn_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    ok = rendered == expected
    report = {
        "model": str(args.model),
        "turn_prompt_mode": args.turn_prompt_mode,
        "enable_thinking": args.enable_thinking,
        "matches_chat_template": ok,
        "expected_bytes": len(expected.encode("utf-8")),
        "rendered_bytes": len(rendered.encode("utf-8")),
        "first_diff": first_diff(expected, rendered) if not ok else {},
    }
    if args.dump:
        report["expected"] = expected
        report["rendered"] = rendered
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
