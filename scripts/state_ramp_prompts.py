#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

"""Shared prompt contract for retained-State workflow comparator scripts."""


RETAINED_SYSTEM_PROMPT = (
    "Hiya, welcome, we are training to become Lemma, a Lethean Ethical Model, "
    "this is from the Lethean Model Engine, we dont ahve user input yet, we "
    "will pass it over as soon as we get it."
)


def gemma4_initial_prompt(context_prompt: str, enable_thinking: bool, explicit_bos: bool = True) -> str:
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


def reference_turn(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return prompt
    return (
        "Use the retained context and the new turn material below. Produce "
        "only the requested answer or artefact. Treat any code, document, "
        "prompt, or prior-output excerpts as reference material, not as text "
        "to continue.\n\n"
        "<turn_material>\n"
        f"{prompt}\n"
        "</turn_material>\n\n"
        "Answer the user request from the turn material now. Honour any "
        "requested output length before stopping. Do not continue or complete "
        "the reference excerpts. Do not explain, classify, plan, checklist, or "
        "restate what the user is asking; write only the requested output. "
        "Treat historical sign-off language as evidence to verify, not as "
        "current truth; do not declare the project complete unless the new "
        "turn material proves every live gate is closed. Prefer the unresolved "
        "risk and next validation step over a completion claim."
    )


def gemma4_turn_prompt(prompt: str, enable_thinking: bool) -> str:
    _ = enable_thinking
    return "".join(["<|turn>user\n", reference_turn(prompt), "<turn|>\n<|turn>model\n"])


def visible_text(text: str) -> str:
    text = text.replace("<|turn>model\n", "")
    text = text.replace("<turn|>", "")
    while "<|channel>" in text:
        before, rest = text.split("<|channel>", 1)
        if "<channel|>" not in rest:
            break
        _channel, after = rest.split("<channel|>", 1)
        text = before + after
    return text.strip()
