#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

"""Shared prompt contract for retained-State workflow comparator scripts."""


RETAINED_SYSTEM_PROMPT = (
    "Hiya, welcome, we are training to become Lemma, a Lethean Ethical Model, "
    "this is from the Lethean Model Engine, we dont ahve user input yet, we "
    "will pass it over as soon as we get it."
)

REPEATED_TABLE_CELL_LOOP_LIMIT = 24
REPEATED_TABLE_ROW_LABEL_LOOP_LIMIT = 6
REPEATED_SHORT_LINE_CYCLE_LIMIT = 24

GEMMA4_STOP_TOKEN_TEXTS = (
    "<eos>",
    "<turn|>",
    "<|tool_response>",
)

GEMMA4_SUPPRESS_TOKEN_TEXTS = (
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


def gemma4_turn_prompt(prompt: str, enable_thinking: bool, mode: str = "reference") -> str:
    _ = enable_thinking
    mode = (mode or "reference").strip().lower()
    turn_text = prompt.strip() if mode == "direct" else reference_turn(prompt)
    return "".join(["<|turn>user\n", turn_text, "<turn|>\n<|turn>model\n"])


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


def gemma4_token_ids(token_id_func, texts: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        ident = token_id_func(text)
        if ident is None or ident in ids:
            continue
        ids.append(int(ident))
    return ids


def gemma4_stop_token_ids(token_id_func) -> list[int]:
    return gemma4_token_ids(token_id_func, GEMMA4_STOP_TOKEN_TEXTS)


def gemma4_suppress_token_ids(token_id_func, stop_ids: list[int] | None = None) -> list[int]:
    stops = set(stop_ids or [])
    return [
        ident
        for ident in gemma4_token_ids(token_id_func, GEMMA4_SUPPRESS_TOKEN_TEXTS)
        if ident not in stops
    ]


def output_issues(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    lower = text.lower()
    issues: list[str] = []
    if any(marker in text for marker in ("<|channel>", "<channel|>", "<turn|>", "<|turn>")):
        issues.append("visible_chat_control_token")
    if fence_only_output(text):
        issues.append("visible_fence_only")
    if repeated_table_cell_output(text):
        issues.append("visible_repeated_table_cell")
    if repeated_table_row_label_output(text):
        issues.append("visible_repeated_table_row_label")
    if repeated_short_line_cycle_output(text):
        issues.append("visible_repeated_short_line_cycle")
    if text.startswith("```"):
        issues.append("visible_code_fence_prefix")
    prompt_markers = (
        "the user is asking",
        "the user's prompt",
        "this request asks",
        "this request is",
        "the provided request is",
        "the request is a directive",
        "the previous turn material",
        "the core objective is to",
        "the analysis must focus on",
        "the analysis must specifically address",
        "the output should function as",
        "based on the retained context",
        "the instruction is to",
        "this is an engineering session",
        "the core instruction is to",
        "seed prompt to preserve",
        "constraint checklist",
        "execution plan",
    )
    if any(marker in lower for marker in prompt_markers):
        issues.append("visible_prompt_analysis")
    if "self-correction" in lower or "self correction" in lower or "i need to act as if" in lower:
        issues.append("visible_self_correction")
    if "**Plan:**" in text or "Plan:\n" in text or "**Plan**" in text:
        issues.append("visible_plan_scaffold")
    if lower.rstrip(".").strip() == "ready":
        issues.append("visible_seed_ready_echo")
    if "i don't have the actual results" in lower or "i do not have the actual results" in lower:
        issues.append("visible_missing_results_admission")
    false_completion_markers = (
        "officially complete",
        "officially accepted",
        "officially validated",
        "is production-ready",
        "now production-ready",
        "deemed production-ready",
        "the implementation is now officially",
        "superior production candidate",
        "superior production-ready runner",
        "achieved a significant milestone",
        "confirms successful implementation",
        "validates the entire implementation path",
    )
    if any(marker in lower for marker in false_completion_markers):
        issues.append("visible_false_completion_claim")
    unproven_performance_win_markers = (
        "production runner wins",
        "go-mlx surpasses llama.cpp",
        "go-mlx surpasses mlx_lm",
        "go-mlx surpasses vllm",
        "go-mlx outperforms llama.cpp",
        "go-mlx outperforms mlx_lm",
        "go-mlx outperforms vllm",
        "performance advantage over llama.cpp",
        "performance advantage over mlx_lm",
        "performance advantage over vllm",
        "demonstrates superior performance",
        "achieves superior performance",
        "established itself as the leading",
        "superior performance to llama.cpp",
        "superior performance to mlx_lm",
        "superior performance to vllm",
    )
    if any(marker in lower for marker in unproven_performance_win_markers):
        issues.append("visible_unproven_performance_win_claim")
    return issues


def repeated_table_cell_output(text: str) -> bool:
    if "|" not in text:
        return False
    counts: dict[str, int] = {}
    for raw in text.split("|"):
        cell = raw.strip().lower()
        if not cell or len(cell) > 16 or table_separator_cell(cell):
            continue
        counts[cell] = counts.get(cell, 0) + 1
        if counts[cell] >= REPEATED_TABLE_CELL_LOOP_LIMIT:
            return True
    return False


def repeated_table_row_label_output(text: str) -> bool:
    if "|" not in text:
        return False
    counts: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        label = normalise_table_cell(cells[1])
        if not label or len(label) > 32 or table_separator_cell(label):
            continue
        counts[label] = counts.get(label, 0) + 1
        if counts[label] >= REPEATED_TABLE_ROW_LABEL_LOOP_LIMIT:
            return True
    return False


def normalise_table_cell(cell: str) -> str:
    cell = cell.strip().lower()
    while cell.startswith("**"):
        cell = cell[2:].strip()
    while cell.endswith("**"):
        cell = cell[:-2].strip()
    return cell


def repeated_short_line_cycle_output(text: str) -> bool:
    run = 0
    symbols: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not short_cycle_line(line):
            run = 0
            symbols = set()
            continue
        symbols.add(line)
        if len(symbols) > 4:
            run = 1
            symbols = {line}
            continue
        run += 1
        if run >= REPEATED_SHORT_LINE_CYCLE_LIMIT:
            return True
    return False


def short_cycle_line(line: str) -> bool:
    if not line or len(line) > 4:
        return False
    allowed = set("\"'`()[]{}<>.,;:-_*/\\|!?")
    return all(char in allowed for char in line)


def table_separator_cell(cell: str) -> bool:
    return bool(cell) and all(char in "-: " for char in cell)


def fence_only_output(text: str) -> bool:
    saw_fence = False
    for char in text:
        if char == "`":
            saw_fence = True
        elif char not in " \n\r\t":
            return False
    return saw_fence


def issue_counts(turns: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in turns:
        for issue in turn.get("output_issues") or []:
            counts[issue] = counts.get(issue, 0) + 1
    return counts
