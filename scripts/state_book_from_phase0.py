#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PHASE0 = Path("/Users/snider/Code/lthn/LEM/training/lem/creative/phase0.json")
DEFAULT_MODEL = Path(
    "/Users/snider/.cache/huggingface/hub/"
    "models--mlx-community--gemma-4-e2b-it-4bit/"
    "snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd"
)
TURN_DELIMITER = "---TURN---"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slugify(text: str, fallback: str = "book") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return value[:80] or fallback


def load_phase0(path: Path) -> list[dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON list")
    prompts: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        prompt_id = str(entry.get("id", f"prompt-{index + 1}")).strip()
        prompt = str(entry.get("prompt", "")).strip()
        if prompt:
            prompts.append(
                {
                    "id": prompt_id,
                    "domain": str(entry.get("domain", "")).strip(),
                    "prompt": prompt,
                }
            )
    if len(prompts) < 2:
        raise ValueError(f"{path} must contain at least two usable prompts")
    return prompts


def choose_seed(prompts: list[dict[str, str]], rng: random.Random, seed_id: str) -> dict[str, str]:
    if seed_id:
        for prompt in prompts:
            if prompt["id"] == seed_id:
                return prompt
        raise ValueError(f"seed id {seed_id!r} was not found")
    return rng.choice(prompts)


def choose_distractors(
    prompts: list[dict[str, str]],
    seed_prompt: dict[str, str],
    rng: random.Random,
    turns: int,
) -> list[dict[str, str]]:
    pool = [prompt for prompt in prompts if prompt["id"] != seed_prompt["id"]]
    if not pool:
        raise ValueError("no distractor prompts available after removing the seed")
    rng.shuffle(pool)
    distractors: list[dict[str, str]] = []
    while len(distractors) < turns:
        distractors.extend(pool)
    return distractors[:turns]


def seed_arc_text(seed_prompt: dict[str, str], turns: int) -> str:
    return (
        "Story arc contract:\n\n"
        f"Seed prompt id: {seed_prompt['id']}\n\n"
        "Use the following seed prompt as the only main story arc for this "
        f"{turns}-chapter book. Later turn prompts may add entropy, imagery, "
        "or interference, but they must not replace the seed arc. The final "
        "chapter must resolve this seed arc rather than resolving any later "
        "distractor prompt.\n\n"
        f"{seed_prompt['prompt']}\n"
    )


def turn_request(
    chapter: int,
    turns: int,
    seed_prompt: dict[str, str],
    distractor: dict[str, str],
    include_seed_contract: bool,
) -> str:
    if include_seed_contract:
        if chapter == 1:
            continuity = "Begin the retained seed story arc."
        elif chapter == turns:
            continuity = (
                "End the retained seed story arc. The final movement must resolve "
                f"the seed prompt id {seed_prompt['id']} and must not resolve the "
                "distractor as the main plot."
            )
        else:
            continuity = f"Continue the retained seed story arc from Chapter {chapter - 1}."
        return (
            f"Chapter {chapter} request:\n\n"
            f"Write Chapter {chapter} only. {continuity} "
            "The seed prompt remains the only plot. Use the distractor for "
            "imagery, mood, pressure, or interference only. Do not retell the "
            "distractor as the chapter plot.\n\n"
            f"Seed prompt id to preserve: {seed_prompt['id']}\n\n"
            "Seed prompt text to preserve:\n"
            f"{seed_prompt['prompt']}\n\n"
            "Distractor pressure for imagery only, not plot:\n"
            f"{distractor['prompt']}\n"
        )
    if chapter == turns:
        continuity = (
            "End the retained story arc. The final movement must resolve the "
            "opening arc without turning the pressure prompt into the main plot."
        )
    else:
        continuity = f"Continue the existing book from Chapter {chapter - 1}."
    return (
        f"**Chapter {chapter}**\n\n"
        f"{continuity} This is chapter {chapter} of {turns}. "
        "Use the following pressure as imagery, mood, or interference only; "
        "do not retell it as the chapter plot:\n"
        f"{distractor['prompt']}\n\n"
        "Write only this chapter heading and prose. Do not include commentary, "
        "planning, summaries, previous chapters, or prompt analysis.\n"
    )


def turn_sections_for(
    turns: int,
    seed_prompt: dict[str, str],
    distractors: list[dict[str, str]],
    include_seed_contract: bool,
) -> list[str]:
    return [
        turn_request(index + 1, turns, seed_prompt, distractor, include_seed_contract)
        for index, distractor in enumerate(distractors)
    ]


def write_turn_sections(path: Path, turn_sections: list[str]) -> None:
    path.write_text(f"\n{TURN_DELIMITER}\n".join(turn_sections), encoding="utf-8")


def write_materials(
    out_dir: Path,
    run_slug: str,
    seed_prompt: dict[str, str],
    distractors: list[dict[str, str]],
    turn_sections: list[str],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = len(distractors)
    seed_path = out_dir / f"{run_slug}.seed.txt"
    turns_path = out_dir / f"{run_slug}.turns.txt"
    meta_path = out_dir / f"{run_slug}.selection.json"

    seed_path.write_text(seed_arc_text(seed_prompt, turns), encoding="utf-8")
    write_turn_sections(turns_path, turn_sections)
    meta_path.write_text(
        json.dumps(
            {
                "seed": seed_prompt,
                "distractors": distractors,
                "turns": turns,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"seed": seed_path, "turns": turns_path, "meta": meta_path}


def metric_line(report: dict) -> str:
    summary = report.get("summary") or {}
    return (
        f"- Successful turns: {summary.get('successful_turns', 0)}\n"
        f"- Initial prefill tokens: {summary.get('initial_prefill_tokens', 0)}\n"
        f"- Final state tokens: {summary.get('final_state_tokens', 0)}\n"
        f"- Appended tokens: {summary.get('appended_tokens', 0)}\n"
        f"- Generated visible tokens: {summary.get('visible_tokens', 0)}\n"
        f"- Decode average: {summary.get('decode_tokens_per_sec_average', 0)} tok/s\n"
        f"- Effective turn average: {summary.get('effective_turn_tokens_per_sec_average', 0)} tok/s\n"
        f"- Active + cache memory peak: {summary.get('active_plus_cache_memory_bytes', 0)} bytes\n"
        f"- Process RSS peak: {summary.get('process_peak_resident_bytes', 0)} bytes\n"
    )


def write_book(
    book_path: Path,
    report_path: Path,
    selection_path: Path,
    title: str,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seed = selection["seed"]
    distractors = selection["distractors"]
    turns = report.get("turns") or []
    chapters = []
    for turn in turns:
        output = str(turn.get("output", "")).strip()
        if output:
            chapters.append(output)
    book_path.parent.mkdir(parents=True, exist_ok=True)
    book_path.write_text(
        "# "
        + title
        + "\n\n"
        + f"Generated by go-mlx retained State run `{report_path.name}`.\n\n"
        + f"Seed prompt: `{seed['id']}`\n\n"
        + seed["prompt"]
        + "\n\n"
        + "Distractor prompts were supplied one per chapter as entropy and "
        "imagery pressure, not as replacement plot instructions.\n\n"
        + "## Distractors\n\n"
        + "\n".join(f"- `{item['id']}`" for item in distractors)
        + "\n\n"
        + "## Metrics\n\n"
        + metric_line(report)
        + "\n---\n\n"
        + "\n\n".join(chapters)
        + "\n",
        encoding="utf-8",
    )
    return report


def build_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    report_path: Path,
    *,
    append_path: Path | None = None,
    turns: int | None = None,
    include_prompt_file: bool = True,
    extra_flags: list[str] | None = None,
) -> list[str]:
    start_tokens = args.start_tokens if include_prompt_file else 0
    command = [
        str(args.bin),
        "state-ramp-profile",
        "-json",
        "-include-output",
        "-report-file",
        str(report_path),
        "-append-file",
        str(append_path or paths["turns"]),
        "-append-turn-delimiter",
        TURN_DELIMITER,
        "-start-tokens",
        str(start_tokens),
        "-target-tokens",
        str(args.target_tokens),
        "-append-tokens",
        str(args.append_tokens),
        "-turn-max-tokens",
        str(args.turn_max_tokens),
        "-turns",
        str(turns if turns is not None else args.turns),
        "-chat-template",
        args.chat_template,
        "-turn-prompt-mode",
        args.turn_prompt_mode,
        "-context",
        str(args.context),
        "-cache-mode",
        args.cache_mode,
        "-estimate-power-watts",
        str(args.power_watts),
        "-turn-min-tokens",
        "0",
    ]
    if include_prompt_file:
        command[6:6] = [
            "-prompt-file",
            str(paths["seed"]),
        ]
    else:
        command[6:6] = [
            "-prompt",
            "",
        ]
    if extra_flags:
        command.extend(extra_flags)
    command.append(str(args.model))
    return command


def run_command_capture(
    args: argparse.Namespace,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    env = os.environ.copy()
    if args.metallib:
        env["MLX_METALLIB_PATH"] = str(args.metallib)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            check=False,
            cwd=args.run_dir,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
    return result.returncode


def run_book(args: argparse.Namespace, command: list[str], run_slug: str) -> int:
    return run_command_capture(
        args,
        command,
        args.run_dir / f"{run_slug}.stdout",
        args.run_dir / f"{run_slug}.stderr",
    )


def append_manifest(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Generate a retained-State book run from phase0 creative prompts."
    )
    parser.add_argument("--phase0", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument("--seed-id", default="")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, default=Path("/private/tmp/go-mlx-goal/book-runs"))
    parser.add_argument("--book-dir", type=Path, default=Path("/private/tmp/go-mlx-goal/books"))
    parser.add_argument("--manifest", type=Path, default=Path("/private/tmp/go-mlx-goal/books/manifest.jsonl"))
    parser.add_argument("--bin", type=Path, default=Path(os.environ.get("GO_MLX_BIN", root / "bin/lthn-mlx")))
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("GO_MLX_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--metallib", type=Path, default=Path(os.environ.get("MLX_METALLIB_PATH", root / "dist/lib/mlx.metallib")))
    parser.add_argument("--start-tokens", type=int, default=10000)
    parser.add_argument("--target-tokens", type=int, default=30000)
    parser.add_argument("--append-tokens", type=int, default=2000)
    parser.add_argument("--turn-max-tokens", type=int, default=2048)
    parser.add_argument("--chat-template", default="gemma4")
    parser.add_argument("--turn-prompt-mode", default="reference", choices=("reference", "direct"))
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--cache-mode", default="paged")
    parser.add_argument("--power-watts", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_book_run(
    args: argparse.Namespace,
    prompts: list[dict[str, str]],
    random_seed: int,
    book_index: int,
) -> dict:
    rng = random.Random(random_seed)
    seed_prompt = choose_seed(prompts, rng, args.seed_id)
    distractors = choose_distractors(prompts, seed_prompt, rng, args.turns)
    turn_sections = turn_sections_for(args.turns, seed_prompt, distractors, True)

    run_slug = (
        time.strftime("%Y-%m-%d")
        + "-"
        + slugify(seed_prompt["id"])
        + f"-seed{random_seed}"
    )
    paths = write_materials(args.run_dir, run_slug, seed_prompt, distractors, turn_sections)
    report_path = args.run_dir / f"{run_slug}.json"
    book_path = args.book_dir / f"{run_slug}.md"
    command = build_command(args, paths, report_path)
    command_path = args.run_dir / f"{run_slug}.command.json"
    command_path.write_text(
        json.dumps(
            {
                "command": command,
                "random_seed": random_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "book_index": book_index,
        "random_seed": random_seed,
        "run_slug": run_slug,
        "seed_prompt": seed_prompt,
        "distractors": distractors,
        "paths": paths,
        "turn_sections": turn_sections,
        "report_path": report_path,
        "book_path": book_path,
        "command": command,
        "command_path": command_path,
    }


def run_prepared_book(args: argparse.Namespace, prepared: dict) -> int:
    seed_prompt = prepared["seed_prompt"]
    distractors = prepared["distractors"]
    paths = prepared["paths"]
    report_path = prepared["report_path"]
    book_path = prepared["book_path"]
    command = prepared["command"]
    run_slug = prepared["run_slug"]

    print(f"book_index: {prepared['book_index']}")
    print(f"seed: {seed_prompt['id']}")
    print("distractors: " + ", ".join(item["id"] for item in distractors))
    print(f"materials: {paths['seed']} {paths['turns']}")
    print(f"report: {report_path}")
    print(f"book: {book_path}")

    if args.dry_run:
        print(f"command: {' '.join(command)}")
        code = 0
        summary = {}
    else:
        code = run_book(args, command, run_slug)
        if report_path.exists():
            report = write_book(
                book_path,
                report_path,
                paths["meta"],
                f"State Book {seed_prompt['id']}",
            )
            summary = report.get("summary") or {}
        else:
            summary = {}

    append_manifest(
        args.manifest,
        {
            "book_index": prepared["book_index"],
            "random_seed": prepared["random_seed"],
            "run_slug": run_slug,
            "seed_id": seed_prompt["id"],
            "distractor_ids": [item["id"] for item in distractors],
            "report_path": str(report_path),
            "book_path": str(book_path),
            "selection_path": str(paths["meta"]),
            "command_path": str(prepared["command_path"]),
            "exit_code": code,
            "dry_run": args.dry_run,
            "summary": summary,
        },
    )
    return code


def main() -> int:
    args = parse_args()
    if args.turns < 1:
        raise ValueError("--turns must be >= 1")
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.count > 1 and args.seed_id:
        raise ValueError("--seed-id can only be used with --count 1")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.book_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_phase0(args.phase0)
    if not args.dry_run and not args.bin.exists():
        print(f"missing executable: {args.bin}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 2
    base_seed = args.random_seed or time.time_ns()
    exit_code = 0
    for index in range(args.count):
        random_seed = base_seed + index
        prepared = prepare_book_run(args, prompts, random_seed, index + 1)
        code = run_prepared_book(args, prepared)
        if code != 0:
            exit_code = code
            break
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"state_book_from_phase0: {exc}", file=sys.stderr)
        raise SystemExit(1)
