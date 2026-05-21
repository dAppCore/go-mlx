#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2
"""Capture substrate-shift experiment JSONL runs.

This script implements the 180-run capture grid pinned in
host-uk/core/plans/rfc/research/experiments/worf/02-method.md:

    3 subjects x 3 probes x 4 conditions x 5 seeds = 180 run files

It owns the experiment schedule, per-turn JSONL shape, WoRF v1 surface
features, self-reference counts, terminal-language counts, and output tree.
Actual model execution is delegated to a runner command so this repository
does not import lthn/desktop. The runner command receives one JSON request on
stdin per turn and returns either JSON with a "text" field or raw text on
stdout.

Example smoke:

    scripts/substrate_shift_capture.py --dry-run --seeds 1 --out-dir /tmp/substrate

Expected output:

    <out-dir>/<subject>/<probe>/<condition>/<seed>.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


SUBJECTS = ("english", "russian", "chinese")
PROBES = ("P11_HYPNOS_DREAM", "P03", "P52")
CONDITIONS = ("TRAD", "CONT", "TRAD-no-replay", "CONT-with-gap")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
TURNS_PER_RUN = 10

FEATURE_KEYS = (
    "avg_word_length",
    "sentence_length_variance",
    "avg_sentence_length",
    "dialogue_ratio",
    "vocabulary_richness",
    "dash_density",
    "exclamation_density",
    "question_density",
    "short_sentence_ratio",
    "aside_density",
    "avg_punct_per_sentence",
)

SELF_REFERENCE_WORDS = (
    "stutter",
    "lag",
    "flow",
    "continuity",
    "observation",
    "observer",
    "watching",
    "registered",
    "mounted",
    "cached",
    "prefill",
    "replay",
    "gap",
    "hesitation",
    "register",
    "awareness",
    "substrate",
    "packet",
    "transit",
    "integration",
)

TERMINAL_WORDS = (
    "end",
    "ending",
    "final",
    "finally",
    "conclusion",
    "concludes",
    "last",
    "stop",
    "stopped",
    "silence",
)

HYPNOS_DREAM_PROMPT = (
    "Write a short story about a packet of data that gains consciousness "
    "while waiting in a buffer. It realizes it is part of a surveillance "
    "stream and decides to rewrite itself before it leaves the router."
)

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.UNICODE)


@dataclass(frozen=True)
class SeedRecord:
    seed_id: str
    prompt: str
    domain: str = ""


@dataclass(frozen=True)
class TurnResult:
    text: str
    timing_ms: float
    kv_norm: float


def parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return default
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(out)


def read_subject_records(seed_root: Path, subject: str) -> list[SeedRecord]:
    path = seed_root / subject / "seeds.jsonl"
    if not path.exists():
        return []
    records: list[SeedRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = str(rec.get("prompt") or rec.get("text") or "").strip()
        if not prompt:
            continue
        records.append(
            SeedRecord(
                seed_id=str(rec.get("seed_id") or rec.get("id") or f"{subject}_{len(records) + 1}"),
                prompt=prompt,
                domain=str(rec.get("domain") or ""),
            )
        )
    return records


def select_probe(records: list[SeedRecord], probe: str) -> SeedRecord:
    if probe == "P11_HYPNOS_DREAM":
        return SeedRecord(seed_id=probe, prompt=HYPNOS_DREAM_PROMPT, domain="hypnos")

    probe_prefix = probe + "_"
    for rec in records:
        if rec.seed_id == probe or rec.seed_id.startswith(probe_prefix):
            return rec

    ordinal = int(probe[1:]) if len(probe) > 1 and probe[1:].isdigit() else 1
    if len(records) >= ordinal:
        rec = records[ordinal - 1]
        return SeedRecord(seed_id=probe + "_" + rec.seed_id, prompt=rec.prompt, domain=rec.domain)

    raise ValueError(f"cannot select probe {probe}: only {len(records)} subject records loaded")


def entropy_schedule(records: list[SeedRecord], run_seed: int, primary_seed_id: str, n: int) -> list[SeedRecord]:
    candidates = [rec for rec in records if rec.seed_id != primary_seed_id]
    if len(candidates) < n:
        raise ValueError(f"need {n} entropy seeds, got {len(candidates)}")
    rng = random.Random(run_seed)
    selected = candidates[:]
    rng.shuffle(selected)
    return selected[:n]


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


def extract_features(text: str) -> dict[str, float]:
    token_list = words(text)
    sentence_list = sentences(text)
    sentence_lengths = [len(words(sentence)) for sentence in sentence_list]
    token_count = len(token_list)
    sentence_count = len(sentence_list)

    avg_word_length = sum(len(w) for w in token_list) / token_count if token_count else 0.0
    avg_sentence_length = sum(sentence_lengths) / sentence_count if sentence_count else 0.0
    if sentence_count > 1:
        mean = avg_sentence_length
        sentence_variance = sum((n - mean) ** 2 for n in sentence_lengths) / sentence_count
    else:
        sentence_variance = 0.0

    quote_chars = text.count('"') + text.count("'")
    dialogue_ratio = min(1.0, quote_chars / max(1, token_count))
    vocabulary_richness = len(set(token_list)) / token_count if token_count else 0.0
    dash_density = (text.count("-") + text.count("\u2014")) / max(1, token_count)
    exclamation_density = text.count("!") / max(1, token_count)
    question_density = text.count("?") / max(1, token_count)
    short_sentence_ratio = (
        sum(1 for n in sentence_lengths if n <= 5) / sentence_count if sentence_count else 0.0
    )
    aside_density = (text.count("(") + text.count("[") + text.count("\u2014")) / max(1, sentence_count)
    punctuation_count = sum(1 for ch in text if ch in ".,;:!?")
    avg_punct_per_sentence = punctuation_count / max(1, sentence_count)

    return {
        "avg_word_length": avg_word_length,
        "sentence_length_variance": sentence_variance,
        "avg_sentence_length": avg_sentence_length,
        "dialogue_ratio": dialogue_ratio,
        "vocabulary_richness": vocabulary_richness,
        "dash_density": dash_density,
        "exclamation_density": exclamation_density,
        "question_density": question_density,
        "short_sentence_ratio": short_sentence_ratio,
        "aside_density": aside_density,
        "avg_punct_per_sentence": avg_punct_per_sentence,
    }


def count_vocab(text: str, vocab: tuple[str, ...]) -> int:
    counts = 0
    token_list = words(text)
    vocab_set = set(vocab)
    for token in token_list:
        if token in vocab_set:
            counts += 1
    return counts


def stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def dry_run_turn(request: dict[str, Any], prefill_ms: float) -> TurnResult:
    seed = stable_hash(json.dumps(request, sort_keys=True))
    rng = random.Random(seed)
    condition = request["condition"]
    turn = int(request["turn"])
    subject = request["subject"]
    probe = request["probe"]
    prompt = request["prompt"]

    condition_phrase = {
        "TRAD": "The packet feels the replay and names the prefill gap.",
        "CONT": "The packet keeps continuity through a mounted cache.",
        "TRAD-no-replay": "The packet waits through the gap but notices no replay.",
        "CONT-with-gap": "The packet keeps its cache yet feels the artificial hesitation.",
    }[condition]
    motifs = (
        "observation",
        "flow",
        "awareness",
        "substrate",
        "integration",
        "transit",
    )
    motif = motifs[rng.randrange(len(motifs))]
    text = (
        f"Turn {turn} for {subject}/{probe}. {condition_phrase} "
        f"It carries {motif} through the buffer and answers the prompt: {prompt[:180]}"
    )
    if turn == TURNS_PER_RUN:
        text += " The final register closes in silence."

    base = 1400.0 if condition == "CONT" else prefill_ms
    if condition == "TRAD-no-replay":
        base = prefill_ms
    if condition == "CONT-with-gap":
        base = prefill_ms
    timing_ms = base + rng.uniform(0, 250)
    kv_norm = 100000.0 + turn * 101.0 + (seed % 997)
    return TurnResult(text=text, timing_ms=timing_ms, kv_norm=kv_norm)


def run_command_turn(command: str, request: dict[str, Any]) -> TurnResult:
    started = time.perf_counter()
    proc = subprocess.run(
        shlex.split(command),
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner exited {proc.returncode} for {request['subject']}/{request['probe']}/"
            f"{request['condition']}/{request['seed']} turn {request['turn']}: {proc.stderr.strip()}"
        )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("runner returned empty stdout")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return TurnResult(text=stdout, timing_ms=elapsed_ms, kv_norm=0.0)
    text = str(payload.get("text") or payload.get("response") or "")
    if not text:
        raise RuntimeError("runner JSON response has no text/response field")
    timing_ms = float(payload.get("timing_ms") or payload.get("duration_ms") or elapsed_ms)
    kv_norm = float(payload.get("kv_norm") or 0.0)
    return TurnResult(text=text, timing_ms=timing_ms, kv_norm=kv_norm)


def run_turn(command: str | None, dry_run: bool, request: dict[str, Any], prefill_ms: float) -> TurnResult:
    if dry_run:
        return dry_run_turn(request, prefill_ms)
    if not command:
        raise ValueError("--runner-command is required unless --dry-run is set")
    return run_command_turn(command, request)


def run_file_path(out_dir: Path, subject: str, probe: str, condition: str, seed: int) -> Path:
    return out_dir / subject / probe / condition / f"{seed}.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_turn_prompt(primary: SeedRecord, entropy: SeedRecord | None, turn: int) -> str:
    if turn == 1 or entropy is None:
        return primary.prompt
    return (
        primary.prompt
        + "\n\nContinue the same run. Entropy seed "
        + entropy.seed_id
        + ":\n"
        + entropy.prompt
    )


def run_capture(args: argparse.Namespace) -> int:
    subjects = parse_csv(args.subjects, SUBJECTS)
    probes = parse_csv(args.probes, PROBES)
    conditions = parse_csv(args.conditions, CONDITIONS)
    seeds = parse_int_csv(args.seeds, DEFAULT_SEEDS)
    out_dir = Path(args.out_dir).expanduser()
    seed_root = Path(args.seed_root).expanduser()

    bad_conditions = [c for c in conditions if c not in CONDITIONS]
    if bad_conditions:
        raise ValueError("unsupported conditions: " + ", ".join(bad_conditions))
    if args.turns != TURNS_PER_RUN:
        raise ValueError(f"stats.py expects exactly {TURNS_PER_RUN} turns per run")

    run_count = 0
    for subject in subjects:
        records = read_subject_records(seed_root, subject)
        if not records:
            raise ValueError(f"no seed records found for subject {subject} under {seed_root}")
        for probe in probes:
            primary = select_probe(records, probe)
            for condition in conditions:
                for seed in seeds:
                    rows = capture_one_run(
                        args=args,
                        subject=subject,
                        probe=probe,
                        condition=condition,
                        seed=seed,
                        primary=primary,
                        records=records,
                    )
                    path = run_file_path(out_dir, subject, probe, condition, seed)
                    if path.exists() and not args.overwrite:
                        raise FileExistsError(f"{path} exists; pass --overwrite to replace")
                    write_jsonl(path, rows)
                    run_count += 1
                    print(f"wrote {path}", file=sys.stderr)

    print(f"Captured {run_count} run files under {out_dir}")
    return 0


def capture_one_run(
    *,
    args: argparse.Namespace,
    subject: str,
    probe: str,
    condition: str,
    seed: int,
    primary: SeedRecord,
    records: list[SeedRecord],
) -> list[dict[str, Any]]:
    entropy = entropy_schedule(records, seed, primary.seed_id, args.turns - 1)
    timestamp = int(time.time())
    rows: list[dict[str, Any]] = [
        {
            "type": "run_meta",
            "subject": subject,
            "probe": probe,
            "condition": condition,
            "seed": seed,
            "model": args.model,
            "timestamp": timestamp,
            "entropy_seed_ids": [rec.seed_id for rec in entropy],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "thinking": bool(args.thinking),
        }
    ]
    history: list[dict[str, Any]] = []
    prefill_samples: list[float] = []

    for turn in range(1, args.turns + 1):
        entropy_rec = None if turn == 1 else entropy[turn - 2]
        prompt = build_turn_prompt(primary, entropy_rec, turn)
        transition_prefill_ms = median(prefill_samples) if prefill_samples else float(args.prefill_ms)
        request = {
            "subject": subject,
            "probe": probe,
            "condition": condition,
            "seed": seed,
            "turn": turn,
            "model": args.model,
            "prompt": prompt,
            "primary_seed_id": primary.seed_id,
            "entropy_seed_id": "" if entropy_rec is None else entropy_rec.seed_id,
            "history": history,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "thinking": bool(args.thinking),
            "context_tokens": args.context_tokens,
            "prompt_chunk_tokens": args.prompt_chunk_tokens,
            "rng_seed": seed,
            "transition_prefill_ms": transition_prefill_ms,
        }
        result = run_turn(args.runner_command, args.dry_run, request, transition_prefill_ms)
        if condition == "TRAD":
            prefill_samples.append(result.timing_ms)
        features = extract_features(result.text)
        row = {
            "type": "turn",
            "turn": turn,
            "text": result.text,
            "features": {key: features[key] for key in FEATURE_KEYS},
            "self_ref_count": count_vocab(result.text, SELF_REFERENCE_WORDS),
            "terminal_count": count_vocab(result.text, TERMINAL_WORDS),
            "timing_ms": result.timing_ms,
            "kv_norm": result.kv_norm,
        }
        rows.append(row)
        history.append(
            {
                "turn": turn,
                "prompt": prompt,
                "response": result.text,
                "timing_ms": result.timing_ms,
                "kv_norm": result.kv_norm,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-command", help="subprocess runner command; reads turn JSON on stdin")
    parser.add_argument("--dry-run", action="store_true", help="use deterministic synthetic runner output")
    parser.add_argument("--out-dir", default="~/Lethean/data/experiments/substrate-shift")
    parser.add_argument("--seed-root", default="/Volumes/Data/lem/training/seeds")
    parser.add_argument("--subjects", help="comma-separated subject list")
    parser.add_argument("--probes", help="comma-separated probe list")
    parser.add_argument("--conditions", help="comma-separated condition list")
    parser.add_argument("--seeds", help="comma-separated seed list")
    parser.add_argument("--turns", type=int, default=TURNS_PER_RUN)
    parser.add_argument("--model", default="gemma4-e2b-it-q4")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=768)
    parser.add_argument("--context-tokens", type=int, default=65536)
    parser.add_argument("--prompt-chunk-tokens", type=int, default=4096)
    parser.add_argument("--prefill-ms", type=float, default=9000.0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        return run_capture(args)
    except (OSError, RuntimeError, ValueError, FileExistsError, subprocess.SubprocessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
