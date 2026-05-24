#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

"""Build retained-State append fixtures from noisy opencode material.

The production state-ramp lane needs the first prompt to hold the large project
context, then each append section should represent the next user turn. Older
diagnostic files mixed the user request and raw truncated GOAL.md fragments in
one user message, which made Gemma 4 validly choose an immediate EOS. This
helper makes the fixture transformation explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DELIMITER = "---TURN---"
DEFAULT_CONTEXT_BYTES = 4096
USER_TURN_RE = re.compile(r"^user\s+turn\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class SectionMeta:
    index: int
    source_bytes: int
    output_bytes: int
    dropped_bytes: int
    extracted_request: bool
    context_bytes: int
    context_excerpt_bytes: int
    context_truncated: bool
    request: str


def split_sections(text: str, delimiter: str) -> list[str]:
    return [section.strip() for section in text.split(delimiter) if section.strip()]


def extract_request(section: str) -> tuple[str, bool, str]:
    lines = section.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        match = USER_TURN_RE.match(line)
        body = "\n".join(lines[idx+1:]).strip()
        if match:
            request = match.group(2).strip()
            return request or line, True, body
        return line, False, body
    return "", False, ""


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes <= 0:
        return "", text.strip() != ""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    return raw[:max_bytes].decode("utf-8", errors="ignore").rstrip(), True


def build_turn(request: str, context: str, mode: str, context_bytes: int) -> tuple[str, int, bool]:
    if mode == "request-only" or not context.strip():
        return request, 0, False
    excerpt, truncated = truncate_utf8(context, context_bytes)
    if not excerpt:
        return request, 0, truncated
    turn = (
        "User request:\n"
        f"{request}\n\n"
        "Context excerpts from this same turn:\n"
        f"{excerpt}\n\n"
        "Answer the user request using the retained state and the context excerpts above. "
        "Do not continue, imitate, or summarise the excerpts unless the request asks for that."
    )
    return turn, len(excerpt.encode("utf-8")), truncated


def build_fixture(sections: list[str], mode: str, context_bytes: int) -> tuple[list[str], list[SectionMeta]]:
    output: list[str] = []
    meta: list[SectionMeta] = []
    for index, section in enumerate(sections, start=1):
        request, extracted, context = extract_request(section)
        if not request:
            continue
        turn, context_excerpt_bytes, context_truncated = build_turn(request, context, mode, context_bytes)
        output.append(turn)
        source_bytes = len(section.encode("utf-8"))
        output_bytes = len(turn.encode("utf-8"))
        meta.append(
            SectionMeta(
                index=index,
                source_bytes=source_bytes,
                output_bytes=output_bytes,
                dropped_bytes=max(0, source_bytes - output_bytes),
                extracted_request=extracted,
                context_bytes=len(context.encode("utf-8")),
                context_excerpt_bytes=context_excerpt_bytes,
                context_truncated=context_truncated,
                request=request,
            )
        )
    return output, meta


def write_delimited(path: Path, sections: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n" + delimiter + "\n").join(sections) + "\n", encoding="utf-8")


def write_meta(path: Path, source: Path, output: Path, delimiter: str, mode: str, context_bytes: int, sections: list[SectionMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_source = sum(section.source_bytes for section in sections)
    total_output = sum(section.output_bytes for section in sections)
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "mode": mode,
                "delimiter": delimiter,
                "context_bytes_limit": context_bytes if mode == "request-context" else 0,
                "sections": [asdict(section) for section in sections],
                "section_count": len(sections),
                "source_bytes": total_source,
                "output_bytes": total_output,
                "dropped_bytes": max(0, total_source - total_output),
                "context_excerpt_bytes": sum(section.context_excerpt_bytes for section in sections),
                "truncated_context_sections": sum(1 for section in sections if section.context_truncated),
                "all_sections_extracted_request": all(section.extracted_request for section in sections),
                "unique_request_count": len({section.request for section in sections}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--meta-file", type=Path, default=None)
    parser.add_argument("--delimiter", default=DEFAULT_DELIMITER)
    parser.add_argument("--mode", choices=("request-only", "request-context"), default="request-only")
    parser.add_argument("--context-bytes", type=int, default=DEFAULT_CONTEXT_BYTES)
    args = parser.parse_args()
    if args.context_bytes < 0:
        parser.error("--context-bytes must be >= 0")

    text = args.append_file.read_text(encoding="utf-8")
    sections = split_sections(text, args.delimiter)
    output, meta = build_fixture(sections, args.mode, args.context_bytes)
    if not output:
        raise SystemExit(f"{args.append_file}: no usable turn requests found")
    write_delimited(args.output_file, output, args.delimiter)
    if args.meta_file is not None:
        write_meta(args.meta_file, args.append_file, args.output_file, args.delimiter, args.mode, args.context_bytes, meta)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "sections": len(output),
                "output": str(args.output_file),
                "meta": str(args.meta_file) if args.meta_file else "",
                "source_bytes": sum(section.source_bytes for section in meta),
                "output_bytes": sum(section.output_bytes for section in meta),
                "dropped_bytes": max(0, sum(section.source_bytes for section in meta) - sum(section.output_bytes for section in meta)),
                "context_excerpt_bytes": sum(section.context_excerpt_bytes for section in meta),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
