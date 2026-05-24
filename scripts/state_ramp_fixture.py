#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2

"""Build retained-State append fixtures from noisy opencode material.

The production state-ramp lane needs the first prompt to hold the large project
context, then each append section should represent the next user turn. Older
diagnostic files mixed the user request and raw truncated GOAL.md fragments in
one user message, which made Gemma 4 validly choose an immediate EOS. This
helper preserves the request stream while making that fixture transformation
explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DELIMITER = "---TURN---"
USER_TURN_RE = re.compile(r"^user\s+turn\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class SectionMeta:
    index: int
    source_bytes: int
    output_bytes: int
    dropped_bytes: int
    extracted_request: bool
    request: str


def split_sections(text: str, delimiter: str) -> list[str]:
    return [section.strip() for section in text.split(delimiter) if section.strip()]


def extract_request(section: str) -> tuple[str, bool]:
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = USER_TURN_RE.match(line)
        if match:
            request = match.group(2).strip()
            return request or line, True
        return line, False
    return "", False


def build_request_only_fixture(sections: list[str]) -> tuple[list[str], list[SectionMeta]]:
    output: list[str] = []
    meta: list[SectionMeta] = []
    for index, section in enumerate(sections, start=1):
        request, extracted = extract_request(section)
        if not request:
            continue
        output.append(request)
        source_bytes = len(section.encode("utf-8"))
        output_bytes = len(request.encode("utf-8"))
        meta.append(
            SectionMeta(
                index=index,
                source_bytes=source_bytes,
                output_bytes=output_bytes,
                dropped_bytes=max(0, source_bytes - output_bytes),
                extracted_request=extracted,
                request=request,
            )
        )
    return output, meta


def write_delimited(path: Path, sections: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n" + delimiter + "\n").join(sections) + "\n", encoding="utf-8")


def write_meta(path: Path, source: Path, output: Path, delimiter: str, sections: list[SectionMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_source = sum(section.source_bytes for section in sections)
    total_output = sum(section.output_bytes for section in sections)
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "mode": "request-only",
                "delimiter": delimiter,
                "sections": [asdict(section) for section in sections],
                "section_count": len(sections),
                "source_bytes": total_source,
                "output_bytes": total_output,
                "dropped_bytes": max(0, total_source - total_output),
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
    args = parser.parse_args()

    text = args.append_file.read_text(encoding="utf-8")
    sections = split_sections(text, args.delimiter)
    output, meta = build_request_only_fixture(sections)
    if not output:
        raise SystemExit(f"{args.append_file}: no usable turn requests found")
    write_delimited(args.output_file, output, args.delimiter)
    if args.meta_file is not None:
        write_meta(args.meta_file, args.append_file, args.output_file, args.delimiter, meta)
    print(
        json.dumps(
            {
                "sections": len(output),
                "output": str(args.output_file),
                "meta": str(args.meta_file) if args.meta_file else "",
                "source_bytes": sum(section.source_bytes for section in meta),
                "output_bytes": sum(section.output_bytes for section in meta),
                "dropped_bytes": sum(section.dropped_bytes for section in meta),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
