#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2

set -euo pipefail

manifest="docs/runtime/2026-05-20-production-benchmark-manifest.json"

root="$(git rev-parse --show-toplevel)"
cd "$root"

if [[ ! -s "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

if ! git ls-files --error-unmatch "$manifest" >/dev/null 2>&1; then
  echo "manifest is not tracked by git: $manifest" >&2
  exit 1
fi

python3 - "$manifest" <<'PY'
import json
import os
import subprocess
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

index_path = manifest.get("canonical_index", "")
if not index_path:
    raise SystemExit("manifest is missing canonical_index")
if not os.path.exists(index_path):
    raise SystemExit(f"missing canonical index: {index_path}")

with open(index_path, "r", encoding="utf-8") as handle:
    index_text = handle.read()

seen = set()
failures = []
json_count = 0
for entry in manifest.get("artifacts", []):
    path = entry.get("path", "")
    kind = entry.get("kind", "")
    identifier = entry.get("id", path)
    if not path:
        failures.append(f"{identifier}: missing path")
        continue
    if path in seen:
        failures.append(f"{identifier}: duplicate path {path}")
    seen.add(path)
    if not os.path.exists(path):
        failures.append(f"{identifier}: missing file {path}")
        continue
    if os.path.getsize(path) == 0:
        failures.append(f"{identifier}: empty file {path}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        failures.append(f"{identifier}: file is not tracked by git: {path}")
    if entry.get("indexed", False) and path not in index_text:
        failures.append(f"{identifier}: path is not referenced by {index_path}")
    if kind == "json":
        json_count += 1
        try:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            failures.append(f"{identifier}: invalid json {path}: {exc}")

if failures:
    print("production benchmark manifest verification failed:", file=sys.stderr)
    for failure in failures:
        print(f" - {failure}", file=sys.stderr)
    raise SystemExit(1)

print(
    f"verified {len(seen)} production benchmark artefacts "
    f"({json_count} json) against {manifest_path}"
)
PY

runtime_status="$(git status --short -- docs/runtime || true)"
if [[ -n "$runtime_status" ]]; then
  runtime_status_count="$(printf '%s\n' "$runtime_status" | wc -l | tr -d ' ')"
  echo "note: docs/runtime still has ${runtime_status_count} non-manifest working-tree changes"
  printf '%s\n' "$runtime_status" | sed -n '1,25p'
  if [[ "$runtime_status_count" -gt 25 ]]; then
    echo "... ${runtime_status_count} total; prune or quarantine in a separate cleanup pass"
  fi
fi
