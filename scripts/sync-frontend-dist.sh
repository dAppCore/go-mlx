#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# sync-frontend-dist.sh — copy the lthn/desktop frontend dist into
# go/cmd/mlx/frontend/dist so the lthn-mlx menubar can embed it via
# go:embed. Single source of truth lives in lthn/desktop; lthn-mlx
# bundles a snapshot at build time.
#
# Run this BEFORE `go build ./cmd/mlx` whenever the frontend has been
# rebuilt or the lthn-mlx menubar surfaces a new ?surface= component
# from the lthn/desktop frontend.
#
# Default sibling layout:
#   ~/Code/core/go-mlx/         (this repo)
#   ~/Code/lthn/desktop/        (frontend source)
#
# Override with: LTHN_DESKTOP_DIST=/path/to/lthn/desktop/frontend/dist \
#                  ./scripts/sync-frontend-dist.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/go/cmd/mlx/frontend/dist"
DEFAULT_SRC="$REPO_ROOT/../../lthn/desktop/frontend/dist"
SRC="${LTHN_DESKTOP_DIST:-$DEFAULT_SRC}"

if [[ ! -d "$SRC" ]]; then
    echo "missing $SRC" >&2
    echo "  expected lthn/desktop checked out as a sibling at ~/Code/lthn/desktop" >&2
    echo "  build the frontend first: cd \$(dirname $SRC) && pnpm build" >&2
    echo "  or override with LTHN_DESKTOP_DIST=/path/to/dist" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"

SIZE=$(du -sh "$DEST" | cut -f1)
COUNT=$(find "$DEST" -type f | wc -l | tr -d ' ')
echo "synced lthn/desktop frontend dist → $DEST"
echo "  size: $SIZE"
echo "  files: $COUNT"
echo "  source: $SRC"
