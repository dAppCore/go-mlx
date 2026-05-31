#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# make-app-bundle.sh — wrap bin/lthn-mlx into an Apple-canonical .app bundle.
#
# Produces:
#   bin/lthn-mlx.app/
#     Contents/
#       Info.plist
#       MacOS/lthn-mlx              ← the cgo binary
#       Resources/mlx.metallib      ← the GPU shader library
#
# At runtime the binary's NSBundle resolution finds the metallib via the
# Apple-canonical Resources/ path — no env vars, no path walking, no
# operator-side metallib shipping.
#
# Prerequisites (script aborts if any missing):
#   - bin/lthn-mlx          (run: cd go && go build -trimpath -ldflags "-extldflags=-mmacosx-version-min=26.0" -o ../bin/lthn-mlx ./cmd/mlx)
#   - dist/lib/mlx.metallib (run: cmake --build build --target install)
#
# Signing + notarisation are out of scope; this script produces the bundle
# structure that the signing pipeline can consume. Typical follow-up:
#   codesign --deep --sign "Developer ID Application: Lethean Ltd (TEAMID)" bin/lthn-mlx.app
#   xcrun notarytool submit bin/lthn-mlx.app.zip --apple-id ... --wait

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO_ROOT/bin/lthn-mlx"
METALLIB="$REPO_ROOT/dist/lib/mlx.metallib"
APP="$REPO_ROOT/bin/lthn-mlx.app"

[[ -f "$BIN" ]] || { echo "missing $BIN — build first with: cd go && go build -trimpath -ldflags \"-extldflags=-mmacosx-version-min=26.0\" -o ../bin/lthn-mlx ./cmd/mlx" >&2; exit 1; }
[[ -f "$METALLIB" ]] || { echo "missing $METALLIB — build first with: cmake --build build --target install" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/lthn-mlx"
cp "$METALLIB" "$APP/Contents/Resources/mlx.metallib"

VERSION="${LTHN_MLX_VERSION:-0.1.0}"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>lthn-mlx</string>
    <key>CFBundleIdentifier</key><string>io.lethean.mlx</string>
    <key>CFBundleName</key><string>lthn-mlx</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>26.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
EOF

BIN_SIZE=$(du -h "$APP/Contents/MacOS/lthn-mlx" | cut -f1)
LIB_SIZE=$(du -h "$APP/Contents/Resources/mlx.metallib" | cut -f1)
TOTAL_SIZE=$(du -sh "$APP" | cut -f1)

echo "built $APP"
echo "  Contents/MacOS/lthn-mlx       $BIN_SIZE"
echo "  Contents/Resources/mlx.metallib  $LIB_SIZE"
echo "  total                         $TOTAL_SIZE"
