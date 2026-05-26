#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# make-pkg-installer.sh — wrap bin/lthn-mlx.app into a signable .pkg installer.
#
# The .pkg places /Applications/lthn-mlx.app and creates a symlink
# /usr/local/bin/lthn-mlx → /Applications/lthn-mlx.app/Contents/MacOS/lthn-mlx
# so `lthn-mlx serve --model ...` works from any terminal after install.
#
# The binary's NSBundle metallib resolution correctly dereferences the symlink
# (via _NSGetExecutablePath, which returns the real path), so the GPU shader
# library at Contents/Resources/mlx.metallib is found from any CWD.
#
# Prerequisites:
#   - bin/lthn-mlx.app  (run: ./scripts/make-app-bundle.sh first)
#
# Optional signing (for distribution):
#   export LTHN_MLX_INSTALLER_IDENTITY='Developer ID Installer: Lethean Ltd (TEAMID)'
#   ./scripts/make-pkg-installer.sh
#   xcrun notarytool submit bin/lthn-mlx.pkg --apple-id ... --team-id TEAMID --wait
#   xcrun stapler staple bin/lthn-mlx.pkg

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$REPO_ROOT/bin/lthn-mlx.app"
PKG="$REPO_ROOT/bin/lthn-mlx.pkg"
VERSION="${LTHN_MLX_VERSION:-0.1.0}"

[[ -d "$APP" ]] || { echo "missing $APP — run: ./scripts/make-app-bundle.sh" >&2; exit 1; }

STAGE="$(mktemp -d -t lthn-mlx-pkg-)"
trap 'rm -rf "$STAGE"' EXIT

# Payload: lthn-mlx.app placed at /Applications/lthn-mlx.app on install.
mkdir -p "$STAGE/payload/Applications"
cp -R "$APP" "$STAGE/payload/Applications/"

# Postinstall: drop a symlink so the CLI is on $PATH after install.
# /usr/local/bin pre-dates Homebrew; macOS PATH includes it by default
# even on fresh installs. The symlink overwrites any prior install.
mkdir -p "$STAGE/scripts"
cat > "$STAGE/scripts/postinstall" <<'EOF'
#!/bin/bash
set -e
mkdir -p /usr/local/bin
rm -f /usr/local/bin/lthn-mlx
ln -s /Applications/lthn-mlx.app/Contents/MacOS/lthn-mlx /usr/local/bin/lthn-mlx
exit 0
EOF
chmod +x "$STAGE/scripts/postinstall"

SIGN_ARGS=()
if [[ -n "${LTHN_MLX_INSTALLER_IDENTITY:-}" ]]; then
    SIGN_ARGS=("--sign" "$LTHN_MLX_INSTALLER_IDENTITY")
fi

pkgbuild \
    --root "$STAGE/payload" \
    --scripts "$STAGE/scripts" \
    --identifier io.lethean.mlx \
    --version "$VERSION" \
    --install-location / \
    "${SIGN_ARGS[@]}" \
    "$PKG"

PKG_SIZE=$(du -h "$PKG" | cut -f1)
echo ""
echo "built $PKG  ($PKG_SIZE)"
echo "  install GUI:  open $PKG"
echo "  install CLI:  sudo installer -pkg $PKG -target /"
echo "  after install, the CLI is on \$PATH at /usr/local/bin/lthn-mlx"

if [[ ${#SIGN_ARGS[@]} -eq 0 ]]; then
    echo ""
    echo "  unsigned. To sign + notarize for distribution:"
    echo "    LTHN_MLX_INSTALLER_IDENTITY='Developer ID Installer: Lethean Ltd (TEAMID)' \\"
    echo "      ./scripts/make-pkg-installer.sh"
    echo "    xcrun notarytool submit $PKG --apple-id ... --team-id TEAMID --wait"
    echo "    xcrun stapler staple $PKG"
fi
