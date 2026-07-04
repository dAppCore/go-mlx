// SPDX-Licence-Identifier: EUPL-1.2

//go:build embed_metallib

// Self-contained metallib: under -tags embed_metallib the shipping build
// bakes the (gzipped) GPU shader library into the binary, so lthn-mlx runs
// from any path with no external mlx.metallib to ship or resolve. Without
// the tag (plain `go build` / `go test`) this file is excluded and MLX
// resolves the metallib externally (colocated / MLX_METALLIB_PATH) as before
// — which keeps the 125MB artifact out of routine dev + CI builds.
//
// The build step gzips dist/lib/mlx.metallib into mlx.metallib.gz next to
// this file before compiling (see Taskfile build:lthn). At process start we
// gunzip it once to a content-addressed cache path and point MLX at it via
// the MLX_METALLIB_PATH hook (lib/mlx device.cpp load_default_library) before
// any Metal device init.
package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
)

//go:embed mlx.metallib.gz
var metallibGz []byte

// init extracts the embedded metallib and sets MLX_METALLIB_PATH before main.
// Best-effort: any failure leaves the env unset so MLX falls back to its
// normal external resolution rather than crashing the process at import time.
func init() {
	// An operator's explicit MLX_METALLIB_PATH outranks the embedded copy —
	// never clobber it (the same set-if-unset contract metal.Init applies to
	// its own resolution).
	if os.Getenv("MLX_METALLIB_PATH") != "" {
		return
	}
	if len(metallibGz) == 0 {
		return
	}
	sum := sha256.Sum256(metallibGz)
	dir := filepath.Join(os.TempDir(), "lthn-mlx", hex.EncodeToString(sum[:8]))
	dst := filepath.Join(dir, "mlx.metallib")

	// Already extracted (content-addressed dir → safe to trust a present file).
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		_ = os.Setenv("MLX_METALLIB_PATH", dst)
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}

	gz, err := gzip.NewReader(bytes.NewReader(metallibGz))
	if err != nil {
		return
	}
	defer func() { _ = gz.Close() }()

	// Write to a temp sibling then rename so a concurrent start never sees a
	// half-written metallib at dst.
	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return
	}
	if _, err := io.Copy(f, gz); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return
	}
	_ = os.Setenv("MLX_METALLIB_PATH", dst)
}
