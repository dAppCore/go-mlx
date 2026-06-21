// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestLayerNormBF16 asserts native.LayerNormBF16 is BYTE-IDENTICAL to pkg/metal.LayerNorm over the
// last axis (parity_test.go pattern, eqBytes — not a tolerance). The gemma4 audio subsampler's
// scale-only LayerNorm (after each strided conv) goes through this.
func TestLayerNormBF16(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 20, 64
	eps := float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(rows*ax, 3))
	w := toBF16Bytes(syntheticFloat32(ax, 5))
	b := toBF16Bytes(syntheticFloat32(ax, 7))

	got, err := LayerNormBF16(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16: %v", err)
	}
	r := mc.AsType(mc.LayerNorm(marr(x, rows, ax), marr(w, ax), marr(b, ax), eps), mc.DTypeBFloat16)
	mc.Materialize(r)
	eqBytes(t, "LayerNormBF16 vs metal.LayerNorm", got, append([]byte(nil), r.RawBytes()...))
}
