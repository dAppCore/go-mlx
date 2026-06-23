// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestConv2dBF16 asserts native.Conv2dBF16 is BYTE-IDENTICAL to pkg/metal.Conv2d (NHWC, groups 1) —
// the parity_test.go pattern (eqBytes). The gemma4 audio subsampler's two strided 3×3 convs go
// through this; the host fp32-accumulate / bf16-round conv matches MLX's, as the LightConv conv1d did.
func TestConv2dBF16(t *testing.T) {
	requireNativeRuntime(t)
	const N, H, W, inC, outC, kh, kw = 1, 8, 8, 2, 4, 3, 3
	in := toBF16Bytes(syntheticFloat32(N*H*W*inC, 3))
	w := toBF16Bytes(syntheticFloat32(outC*kh*kw*inC, 5))

	got, err := Conv2dBF16(in, w, N, H, W, inC, outC, kh, kw, 2, 2, 1, 1)
	if err != nil {
		t.Fatalf("Conv2dBF16: %v", err)
	}
	r := mc.AsType(mc.Conv2d(marr(in, N, H, W, inC), marr(w, outC, kh, kw, inC), 2, 2, 1, 1, 1, 1, 1), mc.DTypeBFloat16)
	mc.Materialize(r)
	eqBytes(t, "Conv2dBF16 vs metal.Conv2d", got, append([]byte(nil), r.RawBytes()...))
}

func TestConv2dF32(t *testing.T) {
	requireNativeRuntime(t)
	const N, H, W, inC, outC, kh, kw = 1, 5, 6, 2, 3, 3, 2
	in := syntheticFloat32(N*H*W*inC, 3)
	w := syntheticFloat32(outC*kh*kw*inC, 5)

	got, err := Conv2dF32(in, w, N, H, W, inC, outC, kh, kw, 2, 1, 1, 0)
	if err != nil {
		t.Fatalf("Conv2dF32: %v", err)
	}
	r := mc.Conv2d(mc.FromValues(in, N, H, W, inC), mc.FromValues(w, outC, kh, kw, inC), 2, 1, 1, 0, 1, 1, 1)
	eqF32(t, "Conv2dF32 vs metal.Conv2d", got, r)
}
