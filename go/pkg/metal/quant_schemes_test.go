// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// quantRoundTrip quantises a known weight to mode, dequantises it, and returns
// the max abs error vs the original plus the number of quant arrays MLX produced
// (3 for affine: w+scales+biases; 2 for scale-only FP4: w+scales). It is the
// independent numeric oracle for a quant format — the format is correct iff a
// round-trip recovers the weight within the format's resolution. Quantize /
// Dequantize errors propagate so an unsupported mode is reported, not masked.
func quantRoundTrip(mode string, groupSize, bits int) (maxErr float32, nArrays int, err error) {
	in := groupSize * 2 // two groups along the last dim
	data := make([]float32, 4*in)
	for i := range data {
		data[i] = float32(i%13)*0.1 - 0.6 // spread across [-0.6, 0.7]
	}
	w := FromValues(data, 4, in)
	defer Free(w)

	wq, scales, biases, err := Quantize(w, groupSize, bits, mode)
	if err != nil {
		return 0, 0, err
	}
	defer Free(wq, scales, biases)
	nArrays = 1
	if scales != nil {
		nArrays++
	}
	if biases != nil {
		nArrays++
	}

	deq := DequantizeMode(wq, scales, biases, groupSize, bits, mode)
	defer Free(deq)
	Materialize(deq)

	orig := w.Floats()
	got := deq.Floats()
	if len(orig) != len(got) {
		return 0, nArrays, core.NewError("dequantized length mismatch")
	}
	for i := range orig {
		if e := float32(math.Abs(float64(orig[i] - got[i]))); e > maxErr {
			maxErr = e
		}
	}
	return maxErr, nArrays, nil
}

// roundTripTolerance bounds a 4-bit round-trip on the [-0.6,0.7] test weights:
// ~16 levels over a ~1.3 range is a ~0.08 step, so a correct quant+dequant lands
// within a step or so. 0.2 proves real recovery (the weight comes back) without
// over-fitting each format's exact resolution.
const roundTripTolerance = 0.2

// TestQuantScheme_RoundTrip_Good is the numeric oracle: affine and the scale-only
// FP4 formats each quantise a weight and recover it within a 4-bit step. mxfp4 /
// nvfp4 carry no zero-point, so they round-trip from the 2-array (w, scales) form
// — the case the binding used to reject as an error.
func TestQuantScheme_RoundTrip_Good(t *testing.T) {
	for _, tc := range []struct {
		mode       string
		gs, bits   int
		wantArrays int
	}{
		{"affine", 32, 4, 3},
		{"mxfp4", 32, 4, 2},
		{"nvfp4", 16, 4, 2},
	} {
		maxErr, nArrays, err := quantRoundTrip(tc.mode, tc.gs, tc.bits)
		if err != nil {
			t.Errorf("%s: round-trip errored: %v", tc.mode, err)
			continue
		}
		if nArrays != tc.wantArrays {
			t.Errorf("%s: %d quant arrays, want %d", tc.mode, nArrays, tc.wantArrays)
		}
		if maxErr <= 0 {
			t.Errorf("%s: round-trip error %.4f is not positive — dequant looks like a passthrough, not real quantisation", tc.mode, maxErr)
		}
		if maxErr > roundTripTolerance {
			t.Errorf("%s: round-trip max abs err %.4f exceeds tolerance %.2f — weight not recovered", tc.mode, maxErr, roundTripTolerance)
		}
	}
}

// TestQuantScheme_Registered_Good asserts the scale-only FP4 formats are
// first-class in the quant registry (not just served by the generic fallback),
// and the registered loader assembles a Linear carrying that mode.
func TestQuantScheme_Registered_Good(t *testing.T) {
	for _, mode := range []string{"affine", "mxfp4", "nvfp4"} {
		load := lookupQuantLoader(mode)
		if load == nil {
			t.Errorf("%s: no registered quant loader", mode)
			continue
		}
		lin, err := load(QuantTensors{Weight: FromValues([]float32{1}, 1, 1), GroupSize: 32, Bits: 4})
		if err != nil || lin == nil {
			t.Errorf("%s: loader returned (%v, %v)", mode, lin, err)
			continue
		}
		if lin.QuantizationMode != mode {
			t.Errorf("%s: loader built Linear with mode %q", mode, lin.QuantizationMode)
		}
		FreeLinear(lin)
	}
}

// TestQuantScheme_Q40_Unsupported documents the boundary: q4_0 (the GGUF block
// format) is NOT an MLX quantization mode — mlx_quantize rejects it — so it is
// intentionally out of this cut. A real q4_0 would need a GGUF-specific packer,
// not the mlx_quantize path. This test records that so q4_0's absence reads as a
// deliberate boundary, not a forgotten format; flip it when a q4_0 packer lands.
func TestQuantScheme_Q40_Unsupported(t *testing.T) {
	if _, _, err := quantRoundTrip("q4_0", 32, 4); err == nil {
		t.Error("q4_0 round-trip unexpectedly succeeded — MLX gained the mode; wire a real q4_0 loader + flip this test")
	}
	if lookupQuantLoader("q4_0") != nil {
		t.Error("q4_0 has a registered loader but no MLX backing — remove it or back it with a GGUF packer")
	}
}
