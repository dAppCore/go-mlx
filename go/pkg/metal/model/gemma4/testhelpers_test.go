// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Test helpers ported from package metal's test suite alongside the Gemma 4
// architecture tests they support (expert-id matvec, native/compiled decode).
// They are duplicated rather than moved because the originals still serve
// non-gemma4 tests in package metal; both copies build the same q4 fixtures and
// float comparisons on the public metal surface.

// float32Fill returns a slice of n copies of value.
func float32Fill(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

// assertFloat32SliceClose fails when got and want differ in length or by more
// than epsilon at any index.
func assertFloat32SliceClose(t *testing.T, got, want []float32, epsilon float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// packMLXAffineQ4TestRows packs a multiple-of-8 slice of 4-bit values into the
// uint32 little-nibble layout MLX expects.
func packMLXAffineQ4TestRows(t *testing.T, values []uint8) []uint32 {
	t.Helper()
	if len(values)%8 != 0 {
		t.Fatalf("q4 test rows must have a multiple of 8 values, got %d", len(values))
	}
	packed := make([]uint32, len(values)/8)
	for i, value := range values {
		if value > 15 {
			t.Fatalf("q4 value %d exceeds 15", value)
		}
		packed[i/8] |= uint32(value) << uint((i%8)*4)
	}
	return packed
}

// quantizedSwitchLinearExpertIDTest builds a deterministic q4 affine
// SwitchLinear fixture for expert-id matvec tests.
func quantizedSwitchLinearExpertIDTest(t *testing.T, experts, outDim, inDim, groupSize, bits, seed int) *metal.SwitchLinear {
	t.Helper()
	if bits != 4 {
		t.Fatalf("test helper currently packs q4 only, got bits=%d", bits)
	}
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*seed + 5) & 15)
	}
	groups := inDim / groupSize
	scales := make([]float32, experts*outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.025 * float32((i%9)+1)
		biases[i] = -0.45 + 0.05*float32((i+seed)%17)
	}
	return metal.NewQuantizedSwitchLinear(
		metal.FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits)),
		metal.FromValues(scales, experts, outDim, groups),
		metal.FromValues(biases, experts, outDim, groups),
		nil,
		groupSize,
		bits,
	)
}

// quantizedSwitchLinearSidecarsAsType converts a SwitchLinear's scale/bias
// sidecars to dtype in place.
func quantizedSwitchLinearSidecarsAsType(linear *metal.SwitchLinear, dtype metal.DType) {
	if linear == nil || linear.Scales == nil || linear.Biases == nil {
		return
	}
	scales := metal.AsType(linear.Scales, dtype)
	biases := metal.AsType(linear.Biases, dtype)
	metal.Free(linear.Scales, linear.Biases)
	linear.Scales = scales
	linear.Biases = biases
}
