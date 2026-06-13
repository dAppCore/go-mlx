// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Quantize packs a dense weight into the (w, scales, biases) triple, the scales
// and biases are per-group ([N, K/groupSize]), and the triple feeds
// QuantizedMatmul end-to-end against a dense matmul of the same shape — proving
// the binding produces weights the engine's quantized path actually consumes.
func TestQuantizeAffineRoundTrip_Good(t *testing.T) {
	const (
		N, K      = 64, 256
		groupSize = 64
		bits      = 4
		M         = 2
	)

	w := RandomNormal(0, 0.1, []int32{N, K}, DTypeFloat32)
	wq, scales, biases, err := Quantize(w, groupSize, bits, "affine")
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	if err := Eval(wq, scales, biases); err != nil {
		t.Fatalf("Eval quantize outputs: %v", err)
	}

	wantGroups := int32(K / groupSize)
	if got := scales.Shape(); len(got) != 2 || got[0] != N || got[1] != wantGroups {
		t.Errorf("scales shape = %v, want [%d %d]", got, N, wantGroups)
	}
	if got := biases.Shape(); len(got) != 2 || got[0] != N || got[1] != wantGroups {
		t.Errorf("biases shape = %v, want [%d %d]", got, N, wantGroups)
	}

	// Structural round-trip: the packed triple is consumable by QuantizedMatmul
	// and the quantized-vs-dense pipeline evaluates to the same [M, N] shape.
	x := RandomNormal(0, 1, []int32{M, K}, DTypeFloat32)
	yq := QuantizedMatmul(x, wq, scales, biases, true, groupSize, bits) // x · dequant(wq)ᵀ
	yd := Matmul(x, Transpose(w))                                       // x · wᵀ
	if err := Eval(yq, yd); err != nil {
		t.Fatalf("Eval round-trip: %v", err)
	}
	if got := yq.Shape(); len(got) != 2 || got[0] != M || got[1] != N {
		t.Errorf("quantized matmul shape = %v, want [%d %d]", got, M, N)
	}
}
