// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// quant_compute_cover_test.go drives the affine QuantCompute interface methods
// (Bits / Quantize / Matmul) that the package-level Quantize tests bypass. Real
// quantize over a synthetic random weight — no model.

func TestQuantCompute_AffineRoundTrip_Good(t *testing.T) {
	requireMetalRuntime(t)

	qc, ok := QuantComputeFor("affine")
	if !ok {
		t.Fatal("affine quant compute did not resolve")
	}
	if qc.Bits() != 0 {
		t.Errorf("affine Bits() = %d, want 0 (config-declared width)", qc.Bits())
	}
	if qc.Kind() != "affine" {
		t.Errorf("affine Kind() = %q, want affine", qc.Kind())
	}

	const (
		N, K      = 64, 256
		groupSize = 64
		bits      = 4
		M         = 2
	)

	w := RandomNormal(0, 0.1, []int32{N, K}, DTypeFloat32)
	defer Free(w)

	// Quantize through the QuantCompute interface.
	wq, scales, biases, err := qc.Quantize(w, groupSize, bits)
	if err != nil {
		t.Fatalf("affine Quantize: %v", err)
	}
	defer Free(wq, scales, biases)
	if err := Eval(wq, scales, biases); err != nil {
		t.Fatalf("Eval quantize outputs: %v", err)
	}

	// Matmul through the interface consumes the packed triple.
	x := RandomNormal(0, 1, []int32{M, K}, DTypeFloat32)
	defer Free(x)
	yq := qc.Matmul(x, wq, scales, biases, true, groupSize, bits)
	defer Free(yq)
	if err := Eval(yq); err != nil {
		t.Fatalf("Eval affine Matmul: %v", err)
	}
	if got := yq.Shape(); len(got) != 2 || got[0] != M || got[1] != N {
		t.Errorf("affine Matmul shape = %v, want [%d %d]", got, M, N)
	}
}

// QuantComputeFor returns false for an unregistered kind.
func TestQuantCompute_UnknownKind_Bad(t *testing.T) {
	if _, ok := QuantComputeFor("not-a-quant-kind"); ok {
		t.Error("QuantComputeFor resolved an unregistered kind")
	}
}
