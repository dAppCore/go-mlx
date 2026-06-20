// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestAddBF16ComputesResidualBytes(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes([]float32{1, -2, 0.5})
	b := toBF16Bytes([]float32{3, 2, -0.25})
	got, err := AddBF16(a, b)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	want := toBF16Bytes([]float32{4, 0, 0.25})
	if !bytes.Equal(got, want) {
		t.Fatalf("AddBF16 bytes = %v (%v), want %v (%v)", got, bf16Floats(got), want, bf16Floats(want))
	}
}

func TestBF16ShapeContracts(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := AddBF16([]byte{0}, []byte{0}); err == nil {
		t.Fatal("expected AddBF16 to reject odd byte length")
	}
	if _, err := MatVecBF16(toBF16Bytes([]float32{1, 2, 3}), toBF16Bytes([]float32{1, 2}), 2, 2); err == nil {
		t.Fatal("expected MatVecBF16 to reject matrix byte length mismatch")
	}
	if _, err := RoPEDimsBF16(toBF16Bytes([]float32{1, 2, 3, 4}), 1, 1, 4, 3, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPEDimsBF16 to reject odd rotaryDim")
	}
}

func TestBF16IdentityKernels(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes([]float32{1, -2, 3, -4})
	rope, err := RoPEBF16(x, 1, 1, 4, 10000, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	if !bytes.Equal(rope, x) {
		t.Fatalf("RoPEBF16 offset zero changed values: got %v want %v", bf16Floats(rope), bf16Floats(x))
	}

	normInput := toBF16Bytes([]float32{1, 1})
	normWeight := toBF16Bytes([]float32{1, 1})
	norm, err := RMSNormBF16(normInput, normWeight, 1, 2, 0)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	if !bytes.Equal(norm, normInput) {
		t.Fatalf("RMSNormBF16 unit vector = %v, want %v", bf16Floats(norm), bf16Floats(normInput))
	}
}
