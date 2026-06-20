// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestMLPBF16PrimitiveKernels(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes([]float32{2, -3, 0.5})
	b := toBF16Bytes([]float32{3, -2, -1})
	mul, err := MulBF16(a, b)
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	wantMul := toBF16Bytes([]float32{6, 6, -0.5})
	if !bytes.Equal(mul, wantMul) {
		t.Fatalf("MulBF16 = %v, want %v", bf16Floats(mul), bf16Floats(wantMul))
	}

	zeros := toBF16Bytes([]float32{0, 0, 0})
	for name, fn := range map[string]func([]byte) ([]byte, error){
		"TanhBF16": TanhBF16,
		"GeluBF16": GeluBF16,
	} {
		got, err := fn(zeros)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Equal(got, zeros) {
			t.Fatalf("%s zeros = %v, want zeros", name, bf16Floats(got))
		}
	}

	gated, err := GeluGateMulBF16(zeros, b)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	assertFloat32Near(t, "GeluGateMulBF16 zero gate", bf16Floats(gated), []float32{0, 0, 0}, 0)
}

func TestGeluGateMulBF16RejectsLengthMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := GeluGateMulBF16(toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{1})); err == nil {
		t.Fatal("expected GeluGateMulBF16 to reject mismatched lengths")
	}
}
