// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestMLPBlockBF16MatchesComposedPrimitives(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 4, 4
	x := toBF16Bytes([]float32{1, -2, 3, -4})
	normW := toBF16Bytes([]float32{1, 1, 1, 1})
	wGate := toBF16Bytes([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	wUp := toBF16Bytes([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	wDown := wUp

	got, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 0)
	if err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}
	normed, err := RMSNormBF16(x, normW, 1, dModel, 0)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	gate, err := MatVecBF16(wGate, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("gate MatVecBF16: %v", err)
	}
	up, err := MatVecBF16(wUp, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("up MatVecBF16: %v", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	down, err := MatVecBF16(wDown, gated, dModel, dFF)
	if err != nil {
		t.Fatalf("down MatVecBF16: %v", err)
	}
	want, err := AddBF16(x, down)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("MLPBlockBF16 = %v, want composed primitives %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestMLPBlockBF16RejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := MLPBlockBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1}), nil, nil, nil, 2, 2, 1e-5); err == nil {
		t.Fatal("expected MLPBlockBF16 to reject x/normWeight shape mismatch")
	}
}
