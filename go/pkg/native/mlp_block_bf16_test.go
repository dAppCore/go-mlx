// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
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

func TestMLPBlockBF16KeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 8, 16
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))

	if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(normW)]
	_, hasGate := residentBufs[key(wGate)]
	_, hasUp := residentBufs[key(wUp)]
	_, hasDown := residentBufs[key(wDown)]
	residentBufMu.Unlock()

	if !hasNorm || !hasGate || !hasUp || !hasDown {
		t.Fatalf("MLPBlockBF16 did not keep fixed weights resident (norm=%v gate=%v up=%v down=%v resident=%d want>=4)", hasNorm, hasGate, hasUp, hasDown, got)
	}
}
