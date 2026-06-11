// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

// Fixed single-token attention kernel tests, relocated from package metal's
// decode_test.go. metal is dot-imported so the relocation keeps the original
// bare metal-symbol usage. These exercise metal.NativeFixedSingleTokenAttention
// (masked + row-update paths) against ScaledDotProductAttention parity.

import (
	"testing"

	. "dappco.re/go/mlx/pkg/metal"
)

func TestDecode_nativeFixedSingleTokenAttentionMasked_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention masked"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keyA := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	valueA := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offsetA := FromValue(0)
	maskA := fixedSingleTokenCausalMaskFromHost(1, 4, 0)
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	maskB := fixedSingleTokenCausalMaskFromHost(1, 4, 1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, maskA, keyB, valueB, offsetB, maskB)

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, maskA, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(masked first) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(masked first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(masked second) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)

	keysValid := Slice(secondKeys, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(secondValues, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(second, secondKeys, secondValues, wantSecond); err != nil {
		t.Fatalf("Eval(masked second) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), wantSecond.Floats())
	floatSliceApprox(t, secondKeys.Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, secondValues.Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}

func TestDecode_nativeFixedSingleTokenAttentionRowUpdate_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention row update"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetFixedAttentionDiagnostics(false, false, true))
	requireMetalRuntime(t)

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keyA := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	valueA := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offsetA := FromValue(0)
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	maskB := fixedSingleTokenCausalMaskFromHost(1, 4, 1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, keyB, valueB, offsetB, maskB)

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(row first) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(row first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	floatSliceApprox(t, firstKeys.Floats(), []float32{1, 0, 0, 0, 0, 0, 0, 0})
	floatSliceApprox(t, firstValues.Floats(), []float32{10, 0, 0, 0, 0, 0, 0, 0})

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(row masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(row masked second) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)

	keysValid := Slice(secondKeys, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(secondValues, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(second, secondKeys, secondValues, wantSecond); err != nil {
		t.Fatalf("Eval(row second) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), wantSecond.Floats())
	floatSliceApprox(t, secondKeys.Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, secondValues.Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}
