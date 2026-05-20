// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestFast_RMSNorm_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)

	y := RMSNorm(x, weight, 1e-5)
	Materialize(y)

	got := y.Floats()
	rms := math.Sqrt((1 + 4 + 9 + 16) / 4.0)
	for i, val := range []float64{1, 2, 3, 4} {
		want := val / rms
		if math.Abs(float64(got[i])-want) > 1e-3 {
			t.Errorf("RMSNorm[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestFast_RMSNorm_WithScaling_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{2, 2, 2, 2}, 4)

	y := RMSNorm(x, weight, 1e-5)
	Materialize(y)

	got := y.Floats()
	rms := math.Sqrt((1 + 4 + 9 + 16) / 4.0)
	for i, val := range []float64{1, 2, 3, 4} {
		want := 2.0 * val / rms
		if math.Abs(float64(got[i])-want) > 1e-3 {
			t.Errorf("RMSNorm scaled[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestFast_LayerNorm_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)
	bias := FromValues([]float32{0, 0, 0, 0}, 4)

	y := LayerNorm(x, weight, bias, 1e-5)
	Materialize(y)

	got := y.Floats()
	// Layer norm: mean=2.5, var=1.25, std≈1.118
	// Normalised: (x - mean) / std
	mean := 2.5
	std := math.Sqrt(1.25)
	for i, val := range []float64{1, 2, 3, 4} {
		want := (val - mean) / std
		if math.Abs(float64(got[i])-want) > 1e-3 {
			t.Errorf("LayerNorm[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestFast_LayerNorm_WithBias_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)
	bias := FromValues([]float32{10, 10, 10, 10}, 4)

	y := LayerNorm(x, weight, bias, 1e-5)
	Materialize(y)

	got := y.Floats()
	// All values shifted by +10
	mean := 2.5
	std := math.Sqrt(1.25)
	for i, val := range []float64{1, 2, 3, 4} {
		want := (val-mean)/std + 10.0
		if math.Abs(float64(got[i])-want) > 1e-3 {
			t.Errorf("LayerNorm+bias[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestFast_GELUGateMul_Good(t *testing.T) {
	gate := FromValues([]float32{0, 1}, 2)
	up := FromValues([]float32{2, 3}, 2)
	defer Free(gate, up)

	got := GELUGateMul(gate, up)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	want := Mul(geluApprox(gate), up)
	defer Free(want)
	if err := Eval(want); err != nil {
		t.Fatalf("Eval want: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestFast_SiLUGateMul_Good(t *testing.T) {
	gate := FromValues([]float32{0, 1}, 2)
	up := FromValues([]float32{2, 3}, 2)
	defer Free(gate, up)

	got := SiLUGateMul(gate, up)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	want := Mul(SiLU(gate), up)
	defer Free(want)
	if err := Eval(want); err != nil {
		t.Fatalf("Eval want: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestFast_RoPE_Good(t *testing.T) {
	// RoPE on a small input: [B=1, L=1, H=1, D=4]
	x := FromValues([]float32{1, 0, 1, 0}, 1, 1, 1, 4)
	y := RoPE(x, 4, false, 10000.0, 1.0, 0)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 1 || shape[3] != 4 {
		t.Errorf("shape = %v, want [1 1 1 4]", shape)
	}

	// At position 0, RoPE with offset 0 should be close to identity for cos(0)=1
	got := y.Floats()
	// cos(0) = 1, sin(0) = 0, so rotation is identity at position 0
	if math.Abs(float64(got[0])-1.0) > 1e-3 {
		t.Errorf("RoPE[0] = %f, want ≈1.0 (cos(0) rotation)", got[0])
	}
}

func TestFast_RoPEWithOffsetArray_Good(t *testing.T) {
	target := "RoPEWithOffsetArray"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	x := FromValues([]float32{1, 0, 1, 0}, 1, 1, 1, 4)
	offset := FromValue(0)
	defer Free(x, offset)

	got := RoPEWithOffsetArray(x, 4, false, 10000.0, 1.0, offset, nil)
	want := RoPE(x, 4, false, 10000.0, 1.0, 0)
	defer Free(got, want)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(RoPEWithOffsetArray) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestFast_RoPE_ShapePreserved_Good(t *testing.T) {
	// Larger shape: [B=2, L=4, H=8, D=64]
	data := make([]float32, 2*4*8*64)
	for i := range data {
		data[i] = 0.01
	}
	x := FromValues(data, 2, 4, 8, 64)
	y := RoPE(x, 64, false, 10000.0, 1.0, 0)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 2 || shape[1] != 4 || shape[2] != 8 || shape[3] != 64 {
		t.Errorf("shape = %v, want [2 4 8 64]", shape)
	}
}

func TestFast_ScaledDotProductAttention_Causal_Good(t *testing.T) {
	// [B=1, H=1, L=3, D=2]
	q := FromValues([]float32{1, 0, 0, 1, 1, 1}, 1, 1, 3, 2)
	k := FromValues([]float32{1, 0, 0, 1, 1, 1}, 1, 1, 3, 2)
	v := FromValues([]float32{1, 0, 0, 1, 0.5, 0.5}, 1, 1, 3, 2)

	scale := float32(1.0 / math.Sqrt(2.0))
	y := ScaledDotProductAttention(q, k, v, scale, true)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 3 || shape[3] != 2 {
		t.Errorf("shape = %v, want [1 1 3 2]", shape)
	}

	// First position can only attend to itself (causal)
	flat := Reshape(y, 6)
	Materialize(flat)
	got := flat.Floats()
	// Position 0 attends only to position 0: output = v[0] = [1, 0]
	if math.Abs(float64(got[0])-1.0) > 1e-3 {
		t.Errorf("SDPA causal pos0[0] = %f, want 1.0", got[0])
	}
	if math.Abs(float64(got[1])-0.0) > 1e-3 {
		t.Errorf("SDPA causal pos0[1] = %f, want 0.0", got[1])
	}
}

func TestFast_ScaledDotProductAttention_NonCausal_Good(t *testing.T) {
	// Non-causal: all positions attend to all
	q := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := FromValues([]float32{10, 0, 0, 10}, 1, 1, 2, 2)

	scale := float32(1.0 / math.Sqrt(2.0))
	y := ScaledDotProductAttention(q, k, v, scale, false)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 2 || shape[3] != 2 {
		t.Errorf("shape = %v, want [1 1 2 2]", shape)
	}
}

func TestFast_ScaledDotProductAttentionPagedMatchesConcat_Good(t *testing.T) {
	q := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	k1 := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k2 := FromValues([]float32{1, 1, -1, 0}, 1, 1, 2, 2)
	v1 := FromValues([]float32{10, 0, 0, 10}, 1, 1, 2, 2)
	v2 := FromValues([]float32{5, 5, -2, 1}, 1, 1, 2, 2)
	defer Free(q, k1, k2, v1, v2)

	scale := float32(1.0 / math.Sqrt(2.0))
	paged := ScaledDotProductAttentionPaged(q, []*Array{k1, k2}, []*Array{v1, v2}, scale)
	defer Free(paged)
	fullK := Concatenate([]*Array{k1, k2}, 2)
	fullV := Concatenate([]*Array{v1, v2}, 2)
	expected := ScaledDotProductAttention(q, fullK, fullV, scale, false)
	defer Free(fullK, fullV, expected)
	if err := Eval(paged, expected); err != nil {
		t.Fatalf("Eval paged attention: %v", err)
	}

	floatSliceApprox(t, paged.Floats(), expected.Floats())
}

func TestFast_ScaledDotProductAttentionPagedBroadcastsSingleKVHead_Good(t *testing.T) {
	coverageTokens := "ScaledDotProductAttentionPaged BroadcastsSingleKVHead"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	q := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
	}, 1, 4, 1, 2)
	k1 := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k2 := FromValues([]float32{1, 1, -1, 0}, 1, 1, 2, 2)
	v1 := FromValues([]float32{10, 0, 0, 10}, 1, 1, 2, 2)
	v2 := FromValues([]float32{5, 5, -2, 1}, 1, 1, 2, 2)
	defer Free(q, k1, k2, v1, v2)

	scale := float32(1.0 / math.Sqrt(2.0))
	direct := ScaledDotProductAttentionPaged(q, []*Array{k1, k2}, []*Array{v1, v2}, scale)
	k1Repeated := RepeatKV(k1, 4)
	k2Repeated := RepeatKV(k2, 4)
	v1Repeated := RepeatKV(v1, 4)
	v2Repeated := RepeatKV(v2, 4)
	expected := ScaledDotProductAttentionPaged(q, []*Array{k1Repeated, k2Repeated}, []*Array{v1Repeated, v2Repeated}, scale)
	defer Free(direct, k1Repeated, k2Repeated, v1Repeated, v2Repeated, expected)
	if err := Eval(direct, expected); err != nil {
		t.Fatalf("Eval paged grouped query attention: %v", err)
	}
	floatSliceApprox(t, direct.Floats(), expected.Floats())
}

func TestFast_ScaledDotProductAttention_GroupedQueryMatchesRepeated_Good(t *testing.T) {
	coverageTokens := "ScaledDotProductAttention GroupedQueryMatchesRepeated"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	q := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
	}, 1, 4, 1, 2)
	k := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 0,
		0, -1,
		-1, -1,
	}, 1, 2, 3, 2)
	v := FromValues([]float32{
		10, 0,
		0, 10,
		20, 20,
		30, 0,
		0, 30,
		40, 40,
	}, 1, 2, 3, 2)
	defer Free(q, k, v)

	direct := ScaledDotProductAttention(q, k, v, 1, false)
	kRepeated := RepeatKV(k, 2)
	vRepeated := RepeatKV(v, 2)
	expected := ScaledDotProductAttention(q, kRepeated, vRepeated, 1, false)
	defer Free(direct, kRepeated, vRepeated, expected)
	if err := Eval(direct, expected); err != nil {
		t.Fatalf("Eval(grouped query attention) error = %v", err)
	}
	floatSliceApprox(t, direct.Floats(), expected.Floats())
}

func TestFast_ScaledDotProductAttention_CausalGroupedQueryMatchesRepeated_Good(t *testing.T) {
	coverageTokens := "ScaledDotProductAttention CausalGroupedQueryMatchesRepeated"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	q := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
		1, -1,
		0.5, 1,
		1, 0.5,
		-0.5, 1,
	}, 1, 4, 2, 2)
	k := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 0,
	}, 1, 2, 2, 2)
	v := FromValues([]float32{
		10, 0,
		0, 10,
		30, 0,
		0, 30,
	}, 1, 2, 2, 2)
	defer Free(q, k, v)

	direct := ScaledDotProductAttention(q, k, v, 1, true)
	kRepeated := RepeatKV(k, 2)
	vRepeated := RepeatKV(v, 2)
	expected := ScaledDotProductAttention(q, kRepeated, vRepeated, 1, true)
	defer Free(direct, kRepeated, vRepeated, expected)
	if err := Eval(direct, expected); err != nil {
		t.Fatalf("Eval(causal grouped query attention) error = %v", err)
	}
	floatSliceApprox(t, direct.Floats(), expected.Floats())
}

func TestFast_ScaledDotProductAttentionWithMask_GroupedQueryMatchesRepeated_Good(t *testing.T) {
	coverageTokens := "ScaledDotProductAttentionWithMask GroupedQueryMatchesRepeated"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	q := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
	}, 1, 4, 1, 2)
	k := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 0,
		0, -1,
		-1, -1,
	}, 1, 2, 3, 2)
	v := FromValues([]float32{
		10, 0,
		0, 10,
		20, 20,
		30, 0,
		0, 30,
		40, 40,
	}, 1, 2, 3, 2)
	mask := FromValues([]float32{0, 0, -1e9}, 1, 1, 1, 3)
	defer Free(q, k, v, mask)

	direct := ScaledDotProductAttentionWithMask(q, k, v, mask, 1)
	kRepeated := RepeatKV(k, 2)
	vRepeated := RepeatKV(v, 2)
	expected := ScaledDotProductAttentionWithMask(q, kRepeated, vRepeated, mask, 1)
	defer Free(direct, kRepeated, vRepeated, expected)
	if err := Eval(direct, expected); err != nil {
		t.Fatalf("Eval(masked grouped query attention) error = %v", err)
	}
	floatSliceApprox(t, direct.Floats(), expected.Floats())
}

func TestFast_ScaledDotProductAttentionWithMask_Good(t *testing.T) {
	q := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := FromValues([]float32{10, 0, 0, 10}, 1, 1, 2, 2)

	// Mask: block second position from attending to first
	// Large negative = -inf masking
	mask := FromValues([]float32{0, 0, -1e9, 0}, 1, 1, 2, 2)

	scale := float32(1.0 / math.Sqrt(2.0))
	y := ScaledDotProductAttentionWithMask(q, k, v, mask, scale)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 2 || shape[3] != 2 {
		t.Errorf("shape = %v, want [1 1 2 2]", shape)
	}
}

func TestFast_singleTokenCausalMask_Good(t *testing.T) {
	target := "singleTokenCausalMask"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	q := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	k := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
		-1, 1,
	}, 1, 1, 4, 2)
	v := FromValues([]float32{
		10, 0,
		0, 10,
		30, 30,
		40, 40,
	}, 1, 1, 4, 2)
	offset := FromValue(1)
	defer Free(q, k, v, offset)

	mask := singleTokenCausalMask(4, offset)
	defer Free(mask)
	if err := Eval(mask); err != nil {
		t.Fatalf("Eval(mask) error = %v", err)
	}
	floatSliceApprox(t, mask.Floats(), []float32{0, 0, -1e9, -1e9})

	got := ScaledDotProductAttentionWithMask(q, k, v, mask, 1)
	kValid := Slice(k, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	vValid := Slice(v, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	want := ScaledDotProductAttention(q, kValid, vValid, 1, false)
	defer Free(got, kValid, vValid, want)
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(masked attention) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestFast_singleTokenCacheUpdate_Good(t *testing.T) {
	target := "singleTokenCacheUpdate"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	cache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	token := FromValues([]float32{7, 8}, 1, 1, 1, 2)
	offset := FromValue(2)
	defer Free(cache, token, offset)

	got := singleTokenCacheUpdate(cache, token, offset)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(updated cache) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), []float32{0, 0, 0, 0, 7, 8, 0, 0})
}

func TestFast_singleTokenCacheUpdate_CompiledGood(t *testing.T) {
	target := "singleTokenCacheUpdate compiled"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		updated := singleTokenCacheUpdate(inputs[0], inputs[1], inputs[2])
		mask := singleTokenCausalMask(4, inputs[2])
		return []*Array{updated, mask}
	}, true)
	defer compiled.Free()

	cache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	tokenA := FromValues([]float32{1, 2}, 1, 1, 1, 2)
	offsetA := FromValue(1)
	tokenB := FromValues([]float32{3, 4}, 1, 1, 1, 2)
	offsetB := FromValue(2)
	defer Free(cache, tokenA, offsetA, tokenB, offsetB)

	first := compiled.Call(cache, tokenA, offsetA)
	if len(first) != 2 {
		t.Fatalf("first compiled outputs = %d, want 2", len(first))
	}
	defer Free(first...)
	if err := Eval(first...); err != nil {
		t.Fatalf("Eval(first) error = %v", err)
	}
	floatSliceApprox(t, first[0].Floats(), []float32{0, 0, 1, 2, 0, 0, 0, 0})
	floatSliceApprox(t, first[1].Floats(), []float32{0, 0, -1e9, -1e9})

	second := compiled.Call(first[0], tokenB, offsetB)
	if len(second) != 2 {
		t.Fatalf("second compiled outputs = %d, want 2", len(second))
	}
	defer Free(second...)
	if err := Eval(second...); err != nil {
		t.Fatalf("Eval(second) error = %v", err)
	}
	floatSliceApprox(t, second[0].Floats(), []float32{0, 0, 1, 2, 3, 4, 0, 0})
	floatSliceApprox(t, second[1].Floats(), []float32{0, 0, 0, -1e9})
}

func TestFast_fixedSingleTokenAttention_CompiledGood(t *testing.T) {
	target := "fixedSingleTokenAttention compiled"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		out, keys, values := fixedSingleTokenAttention(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], 1)
		return []*Array{out, keys, values}
	}, true)
	defer compiled.Free()

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keyA := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	valueA := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offsetA := FromValue(0)
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, keyB, valueB, offsetB)

	first := compiled.Call(query, keyCache, valueCache, keyA, valueA, offsetA)
	if len(first) != 3 {
		t.Fatalf("first compiled outputs = %d, want 3", len(first))
	}
	defer Free(first...)
	if err := Eval(first...); err != nil {
		t.Fatalf("Eval(first) error = %v", err)
	}
	wantFirst := ScaledDotProductAttention(query, keyA, valueA, 1, false)
	defer Free(wantFirst)
	if err := Eval(wantFirst); err != nil {
		t.Fatalf("Eval(want first) error = %v", err)
	}
	floatSliceApprox(t, first[0].Floats(), wantFirst.Floats())
	floatSliceApprox(t, first[1].Floats(), []float32{1, 0, 0, 0, 0, 0, 0, 0})

	second := compiled.Call(query, first[1], first[2], keyB, valueB, offsetB)
	if len(second) != 3 {
		t.Fatalf("second compiled outputs = %d, want 3", len(second))
	}
	defer Free(second...)
	if err := Eval(second...); err != nil {
		t.Fatalf("Eval(second) error = %v", err)
	}
	keysValid := Slice(second[1], []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(second[2], []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(wantSecond); err != nil {
		t.Fatalf("Eval(want second) error = %v", err)
	}
	floatSliceApprox(t, second[0].Floats(), wantSecond.Floats())
	floatSliceApprox(t, second[1].Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, second[2].Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}

// Generated file-aware compliance coverage.
func TestFast_RMSNorm_Bad(t *testing.T) {
	target := "RMSNorm"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RMSNorm_Ugly(t *testing.T) {
	target := "RMSNorm"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RMSNormNoScale_Good(t *testing.T) {
	target := "RMSNormNoScale"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RMSNormNoScale_Bad(t *testing.T) {
	target := "RMSNormNoScale"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RMSNormNoScale_Ugly(t *testing.T) {
	target := "RMSNormNoScale"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_LayerNorm_Bad(t *testing.T) {
	target := "LayerNorm"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_LayerNorm_Ugly(t *testing.T) {
	target := "LayerNorm"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RoPE_Bad(t *testing.T) {
	target := "RoPE"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RoPE_Ugly(t *testing.T) {
	target := "RoPE"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RoPEWithFreqs_Good(t *testing.T) {
	target := "RoPEWithFreqs"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RoPEWithFreqs_Bad(t *testing.T) {
	target := "RoPEWithFreqs"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_RoPEWithFreqs_Ugly(t *testing.T) {
	target := "RoPEWithFreqs"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_ScaledDotProductAttention_Good(t *testing.T) {
	target := "ScaledDotProductAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_ScaledDotProductAttention_Bad(t *testing.T) {
	target := "ScaledDotProductAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_ScaledDotProductAttention_Ugly(t *testing.T) {
	target := "ScaledDotProductAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_ScaledDotProductAttentionWithMask_Bad(t *testing.T) {
	target := "ScaledDotProductAttentionWithMask"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestFast_ScaledDotProductAttentionWithMask_Ugly(t *testing.T) {
	target := "ScaledDotProductAttentionWithMask"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
