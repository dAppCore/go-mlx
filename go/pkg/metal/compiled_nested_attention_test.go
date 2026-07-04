// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The whole-layer compiled decode step (task #65 increment 3) hinges on two
// mechanisms these tests prove synthetically, with no model load:
//
//  1. Nested compile inlines: a CompileShapeless closure whose body calls the
//     C++-side compiled fixed single-token attention traces the inner graph
//     into the outer trace (mlx compile.cpp runs the plain function when any
//     input is a tracer).
//  2. Position enters as data: RoPEWithOffsetArray and the offset/shift-index
//     cache updates take the token position as ARRAY inputs, so a replayed
//     trace computes the right rotation and cache write for every token —
//     nothing position-shaped freezes into the trace.

type nestedAttentionFixture struct {
	batch, qHeads, kvHeads, capacity, headDim int
	scale                                     float32
}

func (f nestedAttentionFixture) tensor(shape []int, seed float32) *Array {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = seed + float32(i%13)*0.25 - float32(i%7)*0.125
	}
	return FromValues(values, shape...)
}

func (f nestedAttentionFixture) query(seed float32) *Array {
	return f.tensor([]int{f.batch, f.qHeads, 1, f.headDim}, seed)
}

func (f nestedAttentionFixture) token(seed float32) *Array {
	return f.tensor([]int{f.batch, f.kvHeads, 1, f.headDim}, seed)
}

func (f nestedAttentionFixture) cache(seed float32) *Array {
	return f.tensor([]int{f.batch, f.kvHeads, f.capacity, f.headDim}, seed)
}

func evalFloats(t *testing.T, label string, a *Array) []float32 {
	t.Helper()
	if a == nil || !a.Valid() {
		t.Fatalf("%s: invalid array", label)
	}
	if err := Eval(a); err != nil {
		t.Fatalf("%s: Eval: %v", label, err)
	}
	return a.Floats()
}

// TestNestedCompile_FixedSingleTokenAttention proves the pre-cap decode
// attention step — dynamic-offset RoPE, offset-indexed cache write, causal
// mask, SDPA — traces inside an outer CompileShapeless closure and replays
// correctly across changing offsets and content.
func TestNestedCompile_FixedSingleTokenAttention(t *testing.T) {
	f := nestedAttentionFixture{batch: 1, qHeads: 4, kvHeads: 2, capacity: 16, headDim: 8, scale: 0.125}

	step := func(in []*Array) []*Array {
		q, cacheK, cacheV, k, v, offset := in[0], in[1], in[2], in[3], in[4], in[5]
		qR := RoPEWithOffsetArray(q, f.headDim, false, 10000, 1.0, offset, nil)
		kR := RoPEWithOffsetArray(k, f.headDim, false, 10000, 1.0, offset, nil)
		out, newK, newV, ok, err := NativeFixedSingleTokenAttention(qR, cacheK, cacheV, kR, v, offset, nil, f.scale)
		Free(qR, kR)
		if err != nil {
			panic(err)
		}
		if !ok {
			panic("fixed single-token attention declined synthetic inputs")
		}
		return []*Array{out, newK, newV}
	}

	compiled := CompileShapeless(step, true)
	defer compiled.Free()

	// Two rounds: different offsets AND different content through the same
	// compiled trace. Replay must match the direct (uncompiled) graph each
	// time — a frozen position or stale cache write diverges immediately.
	for round, tc := range []struct {
		offset int
		seed   float32
	}{
		{offset: 3, seed: 1.0},
		{offset: 11, seed: -2.5},
	} {
		q := f.query(tc.seed)
		cacheK := f.cache(tc.seed + 0.5)
		cacheV := f.cache(tc.seed - 0.5)
		k := f.token(tc.seed + 1.5)
		v := f.token(tc.seed - 1.5)
		offset := FromValue(tc.offset)

		inputs := []*Array{q, cacheK, cacheV, k, v, offset}
		want := step(inputs)
		got := compiled.Call(inputs...)
		if len(got) != len(want) {
			t.Fatalf("round %d: compiled returned %d outputs, want %d", round, len(got), len(want))
		}
		for i, label := range []string{"out", "newK", "newV"} {
			floatSliceApprox(t, evalFloats(t, label+" compiled", got[i]), evalFloats(t, label+" direct", want[i]))
		}
		Free(inputs...)
		Free(want...)
		Free(got...)
	}
}

// TestNestedCompile_FixedSlidingSingleTokenAttention proves the post-cap
// regime: the rotate-and-write cache update driven by shift-index and
// last-index ARRAYS traces and replays inside an outer closure.
func TestNestedCompile_FixedSlidingSingleTokenAttention(t *testing.T) {
	f := nestedAttentionFixture{batch: 1, qHeads: 4, kvHeads: 2, capacity: 8, headDim: 8, scale: 0.125}

	step := func(in []*Array) []*Array {
		q, cacheK, cacheV, k, v, shift, last := in[0], in[1], in[2], in[3], in[4], in[5], in[6]
		out, newK, newV, ok, err := NativeFixedSlidingSingleTokenAttention(q, cacheK, cacheV, k, v, shift, last, f.scale)
		if err != nil {
			panic(err)
		}
		if !ok {
			panic("fixed sliding single-token attention declined synthetic inputs")
		}
		return []*Array{out, newK, newV}
	}

	compiled := CompileShapeless(step, true)
	defer compiled.Free()

	shiftValues := make([]int32, f.capacity)
	for i := range shiftValues {
		shiftValues[i] = int32((i + 1) % f.capacity)
	}

	for round, seed := range []float32{2.0, -3.5} {
		q := f.query(seed)
		cacheK := f.cache(seed + 0.5)
		cacheV := f.cache(seed - 0.5)
		k := f.token(seed + 1.5)
		v := f.token(seed - 1.5)
		shift := FromValues(shiftValues, f.capacity)
		last := FromValue(f.capacity - 1)

		inputs := []*Array{q, cacheK, cacheV, k, v, shift, last}
		want := step(inputs)
		got := compiled.Call(inputs...)
		if len(got) != len(want) {
			t.Fatalf("round %d: compiled returned %d outputs, want %d", round, len(got), len(want))
		}
		for i, label := range []string{"out", "newK", "newV"} {
			floatSliceApprox(t, evalFloats(t, label+" compiled", got[i]), evalFloats(t, label+" direct", want[i]))
		}
		Free(inputs...)
		Free(want...)
		Free(got...)
	}
}
