// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package qwen3

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

func mkLin(out, in int) *metal.Linear {
	vals := make([]float32, out*in)
	for i := range vals {
		vals[i] = float32((i*3+1)%7-3) * 0.04
	}
	return metal.NewLinear(metal.FromValues(vals, out, in), nil)
}

func mkArr(seed float32, shape ...int) *metal.Array {
	n := 1
	for _, s := range shape {
		n *= s
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = seed + float32(i%5)*0.1
	}
	return metal.FromValues(vals, shape...)
}

// buildGatedDeltaMixer makes a small but real-shaped GatedDeltaNet layer: 2 key /
// 4 value heads (×2 repeat), head dim 4, conv kernel 3, hidden 8.
func buildGatedDeltaMixer() *GatedDeltaMixer {
	cfg := &GatedDeltaConfig{Hidden: 8, KeyHeads: 2, KeyHeadDim: 4, ValueHeads: 4, ValueHeadDim: 4, ConvKernel: 3, RMSNormEps: 1e-6}
	w := &GatedDeltaWeights{
		InProjQKV:  mkLin(int(cfg.convDim()), 8), // [32,8]
		InProjA:    mkLin(4, 8),
		InProjB:    mkLin(4, 8),
		InProjZ:    mkLin(16, 8),
		ConvWeight: mkArr(0.2, 32, 1, 3),
		ConvBias:   nil,
		ALog:       mkArr(-0.5, 4),
		DtBias:     mkArr(0.0, 4),
		Norm:       mkArr(1.0, 4),
		OutProj:    mkLin(8, 16),
	}
	return NewGatedDeltaMixer(w, cfg)
}

// TestQwen3_GroupedHeads_Good proves the key→value head repeat is the consecutive
// repeat_interleave: value head j reads key head j/(VH/KH). Distinct per-head
// input values make the mapping observable.
func TestQwen3_GroupedHeads_Good(t *testing.T) {
	const B, L, KH, D, VH = 1, 1, 2, 4, 4 // rep = 2
	// qFlat [B,L,KH*D] = key head 0 → 1.0, key head 1 → 2.0.
	vals := []float32{1, 1, 1, 1, 2, 2, 2, 2}
	qFlat := metal.FromValues(vals, B, L, KH*D)
	defer metal.Free(qFlat)

	heads := groupedHeads(qFlat, B, L, KH, D, VH) // [B,VH,L,D]
	defer metal.Free(heads)
	if err := metal.Eval(heads); err != nil {
		t.Fatalf("eval: %v", err)
	}
	got := heads.Floats() // [B,VH,L,D], b=l=0 → head·D+d
	want := []float32{1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}
	for i, w := range want {
		if math.Abs(float64(got[i]-w)) > 1e-5 {
			t.Fatalf("value head %d dim %d = %g, want %g (repeat order wrong)", i/D, i%D, got[i], w)
		}
	}
}

// TestQwen3_GatedDeltaMixer_Forward_Good proves the forward produces the right
// shape and finite values end to end.
func TestQwen3_GatedDeltaMixer_Forward_Good(t *testing.T) {
	m := buildGatedDeltaMixer()
	x := mkArr(0.1, 1, 3, 8) // [B,L,hidden]
	defer metal.Free(x)

	out, _ := m.Forward(x, &metal.MixerCtx{Cache: metal.NewRecurrentCache(), B: 1, L: 3})
	if out == nil {
		t.Fatal("Forward returned nil")
	}
	defer metal.Free(out)
	if s := out.Shape(); len(s) != 3 || s[0] != 1 || s[1] != 3 || s[2] != 8 {
		t.Fatalf("output shape %v, want [1,3,8]", s)
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval: %v", err)
	}
	for i, f := range out.Floats() {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("output[%d] is not finite: %g", i, f)
		}
	}
}

// TestQwen3_GatedDeltaMixer_StateThreading_Good is the recurrent-correctness gate:
// a single L=2 prefill must equal two L=1 decode steps that thread the
// [conv-state, delta-state] slots — so the conv ring and the gated-delta memory
// both carry across calls exactly. The streaming identity, independent of any
// numerical reference.
func TestQwen3_GatedDeltaMixer_StateThreading_Good(t *testing.T) {
	m := buildGatedDeltaMixer()
	const dModel = 8

	x2 := mkArr(0.15, 1, 2, dModel)
	defer metal.Free(x2)

	cacheA := metal.NewRecurrentCache()
	outA, _ := m.Forward(x2, &metal.MixerCtx{Cache: cacheA, B: 1, L: 2})
	if outA == nil {
		t.Fatal("prefill Forward returned nil")
	}
	defer metal.Free(outA)
	if err := metal.Eval(outA); err != nil {
		t.Fatalf("eval prefill: %v", err)
	}
	want := outA.Floats() // [1,2,dModel]

	cacheB := metal.NewRecurrentCache()
	x0 := metal.Slice(x2, []int32{0, 0, 0}, []int32{1, 1, dModel})
	x1 := metal.Slice(x2, []int32{0, 1, 0}, []int32{1, 2, dModel})
	defer metal.Free(x0, x1)
	out0, _ := m.Forward(x0, &metal.MixerCtx{Cache: cacheB, B: 1, L: 1})
	out1, _ := m.Forward(x1, &metal.MixerCtx{Cache: cacheB, B: 1, L: 1})
	if out0 == nil || out1 == nil {
		t.Fatal("decode Forward returned nil")
	}
	defer metal.Free(out0, out1)
	got0 := append([]float32(nil), func() []float32 { _ = metal.Eval(out0); return out0.Floats() }()...)
	got1 := func() []float32 { _ = metal.Eval(out1); return out1.Floats() }()

	const tol = 2e-3
	for i := 0; i < dModel; i++ {
		if math.Abs(float64(want[i]-got0[i])) > tol {
			t.Fatalf("token0 mismatch at %d: prefill %g step %g", i, want[i], got0[i])
		}
		if math.Abs(float64(want[dModel+i]-got1[i])) > tol {
			t.Fatalf("token1 mismatch at %d: prefill %g step %g", i, want[dModel+i], got1[i])
		}
	}
}
