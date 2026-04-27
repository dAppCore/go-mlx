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
