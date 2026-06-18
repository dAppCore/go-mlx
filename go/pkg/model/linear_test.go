// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/mlx/pkg/safetensors"
)

// TestLoadLinear_QuantAgnostic is the R2 proof: one load path, the format decided per weight
// by .scales, the affine width read from the tensor shapes — so bf16 / 4-bit / 8-bit (and a
// weight one quant leaves bf16 while another quantises) all load with no per-weight branch.
func TestLoadLinear_QuantAgnostic(t *testing.T) {
	const out, in = 4, 64
	mk := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return safetensors.Tensor{Shape: shape, Data: make([]byte, n)} // bytes irrelevant to geometry
	}
	cases := []struct {
		name             string
		t                map[string]safetensors.Tensor
		wantQuant        bool
		wantGS, wantBits int
	}{
		{
			name: "dense bf16 (no .scales)",
			t:    map[string]safetensors.Tensor{"w.weight": mk(out, in)},
		},
		{
			name:      "4-bit affine, group 32",
			t:         map[string]safetensors.Tensor{"w.weight": mk(out, in*4/32), "w.scales": mk(out, in/32), "w.biases": mk(out, in/32)},
			wantQuant: true, wantGS: 32, wantBits: 4,
		},
		{
			name:      "8-bit affine, group 64",
			t:         map[string]safetensors.Tensor{"w.weight": mk(out, in*8/32), "w.scales": mk(out, in/64), "w.biases": mk(out, in/64)},
			wantQuant: true, wantGS: 64, wantBits: 8,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			l := LoadLinear(c.t, "w", in, "affine")
			if l == nil {
				t.Fatal("LoadLinear returned nil for a present weight")
			}
			if l.OutDim != out {
				t.Fatalf("OutDim=%d derived from shape, want %d", l.OutDim, out)
			}
			if l.Quantised() != c.wantQuant {
				t.Fatalf("Quantised()=%v want %v", l.Quantised(), c.wantQuant)
			}
			if c.wantQuant && (l.GroupSize != c.wantGS || l.Bits != c.wantBits) {
				t.Fatalf("geometry gs=%d bits=%d, want gs=%d bits=%d", l.GroupSize, l.Bits, c.wantGS, c.wantBits)
			}
			if !c.wantQuant && l.Kind != "" {
				t.Fatalf("dense weight got Kind=%q, want empty", l.Kind)
			}
		})
	}
}

// TestLoadLinear_AbsentReturnsNil — an optional weight that isn't in the checkpoint loads as
// nil (the caller treats nil as "feature absent"), never a zero-value mistaken for present.
func TestLoadLinear_AbsentReturnsNil(t *testing.T) {
	if l := LoadLinear(map[string]safetensors.Tensor{}, "missing", 64, "affine"); l != nil {
		t.Fatalf("absent weight should return nil, got %+v", l)
	}
}
