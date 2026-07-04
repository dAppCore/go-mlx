// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package qwen3

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// gatedDeltaCtx builds a neutral MixerBuildCtx over synthetic linear_attn weights
// with real-shaped ratios scaled down: 2 key / 4 value heads, dim 4, conv kernel
// 3, hidden 8 → convDim = 2·8 + 16 = 32.
func gatedDeltaCtx() metal.MixerBuildCtx {
	lin := map[string]*metal.Linear{
		"in_proj_qkv": mkLin(32, 8),
		"in_proj_a":   mkLin(4, 8),
		"in_proj_b":   mkLin(4, 8),
		"in_proj_z":   mkLin(16, 8),
		"out_proj":    mkLin(8, 16),
	}
	wt := map[string]*metal.Array{
		"conv1d.weight": mkArr(0.2, 32, 1, 3),
		"A_log":         mkArr(-0.5, 4),
		"dt_bias":       mkArr(0.0, 4),
		"norm.weight":   mkArr(1.0, 4),
	}
	return metal.MixerBuildCtx{
		Linear: func(p string) *metal.Linear { return lin[p] },
		Weight: func(n string) *metal.Array { return wt[n] },
		Cfg:    metal.TransformerConfig{RMSNormEps: 1e-6},
	}
}

// TestQwen3_BuildGatedDelta_Good proves the loader derives the right geometry from
// the weight shapes alone (no config dims) and produces a working mixer.
func TestQwen3_BuildGatedDelta_Good(t *testing.T) {
	mx, err := buildGatedDelta(gatedDeltaCtx())
	if err != nil {
		t.Fatalf("buildGatedDelta: %v", err)
	}
	gd, ok := mx.(*GatedDeltaMixer)
	if !ok || gd.W == nil || gd.W.cfg == nil {
		t.Fatal("expected a *GatedDeltaMixer with weights + cfg")
	}
	c := gd.W.cfg
	if c.Hidden != 8 || c.KeyHeads != 2 || c.KeyHeadDim != 4 || c.ValueHeads != 4 || c.ValueHeadDim != 4 || c.ConvKernel != 3 {
		t.Fatalf("derived geometry wrong: %+v", *c)
	}
	if c.RMSNormEps != 1e-6 {
		t.Fatalf("eps not threaded from ctx.Cfg: %g", c.RMSNormEps)
	}

	// the loaded mixer runs end to end.
	x := mkArr(0.1, 1, 3, 8)
	defer metal.Free(x)
	out, _ := mx.Forward(x, &metal.MixerCtx{Cache: metal.NewRecurrentCache(), B: 1, L: 3})
	if out == nil {
		t.Fatal("loaded mixer Forward returned nil")
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
			t.Fatalf("output[%d] not finite: %g", i, f)
		}
	}
}

// TestQwen3_BuildGatedDelta_Bad proves a missing required weight is refused, not
// silently mis-built.
func TestQwen3_BuildGatedDelta_Bad(t *testing.T) {
	for _, drop := range []string{"in_proj_qkv", "A_log", "norm.weight", "conv1d.weight"} {
		ctx := gatedDeltaCtx()
		origLin, origWt := ctx.Linear, ctx.Weight
		ctx.Linear = func(p string) *metal.Linear {
			if p == drop {
				return nil
			}
			return origLin(p)
		}
		ctx.Weight = func(n string) *metal.Array {
			if n == drop {
				return nil
			}
			return origWt(n)
		}
		if _, err := buildGatedDelta(ctx); err == nil {
			t.Fatalf("expected an error when %q is missing", drop)
		}
	}
}
