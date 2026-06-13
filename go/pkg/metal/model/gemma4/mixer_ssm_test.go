// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// zeroW makes a named [shape...] weight of zeros for the synthetic checkpoints —
// the builders read shapes (not values) to derive geometry, so zeros exercise
// the resolution path without needing meaningful numerics.
func zeroW(shape ...int) *metal.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return metal.FromValues(make([]float32, n), shape...)
}

// TestBuildMamba2Layer_Good is the #44 builder gate: a synthetic Mamba-2
// checkpoint + a config layer_type "mamba2" must resolve end-to-end —
// MixerKindFor names the kind, the registered builder constructs the mixer from
// the weight map, and the result is a recurrent MixerCompute that runs a forward
// through a holder. The shapes are the self-consistent nGroups=1 geometry the
// builder infers (H=2, headDim=2, N=2 → dInner=4, convDim=8, projOut=14).
func TestBuildMamba2Layer_Good(t *testing.T) {
	const D = 4
	prefix := "model.layers.0"
	weights := map[string]*metal.Array{
		prefix + ".mixer.in_proj.weight":  zeroW(14, D),
		prefix + ".mixer.out_proj.weight": zeroW(D, 4),
		prefix + ".mixer.conv1d.weight":   zeroW(8, 1, 3),
		prefix + ".mixer.A_log":           zeroW(2),
		prefix + ".mixer.D":               zeroW(2),
		prefix + ".mixer.dt_bias":         zeroW(2),
		prefix + ".mixer.norm.weight":     zeroW(4),
	}
	cfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{RMSNormEps: 1e-5}, LayerTypes: []string{"mamba2"}}

	if got := cfg.MixerKindFor(0); got != "mamba2" {
		t.Fatalf("MixerKindFor(0) = %q, want mamba2", got)
	}
	build, ok := mixerBuilderFor("mamba2")
	if !ok {
		t.Fatal("mamba2 builder not registered")
	}
	mixer, err := build(MixerBuildCtx{Weights: weights, Prefix: prefix, Cfg: cfg, LayerIdx: 0})
	if err != nil {
		t.Fatalf("buildMamba2Layer: %v", err)
	}
	if mixer.Kind() != "mamba2" || mixer.State() != scheme.StateRecurrent {
		t.Errorf("built mixer = (%q,%v), want (mamba2,recurrent)", mixer.Kind(), mixer.State())
	}

	// The built mixer runs a forward through a recurrent holder (zeros in → a
	// finite, correctly-shaped output + two advanced state slots).
	x := zeroW(1, 2, D)
	defer metal.Free(x)
	rc := metal.NewRecurrentCache()
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	if out == nil {
		t.Fatal("built mamba2 mixer Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != D {
		t.Errorf("forward out shape = %v, want [1 2 %d]", got, D)
	}
	if len(rc.RecurrentState()) != 2 {
		t.Errorf("holder slots = %d, want 2 (conv-state + ssm-state)", len(rc.RecurrentState()))
	}
}

// TestBuildMamba2Layer_Bad: a missing required weight is a loud build error, not
// a silent zero-weight mixer.
func TestBuildMamba2Layer_Bad(t *testing.T) {
	cfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{RMSNormEps: 1e-5}, LayerTypes: []string{"mamba2"}}
	build, _ := mixerBuilderFor("mamba2")
	// in_proj present but conv1d.weight absent → error.
	weights := map[string]*metal.Array{
		"model.layers.0.mixer.in_proj.weight":  zeroW(14, 4),
		"model.layers.0.mixer.out_proj.weight": zeroW(4, 4),
		"model.layers.0.mixer.A_log":           zeroW(2),
	}
	if _, err := build(MixerBuildCtx{Weights: weights, Prefix: "model.layers.0", Cfg: cfg}); err == nil {
		t.Error("expected error for missing conv1d.weight, got nil")
	}
}

// TestBuildRWKV7Layer_Good is the #44 builder gate for RWKV-7: a synthetic
// checkpoint + layer_type "rwkv7" resolves end-to-end. H comes from
// NumAttentionHeads (2); receptance/value widths (H*K=4, H*V=4) give K=V=2.
func TestBuildRWKV7Layer_Good(t *testing.T) {
	const D = 4
	prefix := "model.layers.1"
	weights := map[string]*metal.Array{
		prefix + ".attention.receptance.weight": zeroW(4, D),
		prefix + ".attention.key.weight":        zeroW(4, D),
		prefix + ".attention.value.weight":      zeroW(4, D),
		prefix + ".attention.output.weight":     zeroW(D, 4),
		prefix + ".attention.decay.weight":      zeroW(4, D),
		prefix + ".attention.a_proj.weight":     zeroW(4, D),
		prefix + ".attention.b_proj.weight":     zeroW(4, D),
	}
	cfg := &Gemma4TextConfig{LayerTypes: []string{"full_attention", "rwkv7"}}
	cfg.NumAttentionHeads = 2

	if got := cfg.MixerKindFor(1); got != "rwkv7" {
		t.Fatalf("MixerKindFor(1) = %q, want rwkv7", got)
	}
	build, ok := mixerBuilderFor("rwkv7")
	if !ok {
		t.Fatal("rwkv7 builder not registered")
	}
	mixer, err := build(MixerBuildCtx{Weights: weights, Prefix: prefix, Cfg: cfg, LayerIdx: 1})
	if err != nil {
		t.Fatalf("buildRWKV7Layer: %v", err)
	}
	if mixer.Kind() != "rwkv7" || mixer.State() != scheme.StateRecurrent {
		t.Errorf("built mixer = (%q,%v), want (rwkv7,recurrent)", mixer.Kind(), mixer.State())
	}

	x := zeroW(1, 2, D)
	defer metal.Free(x)
	rc := metal.NewRecurrentCache()
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	if out == nil {
		t.Fatal("built rwkv7 mixer Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[2] != D {
		t.Errorf("forward out shape = %v, want [.. .. %d]", got, D)
	}
	if len(rc.RecurrentState()) != 1 {
		t.Errorf("holder slots = %d, want 1 (wkv-state)", len(rc.RecurrentState()))
	}
}
