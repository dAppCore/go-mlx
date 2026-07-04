// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gla

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// builder_test.go proves the GLA load-time builder composes a working mixer from
// the neutral MixerBuildCtx, including the gk_proj gate wrapped as the GateFn,
// then a Forward through a real recurrent holder. No model load.

const (
	builderHeads  = 2
	builderDim    = 4
	builderDModel = builderHeads * builderDim
)

// identityLinearFor returns a ctx.Linear resolver handing an n×n identity Linear
// for any projection name (q/k/v/o and the gk_proj gate are all D_model square).
func identityLinearFor(n int) func(string) *metal.Linear {
	return func(string) *metal.Linear {
		vals := make([]float32, n*n)
		for i := 0; i < n; i++ {
			vals[i*n+i] = 1
		}
		return metal.NewLinear(metal.FromValues(vals, n, n), nil)
	}
}

func builderCfg() metal.TransformerConfig {
	return metal.TransformerConfig{
		HiddenSize:        builderDModel,
		NumAttentionHeads: builderHeads,
		HeadDim:           builderDim,
	}
}

// TestGla_BuildMixer_Good proves the builder resolves a layer (including the
// gate) and the built mixer runs a 2-token forward through the recurrent holder,
// leaving one state slot.
func TestGla_BuildMixer_Good(t *testing.T) {
	m, err := buildMixer(metal.MixerBuildCtx{Linear: identityLinearFor(builderDModel), Cfg: builderCfg()})
	if err != nil {
		t.Fatalf("buildMixer: %v", err)
	}
	if m.Kind() != "gla" || m.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (gla,recurrent)", m.Kind(), m.State())
	}

	vals := make([]float32, 2*builderDModel)
	for i := range vals {
		vals[i] = 0.1 + float32(i%6)*0.05
	}
	x := metal.FromValues(vals, 1, 2, builderDModel)
	defer metal.Free(x)
	cache := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: cache, B: 1, L: 2})
	if out == nil || !out.Valid() {
		t.Fatal("Forward returned no output")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if got := out.Dim(2); got != builderDModel {
		t.Fatalf("output D_model = %d, want %d", got, builderDModel)
	}
	if len(cache.RecurrentState()) != 1 {
		t.Fatalf("holder carries %d state slots, want 1", len(cache.RecurrentState()))
	}
}

// TestGla_BuildMixer_Bad proves the builder refuses a layer missing the gate
// projection (GLA cannot run without its forget gate) or one with no head
// geometry.
func TestGla_BuildMixer_Bad(t *testing.T) {
	noGate := func(name string) *metal.Linear {
		if name == "gk_proj" {
			return nil
		}
		return identityLinearFor(builderDModel)(name)
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: noGate, Cfg: builderCfg()}); err == nil {
		t.Error("builder accepted a layer missing gk_proj")
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: identityLinearFor(builderDModel), Cfg: metal.TransformerConfig{}}); err == nil {
		t.Error("builder accepted a config with no head geometry")
	}
}

// TestGla_BuildMixer_MissingQKVO proves the builder refuses a layer whose core
// q/k/v/o projections cannot be resolved — the GLA mixer has nothing to project
// through, so it must error rather than build a half-wired mixer. The gk_proj
// gate is present here, so this isolates the q/k/v/o-presence guard from the
// gate guard TestGla_BuildMixer_Bad covers.
func TestGla_BuildMixer_MissingQKVO(t *testing.T) {
	for _, missing := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		resolver := func(name string) *metal.Linear {
			if name == missing {
				return nil
			}
			return identityLinearFor(builderDModel)(name)
		}
		if _, err := buildMixer(metal.MixerBuildCtx{Linear: resolver, Cfg: builderCfg()}); err == nil {
			t.Errorf("builder accepted a layer missing %s", missing)
		}
	}
}

// TestGla_HeadGeometry_DerivesHeadDim proves headGeometry's fallback: when a
// config states NumAttentionHeads + HiddenSize but no explicit HeadDim, the
// per-head width is derived as HiddenSize/NumAttentionHeads. The
// builderCfg-driven tests always set HeadDim explicitly, so this is the only
// caller that exercises the derive branch.
func TestGla_HeadGeometry_DerivesHeadDim(t *testing.T) {
	numHeads, headDim := headGeometry(metal.TransformerConfig{HiddenSize: 8, NumAttentionHeads: 2})
	if numHeads != 2 || headDim != 4 {
		t.Fatalf("headGeometry derived (heads=%d, dim=%d), want (2, 4)", numHeads, headDim)
	}
	// Explicit HeadDim wins over the derived value (no division when stated).
	numHeads, headDim = headGeometry(metal.TransformerConfig{HiddenSize: 8, NumAttentionHeads: 2, HeadDim: 3})
	if numHeads != 2 || headDim != 3 {
		t.Fatalf("headGeometry honoured (heads=%d, dim=%d), want (2, 3)", numHeads, headDim)
	}
}

// TestGla_BuildMixer_Registered proves init() registered the family.
func TestGla_BuildMixer_Registered(t *testing.T) {
	if _, ok := scheme.MixerFor("gla"); !ok {
		t.Fatal("gla not registered in scheme catalogue")
	}
}
