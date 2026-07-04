// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package retnet

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// builder_test.go proves the RetNet load-time builder composes a working mixer
// from the neutral MixerBuildCtx: identity-weight projections from a fake
// ctx.Linear, the derived γ schedule, then a Forward through a real recurrent
// holder. No model load — the build context is synthesised in-test.

const (
	builderHeads  = 2
	builderDim    = 4
	builderDModel = builderHeads * builderDim
)

// identityLinearFor returns a ctx.Linear resolver that hands an n×n identity
// Linear for any projection name — passes the hidden state through unchanged so
// the test isolates the builder + recurrence from projection weights.
func identityLinearFor(n int) func(string) *metal.Linear {
	return func(string) *metal.Linear {
		vals := make([]float32, n*n)
		for i := 0; i < n; i++ {
			vals[i*n+i] = 1
		}
		return metal.NewLinear(metal.FromValues(vals, n, n), nil)
	}
}

// builderCfg is the neutral transformer config the builder reads head geometry
// from.
func builderCfg() metal.TransformerConfig {
	return metal.TransformerConfig{
		HiddenSize:        builderDModel,
		NumAttentionHeads: builderHeads,
		HeadDim:           builderDim,
	}
}

// TestRetnet_BuildMixer_Good proves the builder resolves a layer and the built
// mixer runs a 2-token forward through the recurrent holder.
func TestRetnet_BuildMixer_Good(t *testing.T) {
	ctx := metal.MixerBuildCtx{
		Linear: identityLinearFor(builderDModel),
		Cfg:    builderCfg(),
	}
	m, err := buildMixer(ctx)
	if err != nil {
		t.Fatalf("buildMixer: %v", err)
	}
	if m.Kind() != "retnet" || m.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (retnet,recurrent)", m.Kind(), m.State())
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
	// The holder must now carry the advanced retention state (one slot).
	if len(cache.RecurrentState()) != 1 {
		t.Fatalf("holder carries %d state slots, want 1", len(cache.RecurrentState()))
	}
}

// TestRetnet_BuildMixer_Bad proves the builder refuses an incomplete layer
// (missing a projection, or no head geometry) rather than building a broken
// mixer.
func TestRetnet_BuildMixer_Bad(t *testing.T) {
	// Missing a projection: ctx.Linear returns nil for "v_proj".
	missing := func(name string) *metal.Linear {
		if name == "v_proj" {
			return nil
		}
		return identityLinearFor(builderDModel)(name)
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: missing, Cfg: builderCfg()}); err == nil {
		t.Error("builder accepted a layer missing v_proj")
	}

	// No head geometry in the config.
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: identityLinearFor(builderDModel), Cfg: metal.TransformerConfig{}}); err == nil {
		t.Error("builder accepted a config with no head geometry")
	}
}

// TestRetnet_BuildMixer_Registered proves init() registered the loader so the
// family resolves through the scheme catalogue (the loader-registry round-trip
// itself is exercised by the engine's end-to-end load test).
func TestRetnet_BuildMixer_Registered(t *testing.T) {
	if _, ok := scheme.MixerFor("retnet"); !ok {
		t.Fatal("retnet not registered in scheme catalogue")
	}
}
