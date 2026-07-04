// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// builder_test.go proves the DeltaNet load-time builder composes a working mixer
// from the neutral MixerBuildCtx, including the b_proj write-strength projection
// wrapped as the BetaFn, then a Forward through a real recurrent holder. No model
// load.

const (
	builderHeads  = 2
	builderDim    = 4
	builderDModel = builderHeads * builderDim
)

// builderLinear returns a ctx.Linear resolver: an n×n identity for q/k/v/o, and
// a [H, D_model] rectangular weight for b_proj (which maps the hidden state to
// one write strength per head). MLX matmul is x @ Wᵀ, so a [H, D_model] weight
// turns [B,L,D_model] into [B,L,H].
func builderLinear(n int) func(string) *metal.Linear {
	return func(name string) *metal.Linear {
		if name == "b_proj" {
			vals := make([]float32, builderHeads*n)
			for i := range vals {
				vals[i] = 0.02 + float32(i%5)*0.01
			}
			return metal.NewLinear(metal.FromValues(vals, builderHeads, n), nil)
		}
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

// TestDeltanet_BuildMixer_Good proves the builder resolves a layer (including
// the β projection) and the built mixer runs a 2-token forward through the
// recurrent holder, leaving one state slot.
func TestDeltanet_BuildMixer_Good(t *testing.T) {
	m, err := buildMixer(metal.MixerBuildCtx{Linear: builderLinear(builderDModel), Cfg: builderCfg()})
	if err != nil {
		t.Fatalf("buildMixer: %v", err)
	}
	if m.Kind() != "deltanet" || m.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (deltanet,recurrent)", m.Kind(), m.State())
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

// TestDeltanet_BuildMixer_Bad proves the builder refuses a layer missing the
// write-strength projection or one with no head geometry.
func TestDeltanet_BuildMixer_Bad(t *testing.T) {
	noBeta := func(name string) *metal.Linear {
		if name == "b_proj" {
			return nil
		}
		return builderLinear(builderDModel)(name)
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: noBeta, Cfg: builderCfg()}); err == nil {
		t.Error("builder accepted a layer missing b_proj")
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: builderLinear(builderDModel), Cfg: metal.TransformerConfig{}}); err == nil {
		t.Error("builder accepted a config with no head geometry")
	}
}

// TestDeltanet_BuildMixer_Registered proves init() registered the family.
func TestDeltanet_BuildMixer_Registered(t *testing.T) {
	if _, ok := scheme.MixerFor("deltanet"); !ok {
		t.Fatal("deltanet not registered in scheme catalogue")
	}
}
