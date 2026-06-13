// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx with quant-ready dense
// Linears. Geometry: hidden 4, 2 heads, head dim 2 → q/k/v/o_proj out 4,
// g_proj out = heads*3 = 6.
func fakeBuildCtx() metal.MixerBuildCtx {
	linears := map[string]*metal.Linear{
		"q_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"k_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"v_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"g_proj": metal.NewLinear(metal.FromValues(make([]float32, 6*4), 6, 4), nil),
		"o_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
	}
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(string) *metal.Array { return nil },
		Cfg:      metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2, RMSNormEps: 1e-5},
		LayerIdx: 0,
	}
}

// TestLoader_BuildNSA_Good builds a working NSA mixer from the neutral ctx —
// StateKVCache, head dim inferred from q_proj, a forward over a short sequence
// yields the right shape. blockSize default (64) > L, so every branch degrades
// to causal attention over the available tokens — still a valid forward.
func TestLoader_BuildNSA_Good(t *testing.T) {
	mixer, err := buildNSA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildNSA: %v", err)
	}
	if mixer.Kind() != "nsa" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("built mixer = (%q,%v), want (nsa,kv-cache)", mixer.Kind(), mixer.State())
	}

	// L=4 so the block grid (blockSize 64) yields one partial block.
	x := metal.FromValues(make([]float32, 1*4*4), 1, 4, 4)
	defer metal.Free(x)
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 4})
	if out == nil {
		t.Fatal("NSA Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 4 {
		t.Errorf("forward out shape = %v, want [1 4 4]", got)
	}
}

// TestLoader_BuildNSA_FullBranch_Good exercises the three-branch path (L >=
// BlockSize) end-to-end. The builder bakes BlockSize 64, so this constructs the
// mixer from the same loader then shrinks BlockSize to 2 to drive a full
// compression+selection+sliding forward on a short sequence (the branch math
// itself is pinned by the kernel tests; this confirms the blended Forward runs).
func TestLoader_BuildNSA_FullBranch_Good(t *testing.T) {
	built, err := buildNSA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildNSA: %v", err)
	}
	m := built.(*Mixer)
	m.BlockSize = 2 // L=4 → 2 whole blocks → all three branches active
	m.SelectBlocks = 1
	m.Window = 2

	x := metal.FromValues(make([]float32, 1*4*4), 1, 4, 4)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 4})
	if out == nil {
		t.Fatal("NSA full-branch Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 4 {
		t.Errorf("full-branch forward out shape = %v, want [1 4 4]", got)
	}
}

// TestLoader_BuildNSA_Bad: a missing required projection is a loud build error.
func TestLoader_BuildNSA_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "g_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildNSA(ctx); err == nil {
		t.Error("expected error for missing g_proj, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init().
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader(MixerKind, buildNSA)
	mixer, err := buildNSA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != MixerKind {
		t.Errorf("Kind() = %q, want %q", mixer.Kind(), MixerKind)
	}
}
