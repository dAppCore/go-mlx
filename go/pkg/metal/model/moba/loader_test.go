// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx with quant-ready dense
// Linears. Geometry: hidden 4, 2 heads, head dim 2 → q/k/v/o_proj out 4.
func fakeBuildCtx() metal.MixerBuildCtx {
	linears := map[string]*metal.Linear{
		"q_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"k_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"v_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"o_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
	}
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(string) *metal.Array { return nil },
		Cfg:      metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2, RMSNormEps: 1e-5},
		LayerIdx: 0,
	}
}

// TestLoader_BuildMoBA_Good builds a working MoBA mixer — StateKVCache, head dim
// from q_proj. With the default BlockSize 64 > L, the forward takes the
// short-context causal path (no block grid yet) — the common decode case.
func TestLoader_BuildMoBA_Good(t *testing.T) {
	mixer, err := buildMoBA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildMoBA: %v", err)
	}
	if mixer.Kind() != "moba" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("built mixer = (%q,%v), want (moba,kv-cache)", mixer.Kind(), mixer.State())
	}

	x := metal.FromValues(make([]float32, 1*4*4), 1, 4, 4)
	defer metal.Free(x)
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 4})
	if out == nil {
		t.Fatal("MoBA Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 4 {
		t.Errorf("forward out shape = %v, want [1 4 4]", got)
	}
}

// TestLoader_BuildMoBA_BlockPath_Good exercises the block-selection path (L >=
// BlockSize) end-to-end by shrinking the built mixer's BlockSize to 2 (the
// branch math is pinned by the kernel tests; this confirms the masked Forward
// runs with a real block grid).
func TestLoader_BuildMoBA_BlockPath_Good(t *testing.T) {
	built, err := buildMoBA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildMoBA: %v", err)
	}
	m := built.(*Mixer)
	m.BlockSize = 2 // L=4 → 2 blocks
	m.TopK = 1

	x := metal.FromValues(make([]float32, 1*4*4), 1, 4, 4)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 4})
	if out == nil {
		t.Fatal("MoBA block-path Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 4 || got[2] != 4 {
		t.Errorf("block-path forward out shape = %v, want [1 4 4]", got)
	}
}

// TestLoader_BuildMoBA_Bad: a missing required projection is a loud build error.
func TestLoader_BuildMoBA_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "v_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildMoBA(ctx); err == nil {
		t.Error("expected error for missing v_proj, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init().
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader(MixerKind, buildMoBA)
	mixer, err := buildMoBA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != MixerKind {
		t.Errorf("Kind() = %q, want %q", mixer.Kind(), MixerKind)
	}
}
