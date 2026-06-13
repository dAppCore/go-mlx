// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// Attention layer types map to softmax-hybrid; any other token is a recurrent
// mixer kind; out-of-range / nil default to softmax-hybrid.
func TestMixerKindFor_Good(t *testing.T) {
	cfg := &Gemma4TextConfig{LayerTypes: []string{"full_attention", "sliding_attention", "mamba2", ""}}
	want := map[int]string{0: "softmax-hybrid", 1: "softmax-hybrid", 2: "mamba2", 3: "softmax-hybrid"}
	for idx, exp := range want {
		if got := cfg.MixerKindFor(idx); got != exp {
			t.Errorf("MixerKindFor(%d) = %q, want %q", idx, got, exp)
		}
	}
	if got := cfg.MixerKindFor(99); got != "softmax-hybrid" {
		t.Errorf("MixerKindFor(out-of-range) = %q, want softmax-hybrid", got)
	}
	if got := (*Gemma4TextConfig)(nil).MixerKindFor(0); got != "softmax-hybrid" {
		t.Errorf("MixerKindFor on nil cfg = %q, want softmax-hybrid", got)
	}
}

// A registered builder resolves by kind and returns the family's MixerCompute;
// an unregistered kind reports (nil,false) so the loader refuses the pack.
func TestRegisterMixerBuilder_Good(t *testing.T) {
	const kind = "test-recurrent-mixer"
	RegisterMixerBuilder(kind, func(ctx MixerBuildCtx) (metal.MixerCompute, error) {
		return fakeRecurrentMixer{}, nil
	})
	build, ok := mixerBuilderFor(kind)
	if !ok {
		t.Fatal("registered builder did not resolve")
	}
	mixer, err := build(MixerBuildCtx{})
	if err != nil {
		t.Fatalf("builder: %v", err)
	}
	if mixer.Kind() != kind || mixer.State() != scheme.StateRecurrent {
		t.Errorf("built mixer = (%q,%v), want (%q,recurrent)", mixer.Kind(), mixer.State(), kind)
	}
	if _, ok := mixerBuilderFor("no-such-mixer-kind"); ok {
		t.Error("unregistered mixer kind must not resolve")
	}
}

// NewCache builds the StateRecurrent holder for a recurrent-mixer layer and a KV
// cache for a softmax layer — the mixer-owns-state pairing at the cache
// lifecycle. (Layer 0 keeps a full_attention type so it owns a cache slot; its
// recurrent Mixer is what routes it to the holder, not the layer type.)
func TestGemma4_NewCache_RecurrentMixer_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{NumHiddenLayers: 2},
			SlidingWindow:     32,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "full_attention", Mixer: fakeRecurrentMixer{}},
			{LayerType: "full_attention"},
		},
	}
	caches := model.NewCache()
	if len(caches) != 2 {
		t.Fatalf("len(caches) = %d, want 2", len(caches))
	}
	if _, ok := caches[0].(metal.RecurrentCache); !ok {
		t.Errorf("cache[0] = %T, want a RecurrentCache (recurrent mixer)", caches[0])
	}
	if _, ok := caches[1].(*metal.KVCache); !ok {
		t.Errorf("cache[1] = %T, want *KVCache (softmax)", caches[1])
	}
}

// fakeRecurrentMixer is a minimal MixerCompute for the registry test — it does
// no real compute, only satisfies the contract.
type fakeRecurrentMixer struct{}

func (fakeRecurrentMixer) Kind() string            { return "test-recurrent-mixer" }
func (fakeRecurrentMixer) State() scheme.StateKind { return scheme.StateRecurrent }
func (fakeRecurrentMixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	return x, metal.SharedKV{}
}
