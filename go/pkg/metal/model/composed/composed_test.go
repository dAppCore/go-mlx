// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package composed

import (
	"math"
	"testing"

	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// Synthetic geometry: hidden 4, 2 heads, 2 kv heads, head dim 2, intermediate 8,
// 2 layers, vocab 6 — small enough to compose and run a forward on the CPU-side
// graph without loading a model (AX-11: no model loads, synthetic tensors only).
const (
	tHidden  = 4
	tHeads   = 2
	tKVHeads = 2
	tHeadDim = 2
	tInter   = 8
	tLayers  = 2
	tVocab   = 6
)

func ramp(rows, cols int32) *metal.Array {
	n := rows * cols
	if cols == 0 { // 1-D vector of length rows (norm weights)
		n = rows
	}
	data := make([]float32, int(n))
	for i := range data {
		data[i] = float32(i%7)*0.01 + 0.01
	}
	if cols == 0 {
		return metal.FromValues(data, int(rows))
	}
	return metal.FromValues(data, int(rows), int(cols))
}

// composedTestConfig is the dense config a composed/hybrid checkpoint declares:
// model_type "composed" with per-layer layer_types. RopeTheta + Scale are the
// family-config fields the softmax mixer reads through MixerBuildCtx.Extra.
func composedTestConfig(layerTypes []string) *metal.DenseConfig {
	return &metal.DenseConfig{
		TransformerConfig: metal.TransformerConfig{
			ModelType:             "composed",
			HiddenSize:            tHidden,
			NumHiddenLayers:       tLayers,
			IntermediateSize:      tInter,
			NumAttentionHeads:     tHeads,
			NumKeyValueHeads:      tKVHeads,
			HeadDim:               tHeadDim,
			VocabSize:             tVocab,
			RMSNormEps:            1e-5,
			MaxPositionEmbeddings: 128,
		},
		RopeTheta:  10000,
		Scale:      float32(1.0 / math.Sqrt(float64(tHeadDim))),
		LayerTypes: layerTypes,
	}
}

// fullAttentionWeights builds a complete synthetic weight map for an n-layer
// full-attention composed model (tied embedding, no lm_head).
func fullAttentionWeights(n int32) map[string]*metal.Array {
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight": ramp(tVocab, tHidden),
		"model.norm.weight":         ramp(tHidden, 0),
	}
	for i := int32(0); i < n; i++ {
		p := "model.layers." + itoa(i)
		weights[p+".input_layernorm.weight"] = ramp(tHidden, 0)
		weights[p+".post_attention_layernorm.weight"] = ramp(tHidden, 0)
		weights[p+".self_attn.q_proj.weight"] = ramp(tHeads*tHeadDim, tHidden)
		weights[p+".self_attn.k_proj.weight"] = ramp(tKVHeads*tHeadDim, tHidden)
		weights[p+".self_attn.v_proj.weight"] = ramp(tKVHeads*tHeadDim, tHidden)
		weights[p+".self_attn.o_proj.weight"] = ramp(tHidden, tHeads*tHeadDim)
		weights[p+".mlp.gate_proj.weight"] = ramp(tInter, tHidden)
		weights[p+".mlp.up_proj.weight"] = ramp(tInter, tHidden)
		weights[p+".mlp.down_proj.weight"] = ramp(tHidden, tInter)
	}
	return weights
}

func itoa(i int32) string {
	if i == 0 {
		return "0"
	}
	var b [12]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

// TestComposed_Build_Good is the wire-level gate: a config that declares two
// full-attention layers composes a working model — the registry resolves the
// softmax mixer, the block stack assembles, and a forward yields [B,L,vocab].
func TestComposed_Build_Good(t *testing.T) {
	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	m, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		t.Fatalf("buildComposed: %v", err)
	}
	defer m.CloseModel()

	if m.NumLayers() != tLayers {
		t.Fatalf("NumLayers = %d, want %d", m.NumLayers(), tLayers)
	}
	for i, layer := range m.Layers {
		if layer.Kind != "full_attention" {
			t.Errorf("layer %d kind = %q, want full_attention", i, layer.Kind)
		}
		if layer.Mixer.State() != scheme.StateKVCache {
			t.Errorf("layer %d state = %v, want kv-cache", i, layer.Mixer.State())
		}
	}

	// Cache layout follows each mixer's declared state: full attention → KV cache.
	caches := m.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) != tLayers {
		t.Fatalf("NewCache len = %d, want %d", len(caches), tLayers)
	}
	for i, c := range caches {
		if _, recurrent := c.(metal.RecurrentCache); recurrent {
			t.Errorf("layer %d cache is recurrent, want KV for full_attention", i)
		}
	}

	// Forward [B=1, L=2] → logits [1, 2, vocab]; materialise to force the graph.
	tokens := metal.FromValues([]int32{1, 2}, 1, 2)
	defer metal.Free(tokens)
	out := m.Forward(tokens, caches)
	defer metal.Free(out)
	metal.Materialize(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != tVocab {
		t.Errorf("forward out shape = %v, want [1 2 %d]", got, tVocab)
	}
}

// TestComposed_Uniform_Good: with no layer_types, the model_type is the uniform
// mixer kind for every layer.
func TestComposed_Uniform_Good(t *testing.T) {
	cfg := composedTestConfig(nil)
	cfg.ModelType = "full_attention" // uniform: every layer is this kind
	m, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		t.Fatalf("uniform buildComposed: %v", err)
	}
	defer m.CloseModel()
	for i, layer := range m.Layers {
		if layer.Kind != "full_attention" {
			t.Errorf("uniform layer %d kind = %q, want full_attention", i, layer.Kind)
		}
	}
}

// TestComposed_UntiedOutput_Good: an lm_head.weight in the checkpoint gives the
// model an OWN output projection rather than the tied embedding (the dense
// default when lm_head is absent). The untied output's weight is a distinct
// array from the embedding's — the property CloseModel relies on to free the
// output separately (a tied output shares the embedding's array and is freed
// once, via the embedding).
func TestComposed_UntiedOutput_Good(t *testing.T) {
	weights := fullAttentionWeights(tLayers)
	weights["lm_head.weight"] = ramp(tVocab, tHidden) // untie the output head
	m, err := buildComposed(composedTestConfig([]string{"full_attention", "full_attention"}), weights, nil)
	if err != nil {
		t.Fatalf("untied-output buildComposed: %v", err)
	}
	if m.Output == nil || m.Output.Weight == nil {
		t.Fatal("untied output projection is nil, want lm_head.weight")
	}
	if m.Output.Weight == m.EmbedTokens.Weight {
		t.Error("untied output shares the embedding array, want a distinct lm_head weight")
	}
	m.CloseModel() // exercises the distinct-output free path (not the tied skip)
}

// TestComposed_MissingWeight_Bad: a missing required projection is a loud build
// error, never a silent nil-deref — the property that makes registering the
// composed loader honest without a checkpoint.
func TestComposed_MissingWeight_Bad(t *testing.T) {
	weights := fullAttentionWeights(tLayers)
	delete(weights, "model.layers.1.self_attn.q_proj.weight")
	if _, err := buildComposed(composedTestConfig([]string{"full_attention", "full_attention"}), weights, nil); err == nil {
		t.Error("expected loud error for missing layer 1 q_proj, got nil")
	}
}

// TestComposed_UnregisteredKind_Bad: a layer kind no family registered is refused
// cleanly at build, not loaded as something else.
func TestComposed_UnregisteredKind_Bad(t *testing.T) {
	if _, err := buildComposed(composedTestConfig([]string{"full_attention", "no_such_mixer"}), fullAttentionWeights(tLayers), nil); err == nil {
		t.Error("expected loud error for unregistered mixer kind, got nil")
	}
}

// TestComposed_LayerTypesMismatch_Bad: layer_types that does not cover every
// layer is ambiguous and refused.
func TestComposed_LayerTypesMismatch_Bad(t *testing.T) {
	if _, err := buildComposed(composedTestConfig([]string{"full_attention"}), fullAttentionWeights(tLayers), nil); err == nil {
		t.Error("expected loud error for layer_types length mismatch, got nil")
	}
}

// TestComposed_AmbiguousModelType_Bad: a "composed" model_type with no
// layer_types cannot say what each layer is, and is refused.
func TestComposed_AmbiguousModelType_Bad(t *testing.T) {
	if _, err := buildComposed(composedTestConfig(nil), fullAttentionWeights(tLayers), nil); err == nil {
		t.Error("expected loud error for ambiguous composed config, got nil")
	}
}

// TestComposed_SoftmaxRegistered_Ugly asserts the keystone the composed path
// depends on — the generic softmax mixer self-registered as "full_attention".
func TestComposed_SoftmaxRegistered_Ugly(t *testing.T) {
	if _, ok := metal.MixerLoaderFor("full_attention"); !ok {
		t.Fatal("full_attention mixer loader not registered — composed hybrid cannot build an attention layer")
	}
}

// fakeRecurrentMixer is a minimal StateRecurrent mixer used only to prove the
// heterogeneous path — it declares the recurrent state kind so NewCache must give
// its layer the recurrent holder, not a KV cache. Forward is never exercised here
// (the heterogeneous test asserts build + cache typing, not numerics).
type fakeRecurrentMixer struct{}

func (fakeRecurrentMixer) Kind() string            { return "fake_recurrent" }
func (fakeRecurrentMixer) State() scheme.StateKind { return scheme.StateRecurrent }
func (fakeRecurrentMixer) Forward(x *metal.Array, _ *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	return x, metal.SharedKV{}
}

// TestComposed_Heterogeneous_Good is the thesis of the cut: a config that mixes a
// softmax layer with a recurrent layer composes BOTH through the registry, and
// NewCache types each layer 1:1 by its mixer's declared State — a KV cache for
// the softmax layer, the recurrent holder for the recurrent one. Without this the
// composed model is only Qwen3 with extra indirection; this proves heterogeneous
// dispatch AND heterogeneous cache typing in one go.
func TestComposed_Heterogeneous_Good(t *testing.T) {
	metal.RegisterMixerLoader("fake_recurrent", func(metal.MixerBuildCtx) (metal.MixerCompute, error) {
		return fakeRecurrentMixer{}, nil
	})

	cfg := composedTestConfig([]string{"full_attention", "fake_recurrent"})
	m, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		t.Fatalf("heterogeneous buildComposed: %v", err)
	}
	defer m.CloseModel()

	if m.Layers[0].Kind == m.Layers[1].Kind {
		t.Fatalf("layers should differ in kind, both %q", m.Layers[0].Kind)
	}
	if m.Layers[0].Mixer.State() != scheme.StateKVCache {
		t.Errorf("layer 0 state = %v, want kv-cache", m.Layers[0].Mixer.State())
	}
	if m.Layers[1].Mixer.State() != scheme.StateRecurrent {
		t.Errorf("layer 1 state = %v, want recurrent", m.Layers[1].Mixer.State())
	}

	caches := m.NewCache()
	defer metal.FreeCaches(caches)
	if _, recurrent := caches[0].(metal.RecurrentCache); recurrent {
		t.Error("layer 0 (full_attention) got a recurrent cache, want KV")
	}
	if _, recurrent := caches[1].(metal.RecurrentCache); !recurrent {
		t.Error("layer 1 (fake_recurrent) got a KV cache, want the recurrent holder")
	}
}

// TestComposed_ArchRegistered_Ugly proves the loadModel registration is LIVE, not
// dead code: probeModelType canonicalises a config's model_type through the same
// profile table NormalizeProbeModelType delegates to, so the "composed"/"hybrid"
// arch keys must round-trip for lookupModelLoader to ever resolve them. If this
// fails the registration never fires and only LoadComposed() works.
func TestComposed_ArchRegistered_Ugly(t *testing.T) {
	for _, arch := range []string{"composed", "hybrid"} {
		if got := metal.NormalizeProbeModelType(arch); got != arch {
			t.Fatalf("NormalizeProbeModelType(%q) = %q — composed loadModel registration is unreachable via probeModelType", arch, got)
		}
	}
}

// subpathProbeMixer is a recurrent mixer whose loader REQUIRES ctx.Linear("in_proj")
// to resolve — so it only builds if the composed model discovered the right mixer
// sublayer for the layer. Forward is not exercised here.
type subpathProbeMixer struct{}

func (subpathProbeMixer) Kind() string            { return "subpath_probe" }
func (subpathProbeMixer) State() scheme.StateKind { return scheme.StateRecurrent }
func (subpathProbeMixer) Forward(x *metal.Array, _ *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	return x, metal.SharedKV{}
}

// TestComposed_SubpathResolution_Good proves the cut's thesis-completion: the
// composed model resolves each layer's mixer weights under the sublayer the
// CHECKPOINT actually uses, discovered per layer — softmax under self_attn,
// a recurrent mixer under "mixer" (the real-checkpoint convention) — both in one
// model. Without discovery the probe's bare ctx.Linear("in_proj") would look for
// model.layers.1.in_proj and fail.
func TestComposed_SubpathResolution_Good(t *testing.T) {
	metal.RegisterMixerLoader("subpath_probe", func(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
		if ctx.Linear("in_proj") == nil {
			return nil, core.NewError("subpath_probe: in_proj not resolved — subpath discovery failed")
		}
		return subpathProbeMixer{}, nil
	})

	weights := fullAttentionWeights(tLayers)
	// Re-home layer 1 as a mixer nested under "mixer." (drop its full-attention
	// naming so only the mixer sublayer is present for that layer).
	for _, p := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		delete(weights, "model.layers.1.self_attn."+p+".weight")
	}
	weights["model.layers.1.mixer.in_proj.weight"] = ramp(tHidden, tHidden)

	cfg := composedTestConfig([]string{"full_attention", "subpath_probe"})
	m, err := buildComposed(cfg, weights, nil)
	if err != nil {
		t.Fatalf("buildComposed with nested mixer subpath: %v", err)
	}
	defer m.CloseModel()
	if m.Layers[0].Kind != "full_attention" || m.Layers[1].Kind != "subpath_probe" {
		t.Fatalf("layer kinds = (%q,%q), want (full_attention, subpath_probe)", m.Layers[0].Kind, m.Layers[1].Kind)
	}
}

// TestComposed_AmbiguousSubpath_Bad: a layer carrying TWO candidate mixer
// sublayers (self_attn AND mixer) is refused loudly, never resolved by a random
// map-order pick — the silent wrong-geometry trap the discovery guards against.
func TestComposed_AmbiguousSubpath_Bad(t *testing.T) {
	weights := fullAttentionWeights(tLayers)
	weights["model.layers.0.mixer.in_proj.weight"] = ramp(tHidden, tHidden) // layer 0 now has self_attn AND mixer
	if _, err := buildComposed(composedTestConfig([]string{"full_attention", "full_attention"}), weights, nil); err == nil {
		t.Error("expected ambiguous-subpath error for a layer with two mixer sublayers, got nil")
	}
}
