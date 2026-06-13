// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// weightFor builds a [out,in] float32 weight array under the given checkpoint
// key — enough for a builder's gemma4Linear lookup to succeed and infer
// geometry from the output dimension.
func weightFor(m map[string]*metal.Array, key string, out, in int32) {
	m[key] = metal.Zeros([]int32{out, in}, metal.DTypeFloat32)
}

// freeWeights releases a test weight map.
func freeWeights(m map[string]*metal.Array) {
	for _, w := range m {
		metal.Free(w)
	}
}

// sparseTestCfg is a tiny gemma-4 text config for the sparse builder tests:
// hidden 4, 2 heads, head dim 2. These dims live on the embedded neutral
// metal.TransformerConfig, so they are set through it in the literal.
func sparseTestCfg() *Gemma4TextConfig {
	return &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2},
	}
}

// TestBuildMLALayer_Good builds an MLA mixer from a checkpoint weight map and
// checks the resolved geometry: heads from config, head dim inferred from
// kv_b_proj (out = 2*heads*headDim).
func TestBuildMLALayer_Good(t *testing.T) {
	w := map[string]*metal.Array{}
	defer freeWeights(w)
	const p = "model.layers.0.self_attn"
	weightFor(w, p+".kv_a_proj_with_mqa.weight", 6, 4) // latent 6
	weightFor(w, p+".kv_b_proj.weight", 8, 6)          // 2*heads*headDim = 2*2*2 = 8
	weightFor(w, p+".q_a_proj.weight", 6, 4)
	weightFor(w, p+".q_b_proj.weight", 4, 6) // heads*headDim = 4
	weightFor(w, p+".o_proj.weight", 4, 4)

	mixer, err := buildMLALayer(MixerBuildCtx{Weights: w, Prefix: "model.layers.0", Cfg: sparseTestCfg()})
	if err != nil {
		t.Fatalf("buildMLALayer: %v", err)
	}
	if mixer.Kind() != "mla" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("MLA mixer = (%q,%v), want (mla,kv-cache)", mixer.Kind(), mixer.State())
	}
}

// TestBuildMLALayer_MissingWeight_Bad confirms a missing required projection is
// a loud build error, not a silent zero-weight mixer.
func TestBuildMLALayer_MissingWeight_Bad(t *testing.T) {
	w := map[string]*metal.Array{}
	defer freeWeights(w)
	const p = "model.layers.0.self_attn"
	weightFor(w, p+".kv_a_proj_with_mqa.weight", 6, 4)
	// kv_b_proj deliberately absent.
	weightFor(w, p+".q_a_proj.weight", 6, 4)
	weightFor(w, p+".q_b_proj.weight", 4, 6)
	weightFor(w, p+".o_proj.weight", 4, 4)

	if _, err := buildMLALayer(MixerBuildCtx{Weights: w, Prefix: "model.layers.0", Cfg: sparseTestCfg()}); err == nil {
		t.Fatal("expected error for missing kv_b_proj")
	}
}

// TestBuildNSALayer_Good builds an NSA mixer and checks identity + that it
// declares a KV cache (softmax-family).
func TestBuildNSALayer_Good(t *testing.T) {
	w := map[string]*metal.Array{}
	defer freeWeights(w)
	const p = "model.layers.1.self_attn"
	for _, proj := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		weightFor(w, p+"."+proj+".weight", 4, 4) // heads*headDim = 4
	}
	weightFor(w, p+".g_proj.weight", 6, 4) // gate: heads*3 = 6

	mixer, err := buildNSALayer(MixerBuildCtx{Weights: w, Prefix: "model.layers.1", Cfg: sparseTestCfg()})
	if err != nil {
		t.Fatalf("buildNSALayer: %v", err)
	}
	if mixer.Kind() != "nsa" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("NSA mixer = (%q,%v), want (nsa,kv-cache)", mixer.Kind(), mixer.State())
	}
}

// TestBuildMoBALayer_Good builds a MoBA mixer and checks identity + KV cache.
func TestBuildMoBALayer_Good(t *testing.T) {
	w := map[string]*metal.Array{}
	defer freeWeights(w)
	const p = "model.layers.2.self_attn"
	for _, proj := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		weightFor(w, p+"."+proj+".weight", 4, 4)
	}

	mixer, err := buildMoBALayer(MixerBuildCtx{Weights: w, Prefix: "model.layers.2", Cfg: sparseTestCfg()})
	if err != nil {
		t.Fatalf("buildMoBALayer: %v", err)
	}
	if mixer.Kind() != "moba" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("MoBA mixer = (%q,%v), want (moba,kv-cache)", mixer.Kind(), mixer.State())
	}
}

// TestBuildGSALayer_Good builds a GSA mixer and checks it declares the recurrent
// state kind (so the loader pairs it with the #39 holder), with slots inferred
// from f_proj (out = heads*slots).
func TestBuildGSALayer_Good(t *testing.T) {
	w := map[string]*metal.Array{}
	defer freeWeights(w)
	const p = "model.layers.3.self_attn"
	for _, proj := range []string{"q_proj", "k_proj", "v_proj", "o_proj", "g_proj"} {
		weightFor(w, p+"."+proj+".weight", 4, 4)
	}
	weightFor(w, p+".f_proj.weight", 8, 4) // heads*slots = 2*4 → slots 4

	mixer, err := buildGSALayer(MixerBuildCtx{Weights: w, Prefix: "model.layers.3", Cfg: sparseTestCfg()})
	if err != nil {
		t.Fatalf("buildGSALayer: %v", err)
	}
	if mixer.Kind() != "gsa" || mixer.State() != scheme.StateRecurrent {
		t.Fatalf("GSA mixer = (%q,%v), want (gsa,recurrent)", mixer.Kind(), mixer.State())
	}
}

// TestSparseBuilders_ResolveByKind_Good proves all four kinds resolve through
// the loader's builder registry end-to-end — the layer_type → mixer path.
func TestSparseBuilders_ResolveByKind_Good(t *testing.T) {
	for _, kind := range []string{"mla", "nsa", "moba", "gsa"} {
		if _, ok := mixerBuilderFor(kind); !ok {
			t.Errorf("builder for %q not registered", kind)
		}
	}
}

// TestSparseLayerTypesResolveMixerKind_Good confirms a config declaring the
// sparse layer types maps each to its mixer kind (and attention types still map
// to softmax-hybrid).
func TestSparseLayerTypesResolveMixerKind_Good(t *testing.T) {
	cfg := &Gemma4TextConfig{LayerTypes: []string{"full_attention", "mla", "nsa", "moba", "gsa"}}
	want := map[int]string{0: "softmax-hybrid", 1: "mla", 2: "nsa", 3: "moba", 4: "gsa"}
	for idx, exp := range want {
		if got := cfg.MixerKindFor(idx); got != exp {
			t.Errorf("MixerKindFor(%d) = %q, want %q", idx, got, exp)
		}
	}
}
