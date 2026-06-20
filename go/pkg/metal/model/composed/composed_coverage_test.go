// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Coverage companions for composed.go. composed_test.go drives the disk-free
// composition core (buildComposed) from synthetic metal.FromValues weights; this
// file closes the two gaps that core cannot reach:
//
//   - the on-disk entry points — LoadComposed (config → tokenizer → weights →
//     compose → materialise) and the init() loader closure that metal.LoadAndInit
//     dispatches a "composed"/"hybrid" config to — driven from a synthetic
//     t.TempDir() checkpoint (SaveSafetensors + JSON configs), the AX-11-clean
//     fixture style the sibling qwen3 / gemma4 loaders use, never a model off disk;
//   - the config/shape-derived and defensive branches of buildComposed,
//     discoverMixerSubpath, resolveLayerKinds, ApplyLoRA and CloseModel that the
//     happy-path fixtures do not exercise (vocab-from-embed, quantized embed +
//     lm_head, missing embed, bare-mixer subpath, a non-positive layer count, and
//     the nil-layer guards).
package composed

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// requireMetalRuntime gates the disk-load and quantize tests: they read a
// checkpoint and run the composed forward / quantizer on a Metal device, so they
// skip unless built with -tags metal_runtime on a usable device.
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// composedConfigJSON is the on-disk config a composed checkpoint declares: a
// "composed" model_type with one layer_types entry per layer. The geometry
// matches the synthetic fixture (hidden 4, 2 heads, head dim 2, intermediate 8,
// 2 layers) so the canonical fullAttentionWeights pack loads against it. With no
// vocab_size key the loader backfills it from the embedding rows; a caller that
// wants the explicit-vocab path adds the field.
const composedConfigJSON = `{
	"model_type": "composed",
	"hidden_size": 4,
	"num_hidden_layers": 2,
	"intermediate_size": 8,
	"num_attention_heads": 2,
	"num_key_value_heads": 2,
	"head_dim": 2,
	"vocab_size": 6,
	"max_position_embeddings": 128,
	"rms_norm_eps": 1e-5,
	"layer_types": ["full_attention", "full_attention"]
}`

// writeComposedTokenizer writes the same minimal BPE tokenizer the sibling
// loaders use, so LoadComposed's metal.LoadTokenizer step resolves.
func writeComposedTokenizer(t testing.TB, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {
			"type": "BPE",
			"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<pad>", "special": true},
			{"id": 1, "content": "<eos>", "special": true},
			{"id": 2, "content": "<bos>", "special": true}
		]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
}

// writeComposedCheckpoint lays a full synthetic composed checkpoint into dir:
// config.json (the given body), tokenizer.json, and model.safetensors holding
// the weight pack. The weights are freed via t.Cleanup once SaveSafetensors has
// serialised them.
func writeComposedCheckpoint(t testing.TB, dir, configJSON string, weights map[string]*metal.Array) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), configJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeComposedTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	t.Cleanup(func() {
		for _, a := range weights {
			metal.Free(a)
		}
	})
}

// TestComposed_LoadComposed_Good drives the disk entry point end-to-end from a
// synthetic on-disk checkpoint: ResolveModelRoot → read config.json →
// ParseDenseConfig → LoadTokenizer → LoadModelWeights → buildComposed →
// Materialize. It is the coverage path for the whole LoadComposed body that the
// buildComposed unit fixtures bypass (they call the disk-free core directly).
func TestComposed_LoadComposed_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeComposedCheckpoint(t, dir, composedConfigJSON, fullAttentionWeights(tLayers))

	m, err := LoadComposed(dir)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	defer m.CloseModel()

	if m.NumLayers() != tLayers {
		t.Fatalf("NumLayers = %d, want %d", m.NumLayers(), tLayers)
	}
	if m.ModelType() != "composed" {
		t.Fatalf("ModelType = %q, want composed", m.ModelType())
	}
	if m.Tokenizer() == nil {
		t.Error("Tokenizer is nil after a disk load, want the loaded tokenizer")
	}

	// The loaded model runs a forward exactly like the synthetic fixture does.
	caches := m.NewCache()
	defer metal.FreeCaches(caches)
	tokens := metal.FromValues([]int32{1, 2}, 1, 2)
	defer metal.Free(tokens)
	out := m.Forward(tokens, caches)
	defer metal.Free(out)
	metal.Materialize(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != tVocab {
		t.Errorf("loaded-model forward out shape = %v, want [1 2 %d]", got, tVocab)
	}
}

// TestComposed_LoadComposed_Dispatch_Good routes a synthetic composed checkpoint
// through metal.LoadAndInit, which resolves the "composed" arch to the loader
// registered in init() and invokes its closure — the init() dispatch arm the
// direct LoadComposed call bypasses. A successful materialising load proves the
// closure's happy path (LoadComposed → return model, no error wrap).
func TestComposed_LoadComposed_Dispatch_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeComposedCheckpoint(t, dir, composedConfigJSON, fullAttentionWeights(tLayers))

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(composed dispatch) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "composed" {
		t.Errorf("dispatched-load ModelType = %q, want composed", model.ModelType())
	}
}

// TestComposed_LoadComposed_Dispatch_Bad routes a "composed" config with no
// safetensors through metal.LoadAndInit: the init() loader closure runs,
// LoadComposed fails inside it (no weights), and the closure wraps the failure
// with its "load composed model" context — the closure's error arm.
func TestComposed_LoadComposed_Dispatch_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), composedConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeComposedTokenizer(t, dir)

	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(composed with no weights) error = nil, want a wrapped loader failure")
	}
}

// TestComposed_LoadComposed_MissingConfig_Bad: LoadComposed on a directory with
// no config.json fails at the read step and returns a wrapped "load config"
// error — the first error arm of the disk loader.
func TestComposed_LoadComposed_MissingConfig_Bad(t *testing.T) {
	if _, err := LoadComposed(t.TempDir()); err == nil {
		t.Fatal("LoadComposed(empty dir) error = nil, want a missing-config failure")
	}
}

// TestComposed_LoadComposed_BadConfig_Bad: LoadComposed on a malformed config.json
// fails at ParseDenseConfig and returns a wrapped "parse config" error — the
// parse error arm, distinct from the missing-file arm above.
func TestComposed_LoadComposed_BadConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), `{ not json`); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if _, err := LoadComposed(dir); err == nil {
		t.Fatal("LoadComposed(malformed config) error = nil, want a parse-config failure")
	}
}

// TestComposed_LoadComposed_MissingTokenizer_Bad: a valid config with no
// tokenizer.json fails at LoadTokenizer — the tokenizer error arm.
func TestComposed_LoadComposed_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), composedConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if _, err := LoadComposed(dir); err == nil {
		t.Fatal("LoadComposed(no tokenizer) error = nil, want a load-tokenizer failure")
	}
}

// TestComposed_LoadComposed_BuildFails_Bad covers LoadComposed's last error arm:
// config, tokenizer and weights all load cleanly, but the composition itself
// fails — here a checkpoint whose safetensors carries the norms (so
// LoadModelWeights succeeds and finds a file) yet omits model.embed_tokens.weight,
// so buildComposed returns the missing-embed error and LoadComposed wraps it with
// "compose model". This is the only LoadComposed branch the happy-path and
// pre-build-failure tests do not reach.
func TestComposed_LoadComposed_BuildFails_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), composedConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeComposedTokenizer(t, dir)
	// A safetensors with the final norm only: enough for LoadModelWeights to load
	// a file, but no embed_tokens.weight, so buildComposed fails at the embed guard.
	weights := map[string]*metal.Array{"model.norm.weight": ramp(tHidden, 0)}
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	for _, a := range weights {
		metal.Free(a)
	}

	if _, err := LoadComposed(dir); err == nil {
		t.Fatal("LoadComposed(weights load but compose fails) error = nil, want a wrapped compose-model failure")
	}
}

// TestComposed_LoadComposed_VocabFromEmbed_Good drops vocab_size from the config
// so buildComposed backfills it from the embedding weight's first dimension — the
// cfg.VocabSize == 0 fallback branch, driven through the disk loader. The embed
// weight is [vocab, hidden], so the loaded model's reported vocab equals the row
// count.
func TestComposed_LoadComposed_VocabFromEmbed_Good(t *testing.T) {
	requireMetalRuntime(t)

	const noVocabConfig = `{
		"model_type": "composed",
		"hidden_size": 4,
		"num_hidden_layers": 2,
		"intermediate_size": 8,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 2,
		"max_position_embeddings": 128,
		"rms_norm_eps": 1e-5,
		"layer_types": ["full_attention", "full_attention"]
	}`
	dir := t.TempDir()
	writeComposedCheckpoint(t, dir, noVocabConfig, fullAttentionWeights(tLayers))

	m, err := LoadComposed(dir)
	if err != nil {
		t.Fatalf("LoadComposed(no vocab_size) error = %v", err)
	}
	defer m.CloseModel()

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.VocabSize != tVocab {
		t.Fatalf("VocabSize = %d, want %d backfilled from the embed weight rows", info.VocabSize, tVocab)
	}
}

// Quantization geometry: mlx_quantize requires the inner dimension to be a
// multiple of the group size (min 32), so the tiny synthetic geometry cannot be
// quantized. This dedicated, larger shape (hidden 64, 2 heads, head dim 32, inter
// 128, vocab 64, group 64) is the smallest one that quantizes cleanly — the same
// shape the sibling qwen3 quantized fixture uses.
const (
	qHidden  = int32(64)
	qHeads   = int32(2)
	qHeadDim = int32(32)
	qInter   = int32(128)
	qVocab   = int32(64)
	qGroup   = 64
	qBits    = 4
)

// quantizedComposedWeights packs a 2-layer full-attention composed weight set
// into a quantized checkpoint at the quantizable geometry: every projection
// (embed, the per-layer q/k/v/o + MLP, and an untied lm_head) is packed into its
// (weight, scales, biases) triplet, so LoadComposed takes the quantized-embed
// branch (embed_tokens.scales present) and the quantized-lm_head branch
// (NewQuantizedLinearWithMode) that the bf16 fixture skips. The norms stay dense
// (they are not quantized in a real checkpoint).
func quantizedComposedWeights(t testing.TB) map[string]*metal.Array {
	t.Helper()
	weights := map[string]*metal.Array{}
	addQuant := func(name string, out, in int32) {
		dense := ramp(out, in)
		wq, sc, bi, err := metal.Quantize(dense, qGroup, qBits, "")
		if err != nil {
			t.Fatalf("Quantize(%s): %v", name, err)
		}
		metal.Free(dense)
		weights[name+".weight"] = wq
		weights[name+".scales"] = sc
		weights[name+".biases"] = bi
	}
	addQuant("model.embed_tokens", qVocab, qHidden)
	addQuant("lm_head", qVocab, qHidden)
	for i := int32(0); i < 2; i++ {
		p := "model.layers." + itoa(i)
		addQuant(p+".self_attn.q_proj", qHeads*qHeadDim, qHidden)
		addQuant(p+".self_attn.k_proj", qHeads*qHeadDim, qHidden)
		addQuant(p+".self_attn.v_proj", qHeads*qHeadDim, qHidden)
		addQuant(p+".self_attn.o_proj", qHidden, qHeads*qHeadDim)
		addQuant(p+".mlp.gate_proj", qInter, qHidden)
		addQuant(p+".mlp.up_proj", qInter, qHidden)
		addQuant(p+".mlp.down_proj", qHidden, qInter)
		weights[p+".input_layernorm.weight"] = ramp(qHidden, 0)
		weights[p+".post_attention_layernorm.weight"] = ramp(qHidden, 0)
	}
	weights["model.norm.weight"] = ramp(qHidden, 0)
	return weights
}

// TestComposed_LoadComposed_Quantized_Good drives the quantized load path: the
// config declares quantization, the checkpoint carries quantized embed + lm_head
// + projection triplets, so buildComposed takes the quantized-embedding branch
// (GroupSize/Bits copied onto the Embedding) and the quantized-lm_head branch
// (NewQuantizedLinearWithMode for the untied output). The bf16 fixtures cover
// neither.
func TestComposed_LoadComposed_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	const quantConfig = `{
		"model_type": "composed",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"vocab_size": 64,
		"max_position_embeddings": 128,
		"rms_norm_eps": 1e-5,
		"quantization": {"bits": 4, "group_size": 64},
		"layer_types": ["full_attention", "full_attention"]
	}`
	dir := t.TempDir()
	writeComposedCheckpoint(t, dir, quantConfig, quantizedComposedWeights(t))

	m, err := LoadComposed(dir)
	if err != nil {
		t.Fatalf("LoadComposed(quantized) error = %v", err)
	}
	defer m.CloseModel()

	if m.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved")
	}
	if m.EmbedTokens.Bits != qBits || m.EmbedTokens.GroupSize != qGroup {
		t.Errorf("embed quant = %d/%d, want %d/%d", m.EmbedTokens.Bits, m.EmbedTokens.GroupSize, qBits, qGroup)
	}
	// An untied lm_head was supplied, so the output is its own quantized linear,
	// distinct from the embedding array.
	if m.Output == nil || m.Output.Weight == nil {
		t.Fatal("quantized untied output is nil, want the lm_head linear")
	}
	if m.Output.Weight == m.EmbedTokens.Weight {
		t.Error("quantized output shares the embedding array, want a distinct lm_head weight")
	}

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.QuantBits != qBits || info.QuantGroup != qGroup {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, qBits, qGroup)
	}
}

// TestComposed_Build_VocabFromEmbed_Good covers the disk-free cfg.VocabSize == 0
// backfill in buildComposed: the config omits vocab_size and the embedding rows
// supply it. (The disk test above proves it through LoadComposed; this pins the
// branch in the core without the I/O.)
func TestComposed_Build_VocabFromEmbed_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	cfg.VocabSize = 0 // force the embed-row backfill
	m, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		t.Fatalf("buildComposed(no vocab_size): %v", err)
	}
	defer m.CloseModel()
	if cfg.VocabSize != tVocab {
		t.Fatalf("backfilled VocabSize = %d, want %d from the embed rows", cfg.VocabSize, tVocab)
	}
}

// TestComposed_Build_MissingEmbed_Bad: a checkpoint with no
// model.embed_tokens.weight is refused loudly at build — the embedding is the
// one weight with no fallback, so its absence is a build error, never a nil
// Embedding the forward would deref.
func TestComposed_Build_MissingEmbed_Bad(t *testing.T) {
	weights := fullAttentionWeights(tLayers)
	delete(weights, "model.embed_tokens.weight")
	defer func() {
		for _, a := range weights {
			metal.Free(a)
		}
	}()
	if _, err := buildComposed(composedTestConfig([]string{"full_attention", "full_attention"}), weights, nil); err == nil {
		t.Error("expected loud error for missing model.embed_tokens.weight, got nil")
	}
}

// TestComposed_DiscoverMixerSubpath_Bare_Good covers discoverMixerSubpath's
// case-0 (bare) arm directly: a layer whose only sub-keys are the norms (no
// mixer sublayer ≥3 components deep, no mlp) yields an empty subpath, telling the
// caller the mixer weights are bare leaves under the layer prefix. The build
// fixtures always nest the mixer under self_attn, so this arm is otherwise unhit.
func TestComposed_DiscoverMixerSubpath_Bare_Good(t *testing.T) {
	weights := map[string]*metal.Array{
		// Only the two norms for layer 0 — both are leaves (2 components past the
		// layer prefix), so neither registers as a mixer sublayer.
		"model.layers.0.input_layernorm.weight":          ramp(tHidden, 0),
		"model.layers.0.post_attention_layernorm.weight": ramp(tHidden, 0),
	}
	defer func() {
		for _, a := range weights {
			metal.Free(a)
		}
	}()
	sub, err := discoverMixerSubpath(weights, 0)
	if err != nil {
		t.Fatalf("discoverMixerSubpath(bare layer): %v", err)
	}
	if sub != "" {
		t.Fatalf("bare-layer subpath = %q, want \"\" (mixer weights are bare leaves)", sub)
	}
}

// TestComposed_ResolveLayerKinds_NonPositive_Bad covers resolveLayerKinds'
// num_hidden_layers <= 0 guard: a config that declares no layers cannot map any
// layer to a kind and is refused, never returning an empty kinds slice the build
// loop would silently skip.
func TestComposed_ResolveLayerKinds_NonPositive_Bad(t *testing.T) {
	cfg := composedTestConfig(nil)
	cfg.ModelType = "full_attention"
	cfg.NumHiddenLayers = 0
	if _, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil); err == nil {
		t.Error("expected loud error for num_hidden_layers <= 0, got nil")
	}
}

// TestComposed_ApplyLoRA_NilLayer_Good covers ApplyLoRA's nil-layer / nil-MLP
// guard: a model whose layer slice holds a nil layer and a layer with no MLP is
// skipped over both, wrapping only the real MLP — the adapter binds the three
// targets of the one wrappable layer and nothing from the skipped ones.
func TestComposed_ApplyLoRA_NilLayer_Good(t *testing.T) {
	model := &ComposedModel{
		Layers: []*composedLayer{
			nil,        // nil layer → skipped
			{MLP: nil}, // layer with no MLP → skipped
			{MLP: &metal.SiLUMLP{ // the one wrappable layer
				GateProj: metal.NewLinear(ramp(tInter, tHidden), nil),
				UpProj:   metal.NewLinear(ramp(tInter, tHidden), nil),
				DownProj: metal.NewLinear(ramp(tHidden, tInter), nil),
			}},
		},
	}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         4,
		Scale:        2,
		TargetLayers: []string{"gate_proj", "up_proj", "down_proj"},
	})
	if len(adapter.Layers) != 3 {
		t.Fatalf("adapter entries = %d, want 3 (only the one non-nil MLP layer wrapped)", len(adapter.Layers))
	}
	// The wrapped entries must be keyed under layer index 2 (the wrappable layer),
	// never the skipped indices 0 / 1.
	if _, ok := adapter.Layers["model.layers.2.mlp.gate_proj"]; !ok {
		t.Error("expected the wrappable layer (index 2) gate_proj to be wrapped")
	}
}

// TestComposed_CloseModel_NilLayer_Good covers CloseModel's nil-layer guard: a
// model whose layer slice holds a nil entry alongside a real one closes cleanly,
// skipping the nil and freeing the real layer's arrays — the defensive path a
// half-built model relies on.
func TestComposed_CloseModel_NilLayer_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	m, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		t.Fatalf("buildComposed: %v", err)
	}
	// Inject a nil layer so CloseModel walks the nil-guard arm before the real ones.
	m.Layers = append([]*composedLayer{nil}, m.Layers...)
	m.CloseModel() // must not panic on the nil layer
}
