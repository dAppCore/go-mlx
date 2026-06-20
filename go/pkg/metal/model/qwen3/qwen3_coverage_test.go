// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Coverage companions for qwen3.go — they drive the registered-loader closures
// (the init() dispatch arms) and the loader's config/shape-derived branches that
// the direct LoadQwen3 fixtures in qwen3_test.go do not reach. All synthetic
// (SaveSafetensors + temp-dir configs), no live model — the AX-11-clean fixture
// style of the sibling tests.

package qwen3

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// TestQwen3_DenseLoaderDispatch_Good routes a synthetic plain-qwen3 checkpoint
// through metal.LoadAndInit, which resolves the arch to the qwen3 family loader
// registered in init() and invokes its closure (the init() dispatch arm the
// direct LoadQwen3 fixtures bypass). A successful materialising load proves the
// closure's happy path.
func TestQwen3_DenseLoaderDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	weights := tinyDenseWeights(64, 128, 100)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3 dispatch) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() == "" {
		t.Error("ModelType empty after dispatched load")
	}
}

// TestQwen3_DenseLoaderDispatch_Bad routes a plain-qwen3 config with no tokenizer
// (and no safetensors) through metal.LoadAndInit: the family loader closure runs,
// LoadQwen3 fails inside it, and the closure wraps the failure with its
// "load qwen3 native model" context — the closure's error arm.
func TestQwen3_DenseLoaderDispatch_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")

	_, err := metal.LoadAndInit(dir)
	if err == nil {
		t.Fatal("LoadAndInit(qwen3 with no tokenizer) error = nil, want a wrapped loader failure")
	}
}

// TestQwen3_ComposedLoaderDispatch_Bad routes a qwen3_6 config (no safetensors)
// through metal.LoadAndInit: the dispatch resolves the qwen3_6 family to the
// composed loader registered in init(); composed.LoadComposed fails (no weights)
// and the closure wraps it with "load qwen3_6 composed model" — the composed
// dispatch arm's error path.
func TestQwen3_ComposedLoaderDispatch_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_6",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"max_position_embeddings": 4096,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	writeMinimalTokenizer(t, dir)

	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(qwen3_6 with no weights) error = nil, want a wrapped composed-loader failure")
	}
}

// TestQwen3_MoEStagedLoaderDispatch_Bad routes a qwen3_6_moe config that omits
// the sparse-expert metadata through metal.LoadAndInit: the dispatch resolves
// the qwen3_6_moe family to loadQwen36MoEStagedModel (registered in init());
// validation rejects the config and the closure wraps it with "validate
// qwen3_6_moe native load" — the staged-MoE dispatch arm's error path.
func TestQwen3_MoEStagedLoaderDispatch_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_6_moe",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	writeMinimalTokenizer(t, dir)

	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(qwen3_6_moe without expert metadata) error = nil, want a wrapped validation failure")
	}
}

// TestQwen3_LoadQwen3_VocabFromEmbed_Good drops vocab_size from the config so the
// loader backfills it from the embedding weight's first dimension — the
// cfg.VocabSize == 0 fallback branch. The embed weight is [vocab, hidden], so the
// loaded model's reported vocab must equal the embedding row count.
func TestQwen3_LoadQwen3_VocabFromEmbed_Good(t *testing.T) {
	requireMetalRuntime(t)

	const vocab, hidden, inter = int32(48), int32(64), int32(128)
	dir := t.TempDir()
	// config.json with no vocab_size key — the loader must derive it from embed.
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 32,
		"rms_norm_eps": 1e-6
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyDenseWeights(hidden, inter, vocab)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3(dir)
	if err != nil {
		t.Fatalf("LoadQwen3(no vocab_size) error = %v", err)
	}
	defer closeQwen3(model)

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != int(vocab) {
		t.Fatalf("VocabSize = %d, want %d backfilled from the embed weight rows", info.VocabSize, vocab)
	}
}
