// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// synthGemma4TextConfigJSON is the canonical tiny gemma4-text trunk
// (matching synthGemma4TrunkWeights — hidden 8, 2 layers, vocab 10) declared
// as plain text "gemma4_text". No vision/audio sections, so the loaded model
// carries no tower; multimodalGreedyDecode then runs its decode loop with
// nil modalities, falling straight through ForwardUnifiedVideoMultiModal to
// the dense text forward.
const synthGemma4TextConfigJSON = `{
	"model_type": "gemma4_text",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 1,
	"num_key_value_heads": 1,
	"head_dim": 4,
	"global_head_dim": 8,
	"vocab_size": 10,
	"max_position_embeddings": 16,
	"rms_norm_eps": 1e-6,
	"sliding_window": 4,
	"sliding_window_pattern": 2,
	"num_kv_shared_layers": 0,
	"hidden_size_per_layer_input": 0,
	"use_double_wide_mlp": false,
	"layer_types": ["sliding_attention", "full_attention"]
}`

// writeSyntheticGemma4TextModel writes a complete, loadable 2-layer
// gemma4_text checkpoint (config + tokenizer + the shared tiny trunk) into a
// fresh temp dir and returns the dir. Loadable through gemma4.LoadGemma4.
func writeSyntheticGemma4TextModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), synthGemma4TextConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), synthTokenizerJSON); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
	w := synthGemma4TrunkWeights()
	defer func() {
		for _, a := range w {
			metal.Free(a)
		}
	}()
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), w); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	return dir
}

// multimodalGreedyDecode over a tiny gemma4-text synthetic with NO modality
// arrays runs the real prefill + greedy-decode loop (ForwardUnifiedVideoMultiModal
// falls through to the dense text forward when no tower is present). This is
// the shared verb loop the audio + vision commands drive; covered here without
// a vision/audio tower fixture.
func TestMultimodalGreedyDecode_NoModalities_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	dir := writeSyntheticGemma4TextModel(t)
	m, err := gemma4.LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer m.CloseModel()

	ids := m.Tok.Encode("hello")
	if len(ids) == 0 {
		ids = []int32{2, 3, 4}
	}
	res, err := multimodalGreedyDecode(context.Background(), m, ids, nil, nil, nil, 6)
	if err != nil {
		t.Fatalf("multimodalGreedyDecode: %v", err)
	}
	if res.PrefillDur <= 0 {
		t.Fatalf("PrefillDur = %v, want a positive prefill duration", res.PrefillDur)
	}
	// The decode loop ran (generated up to the bound or hit a stop token); the
	// generated slice is non-nil either way.
	if res.Generated == nil {
		t.Fatal("Generated is nil, want an (possibly empty) generated slice")
	}
}

// multimodalGreedyDecode honours context cancellation: a cancelled context
// returns the cancellation error from the decode loop before reaching the
// token bound.
func TestMultimodalGreedyDecode_Cancelled_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	dir := writeSyntheticGemma4TextModel(t)
	m, err := gemma4.LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer m.CloseModel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ids := m.Tok.Encode("hello")
	if len(ids) == 0 {
		ids = []int32{2, 3, 4}
	}
	if _, err := multimodalGreedyDecode(ctx, m, ids, nil, nil, nil, 64); err == nil {
		t.Fatal("multimodalGreedyDecode with a cancelled context returned nil error, want cancellation")
	}
}
