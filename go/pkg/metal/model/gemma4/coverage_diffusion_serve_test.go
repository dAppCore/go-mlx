// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the block-diffusion serve bridge (diffusion_serve.go). The
// existing synthetic diffusion model uses a 16-position trunk that cannot commit
// the hardcoded 64-canvas serve profile; this builds a larger synthetic trunk
// (256 positions) so GenerateBlockDiffusion's streaming body runs end to end.
// Helpers prefixed cov4ds.

// cov4dsLoadBigDiffusionModel writes a synthetic diffusion checkpoint with a
// 256-position trunk and loads it, so the 64-canvas serve profile fits.
func cov4dsLoadBigDiffusionModel(t *testing.T) *DiffusionGemmaModel {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "diffusion_gemma",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"max_position_embeddings": 256,
		"rms_norm_eps": 1e-6,
		"sliding_window": 128,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"canvas_length": 64,
		"eos_token_id": 1,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	cov4SaveAndFree(t, core.JoinPath(dir, "model.safetensors"), gemma4DiffusionTinyWeights())
	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// TestDiffusionServe_GenerateBlockDiffusion_Good drives the streaming serve path:
// the canvas iterator yields at least one token, the OnCanvas hook commits, and
// the neutral readbacks report the run's metrics afterwards.
func TestDiffusionServe_GenerateBlockDiffusion_Good(t *testing.T) {
	m := cov4dsLoadBigDiffusionModel(t)

	// MaxTokens 64 == one canvas; a fixed seed keeps the run deterministic.
	opts := metal.BlockDiffusionOptions{MaxTokens: 64, Seed: 7, SeedSet: true}
	seq := m.GenerateBlockDiffusion(context.Background(), "hello world", opts)

	count := 0
	for tok := range seq {
		count++
		if tok.ID < 0 || tok.ID >= m.Cfg.VocabSize {
			t.Fatalf("token id %d out of [0,%d)", tok.ID, m.Cfg.VocabSize)
		}
		if count >= 4 {
			break // consumer stop exercises the yield=false cancel path
		}
	}
	if count == 0 {
		t.Fatal("GenerateBlockDiffusion yielded no tokens, want at least one canvas token")
	}
	// after the run the readbacks reflect it (no terminal error path here).
	if err := m.BlockDiffusionErr(); err != nil {
		t.Fatalf("BlockDiffusionErr after run = %v, want nil", err)
	}
}

// TestDiffusionServe_GenerateBlockDiffusion_NilCtx_Good covers the ctx==nil
// branch (the serve falls back to context.Background) and a temperature override.
func TestDiffusionServe_GenerateBlockDiffusion_NilCtx_Good(t *testing.T) {
	m := cov4dsLoadBigDiffusionModel(t)

	opts := metal.BlockDiffusionOptions{MaxTokens: 64, Temperature: 0.5, Seed: 3, SeedSet: true}
	//nolint:staticcheck // exercising the nil-ctx fallback deliberately
	seq := m.GenerateBlockDiffusion(nil, "hello world", opts)
	count := 0
	for tok := range seq {
		_ = tok
		count++
		break // first canvas is enough to cover the body
	}
	if count == 0 {
		t.Fatal("GenerateBlockDiffusion(nil ctx) yielded no tokens, want the nil-ctx fallback to still produce a canvas")
	}
	if err := m.BlockDiffusionErr(); err != nil {
		t.Fatalf("BlockDiffusionErr = %v, want nil", err)
	}
}
