// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4DiffusionConfigJSON is a synthetic two-layer diffusion_gemma config —
// the dense gemma4 shape plus the diffusion-specific canvas_length + eos_token_id
// the loader reads through parseDiffusionConfigExtras.
const gemma4DiffusionConfigJSON = `{
	"model_type": "diffusion_gemma",
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
	"canvas_length": 4,
	"eos_token_id": 1,
	"layer_types": ["sliding_attention", "full_attention"]
}`

// gemma4DiffusionTinyWeights extends gemma4TinyWeights with the diffusion extras
// the loader collects on top of the weight-tied trunk: the self-conditioning
// block (pre-norm + gate/up/down MLP at the trunk's hidden/intermediate dims)
// and the encoder-role per-layer scalars (raw pre-sanitize keys — the loader
// extracts them before the trunk sanitize can free them).
func gemma4DiffusionTinyWeights() map[string]*metal.Array {
	w := gemma4TinyWeights()
	w["self_conditioning.pre_norm.weight"] = seqArray(0.02, 8)
	w["self_conditioning.gate_proj.weight"] = seqArray(0.7, 16, 8)
	w["self_conditioning.up_proj.weight"] = seqArray(0.8, 16, 8)
	w["self_conditioning.down_proj.weight"] = seqArray(0.9, 8, 16)
	w["model.encoder.language_model.layers.0.layer_scalar"] = metal.FromValues([]float32{1}, 1)
	w["model.encoder.language_model.layers.1.layer_scalar"] = metal.FromValues([]float32{1}, 1)
	return w
}

// loadGemma4DiffusionTestModel writes the synthetic diffusion checkpoint and
// loads it, registering Close cleanup. This is the synthetic force-Eval harness
// the diffusion-family tests reuse — no real 26B checkpoint needed (the live
// model_eval probes in diffusion_live_test.go cover the real weights).
func loadGemma4DiffusionTestModel(t *testing.T) *DiffusionGemmaModel {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4DiffusionConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4DiffusionTinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// TestDiffusion_LoadDiffusionGemma_Good loads the synthetic checkpoint and asserts
// the diffusion extras are wired: the declared canvas length and eos id parsed,
// the self-conditioning block present, and one encoder-role scalar per trunk
// layer collected.
func TestDiffusion_LoadDiffusionGemma_Good(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	if m.CanvasLength != 4 {
		t.Fatalf("CanvasLength = %d, want 4", m.CanvasLength)
	}
	if len(m.EOSTokens) != 1 || m.EOSTokens[0] != 1 {
		t.Fatalf("EOSTokens = %v, want [1]", m.EOSTokens)
	}
	if m.SelfCondPreNorm == nil || m.SelfCondMLP == nil {
		t.Fatal("self-conditioning block not loaded")
	}
	if len(m.EncoderLayerScalars) != len(m.Layers) {
		t.Fatalf("EncoderLayerScalars = %d, want one per layer (%d)", len(m.EncoderLayerScalars), len(m.Layers))
	}
}

// TestDiffusion_LoadDiffusionGemma_Bad pins the fail-loud guard: a checkpoint with
// the trunk but no self-conditioning block is rejected rather than loading a
// half-built diffusion model.
func TestDiffusion_LoadDiffusionGemma_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4DiffusionConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	// Trunk only — no self_conditioning.* and no encoder scalars.
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	m, err := LoadDiffusionGemma(dir)
	if err == nil {
		_ = m.Close()
		t.Fatal("LoadDiffusionGemma loaded a checkpoint missing the self-conditioning block, want error")
	}
}

// TestDiffusion_DenoiseForward_Good runs one bidirectional canvas forward against
// a prompt-prefilled cache and asserts full-canvas logits [1, L, vocab]. This is
// the core denoising compute the sampler consumes — the masks are built inside
// DenoiseForward from the cache offset.
func TestDiffusion_DenoiseForward_Good(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	const canvasLen = 4
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(canvasLen + 8)
	}
	defer metal.FreeCaches(caches)

	canvas := metal.FromValues([]int32{2, 3, 4, 5}, 1, canvasLen)
	logits := m.DenoiseForward(canvas, nil, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer metal.Free(canvas, logits)

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != canvasLen || shape[2] != 10 {
		t.Fatalf("denoise logits shape = %v, want [1 %d 10]", shape, canvasLen)
	}
}

// TestDiffusion_DenoiseForwardWithMasks_Good drives the caller-owned-mask variant
// the generation loop uses (masks built once, reused per step), then feeds the
// shaped logits to SampleDenoiseStep so the self-conditioning embedding flows back
// into a second forward — the step-to-step contract.
func TestDiffusion_DenoiseForwardWithMasks_Good(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	const canvasLen = 4
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(canvasLen + 8)
	}
	defer metal.FreeCaches(caches)

	globalMask := diffusionGlobalCanvasMask(1, canvasLen, canvasLen)
	localMask := diffusionBlockLocalCanvasMask(1, canvasLen, canvasLen, 0, m.Cfg.SlidingWindow)
	defer metal.Free(globalMask, localMask)

	canvas := []int32{2, 3, 4, 5}
	canvasArr := metal.FromValues(canvas, 1, canvasLen)
	logits := m.DenoiseForwardWithMasks(canvasArr, nil, caches, globalMask, localMask)
	metal.Free(canvasArr)

	cfg := DefaultDiffusionStepConfig(m.Cfg.VocabSize)
	cfg.Seed = 3
	res, err := m.SampleDenoiseStep(logits, canvas, 0, 1.0, cfg)
	metal.Free(logits)
	if err != nil {
		t.Fatalf("SampleDenoiseStep: %v", err)
	}
	defer metal.Free(res.SCEmb)

	if len(res.Canvas) != canvasLen {
		t.Fatalf("next canvas len = %d, want %d", len(res.Canvas), canvasLen)
	}
	if res.SCEmb == nil || !res.SCEmb.Valid() {
		t.Fatal("self-conditioning embedding invalid after a step")
	}

	// Roll the caches back (the masked-write transaction in reverse) and run a
	// second masked forward seeded by the step's self-conditioning signal.
	for _, c := range caches {
		c.(*metal.FixedKVCache).TruncateTo(0)
	}
	canvasArr2 := metal.FromValues(res.Canvas, 1, canvasLen)
	logits2 := m.DenoiseForwardWithMasks(canvasArr2, res.SCEmb, caches, globalMask, localMask)
	if err := metal.Eval(logits2); err != nil {
		t.Fatalf("Eval second forward: %v", err)
	}
	metal.Free(canvasArr2, logits2)
}

// TestDiffusion_GenerateDiffusion_Good runs the full block-diffusion loop end to
// end on the synthetic model: causal prompt prefill, one canvas of
// denoise-and-commit, capped at a single canvas. It asserts the metrics reflect
// real work (prefill tokens counted, at least one step taken) and the emitted
// ids stay in vocabulary.
func TestDiffusion_GenerateDiffusion_Good(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	const canvasLen = 4
	promptTokens := m.Tok.Encode("hello world")
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(len(promptTokens) + canvasLen + 16)
	}
	defer metal.FreeCaches(caches)

	cfg := DiffusionGenerateConfig{
		Step:         DefaultDiffusionStepConfig(m.Cfg.VocabSize),
		CanvasLength: canvasLen,
		MaxSteps:     3,
		MaxCanvases:  1,
		StopTokens:   []int32{-1}, // run the full canvas budget
	}
	cfg.Step.Seed = 9

	emitted, metrics, err := m.GenerateDiffusion(context.Background(), "hello world", caches, cfg)
	if err != nil {
		t.Fatalf("GenerateDiffusion: %v", err)
	}
	if metrics.PrefillTokens == 0 {
		t.Fatal("metrics.PrefillTokens = 0, want the prompt counted")
	}
	if metrics.TotalSteps == 0 || metrics.Canvases != 1 {
		t.Fatalf("metrics steps/canvases = %d/%d, want >0 / 1", metrics.TotalSteps, metrics.Canvases)
	}
	for i, id := range emitted {
		if id < 0 || id >= m.Cfg.VocabSize {
			t.Fatalf("emitted[%d] = %d, want 0 <= id < vocab %d", i, id, m.Cfg.VocabSize)
		}
	}
}

// TestDiffusion_GenerateDiffusion_Bad pins the empty-prompt guard: a prompt that
// encodes to zero tokens is rejected before any forward.
func TestDiffusion_GenerateDiffusion_Bad(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(16)
	}
	defer metal.FreeCaches(caches)

	_, _, err := m.GenerateDiffusion(context.Background(), "", caches, DiffusionGenerateConfig{MaxCanvases: 1})
	if err == nil {
		t.Fatal("GenerateDiffusion(empty prompt) error = nil, want zero-token rejection")
	}
}

// TestDiffusion_BlockDiffusionReadbacks_ZeroState pins the neutral serve
// readbacks on a fresh model: before any run the run-state is lazily created and
// reports no error and empty metrics. The streaming happy path (GenerateBlockDiffusion)
// hardcodes the tuned 64-canvas serve profile, which a 16-position synthetic
// trunk cannot commit, so the streaming path stays covered by the live
// model_eval tests (diffusion_live_test.go) and only the readback surface is
// asserted synthetically here.
func TestDiffusion_BlockDiffusionReadbacks_ZeroState(t *testing.T) {
	m := loadGemma4DiffusionTestModel(t)

	if err := m.BlockDiffusionErr(); err != nil {
		t.Fatalf("BlockDiffusionErr before any run = %v, want nil", err)
	}
	if got := m.BlockDiffusionMetrics(); got.EmittedTokens != 0 || got.PromptTokens != 0 || got.Canvases != 0 {
		t.Fatalf("BlockDiffusionMetrics before any run = %+v, want zero", got)
	}
}
