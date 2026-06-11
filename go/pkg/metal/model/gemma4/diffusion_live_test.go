// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && model_eval

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestDiffusionDenoiseStep_LiveModel proves Unit B end-to-end on the real
// checkpoint: a causal prompt prefill into fixed caches, then two denoising
// steps — bidirectional canvas forward (pending-armed), reference sampler
// (annealing temperature, entropy-bound acceptance, renoise), self-
// conditioning fed from step 1 into step 2 — with the cache prefix proven
// intact after each discard.
//
//	go test -tags model_eval -run 'TestDiffusionDenoiseStep_LiveModel$' -count=1 ./pkg/metal/model/gemma4
func TestDiffusionDenoiseStep_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/diffusiongemma-26B-A4B-it-4bit")
	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	defer closeGemma4(m.Gemma4Model)

	const canvasLen = 64
	prompt := "Write a short story about a clockmaker."
	promptTokens := m.Tok.Encode(prompt)
	if len(promptTokens) == 0 {
		t.Fatal("prompt encoded to zero tokens")
	}

	// Hand-built fixed caches (1:1, no shared layers in this trunk) — the
	// probe exercises the family layer, not the engine cache policy.
	if m.Cfg.NumKVSharedLayers > 0 {
		t.Fatalf("probe assumes no KV-share; config declares %d shared layers", m.Cfg.NumKVSharedLayers)
	}
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(len(promptTokens) + canvasLen + 64)
	}
	defer metal.FreeCaches(caches)

	// Causal prompt prefill — encoder-mode: writes the prompt KV prefix.
	promptArr := metal.FromValues(promptTokens, 1, len(promptTokens))
	prefillLogits := m.Forward(promptArr, caches)
	if err := metal.Eval(prefillLogits); err != nil {
		t.Fatalf("prefill eval: %v", err)
	}
	metal.Free(promptArr, prefillLogits)
	promptOffset := caches[0].Offset()
	if promptOffset != len(promptTokens) {
		t.Fatalf("prompt offset = %d, want %d", promptOffset, len(promptTokens))
	}

	cfg := DefaultDiffusionStepConfig(m.Cfg.VocabSize)
	cfg.Seed = 7
	canvas := make([]int32, canvasLen)
	for i := range canvas {
		canvas[i] = int32((i * 2654435761) % int(cfg.TextVocabSize))
	}

	truncate := func() {
		for i, c := range caches {
			if !c.(*metal.FixedKVCache).TruncateTo(promptOffset) {
				t.Fatalf("cache %d: TruncateTo(%d) declined (offset %d, pre-cap expected)", i, promptOffset, c.Offset())
			}
		}
	}

	var scEmb *metal.Array
	var lastAccepted int
	noiseSchedule := []float32{1.0, 0.75}
	for step, noise := range noiseSchedule {
		canvasArr := metal.FromValues(canvas, 1, canvasLen)
		logits := m.DenoiseForward(canvasArr, scEmb, caches)
		metal.Free(canvasArr)
		var lShape [metal.MaxTensorRank]int32
		shape := logits.ShapeInto(lShape[:0])
		if len(shape) != 3 || shape[0] != 1 || shape[1] != canvasLen || shape[2] != m.Cfg.VocabSize {
			t.Fatalf("step %d logits shape = %v, want [1 %d %d]", step, shape, canvasLen, m.Cfg.VocabSize)
		}
		res, err := m.SampleDenoiseStep(logits, canvas, step, noise, cfg)
		metal.Free(logits)
		if err != nil {
			t.Fatalf("step %d sample: %v", step, err)
		}
		truncate()

		if got := caches[0].Offset(); got != promptOffset {
			t.Fatalf("step %d: cache offset = %d after truncate, want prompt prefix %d", step, got, promptOffset)
		}
		if len(res.Canvas) != canvasLen {
			t.Fatalf("step %d: canvas len = %d, want %d", step, len(res.Canvas), canvasLen)
		}
		if res.Accepted <= 0 {
			t.Fatalf("step %d: accepted %d tokens, want at least the most confident one", step, res.Accepted)
		}
		var scShape [metal.MaxTensorRank]int32
		sc := res.SCEmb.ShapeInto(scShape[:0])
		if len(sc) != 3 || sc[0] != 1 || sc[1] != canvasLen || sc[2] != m.Cfg.HiddenSize {
			t.Fatalf("step %d: sc embedding shape = %v, want [1 %d %d]", step, sc, canvasLen, m.Cfg.HiddenSize)
		}
		t.Logf("step %d (noise %.2f): accepted %d · changed %d · entropy-driven acceptance live", step, noise, res.Accepted, res.Changed)
		t.Logf("step %d: canvas[0:8]  in=%v out=%v", step, canvas[:8], res.Canvas[:8])

		metal.Free(scEmb)
		scEmb = res.SCEmb
		canvas = res.Canvas
	}
	metal.Free(scEmb)
	_ = lastAccepted
}
