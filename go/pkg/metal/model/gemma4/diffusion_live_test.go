// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && model_eval

package gemma4

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
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

// TestDiffusionBook_PerTurnMemoryProbe_LiveModel is the #77 live
// instrument: a serve-book-shaped multi-turn run against the real
// DiffusionGemma checkpoint, in-process (no serve), with per-phase memory
// readings from inside the denoise loop. Each turn mirrors
// GenerateBlockDiffusion's request shape exactly — fresh fixed caches
// sized prompt+canvases, the tuned decode profile — and the conversation
// grows per turn like book chapters. Turn 5 runs after a ClearCache to
// show how much of the accumulation is allocator-cache (the surviving
// theory after the synthetic probe eliminated retirees).
//
// A 40GiB watchdog aborts the run safely (the historical books killed
// the box at 42-148GB; the AR twin runs these chapters flat).
//
//	go test -tags model_eval -run 'TestDiffusionBook_PerTurnMemoryProbe_LiveModel$' -count=1 -v -timeout 30m ./pkg/metal/model/gemma4
func TestDiffusionBook_PerTurnMemoryProbe_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/diffusiongemma-26B-A4B-it-4bit")

	const watchdogBytes = uint64(40) << 30
	metal.SetMemoryLimit(watchdogBytes)

	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	defer closeGemma4(m.Gemma4Model)
	t.Logf("loaded: active=%dMiB cache=%dMiB", metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20)

	// Synthetic chapter filler — deterministic, ~1.5k tokens per turn so the
	// prefix grows like a book without needing real chapter text.
	filler := ""
	for i := 0; i < 220; i++ {
		filler += "The clockmaker wound every spring in the workshop while the brass wheels turned through the quiet afternoon. "
	}

	const (
		canvasLen   = int32(DefaultCanvasLength)
		maxCanvases = 3
		turns       = 5
	)

	conversation := "Write chapter 1 of a story about a clockmaker.\n"
	for turn := 1; turn <= turns; turn++ {
		if turn == turns {
			// Turn 5: the allocator-cache discriminator.
			metal.ClearCache()
			t.Logf("--- ClearCache before turn %d: active=%dMiB cache=%dMiB", turn, metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20)
		}
		prompt := conversation + filler + core.Sprintf("\nWrite chapter %d.\n", turn)
		promptTokens := m.Tok.Encode(prompt)

		// GenerateBlockDiffusion's request-cache shape, verbatim.
		capacity := len(promptTokens) + (int(canvasLen)+8)*maxCanvases + 64
		caches := make([]metal.Cache, len(m.Layers))
		for i := range caches {
			caches[i] = metal.NewFixedKVCache(capacity)
		}

		metal.ResetPeakMemory()
		turnStartActive := metal.GetActiveMemory()
		turnStartCache := metal.GetCacheMemory()

		cfg := DiffusionGenerateConfig{
			Step:         DefaultDiffusionStepConfig(m.Cfg.VocabSize),
			CanvasLength: canvasLen,
			MaxCanvases:  maxCanvases,
			// No stop tokens: every turn runs the full canvas budget so the
			// per-turn workload is uniform and the profiles compare.
			StopTokens: []int32{-1},
		}
		cfg.Step.Seed = 42 + uint64(turn)
		watchdogTripped := false
		ctx, cancel := context.WithCancel(context.Background())
		cfg.OnStep = func(canvasIdx, step int, _ DiffusionStepResult, _ time.Duration) {
			if active := metal.GetActiveMemory(); active > watchdogBytes {
				watchdogTripped = true
				t.Errorf("WATCHDOG: active=%dMiB at turn %d canvas %d step %d — aborting", active>>20, turn, canvasIdx, step)
				cancel()
			}
		}
		cfg.OnCanvas = func(canvasIdx int, kept []int32, steps int, d time.Duration) {
			t.Logf("  turn %d canvas %d: steps=%d kept=%d %.1fs active=%dMiB cache=%dMiB",
				turn, canvasIdx, steps, len(kept), d.Seconds(), metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20)
		}

		emitted, dm, err := m.GenerateDiffusion(ctx, prompt, caches, cfg)
		metal.FreeCaches(caches)
		cancel()
		if err != nil && !watchdogTripped {
			t.Fatalf("turn %d: GenerateDiffusion: %v", turn, err)
		}

		reply := m.Tok.Decode(emitted)
		conversation = prompt + reply + "\n"

		t.Logf("turn %d: prefix=%dtok emitted=%d prefill=%.1fs denoise=%.1fs | active %dMiB->%dMiB peak=%dMiB | cache %dMiB->%dMiB",
			turn, dm.PrefillTokens, dm.EmittedTokens, dm.PrefillDur.Seconds(), dm.DenoiseDur.Seconds(),
			turnStartActive>>20, metal.GetActiveMemory()>>20, metal.GetPeakMemory()>>20,
			turnStartCache>>20, metal.GetCacheMemory()>>20)

		if watchdogTripped {
			t.Fatal("watchdog tripped — profile above is the evidence")
		}
	}
}

// TestDiffusionServeBridge_ClearsAllocatorCachePerRequest_LiveModel
// verifies the #77 fix at the serve seam: GenerateBlockDiffusion drops the
// allocator cache with each request, so a multi-turn serve book starts
// every turn from a clean floor instead of parking ~10GB of never-refit
// buckets per chapter (the per-turn probe above measured the unbounded
// shape: active flat at the weights, cache 0→27GB over four turns).
//
//	go test -tags model_eval -run 'TestDiffusionServeBridge_ClearsAllocatorCachePerRequest_LiveModel$' -count=1 -v -timeout 15m ./pkg/metal/model/gemma4
func TestDiffusionServeBridge_ClearsAllocatorCachePerRequest_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/diffusiongemma-26B-A4B-it-4bit")
	metal.SetMemoryLimit(uint64(40) << 30)
	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	defer closeGemma4(m.Gemma4Model)

	filler := ""
	for i := 0; i < 220; i++ {
		filler += "The clockmaker wound every spring in the workshop while the brass wheels turned through the quiet afternoon. "
	}

	conversation := "Write chapter 1 of a story about a clockmaker.\n"
	const floorMiB = uint64(1024)
	for turn := 1; turn <= 2; turn++ {
		prompt := conversation + filler + core.Sprintf("\nWrite chapter %d.\n", turn)
		var reply string
		for tok := range m.GenerateBlockDiffusion(context.Background(), prompt, metal.BlockDiffusionOptions{
			MaxTokens: 128,
			Seed:      uint64(turn),
			SeedSet:   true,
		}) {
			reply += tok.Text
		}
		if err := m.BlockDiffusionErr(); err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		conversation = prompt + reply + "\n"
		cache := metal.GetCacheMemory() >> 20
		t.Logf("turn %d: emitted=%d active=%dMiB cache=%dMiB after request", turn, m.BlockDiffusionMetrics().EmittedTokens, metal.GetActiveMemory()>>20, cache)
		if cache > floorMiB {
			t.Errorf("turn %d: allocator cache %dMiB after request, want < %dMiB — the per-request clear is not holding", turn, cache, floorMiB)
		}
	}
}

// TestDiffusionPerStepMemory_LiveModel is the #77 step-grain instrument:
// ONE serve-shaped request (13 canvases — the book chapter's 800-token
// budget) against a ch5-sized prefix, logging active + allocator-cache
// memory on EVERY denoise step. The falsified per-request fix taught us
// the growth is within-request; this run shows the slope and whether it
// lives in active memory (the un-evaluated cache-graph chain suspect) or
// the allocator cache, and whether it steps per canvas or per step.
//
//	go test -tags model_eval -run 'TestDiffusionPerStepMemory_LiveModel$' -count=1 -v -timeout 30m ./pkg/metal/model/gemma4
func TestDiffusionPerStepMemory_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/diffusiongemma-26B-A4B-it-4bit")
	const watchdogBytes = uint64(38) << 30
	metal.SetMemoryLimit(uint64(40) << 30)

	m, err := LoadDiffusionGemma(dir)
	if err != nil {
		t.Fatalf("LoadDiffusionGemma: %v", err)
	}
	defer closeGemma4(m.Gemma4Model)
	t.Logf("loaded: active=%dMiB cache=%dMiB", metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20)

	// ~3.5k-token prompt — the prefix where the C015 book cliffed (ch5).
	filler := ""
	for i := 0; i < 180; i++ {
		filler += "The clockmaker wound every spring in the workshop while the brass wheels turned through the quiet afternoon. "
	}
	prompt := filler + "\nWrite the next chapter of the story.\n"

	const (
		canvasLen   = int32(DefaultCanvasLength)
		maxCanvases = 13 // ceil(800/64) — the serve's chapter budget
	)
	promptTokens := m.Tok.Encode(prompt)
	capacity := len(promptTokens) + (int(canvasLen)+8)*maxCanvases + 64
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(capacity)
	}
	defer metal.ClearCache()
	defer metal.FreeCaches(caches)

	cfg := DiffusionGenerateConfig{
		Step:         DefaultDiffusionStepConfig(m.Cfg.VocabSize),
		CanvasLength: canvasLen,
		MaxCanvases:  maxCanvases,
		StopTokens:   []int32{-1}, // run the full budget
	}
	cfg.Step.Seed = 42

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var tripped bool
	cfg.OnStep = func(canvasIdx, step int, res DiffusionStepResult, d time.Duration) {
		active := metal.GetActiveMemory()
		t.Logf("STEP c%02d s%02d active=%6dMiB cache=%6dMiB accepted=%2d entropy=%.3f fwd=%4dms smp=%4dms",
			canvasIdx, step, active>>20, metal.GetCacheMemory()>>20,
			res.Accepted, res.MeanEntropy, res.ForwardDur.Milliseconds(), res.SampleDur.Milliseconds())
		if active > watchdogBytes {
			tripped = true
			t.Errorf("WATCHDOG: active=%dMiB at canvas %d step %d", active>>20, canvasIdx, step)
			cancel()
		}
	}
	cfg.OnCanvas = func(canvasIdx int, kept []int32, steps int, d time.Duration) {
		t.Logf("CANVAS %02d: steps=%d kept=%d %.1fs active=%dMiB cache=%dMiB",
			canvasIdx, steps, len(kept), d.Seconds(), metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20)
	}

	_, dm, err := m.GenerateDiffusion(ctx, prompt, caches, cfg)
	if err != nil && !tripped {
		t.Fatalf("GenerateDiffusion: %v", err)
	}
	t.Logf("END: prefix=%d emitted=%d canvases=%d steps=%d prefill=%.1fs denoise=%.1fs | active=%dMiB cache=%dMiB peak=%dMiB",
		dm.PrefillTokens, dm.EmittedTokens, dm.Canvases, dm.TotalSteps,
		dm.PrefillDur.Seconds(), dm.DenoiseDur.Seconds(),
		metal.GetActiveMemory()>>20, metal.GetCacheMemory()>>20, metal.GetPeakMemory()>>20)
}
