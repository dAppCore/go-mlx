// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// The block-diffusion generation loop (docs/RFC.diffusion-gemma.md, Unit C).
//
// Canvases generate autoregressively: each one starts as uniform-random
// tokens and denoises in place (Unit B steps) against the committed prefix,
// then commits causally into the KV cache — the encoder role. The trunk's
// loaded LayerScalars are the DECODER (denoise) role; encoder-mode forwards
// (prompt prefill, canvas commits) swap in the encoder-role scalars.

// The measured decode profile (wave-2 sweep, docs/RFC.diffusion-gemma.md):
// canvas 64 at 16 max steps converges in ~7-10 steps per canvas and runs ~2x
// the checkpoint's declared profile (canvas 256, 48 steps) on the 4bit
// checkpoint, with no quality loss — shorter canvases are more
// autoregressive, not less coherent. Pushing harder backfires: steps 12
// anneals too fast (the canvas destabilises and re-converges), canvas 32
// trades the step-cost win back as step count. The per-step floor is
// ~60ms fixed + ~0.85ms/token (kernel-level — the next wave's target).
const (
	// DefaultCanvasLength is the tuned per-canvas token count. The
	// checkpoint declares 256; pass that explicitly for the reference shape.
	DefaultCanvasLength = 64
	// DefaultMaxSteps bounds (and paces — the noise schedule is
	// 1 - step/MaxSteps) the denoising loop. The reference ships 48.
	DefaultMaxSteps = 16
)

// DiffusionGenerateConfig drives GenerateDiffusion.
type DiffusionGenerateConfig struct {
	Step DiffusionStepConfig
	// CanvasLength 0 adopts DefaultCanvasLength (the checkpoint's declared
	// canvas is m.CanvasLength — pass it explicitly for the reference shape).
	CanvasLength int32
	// MaxSteps 0 adopts DefaultMaxSteps. The noise schedule derives from it,
	// so it paces the anneal as well as capping the loop.
	MaxSteps int
	// StabilityThreshold is how many consecutive steps the argmax canvas
	// must hold unchanged before convergence (reference default 1).
	StabilityThreshold int
	// ConfidenceThreshold is the mean per-token entropy the canvas must
	// fall below for convergence (reference default 0.005).
	ConfidenceThreshold float32
	// MaxCanvases bounds the response length (canvases x canvas tokens).
	MaxCanvases int
	// StopTokens end the response; defaults to the checkpoint's eos ids.
	StopTokens []int32
	// OnStep observes every denoising step (the diffuse verb's trace).
	OnStep func(canvasIdx, step int, res DiffusionStepResult, d time.Duration)
	// OnCanvas observes every committed canvas.
	OnCanvas func(canvasIdx int, kept []int32, steps int, d time.Duration)
}

// DiffusionMetrics reports one GenerateDiffusion run.
type DiffusionMetrics struct {
	Canvases       int
	TotalSteps     int
	EmittedTokens  int
	PrefillTokens  int
	PrefillDur     time.Duration
	DenoiseDur     time.Duration
	CommitDur      time.Duration
	TotalDur       time.Duration
	StoppedOnToken bool
}

// GenerateDiffusion runs the full block-diffusion loop: causal prompt
// prefill (encoder role), then canvases of denoise-and-commit until a stop
// token, MaxCanvases, or ctx cancellation. Returns the generated token ids
// (stop token excluded) — the caller decodes text and owns the caches.
func (m *DiffusionGemmaModel) GenerateDiffusion(ctx context.Context, prompt string, caches []metal.Cache, cfg DiffusionGenerateConfig) ([]int32, DiffusionMetrics, error) {
	const op = "gemma4.GenerateDiffusion"
	var metrics DiffusionMetrics
	start := time.Now()

	canvasLen := cfg.CanvasLength
	if canvasLen <= 0 {
		canvasLen = DefaultCanvasLength
	}
	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = DefaultMaxSteps
	}
	stability := cfg.StabilityThreshold
	if stability <= 0 {
		stability = 1
	}
	confidence := cfg.ConfidenceThreshold
	if confidence <= 0 {
		confidence = 0.005
	}
	maxCanvases := cfg.MaxCanvases
	if maxCanvases <= 0 {
		maxCanvases = 1
	}
	stops := cfg.StopTokens
	if len(stops) == 0 {
		stops = m.EOSTokens
	}
	stepCfg := cfg.Step
	if stepCfg.TextVocabSize <= 0 {
		stepCfg.TextVocabSize = m.Cfg.VocabSize
	}

	// Prompt prefill — encoder mode: causal, writes the prefix.
	promptTokens := m.Tok.Encode(prompt)
	if len(promptTokens) == 0 {
		return nil, metrics, core.E(op, "prompt encoded to zero tokens", nil)
	}
	prefillStart := time.Now()
	var prefillErr error
	m.withEncoderScalars(func() {
		promptArr := metal.FromValues(promptTokens, 1, len(promptTokens))
		logits := m.Forward(promptArr, caches)
		prefillErr = metal.Eval(logits)
		metal.Free(promptArr, logits)
	})
	if prefillErr != nil {
		return nil, metrics, core.E(op, "prompt prefill", prefillErr)
	}
	metrics.PrefillTokens = len(promptTokens)
	metrics.PrefillDur = time.Since(prefillStart)

	emitted := make([]int32, 0, int(canvasLen)*maxCanvases)

	for canvasIdx := 0; canvasIdx < maxCanvases; canvasIdx++ {
		prefix := caches[0].Offset()
		canvasStart := time.Now()

		// Fresh noise canvas, keyed off the seed + canvas index.
		initKey := metal.RandomKey(stepCfg.Seed ^ (uint64(canvasIdx+1) << 32))
		initF := metal.RandomUniformWithKey(0, float32(stepCfg.TextVocabSize), []int32{canvasLen}, metal.DTypeFloat32, initKey)
		if err := metal.Eval(initF); err != nil {
			metal.Free(initKey, initF)
			return emitted, metrics, core.E(op, "initial canvas", err)
		}
		canvas := make([]int32, canvasLen)
		for i, v := range initF.Floats()[:canvasLen] {
			id := int32(v)
			if id >= stepCfg.TextVocabSize {
				id = stepCfg.TextVocabSize - 1
			}
			canvas[i] = id
		}
		metal.Free(initKey, initF)

		// The denoising loop. Step keys fold the canvas index so every
		// canvas draws an independent chain. The masks depend only on the
		// canvas position — build once, reuse every step.
		canvasStepCfg := stepCfg
		canvasStepCfg.Seed = stepCfg.Seed + uint64(canvasIdx)*0x9E3779B97F4A7C15
		globalMask := diffusionGlobalCanvasMask(1, canvasLen, int32(prefix)+canvasLen)
		localMask := diffusionBlockLocalCanvasMask(1, canvasLen, int32(prefix)+canvasLen, int32(prefix), m.Cfg.SlidingWindow)

		// Convergence per the reference: the argmax canvas unchanged for
		// StabilityThreshold consecutive steps AND mean entropy under
		// ConfidenceThreshold — or the step cap. The COMMIT is always the
		// clean argmax canvas, never the noisy sampled one.
		var scEmb *metal.Array
		var prevGreedy []int32
		var lastGreedy []int32
		stableRun := 0
		steps := 0
		for step := 0; step < maxSteps; step++ {
			select {
			case <-ctx.Done():
				metal.Free(scEmb, globalMask, localMask)
				return emitted, metrics, ctx.Err()
			default:
			}
			stepStart := time.Now()
			noise := 1.0 - float32(step)/float32(maxSteps)

			canvasArr := metal.FromValues(canvas, 1, int(canvasLen))
			logits := m.DenoiseForwardWithMasks(canvasArr, scEmb, caches, globalMask, localMask)
			metal.Free(canvasArr)
			forwardDur := time.Since(stepStart)
			// Split the forward eval: host command-encode vs GPU-wait (lever diagnostic).
			encStart := time.Now()
			_ = metal.EvalAsync(logits)
			encodeDur := time.Since(encStart)
			gpuStart := time.Now()
			metal.Synchronize(metal.DefaultStream())
			gpuDur := time.Since(gpuStart)
			res, err := m.SampleDenoiseStep(logits, canvas, step, noise, canvasStepCfg)
			metal.Free(logits)
			if err != nil {
				metal.Free(scEmb, globalMask, localMask)
				return emitted, metrics, err
			}
			for i, c := range caches {
				if !c.(*metal.FixedKVCache).TruncateTo(prefix) {
					metal.Free(scEmb, res.SCEmb, globalMask, localMask)
					return emitted, metrics, core.E(op, core.Sprintf("cache %d declined TruncateTo(%d)", i, prefix), nil)
				}
			}
			res.ForwardDur = forwardDur
			res.SampleDur = time.Since(stepStart) - forwardDur
			res.EncodeDur = encodeDur
			res.GpuDur = gpuDur
			steps++
			metrics.TotalSteps++
			if cfg.OnStep != nil {
				cfg.OnStep(canvasIdx, step, res, time.Since(stepStart))
			}

			if prevGreedy != nil && int32SlicesEqual(res.Greedy, prevGreedy) {
				stableRun++
			} else {
				stableRun = 0
			}
			prevGreedy = res.Greedy
			lastGreedy = res.Greedy
			metal.Free(scEmb)
			scEmb = res.SCEmb
			if stableRun >= stability && res.MeanEntropy < confidence {
				break
			}
			canvas = res.Canvas
		}
		metal.Free(scEmb, globalMask, localMask)
		if lastGreedy != nil {
			canvas = lastGreedy
		}
		metrics.DenoiseDur += time.Since(canvasStart)

		// Stop-token truncate: keep up to (excluding) the first stop.
		kept := canvas
		stopped := false
		for i, id := range canvas {
			if tokenInSet(id, stops) {
				kept = canvas[:i]
				stopped = true
				break
			}
		}

		// Commit the kept tokens causally — encoder role. The reference
		// commits the full canvas including pads; committing only the kept
		// prefix keeps the cache clean for the next canvas and changes
		// nothing for a stopped response (generation ends here).
		if len(kept) > 0 {
			commitStart := time.Now()
			var commitErr error
			m.withEncoderScalars(func() {
				keptArr := metal.FromValues(kept, 1, len(kept))
				logits := m.Forward(keptArr, caches)
				commitErr = metal.Eval(logits)
				metal.Free(keptArr, logits)
			})
			if commitErr != nil {
				return emitted, metrics, core.E(op, "canvas commit", commitErr)
			}
			metrics.CommitDur += time.Since(commitStart)
		}

		emitted = append(emitted, kept...)
		metrics.Canvases++
		metrics.EmittedTokens = len(emitted)
		if cfg.OnCanvas != nil {
			cfg.OnCanvas(canvasIdx, kept, steps, time.Since(canvasStart))
		}
		if stopped {
			metrics.StoppedOnToken = true
			break
		}
	}

	metrics.TotalDur = time.Since(start)
	return emitted, metrics, nil
}

// withEncoderScalars runs fn with the encoder-role layer scalars swapped in.
// One weight-tied trunk serves both roles; the per-layer scalar is the only
// parametric difference (the decoder set loads onto the layers, the encoder
// set rides DiffusionGemmaModel). Single-flight generation only.
func (m *DiffusionGemmaModel) withEncoderScalars(fn func()) {
	if len(m.EncoderLayerScalars) != len(m.Layers) {
		fn()
		return
	}
	for i, l := range m.Layers {
		l.LayerScalar, m.EncoderLayerScalars[i] = m.EncoderLayerScalars[i], l.LayerScalar
	}
	defer func() {
		for i, l := range m.Layers {
			l.LayerScalar, m.EncoderLayerScalars[i] = m.EncoderLayerScalars[i], l.LayerScalar
		}
	}()
	fn()
}

func int32SlicesEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func tokenInSet(id int32, set []int32) bool {
	for _, s := range set {
		if id == s {
			return true
		}
	}
	return false
}
