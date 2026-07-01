// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	gemma4chat "dappco.re/go/mlx/pkg/model/gemma4/chat"
)

// runDiffuseCommand generates text through the block-diffusion sampler:
// canvases of tokens denoised in parallel against the committed prefix, then
// committed causally — the DiffusionGemma decoding loop, with a per-step
// trace (accepted, changed, ms/step) that shows the denoiser converging.
func runDiffuseCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("diffuse", flag.ContinueOnError)
	fs.SetOutput(stderr)
	prompt := fs.String("prompt", "Write a haiku about clockwork.", "user prompt")
	maxCanvases := fs.Int("max-canvases", 8, "response length bound, in canvases")
	steps := fs.Int("steps", 0, "max denoising steps per canvas (0 = tuned default 16; paces the anneal)")
	canvas := fs.Int("canvas", 0, "canvas length (0 = tuned default 64; the checkpoint declares 256)")
	entropy := fs.Float64("entropy", 0.3, "acceptance entropy budget per step (0.5+ backfires)")
	seed := fs.Uint64("seed", 0, "PRNG key chain root (0 = time-derived)")
	chatFlag := fs.Bool("chat", true, "format the prompt with the model chat template")
	trace := fs.Bool("trace", false, "print one line per denoising step")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: lthn-mlx diffuse [flags] <model-path>\n\n")
		core.WriteString(stderr, "Generate text with the block-diffusion sampler (DiffusionGemma):\n")
		core.WriteString(stderr, "whole canvases denoise in parallel and commit autoregressively.\n\n")
		core.WriteString(stderr, "Flags:\n")
		fs.PrintDefaults()
		core.WriteString(stderr, "\nExample:\n")
		core.WriteString(stderr, "    lthn-mlx diffuse -trace -prompt 'Explain entropy briefly.' <model>\n")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return 2
	}

	m, err := gemma4.LoadDiffusionGemma(fs.Arg(0))
	if err != nil {
		core.Print(stderr, "%s diffuse: load: %v", cliName(), err)
		return 1
	}
	defer m.Close()

	formatted := *prompt
	if *chatFlag {
		formatted = gemma4chat.Format(
			[]chat.Message{{Role: "user", Content: *prompt}},
			chat.Config{},
		)
	}

	canvasLen := int32(*canvas)
	if canvasLen <= 0 {
		canvasLen = gemma4.DefaultCanvasLength
	}
	promptTokens := len(m.Tok.Encode(formatted))
	capacity := promptTokens + (int(canvasLen)+8)*(*maxCanvases) + 64
	caches := make([]metal.Cache, m.NumLayers())
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(capacity)
	}
	defer metal.FreeCaches(caches)

	cfg := gemma4.DiffusionGenerateConfig{
		Step:         gemma4.DefaultDiffusionStepConfig(0),
		CanvasLength: canvasLen,
		MaxSteps:     *steps, // 0 resolves to the tuned DefaultMaxSteps
		MaxCanvases:  *maxCanvases,
	}
	cfg.Step.EntropyBound = float32(*entropy)
	cfg.Step.Seed = *seed
	if cfg.Step.Seed == 0 {
		cfg.Step.Seed = uint64(time.Now().UnixNano())
	}
	if *trace {
		cfg.OnStep = func(canvasIdx, step int, res gemma4.DiffusionStepResult, d time.Duration) {
			core.WriteString(stderr, core.Sprintf(
				"canvas %d · step %2d · acc %3d · H %.3f · build %4.1f + encode %5.1f + gpu %5.1f = %5.1f ms\n",
				canvasIdx, step, res.Accepted, res.MeanEntropy,
				float64(res.ForwardDur.Microseconds())/1000.0,
				float64(res.EncodeDur.Microseconds())/1000.0,
				float64(res.GpuDur.Microseconds())/1000.0,
				float64(d.Microseconds())/1000.0))
		}
	}
	cfg.OnCanvas = func(canvasIdx int, kept []int32, steps int, d time.Duration) {
		core.WriteString(stderr, core.Sprintf(
			"canvas %d done · %d tokens kept · %d steps · %.2fs\n",
			canvasIdx, len(kept), steps, d.Seconds()))
	}

	ids, metrics, err := m.GenerateDiffusion(ctx, formatted, caches, cfg)
	if err != nil {
		core.Print(stderr, "%s diffuse: %v", cliName(), err)
		return 1
	}

	out := m.Tok.Decode(ids)
	core.WriteString(stdout, out)
	core.WriteString(stdout, "\n\n")
	rate := 0.0
	denoise := metrics.DenoiseDur.Seconds()
	if metrics.TotalDur > 0 {
		rate = float64(metrics.EmittedTokens) / metrics.TotalDur.Seconds()
	}
	msPerStep := 0.0
	if metrics.TotalSteps > 0 {
		msPerStep = metrics.DenoiseDur.Seconds() * 1000.0 / float64(metrics.TotalSteps)
	}
	core.WriteString(stdout, core.Sprintf(
		"diffusion %.1f tok/s overall  ·  %d tokens / %d canvases / %d steps  ·  %.1f ms/step  ·  denoise %.2fs + commit %.2fs + prefill %dms  ·  stopped=%v\n",
		rate, metrics.EmittedTokens, metrics.Canvases, metrics.TotalSteps, msPerStep,
		denoise, metrics.CommitDur.Seconds(), metrics.PrefillDur.Milliseconds(), metrics.StoppedOnToken))
	return 0
}
