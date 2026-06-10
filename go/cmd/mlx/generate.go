// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx"
)

// runGenerateCommand loads a model and generates from a prompt with no HTTP
// serve in the path, reporting decode-only tok/s (prefill excluded) for
// like-for-like comparison against other engines on the same model + quant
// (e.g. llama-cli / llama-bench). It prints the generated text too, so it
// doubles as a quick one-shot run.
//
//	lthn-mlx generate ~/models/gemma-4-e2b-it-4bit
//	lthn-mlx generate -max-tokens 256 ~/models/lemer-lite
func runGenerateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("generate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "user prompt")
	maxTokens := fs.Int("max-tokens", 128, "tokens to generate")
	temp := fs.Float64("temp", 1.0, "sampling temperature (0 = greedy/argmax — fastest, fair vs llama-bench)")
	think := fs.Bool("think", false, "enable the thinking channel (off keeps the decode rate clean)")
	contextLen := fs.Int("context", 0, "context length override (0 = model default)")
	tracePhases := fs.Bool("trace", false, "print the per-token decode time budget — GPU wait vs host-serial work (runs greedy and sampled lanes; ignores -temp)")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s generate [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Load a model and generate from a prompt with no HTTP serve in the path,\n")
		core.WriteString(stderr, "reporting decode-only tok/s (prefill excluded) for like-for-like benching\n")
		core.WriteString(stderr, "against other engines on the same model + quant (e.g. llama-bench). The\n")
		core.WriteString(stderr, "generated text is printed too, so it also serves as a quick one-shot run.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s generate ~/models/gemma-4-e2b-it-4bit\n", name))
		core.WriteString(stderr, "    # one-shot generate + decode tok/s\n")
		core.WriteString(stderr, core.Sprintf("  %s generate -max-tokens 256 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, "    # 256-token decode rate, for like-for-like comparison\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s generate: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	loadOpts := []mlx.LoadOption{}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, mlx.WithContextLength(*contextLen))
	}
	if *tracePhases {
		return runGenerateTrace(ctx, fs.Arg(0), *prompt, *maxTokens, loadOpts, stdout, stderr)
	}
	tm, err := mlx.LoadModelAsTextModel(fs.Arg(0), loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}

	off := !*think
	msgs := []inference.Message{{Role: "user", Content: *prompt}}

	// run generates up to limit tokens and times prefill (start → first token)
	// separately from decode (first → last token), so the reported rate is the
	// steady-state decode rate, comparable to llama-bench's tg.
	run := func(limit int, collect *[]byte) (n int, prefill, decode time.Duration) {
		start := time.Now()
		var first time.Time
		for tok := range tm.Chat(ctx, msgs, inference.WithMaxTokens(limit), inference.WithEnableThinking(&off), inference.WithTemperature(float32(*temp))) {
			if n == 0 {
				first = time.Now()
				prefill = first.Sub(start)
			}
			if collect != nil {
				*collect = append(*collect, tok.Text...)
			}
			n++
		}
		decode = time.Since(first)
		return n, prefill, decode
	}

	run(8, nil) // warm the kernels — first call pays compilation + allocation
	if err := tm.Err(); err != nil {
		core.Print(stderr, "%s generate: warm: %v", cliName(), err)
		return 1
	}
	var out []byte
	n, prefill, decode := run(*maxTokens, &out)
	if err := tm.Err(); err != nil {
		core.Print(stderr, "%s generate: %v", cliName(), err)
		return 1
	}
	if n < 2 {
		core.Print(stderr, "%s generate: produced only %d tokens", cliName(), n)
		return 1
	}

	core.WriteString(stdout, string(out))
	core.WriteString(stdout, "\n\n")
	core.WriteString(stdout, core.Sprintf(
		"decode %.1f tok/s  (%d tok / %.3fs, prefill %dms excluded)  ·  total %.1f tok/s\n",
		float64(n-1)/decode.Seconds(), n, decode.Seconds(), prefill.Milliseconds(),
		float64(n)/(prefill+decode).Seconds(),
	))
	return 0
}

// runGenerateTrace loads the model once via the root API and prints the
// per-token decode time budget from the engine's phase trace: how long the
// host blocks waiting on the GPU result versus how long it spends in serial
// host work (graph build, detokenise, yield) while the GPU sits idle. The
// split locates where decode tok/s goes. Both lanes run on the same load.
func runGenerateTrace(ctx context.Context, modelPath, prompt string, maxTokens int, loadOpts []mlx.LoadOption, stdout, stderr io.Writer) int {
	m, err := mlx.LoadModel(modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}
	defer m.Close()

	msgs := []inference.Message{{Role: "user", Content: prompt}}
	run := func(temp float32, limit int, trace bool) bool {
		opts := []mlx.GenerateOption{mlx.WithMaxTokens(limit), mlx.WithTemperature(temp)}
		if trace {
			opts = append(opts, mlx.WithTokenPhaseTrace())
		}
		for range m.ChatTokens(ctx, msgs, opts...) {
		}
		if err := m.Err(); err != nil {
			core.Print(stderr, "%s generate: %v", cliName(), err)
			return false
		}
		return true
	}

	if !run(0, 8, false) { // warm: kernel compilation + allocation
		return 1
	}
	lanes := []struct {
		name string
		temp float32
	}{
		{"greedy (temp=0)", 0},
		{"sampled (temp=1)", 1},
	}
	for _, lane := range lanes {
		if !run(lane.temp, maxTokens, true) {
			return 1
		}
		printTokenPhaseBudget(stdout, lane.name, m.Metrics())
	}
	return 0
}

// printTokenPhaseBudget averages the engine's per-token phase trace over the
// warm tokens (step 0 and the final token are skipped) and reports the
// GPU-wait vs host-serial split plus each phase's share.
func printTokenPhaseBudget(stdout io.Writer, lane string, metrics mlx.Metrics) {
	type row struct {
		name string
		get  func(mlx.TokenPhaseTrace) time.Duration
	}
	rows := []row{
		{"token-read wait (GPU busy)", func(p mlx.TokenPhaseTrace) time.Duration { return p.TokenReadDuration }},
		{"sample eval wait (GPU busy)", func(p mlx.TokenPhaseTrace) time.Duration { return p.SampleEvalDuration }},
		{"forward graph build (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.ForwardDuration }},
		{"logits slice (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.LogitsDuration }},
		{"sample build (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.SampleDuration }},
		{"detach (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.DetachDuration }},
		{"decode text (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.DecodeTextDuration }},
		{"yield to consumer (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.YieldDuration }},
		{"next input upload (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.NextInputDuration }},
		{"prefetch submit (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchDuration }},
		{"  prefetch: logits graph", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchLogitsDuration }},
		{"  prefetch: cache state", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchCacheDuration }},
		{"materialize (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.MaterializeDuration }},
		{"cache probe (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.CacheProbeDuration }},
		{"probe token (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.ProbeTokenDuration }},
		{"other (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.OtherDuration }},
	}

	var n int
	var total, gpu time.Duration
	sums := make([]time.Duration, len(rows))
	for _, p := range metrics.TokenPhases {
		if p.Step == 0 || p.FinalToken {
			continue
		}
		n++
		total += p.TotalDuration
		gpu += p.TokenReadDuration + p.SampleEvalDuration
		for i, r := range rows {
			sums[i] += r.get(p)
		}
	}
	if n == 0 {
		core.WriteString(stdout, core.Sprintf("%s: no warm token phases captured\n", lane))
		return
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000.0 / float64(n) }
	avgTotal := ms(total)
	avgGPU := ms(gpu)
	avgHost := avgTotal - avgGPU
	core.WriteString(stdout, core.Sprintf("\n%s — %d warm tokens · %.3f ms/token · %.1f tok/s\n",
		lane, n, avgTotal, 1000.0/avgTotal))
	core.WriteString(stdout, core.Sprintf("  GPU wait   %8.3f ms  %5.1f%%\n", avgGPU, 100*avgGPU/avgTotal))
	core.WriteString(stdout, core.Sprintf("  host serial%8.3f ms  %5.1f%%   <- GPU idle; tok/s ceiling if zeroed: %.1f\n",
		avgHost, 100*avgHost/avgTotal, 1000.0/avgGPU))
	for i, r := range rows {
		avg := ms(sums[i])
		if avg < 0.001 {
			continue
		}
		core.WriteString(stdout, core.Sprintf("    %-28s %8.3f ms  %5.1f%%\n", r.name, avg, 100*avg/avgTotal))
	}
}
