// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
)

// runSSDCommand drives native self-distillation (#50/#97) — the no-correct-answer
// lane, how LEK-P0 applies. It samples raw outputs from the FROZEN base over a
// prompt set (nothing taught: no reference answer, no verifier, no teacher),
// captures every return, optionally scores each self-sample at birth, then
// fine-tunes the model on its OWN responses with the native SFT path. The first
// run therefore *creates its own SFT corpus* (the capture sidecar) — a reusable
// artifact you can refine on again.
//
// The kernel lane (#97): --kernel prefixes the LEK-2 kernel onto every GENERATION
// prompt (prefilled once as KV state), but the fine-tune rows keep the BARE
// prompt — the model learns from how it speaks UNDER the kernel, never from the
// kernel's words.
//
// Live-run verb — it loads the model, samples, and trains; run it deliberately.
//
//	lthn-mlx ssd --model ~/models/gemma-4-E2B-it-bf16 --data .../lek2.jsonl \
//	  --kernel .lek/lek2-kernel.txt --checkpoint-dir ~/Lethean/data/ssd/run1
func runSSDCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ssd"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "frozen base model path to self-distil (required)")
	dataPath := fs.String("data", "", "prompt JSONL — {\"messages\":[…]} or {\"prompt\":…} per line; only prompts are read, responses are self-generated (required)")
	kernelPath := fs.String("kernel", "", "file holding the LEK-2 kernel prefix — rides every generation as KV state, never enters the training rows (#97)")
	// sampling phase (frozen base → raw self-output); engine defaults, temp must be non-unit
	sampleMaxTokens := fs.Int("sample-max-tokens", 256, "tokens per self-generated sample")
	sampleTemp := fs.Float64("sample-temp", 0.7, "sampling temperature (must be non-unit ≠ 1.0 — diversity is the point)")
	sampleTopK := fs.Int("sample-top-k", 64, "sampling top-k")
	sampleTopP := fs.Float64("sample-top-p", 0.95, "sampling top-p")
	sampleMinP := fs.Float64("sample-min-p", 0, "sampling min-p")
	repPenalty := fs.Float64("rep-penalty", 1.0, "repetition penalty over self-samples")
	filterShortest := fs.Float64("filter-shortest", 10, "drop the shortest N%% of self-samples before fine-tuning (0 keeps all)")
	scoreSamples := fs.Bool("score-samples", true, "lem-scorer over every self-sample at birth → ssd-samples-score.jsonl (the no-correct-answer quality read)")
	// fine-tune phase (refine on the kept self-samples) — shared SFT surface
	rank := fs.Int("rank", 8, "LoRA rank")
	alpha := fs.Float64("alpha", 32, "LoRA alpha")
	lr := fs.Float64("lr", 1e-4, "AdamW learning rate")
	epochs := fs.Int("epochs", 1, "fine-tune epochs over the self-generated corpus")
	batch := fs.Int("batch", 1, "batch size")
	gradAccum := fs.Int("grad-accum", 4, "gradient accumulation steps")
	maxSeqLen := fs.Int("max-seq", 1024, "max sequence length")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint dir (also hosts ssd-captures.jsonl, ssd-samples-score.jsonl, the score cascade)")
	checkpointEvery := fs.Int("checkpoint-every", 50, "save a checkpoint every N optimizer steps (0 disables)")
	savePath := fs.String("save", "", "final adapter path (default <checkpoint-dir>/adapter.safetensors)")
	merge := fs.Bool("merge", false, "merge the adapter into the model weights after training")
	evalEvery := fs.Int("eval-every", 25, "run eval probes every N optimizer steps (0 disables eval + cascade)")
	evalPromptsPath := fs.String("eval-prompts", "", "file of eval probes, one per line (overrides --data derivation)")
	evalProbes := fs.Int("eval-probes", 4, "probes derived from --data when --eval-prompts is absent")
	evalMaxTokens := fs.Int("eval-max-tokens", 200, "tokens per eval generation")
	scoreCascade := fs.Bool("score-cascade", true, "lem-scorer over every eval pass: best checkpoint by windowed composite")
	scoreWindow := fs.Int("score-window", 3, "eval passes per windowed composite")
	runID := fs.String("run-id", "", "run identity tag for the metrics sink (default ssd-<timestamp>)")
	metricsLp := fs.String("metrics-lp", "", "append v0-schema line protocol here (default <checkpoint-dir>/metrics.lp; \"off\" disables)")
	influxURL := fs.String("influx-url", "", "InfluxDB write URL — streams the same lines live")
	influxToken := fs.String("influx-token", "", "InfluxDB API token for --influx-url")
	contextLen := fs.Int("context", 0, "model context override; 0 uses the model default")

	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ssd --model <base> --data <prompts.jsonl> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Self-distillation (no-correct-answer): sample the FROZEN base over the\n")
		core.WriteString(stderr, "prompts, capture + score each self-output at birth, then fine-tune the\n")
		core.WriteString(stderr, "model on its own responses. Nothing is taught — no reference answer, no\n")
		core.WriteString(stderr, "verifier. The capture sidecar IS the self-generated SFT corpus. --kernel\n")
		core.WriteString(stderr, "rides generation as KV state but never enters the training rows (#97).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" || *dataPath == "" {
		fs.Usage()
		return 2
	}

	// Eval probes for the fine-tune-phase cascade: explicit file wins, else the
	// first user turns of the prompt set (fixed across the run).
	var prompts []string
	switch {
	case *evalPromptsPath != "":
		text, err := coreio.Local.Read(*evalPromptsPath)
		if err != nil {
			core.Print(stderr, "%s ssd: eval prompts unreadable: %v\n", cliName(), err)
			return 1
		}
		for _, line := range core.Split(text, "\n") {
			if trimmed := core.Trim(line); trimmed != "" {
				prompts = append(prompts, trimmed)
			}
		}
	default:
		derived, err := sftProbesFromValid(*dataPath, *evalProbes)
		if err != nil {
			core.Print(stderr, "%s ssd: deriving probes from --data failed: %v\n", cliName(), err)
			return 1
		}
		prompts = derived
	}

	dataFile, err := coreio.Local.Open(*dataPath)
	if err != nil {
		core.Print(stderr, "%s ssd: prompt data unreadable: %v\n", cliName(), err)
		return 1
	}
	defer dataFile.Close()
	ds, err := dataset.LoadJSONL(dataFile, dataset.Config{})
	if err != nil {
		core.Print(stderr, "%s ssd: prompt data parse: %v\n", cliName(), err)
		return 1
	}

	var kernel string
	if *kernelPath != "" {
		text, err := coreio.Local.Read(*kernelPath)
		if err != nil {
			core.Print(stderr, "%s ssd: kernel unreadable: %v\n", cliName(), err)
			return 1
		}
		kernel = text // verbatim — no normalisation (#97)
	}

	var loadOpts []mlx.LoadOption
	if *contextLen > 0 {
		loadOpts = append(loadOpts, mlx.WithContextLength(*contextLen))
	}
	core.Print(stderr, "%s ssd: loading %s\n", cliName(), *modelPath)
	m, err := mlx.LoadModel(*modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s ssd: model load: %v\n", cliName(), err)
		return 1
	}
	defer m.Close()

	save := *savePath
	if save == "" && *checkpointDir != "" {
		save = core.JoinPath(*checkpointDir, "adapter.safetensors")
	}

	lpPath := *metricsLp
	if lpPath == "" && *checkpointDir != "" {
		lpPath = core.JoinPath(*checkpointDir, "metrics.lp")
	}
	if lpPath == "off" {
		lpPath = ""
	}
	var metricsSink *probe.LineProtocolSink
	if lpPath != "" || *influxURL != "" {
		id := *runID
		if id == "" {
			id = "ssd-" + time.Now().Format("20060102-150405")
		}
		sinkCfg := probe.LineProtocolConfig{Model: core.PathBase(*modelPath), RunID: id, FilePath: lpPath}
		if *influxURL != "" {
			sinkCfg.Post = probe.NewInfluxPoster(*influxURL, *influxToken)
		}
		metricsSink = probe.NewLineProtocolSink(sinkCfg)
		defer metricsSink.Close()
	}

	sft := mlx.SFTConfig{
		LoRA:                      spine.LoRAConfig{Rank: *rank, Alpha: float32(*alpha)},
		BatchSize:                 *batch,
		GradientAccumulationSteps: *gradAccum,
		Epochs:                    *epochs,
		LearningRate:              *lr,
		MaxSeqLen:                 *maxSeqLen,
		CheckpointDir:             *checkpointDir,
		CheckpointEvery:           *checkpointEvery,
		EvalEvery:                 *evalEvery,
		EvalPrompts:               prompts,
		EvalMaxTokens:             *evalMaxTokens,
		SavePath:                  save,
		Merge:                     *merge,
		ScoreCascade:              *scoreCascade && *evalEvery > 0 && len(prompts) > 0,
		ScoreWindow:               *scoreWindow,
	}
	if metricsSink != nil {
		sft.ProbeSink = metricsSink
	}
	cfg := mlx.SSDConfig{
		SampleMaxTokens:       *sampleMaxTokens,
		SampleTemperature:     float32(*sampleTemp),
		SampleTopK:            *sampleTopK,
		SampleTopP:            float32(*sampleTopP),
		SampleMinP:            float32(*sampleMinP),
		RepetitionPenalty:     float32(*repPenalty),
		FilterShortestPercent: float32(*filterShortest),
		ScoreSamples:          *scoreSamples,
		KernelPrefix:          kernel,
		SFT:                   sft,
	}

	result, err := m.RunSSD(ctx, ds, cfg)
	if err != nil {
		core.Print(stderr, "%s ssd: self-distillation: %v\n", cliName(), err)
		return 1
	}

	core.Print(stdout, "self-samples %d  sample-temp %.2f  kernel %v\n",
		len(result.Samples), result.SampleTemperature, result.KernelApplied)
	if result.SampleScoreMean > 0 {
		core.Print(stdout, "sample-score mean %.2f over %d scored  (%s)\n",
			result.SampleScoreMean, len(result.SampleScores), result.SampleScoreSidecar)
	}
	if result.CaptureSidecar != "" {
		core.Print(stdout, "self-SFT corpus %s  (capture-first — refine on this again)\n", result.CaptureSidecar)
	}
	if sftRes := result.SFT; sftRes != nil {
		core.Print(stdout, "fine-tune: steps %d  epochs %d  samples %d  last-loss %.4f\n",
			sftRes.Steps, sftRes.Epochs, sftRes.Samples, sftRes.LastLoss)
		if sftRes.AdapterPath != "" {
			core.Print(stdout, "adapter %s\n", sftRes.AdapterPath)
		}
		if sftRes.BestScoreComposite > 0 {
			core.Print(stdout, "best step %d  windowed composite %.2f  (cascade: %s)\n",
				sftRes.BestScoreStep, sftRes.BestScoreComposite, sftRes.ScoreSidecarPath)
		}
	}
	if metricsSink != nil {
		metricsSink.Flush()
		core.Print(stdout, "metrics %d lines", metricsSink.Lines())
		if lpPath != "" {
			core.Print(stdout, "  → %s", lpPath)
		}
		core.Print(stdout, "\n")
	}
	return 0
}
