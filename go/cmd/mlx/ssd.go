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
	"dappco.re/go/inference/probe"
)

// runSSDCommand drives native self-distillation sampling (#50/#97) — the
// no-correct-answer lane, how LEK-P0 applies. It samples raw outputs from the
// FROZEN base over a prompt set (nothing taught: no reference answer, no
// verifier, no teacher), captures every return, and scores each self-sample at
// birth. It STOPS at the scored trace (ssd-captures.jsonl + ssd-samples-score.jsonl):
// a stronger lab model picks steps from the trace and re-performs the sequence
// into the SFT artifact, which a separate `sft` run trains on. SSD never trains.
//
// The kernel lane (#97): --kernel prefixes the LEK-2 kernel onto every GENERATION
// prompt (prefilled once as KV state), but the captured rows keep the BARE
// prompt — the trace records how it speaks UNDER the kernel, never the kernel's words.
//
// Live-run verb — it loads the model and samples; run it deliberately.
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
	// output: ssd stops at the scored trace — captures + sample scores land
	// in this dir (the lab's pick-steps surface). Training is a separate sft run.
	checkpointDir := fs.String("checkpoint-dir", "", "output dir for the scored trace — ssd-captures.jsonl + ssd-samples-score.jsonl")
	runID := fs.String("run-id", "", "run identity tag for the metrics sink (default ssd-<timestamp>)")
	metricsLp := fs.String("metrics-lp", "", "append v0-schema line protocol here (default <checkpoint-dir>/metrics.lp; \"off\" disables)")
	influxURL := fs.String("influx-url", "", "InfluxDB write URL — streams the same lines live")
	influxToken := fs.String("influx-token", "", "InfluxDB API token for --influx-url")
	contextLen := fs.Int("context", 0, "model context override; 0 uses the model default")

	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ssd --model <base> --data <prompts.jsonl> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Self-distillation sampling (no-correct-answer): sample the FROZEN base over\n")
		core.WriteString(stderr, "the prompts, capture + score each self-output at birth, and STOP at the\n")
		core.WriteString(stderr, "scored trace. Nothing is taught — no reference answer, no verifier. The lab\n")
		core.WriteString(stderr, "refines the trace into the SFT artifact; a separate `sft` run trains on it.\n")
		core.WriteString(stderr, "--kernel rides generation as KV state but never enters the captured rows (#97).\n")
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

	// SSD owns only sampling + scoring; the SFT sub-config carries just the
	// trace output dir (training is a separate sft run on the lab's artifact).
	sft := mlx.SFTConfig{CheckpointDir: *checkpointDir}
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
		core.Print(stdout, "ssd trace %s  (the lab picks steps from this + the scores)\n", result.CaptureSidecar)
	}
	core.Print(stdout, "next: refine the trace in the lab, then  %s sft --data <artifact> --model %s\n", cliName(), *modelPath)
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
