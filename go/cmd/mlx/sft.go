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

// runSFTCommand drives the native LoRA SFT loop (#50) over a {messages}
// JSONL set, with the score cascade riding every eval pass — the lem-scorer
// scores each (probe, output) at generation time into a sidecar, saved
// checkpoints carry the windowed composite, and the run names the best
// checkpoint by semantic analysis, not loss guesswork.
//
// This is a live-run verb — it loads the model and trains; the operator
// runs it deliberately.
//
//	lthn-mlx sft --model ~/models/LEM-gemma3-1b --data /Volumes/Data/lem/training/gold-full/train.jsonl \
//	  --valid /Volumes/Data/lem/training/gold-full/valid.jsonl \
//	  --rank 16 --epochs 2 --checkpoint-dir ~/Lethean/data/sft/run1
func runSFTCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("sft"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model path to fine-tune (required)")
	dataPath := fs.String("data", "", "training JSONL — {\"messages\":[{role,content}…]} per line (required)")
	validPath := fs.String("valid", "", "validation JSONL; feeds the val-loss curve AND derives eval probes from its first user turns when --eval-prompts is absent")
	valSamples := fs.Int("val-samples", 32, "samples in the fixed validation subset forwarded for val loss")
	valEvery := fs.Int("val-every", 0, "run the val-loss pass every N optimizer steps (0 follows --eval-every)")
	evalPromptsPath := fs.String("eval-prompts", "", "file of eval probes, one per line (overrides --valid derivation)")
	evalEvery := fs.Int("eval-every", 25, "run the eval probes every N optimizer steps (0 disables eval + cascade)")
	evalMaxTokens := fs.Int("eval-max-tokens", 200, "tokens per eval generation")
	evalProbes := fs.Int("eval-probes", 4, "probes derived from --valid when --eval-prompts is absent")
	scoreCascade := fs.Bool("score-cascade", true, "lem-scorer over every eval pass: sidecar JSONL + best checkpoint by windowed composite")
	scoreWindow := fs.Int("score-window", 3, "eval passes per windowed composite")
	runID := fs.String("run-id", "", "run identity tag for the metrics sink (default sft-<timestamp>)")
	metricsLp := fs.String("metrics-lp", "", "append v0-schema line protocol here (default <checkpoint-dir>/metrics.lp; \"off\" disables)")
	influxURL := fs.String("influx-url", "", "InfluxDB write URL — streams the same lines live (e.g. http://localhost:8086/api/v2/write?org=lem&bucket=training)")
	influxToken := fs.String("influx-token", "", "InfluxDB API token for --influx-url")
	rank := fs.Int("rank", 16, "LoRA rank")
	alpha := fs.Float64("alpha", 32, "LoRA alpha")
	lr := fs.Float64("lr", 1e-4, "AdamW learning rate")
	epochs := fs.Int("epochs", 1, "training epochs")
	batch := fs.Int("batch", 1, "batch size")
	gradAccum := fs.Int("grad-accum", 4, "gradient accumulation steps")
	maxSeqLen := fs.Int("max-seq", 1024, "max sequence length (longer samples truncate)")
	packing := fs.Bool("packing", false, "sequence packing (denser batches, order mixed)")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory (also hosts the score-cascade sidecar)")
	checkpointEvery := fs.Int("checkpoint-every", 50, "save a checkpoint every N optimizer steps (0 disables)")
	savePath := fs.String("save", "", "final adapter path (default <checkpoint-dir>/adapter.safetensors when a dir is set)")
	resumePath := fs.String("resume", "", "resume from a saved adapter checkpoint")
	merge := fs.Bool("merge", false, "merge the adapter into the model weights after training")
	contextLen := fs.Int("context", 0, "model context override; 0 uses the model default")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s sft --model <path> --data <train.jsonl> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Native LoRA SFT with the score cascade: the lem-scorer reads every eval\n")
		core.WriteString(stderr, "generation at the moment it happens, so the checkpoint is not a guess —\n")
		core.WriteString(stderr, "it's from semantic analysis. Scores land in score-cascade.jsonl beside\n")
		core.WriteString(stderr, "the checkpoints; each checkpoint's metadata carries its windowed\n")
		core.WriteString(stderr, "composite; the summary names the best step.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s sft --model ~/models/LEM-gemma3-1b \\\n", name))
		core.WriteString(stderr, "      --data /Volumes/Data/lem/training/gold-full/train.jsonl \\\n")
		core.WriteString(stderr, "      --valid /Volumes/Data/lem/training/gold-full/valid.jsonl \\\n")
		core.WriteString(stderr, "      --rank 16 --epochs 2 --checkpoint-dir ~/Lethean/data/sft/gold-1\n")
		core.WriteString(stderr, "    # gold-full LEM set; probes derive from valid; cascade on by default\n")
		core.WriteString(stderr, core.Sprintf("  %s sft --model <m> --data train.jsonl --eval-every 0\n", name))
		core.WriteString(stderr, "    # plain loss-only run — no eval, no cascade\n")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" || *dataPath == "" {
		fs.Usage()
		return 2
	}

	// Eval probes: an explicit file wins; otherwise the first user turns of
	// the validation set (the probes must stay FIXED across the run — the
	// cascade compares like with like).
	var prompts []string
	switch {
	case *evalPromptsPath != "":
		text, err := coreio.Local.Read(*evalPromptsPath)
		if err != nil {
			core.Print(stderr, "%s sft: eval prompts unreadable: %v\n", cliName(), err)
			return 1
		}
		for _, line := range core.Split(text, "\n") {
			if trimmed := core.Trim(line); trimmed != "" {
				prompts = append(prompts, trimmed)
			}
		}
	case *validPath != "":
		derived, err := sftProbesFromValid(*validPath, *evalProbes)
		if err != nil {
			core.Print(stderr, "%s sft: deriving probes from --valid failed: %v\n", cliName(), err)
			return 1
		}
		prompts = derived
	}

	dataFile, err := coreio.Local.Open(*dataPath)
	if err != nil {
		core.Print(stderr, "%s sft: training data unreadable: %v\n", cliName(), err)
		return 1
	}
	defer dataFile.Close()
	ds, err := dataset.LoadJSONL(dataFile, dataset.Config{})
	if err != nil {
		core.Print(stderr, "%s sft: training data parse: %v\n", cliName(), err)
		return 1
	}

	// The validation set feeds two instruments: the probe prompts above
	// and the val-loss curve — the train/val pair the cascade is read on.
	var validData dataset.Dataset
	if *validPath != "" {
		validFile, err := coreio.Local.Open(*validPath)
		if err != nil {
			core.Print(stderr, "%s sft: validation data unreadable: %v\n", cliName(), err)
			return 1
		}
		defer validFile.Close()
		validData, err = dataset.LoadJSONL(validFile, dataset.Config{})
		if err != nil {
			core.Print(stderr, "%s sft: validation data parse: %v\n", cliName(), err)
			return 1
		}
	}

	var loadOpts []mlx.LoadOption
	if *contextLen > 0 {
		loadOpts = append(loadOpts, mlx.WithContextLength(*contextLen))
	}
	core.Print(stderr, "%s sft: loading %s\n", cliName(), *modelPath)
	m, err := mlx.LoadModel(*modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s sft: model load: %v\n", cliName(), err)
		return 1
	}
	defer m.Close()

	save := *savePath
	if save == "" && *checkpointDir != "" {
		save = core.JoinPath(*checkpointDir, "adapter.safetensors")
	}

	// The metrics sink (#97): train/val loss + cascade scores as v0-schema
	// line protocol — the lp file is the durable copy, the Influx URL the
	// live hot store; both optional, training never depends on either.
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
			id = "sft-" + time.Now().Format("20060102-150405")
		}
		sinkCfg := probe.LineProtocolConfig{
			Model:    core.PathBase(*modelPath),
			RunID:    id,
			FilePath: lpPath,
		}
		if *influxURL != "" {
			sinkCfg.Post = probe.NewInfluxPoster(*influxURL, *influxToken)
		}
		metricsSink = probe.NewLineProtocolSink(sinkCfg)
		defer metricsSink.Close()
	}
	cfg := mlx.SFTConfig{
		LoRA:                      spine.LoRAConfig{Rank: *rank, Alpha: float32(*alpha)},
		BatchSize:                 *batch,
		GradientAccumulationSteps: *gradAccum,
		Epochs:                    *epochs,
		LearningRate:              *lr,
		MaxSeqLen:                 *maxSeqLen,
		SequencePacking:           *packing,
		CheckpointDir:             *checkpointDir,
		CheckpointEvery:           *checkpointEvery,
		EvalEvery:                 *evalEvery,
		EvalPrompts:               prompts,
		EvalMaxTokens:             *evalMaxTokens,
		SavePath:                  save,
		ResumePath:                *resumePath,
		Merge:                     *merge,
		ScoreCascade:              *scoreCascade && *evalEvery > 0 && len(prompts) > 0,
		ScoreWindow:               *scoreWindow,
		ValidData:                 validData,
		ValidSamples:              *valSamples,
		ValEvery:                  *valEvery,
	}
	if metricsSink != nil {
		cfg.ProbeSink = metricsSink
	}

	result, err := m.TrainSFT(ctx, ds, cfg)
	if err != nil {
		core.Print(stderr, "%s sft: training: %v\n", cliName(), err)
		return 1
	}

	core.Print(stdout, "steps %d  epochs %d  samples %d  last-loss %.4f\n",
		result.Steps, result.Epochs, result.Samples, result.LastLoss)
	if len(result.ValLosses) > 0 {
		first := result.ValLosses[0]
		last := result.ValLosses[len(result.ValLosses)-1]
		core.Print(stdout, "val-loss %.4f → %.4f  (%d passes)\n", first.Loss, last.Loss, len(result.ValLosses))
	}
	if result.AdapterPath != "" {
		core.Print(stdout, "adapter %s\n", result.AdapterPath)
	}
	for _, cp := range result.CheckpointMetadata {
		if cp.ScoreComposite > 0 {
			core.Print(stdout, "checkpoint step %d  score %.2f  %s\n", cp.Step, cp.ScoreComposite, cp.Path)
		}
	}
	if result.BestScoreComposite > 0 {
		core.Print(stdout, "best step %d  windowed composite %.2f  (cascade: %s)\n",
			result.BestScoreStep, result.BestScoreComposite, result.ScoreSidecarPath)
	}
	if metricsSink != nil {
		metricsSink.Flush()
		core.Print(stdout, "metrics %d lines", metricsSink.Lines())
		if lpPath != "" {
			core.Print(stdout, "  → %s", lpPath)
		}
		if dropped := metricsSink.Dropped(); dropped > 0 {
			core.Print(stdout, "  (%d dropped)", dropped)
		}
		core.Print(stdout, "\n")
	}
	return 0
}

// sftProbesFromValid derives the fixed eval probe set: the first n distinct
// user turns of the validation JSONL.
func sftProbesFromValid(path string, n int) ([]string, error) {
	if n <= 0 {
		n = 4
	}
	text, err := coreio.Local.Read(path)
	if err != nil {
		return nil, err
	}
	var prompts []string
	for _, line := range core.Split(text, "\n") {
		if len(prompts) >= n {
			break
		}
		line = core.Trim(line)
		if line == "" {
			continue
		}
		var row struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if r := core.JSONUnmarshal([]byte(line), &row); !r.OK {
			continue
		}
		for _, msg := range row.Messages {
			if msg.Role == "user" && core.Trim(msg.Content) != "" {
				prompts = append(prompts, msg.Content)
				break
			}
		}
	}
	if len(prompts) == 0 {
		return nil, core.NewError("mlx: no user turns found in validation set")
	}
	return prompts, nil
}
