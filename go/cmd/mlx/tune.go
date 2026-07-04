// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// runTuneCommand measures AR decode against each MTP draft block ON THE REAL
// MODEL and persists the winner as a tuning profile the serve auto-applies.
// This is a live-run verb — it loads the pair and generates; the operator
// runs it deliberately, once per machine + model.
//
//	lthn-mlx tune --model ~/models/gemma-4-31b-it-q4
//	lthn-mlx tune --model ~/models/gemma-4-31b-it-q4 --depths 4,5,6 --max-tokens 256
//	lthn-mlx tune --model <target> --draft <assistant> --json
func runTuneCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "Gemma 4 target model path (required)")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside the target, a path forces it")
	depths := fs.String("depths", "4,5,6", "comma-separated draft blocks to sweep (verify forward = carried lead + block-1 proposals)")
	maxTokens := fs.Int("max-tokens", 256, "tokens per measurement run")
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "measurement prompt")
	workload := fs.String("workload", string(inference.TuningWorkloadChat), "workload the profile is scored + persisted under")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory (default ~/Lethean/data/tuning)")
	jsonOut := fs.Bool("json", false, "emit JSONL tuning events instead of the text summary")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune --model <path> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Measure plain AR decode against each MTP draft block on the real model,\n")
		core.WriteString(stderr, "then persist the winner as a tuning profile. serve applies the tuned\n")
		core.WriteString(stderr, "block automatically on later boots (--no-auto-profile opts out). If AR\n")
		core.WriteString(stderr, "wins, the verb says so and persists nothing.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s tune --model ~/models/gemma-4-31b-it-q4\n", name))
		core.WriteString(stderr, "    # detect the drafter, sweep blocks 4-6, persist the winner\n")
		core.WriteString(stderr, core.Sprintf("  %s tune --model <target> --draft <assistant> --depths 3,4,5,6 --json\n", name))
		core.WriteString(stderr, "    # explicit pair + sweep, JSONL events for tooling\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s tune: unexpected positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s tune: --model is required\n", cliName()))
		fs.Usage()
		return 2
	}
	tuneWorkload := inference.TuningWorkload(core.Trim(*workload))
	if !cliValidTuningWorkload(tuneWorkload) {
		core.Print(stderr, "%s tune: unsupported workload %q", cliName(), *workload)
		return 2
	}
	blocks, err := parseTuneDraftBlocks(*depths)
	if err != nil {
		core.Print(stderr, "%s tune: %v", cliName(), err)
		return 2
	}
	detection := resolveServeDraft(*modelPath, *draftPath, true)
	if !detection.Active() {
		core.Print(stderr, "%s tune: no MTP drafter found for %s — nothing to tune (place an assistant/ beside the target or pass --draft)", cliName(), *modelPath)
		return 1
	}
	dir := core.Trim(*profileDir)
	if dir == "" {
		dir = standardTuningProfileDir()
	}

	results, err := runTuneMeasurements(ctx, tuneMeasurementRequest{
		ModelPath: *modelPath,
		DraftPath: detection.DraftPath,
		Prompt:    *prompt,
		MaxTokens: *maxTokens,
		Blocks:    blocks,
		Workload:  tuneWorkload,
	})
	if err != nil {
		core.Print(stderr, "%s tune: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		for i := range results {
			if err := writeTuningEventJSONL(stdout, inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: results[i].Candidate, Result: &results[i]}); err != nil {
				core.Print(stderr, "%s tune: %v", cliName(), err)
				return 1
			}
		}
	} else {
		printTuneRunSummary(stdout, *modelPath, results)
	}

	selected, found := cliSelectTuningResult(results)
	if !found {
		core.Print(stderr, "%s tune: every candidate failed — nothing to persist", cliName())
		return 1
	}
	if selected.Candidate.ID == tuneARCandidateID {
		// AR wins: say so honestly, persist NOTHING — the drafter does not
		// earn its keep on this machine + model and serve must not engage a
		// tuned block that loses.
		if *jsonOut {
			_ = writeTuningEventJSONL(stdout, inference.TuningEvent{Kind: inference.TuningEventSelected, Candidate: selected.Candidate, Result: &selected, Labels: map[string]string{"persisted": "false", "reason": "plain AR decode won — no profile written"}})
		} else {
			core.WriteString(stdout, core.Sprintf("selected: plain AR decode (%.1f tok/s) — MTP loses on this machine + model, no profile written\n", selected.Measurements.DecodeTokensPerSec))
		}
		return 0
	}

	machineHash, hashErr := currentMachineProfileHash(ctx)
	if hashErr != nil {
		core.Print(stderr, "%s tune: machine hash unavailable (%v) — profile written without one", cliName(), hashErr)
	}
	labels := cliTuningSelectionLabels(results, selected)
	labels["source"] = "lthn-mlx tune"
	profile := cliBuildTuningProfile(inference.TuningPlan{}, *modelPath, machineHash, tuneWorkload, selected, labels, time.Now())
	path := cliTuningProfilePath(dir, profile)
	if err := writeTuningProfile(path, profile); err != nil {
		core.Print(stderr, "%s tune: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		_ = writeTuningEventJSONL(stdout, inference.TuningEvent{Kind: inference.TuningEventSelected, Candidate: selected.Candidate, Result: &selected, Labels: map[string]string{"persisted": "true", "profile_path": path}})
	} else {
		core.WriteString(stdout, core.Sprintf("selected: %s (%.1f tok/s vs AR %.1f) — profile %s\n",
			selected.Candidate.ID, selected.Measurements.DecodeTokensPerSec, tuneARDecodeRate(results), path))
		core.WriteString(stdout, "serve applies the tuned block automatically; --no-auto-profile opts out\n")
	}
	return 0
}

const (
	tuneARCandidateID = "ar"
	// tuneDraftBlockLabel carries the winning block inside the persisted
	// profile (Candidate.Labels) — serve reads it back at boot.
	tuneDraftBlockLabel = "mtp_draft_block"
)

// standardTuningProfileDir is where tune writes and serve reads profiles.
func standardTuningProfileDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "tuning")
}

// parseTuneDraftBlocks parses --depths into draft blocks, bounded to the
// MTPLX block semantics (2..8: a block of 1 has no proposals to verify).
func parseTuneDraftBlocks(value string) ([]int, error) {
	parts := core.Split(core.Trim(value), ",")
	blocks := make([]int, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part == "" {
			continue
		}
		parsed := core.ParseInt(part, 10, 32)
		if !parsed.OK {
			return nil, core.Errorf("invalid draft block %q", part)
		}
		block := int(parsed.Value.(int64))
		if block < 2 || block > 8 {
			return nil, core.Errorf("draft block %d out of range 2..8", block)
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		return nil, core.NewError("no draft blocks to sweep")
	}
	return blocks, nil
}

// tuneMeasurementRequest is one tune invocation's bench plan.
type tuneMeasurementRequest struct {
	ModelPath string
	DraftPath string
	Prompt    string
	MaxTokens int
	Blocks    []int
	Workload  inference.TuningWorkload
}

// runTuneMeasurements is swapped by tests for a fake runner — the live
// implementation loads the pair once and generates for real.
var runTuneMeasurements = runTuneMeasurementsLive

// runTuneMeasurementsLive loads the pair ONCE, measures plain AR decode on
// the target, then each draft block through the MTP lane (greedy — the
// committed sequence equals AR, so the rates compare like-for-like).
func runTuneMeasurementsLive(ctx context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
	pair, err := mlx.LoadSpeculativePair(req.ModelPath, req.DraftPath, mlx.SpeculativePairConfig{})
	if err != nil {
		return nil, core.E("tune", "load pair", err)
	}
	defer pair.Close()
	if req.MaxTokens <= 0 {
		req.MaxTokens = 256
	}

	// Warm the kernels — the first generation pays compilation + allocation.
	for range pair.Target.GenerateTokens(ctx, req.Prompt, mlx.WithMaxTokens(8), mlx.WithTemperature(0)) {
	}
	if err := pair.Target.Err(); err != nil {
		return nil, core.E("tune", "warm", err)
	}

	results := make([]inference.TuningResult, 0, 1+len(req.Blocks))

	// AR baseline.
	for range pair.Target.GenerateTokens(ctx, req.Prompt, mlx.WithMaxTokens(req.MaxTokens), mlx.WithTemperature(0)) {
	}
	if err := pair.Target.Err(); err != nil {
		return nil, core.E("tune", "AR baseline", err)
	}
	results = append(results, tuneResultFromMetrics(tuneARCandidateID, nil, pair.Target.Metrics(), req.Workload))

	// Each draft block through the MTP lane.
	for _, block := range req.Blocks {
		_, genErr := pair.Generate(ctx, req.Prompt, mlx.SpeculativeDecodeConfig{
			MaxTokens:   req.MaxTokens,
			DraftTokens: block - 1,
		})
		candidateID := core.Sprintf("mtp-block-%d", block)
		labels := map[string]string{tuneDraftBlockLabel: core.Sprintf("%d", block)}
		if genErr != nil {
			results = append(results, inference.TuningResult{
				Candidate: inference.TuningCandidate{ID: candidateID, Workload: req.Workload, Labels: labels},
				Error:     genErr.Error(),
			})
			continue
		}
		results = append(results, tuneResultFromMetrics(candidateID, labels, pair.Metrics(), req.Workload))
	}
	return results, nil
}

// tuneResultFromMetrics shapes one measured run into the tuning machinery's
// result type, scored through the existing workload scorer.
func tuneResultFromMetrics(candidateID string, labels map[string]string, metrics mlx.Metrics, workload inference.TuningWorkload) inference.TuningResult {
	measurements := inference.TuningMeasurements{
		DecodeTokensPerSec:     metrics.DecodeTokensPerSec,
		PrefillTokensPerSec:    metrics.PrefillTokensPerSec,
		FirstTokenMilliseconds: float64(metrics.FirstTokenDuration.Milliseconds()),
		PeakMemoryBytes:        metrics.PeakMemoryBytes,
	}
	return inference.TuningResult{
		Candidate:    inference.TuningCandidate{ID: candidateID, Workload: workload, Labels: labels},
		Measurements: measurements,
		Score:        inference.ScoreTuningMeasurements(workload, measurements),
	}
}

// tuneARDecodeRate reads the AR baseline's decode rate out of a result set.
func tuneARDecodeRate(results []inference.TuningResult) float64 {
	for _, result := range results {
		if result.Candidate.ID == tuneARCandidateID {
			return result.Measurements.DecodeTokensPerSec
		}
	}
	return 0
}
