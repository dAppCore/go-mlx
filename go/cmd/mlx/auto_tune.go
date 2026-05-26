// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	iofs "io/fs"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
)

// standardProfileDir returns ~/Lethean/data/profiles/ — the canonical
// location lthn-mlx writes auto-tuned profiles to and reads them from
// at serve boot. Matches lthn/desktop's pkg/paths convention so the
// two binaries share the same machine-tuned configurations without
// per-binary profile dir flags.
//
// The directory is created on first read.
func standardProfileDir() string {
	dir := core.PathJoin(core.Env("HOME"), "Lethean", "data", "profiles")
	_ = core.MkdirAll(dir, 0o755)
	return dir
}

// runAutoTuneCommand is the zero-action wrapper around tune-plan +
// tune-run. It always tunes against the current machine, always writes
// the best profile to standardProfileDir(), and uses sensible defaults
// for everything else — the user runs one command, waits, and the
// next `lthn-mlx serve <model>` boots with the optimal config without
// any --profile flag.
//
//	lthn-mlx auto-tune ~/models/lemer-lite                  # chat workload (default)
//	lthn-mlx auto-tune --workload long_context ~/models/lemer-lite
//	lthn-mlx auto-tune --max-candidates 5 ~/models/lemer-lite  # cap exploration time
func runAutoTuneCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	defaultBench := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("auto-tune"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	workload := fs.String("workload", string(inference.TuningWorkloadChat),
		"workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	maxCandidates := fs.Int("max-candidates", 0,
		"cap candidate exploration; 0 lets the planner choose")
	splitFFNCaches := fs.String("split-ffn-caches", "",
		"comma-separated CPU FFN cache layer counts to rank (e.g. 0,8,16)")
	prompt := fs.String("prompt", defaultBench.Prompt,
		"smoke prompt for candidate measurements")
	maxTokens := fs.Int("max-tokens", defaultBench.MaxTokens,
		"generated tokens per candidate measurement")
	runs := fs.Int("runs", defaultBench.Runs,
		"measurement runs per candidate")
	profileDir := fs.String("profile-dir", "",
		"override the standard profiles directory (~/Lethean/data/profiles)")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s auto-tune [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Zero-action auto-tune. Runs tune-plan + tune-run for the current\n")
		core.WriteString(stderr, "machine + given model + workload, then writes the best-scoring\n")
		core.WriteString(stderr, "profile to ~/Lethean/data/profiles/ (the standard location serve\n")
		core.WriteString(stderr, "auto-discovers from). After this completes, `lthn-mlx serve <model>`\n")
		core.WriteString(stderr, "boots with the tuned config — no --profile flag needed.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Trade-off: spends 1-10 minutes (depending on candidate count and\n")
		core.WriteString(stderr, "runs-per-candidate) for measurably better tok/s + memory shape on\n")
		core.WriteString(stderr, "this specific machine.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s auto-tune ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default chat workload, save best to ~/Lethean/data/profiles/\n"))
		core.WriteString(stderr, core.Sprintf("  %s auto-tune --workload coding ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # optimise for coding workload\n"))
		core.WriteString(stderr, core.Sprintf("  %s auto-tune --max-candidates 5 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # cap candidate exploration (faster, less thorough)\n"))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Next: `lthn-mlx serve <model>` automatically picks up the saved profile.\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s auto-tune: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s auto-tune: %v", cliName(), err)
		return 2
	}
	if len(workloads) == 0 {
		workloads = []inference.TuningWorkload{inference.TuningWorkloadChat}
	}
	caches, err := cliSplitFFNCacheLayers(*splitFFNCaches)
	if err != nil {
		core.Print(stderr, "%s auto-tune: %v", cliName(), err)
		return 2
	}

	modelPath := fs.Arg(0)
	dir := core.Trim(*profileDir)
	if dir == "" {
		dir = standardProfileDir()
	}

	machineHash, err := currentMachineProfileHash(ctx)
	if err != nil {
		core.Print(stderr, "%s auto-tune: machine hash: %v", cliName(), err)
		return 1
	}

	core.WriteString(stdout, core.Sprintf("auto-tune: model=%s workload=%s machine=%s\n", modelPath, workloads[0], machineHash))
	core.WriteString(stdout, "  planning candidates…\n")

	plan, err := runPlanLocalTuning(ctx, inference.TuningPlanRequest{
		Model:     inference.ModelIdentity{Path: modelPath},
		Workloads: workloads,
		Budget: inference.TuningBudget{
			MaxCandidates:     *maxCandidates,
			SmokeTokens:       *maxTokens,
			Runs:              *runs,
			AllowStateBench:   true,
			AllowModelReloads: true,
		},
	})
	if err != nil {
		core.Print(stderr, "%s auto-tune: plan: %v", cliName(), err)
		return 1
	}
	if len(caches) > 0 {
		plan = appendSplitFFNTuningCandidates(ctx, plan, modelPath, caches)
	}
	candidates := cliLimitTuningCandidates(plan.Candidates, *maxCandidates)
	if len(candidates) == 0 {
		core.Print(stderr, "%s auto-tune: no tuning candidates", cliName())
		return 1
	}
	core.WriteString(stdout, core.Sprintf("  measuring %d candidates (this can take several minutes)…\n", len(candidates)))

	benchCfg := bench.DefaultConfig()
	benchCfg.Model = core.PathBase(modelPath)
	benchCfg.ModelPath = modelPath
	benchCfg.Prompt = *prompt
	benchCfg.CachePrompt = *prompt
	benchCfg.MaxTokens = *maxTokens
	benchCfg.Runs = *runs

	results, err := runLocalTuning(ctx, mlx.LocalTuningRunConfig{
		ModelPath:  modelPath,
		Workload:   workloads[0],
		Candidates: candidates,
		Bench:      benchCfg,
		Emit: func(event inference.TuningEvent) bool {
			if event.Kind == inference.TuningEventResult && event.Result != nil {
				core.WriteString(stdout, core.Sprintf("    candidate %s: score=%.3f\n", event.Candidate.ID, event.Result.Score.Score))
			}
			return true
		},
	})
	if err != nil {
		core.Print(stderr, "%s auto-tune: %v", cliName(), err)
		return 1
	}

	selected, ok := cliSelectTuningResult(results)
	if !ok {
		core.Print(stderr, "%s auto-tune: no successful tuning result to persist", cliName())
		return 1
	}
	selectionLabels := cliTuningSelectionLabels(results, selected)
	profile := cliBuildTuningProfile(plan, modelPath, machineHash, workloads[0], selected, selectionLabels, time.Now())
	profilePath := cliTuningProfilePath(dir, profile)
	if err := writeTuningProfile(profilePath, profile); err != nil {
		core.Print(stderr, "%s auto-tune: %v", cliName(), err)
		return 1
	}

	core.WriteString(stdout, core.Sprintf("\nselected: %s (score=%.3f)\n", selected.Candidate.ID, selected.Score.Score))
	core.WriteString(stdout, core.Sprintf("saved:    %s\n", profilePath))
	core.WriteString(stdout, core.Sprintf("\nrun `%s serve %s` — the profile auto-applies.\n", cliName(), modelPath))
	return 0
}

// findStandardProfileForModel scans the standard profile dir for a
// profile matching the current machine + the given model path. Returns
// the profile path + true if found, or empty + false. Errors are
// swallowed — the caller falls through to defaults if discovery fails.
//
// Match logic: machine hash must equal currentMachineProfileHash;
// model path must equal the requested path. Workload defaults to chat
// unless multiple match in which case the highest score wins.
func findStandardProfileForModel(ctx context.Context, modelPath string) (string, bool) {
	if core.Trim(modelPath) == "" {
		return "", false
	}
	machineHash, err := currentMachineProfileHash(ctx)
	if err != nil || machineHash == "" {
		return "", false
	}
	dir := standardProfileDir()
	entries := core.ReadDir(core.DirFS(dir), ".")
	if !entries.OK {
		return "", false
	}
	dirEntries, ok := entries.Value.([]iofs.DirEntry)
	if !ok {
		return "", false
	}
	var bestPath string
	var bestScore float64
	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !core.HasSuffix(name, ".json") {
			continue
		}
		candidatePath := core.PathJoin(dir, name)
		report, err := readTuneProfileReport(candidatePath)
		if err != nil || report.Profile == nil {
			continue
		}
		if report.Profile.Key.MachineHash != machineHash {
			continue
		}
		if report.ModelPath != modelPath && report.Profile.Key.Model.Path != modelPath {
			continue
		}
		if bestPath == "" || report.Profile.Score.Score > bestScore {
			bestPath = candidatePath
			bestScore = report.Profile.Score.Score
		}
	}
	return bestPath, bestPath != ""
}
