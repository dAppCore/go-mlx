// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"os/signal"
	"syscall"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	args := core.Args()
	if len(args) > 0 {
		if name := core.PathBase(args[0]); name != "" {
			commandName = name
		}
	}
	core.Exit(runCommand(ctx, args[1:], core.Stdout(), core.Stderr()))
}

var commandName = "go-mlx"

func cliName() string {
	name := core.Trim(commandName)
	if name == "" {
		return "go-mlx"
	}
	return name
}

func cliCommandName(command string) string {
	if command == "" {
		return cliName()
	}
	return cliName() + " " + command
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		// Launched from Finder via the .app bundle → default to menubar.
		// CLI invocation with no args → show help.
		if isInsideAppBundle() {
			return runMenubarCommand(ctx, args, stdout, stderr)
		}
		printUsage(stdout)
		return 0
	}
	switch args[0] {
	case "auto-tune":
		return runAutoTuneCommand(ctx, args[1:], stdout, stderr)
	case "auto-round":
		return runAutoRoundCommand(args[1:], stdout, stderr)
	case "menubar":
		return runMenubarCommand(ctx, args[1:], stdout, stderr)
	case "discover":
		return runDiscoverCommand(ctx, args[1:], stdout, stderr)
	case "ffn-estimate":
		return runFFNEstimateCommand(ctx, args[1:], stdout, stderr)
	case "pack":
		return runPackCommand(ctx, args[1:], stdout, stderr)
	case "ssd-recipes":
		return runSSDRecipesCommand(args[1:], stdout, stderr)
	case "ssd-eval":
		return runSSDEvalCommand(args[1:], stdout, stderr)
	case "memory-pretrain-build":
		return runMemoryPretrainBuildCommand(ctx, args[1:], stdout, stderr)
	case "profile-list":
		return runProfileListCommand(ctx, args[1:], stdout, stderr)
	case "profile-select":
		return runProfileSelectCommand(ctx, args[1:], stdout, stderr)
	case "replace-plan":
		return runReplacePlanCommand(ctx, args[1:], stdout, stderr)
	case "serve":
		return runServeCommand(ctx, args[1:], stdout, stderr)
	case "slice":
		return runSliceCommand(ctx, args[1:], stdout, stderr)
	case "slice-smoke":
		return runSliceSmokeCommand(ctx, args[1:], stdout, stderr)
	case "state-pack":
		return runStatePackCommand(ctx, args[1:], stdout, stderr)
	case "tune-plan":
		return runTunePlanCommand(ctx, args[1:], stdout, stderr)
	case "tune-profile":
		return runTuneProfileCommand(ctx, args[1:], stdout, stderr)
	case "tune-run":
		return runTuneRunCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s: unknown command %q", cliName(), args[0])
		printUsage(stderr)
		return 2
	}
}

type cpuFFNMemoryEstimateReport struct {
	Version              int                          `json:"version"`
	SourcePath           string                       `json:"source_path"`
	CPUFFNCache          int                          `json:"cpu_ffn_cache"`
	CPUFFNMemoryEstimate *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory_estimate,omitempty"`
	Error                string                       `json:"error,omitempty"`
}

type sliceSmokeReport struct {
	Version                   int                          `json:"version"`
	SourcePath                string                       `json:"source_path"`
	OutputPath                string                       `json:"output_path"`
	Preset                    inference.ModelSlicePreset   `json:"preset"`
	SliceDuration             time.Duration                `json:"slice_duration"`
	LoadDuration              time.Duration                `json:"load_duration,omitempty"`
	BenchDuration             time.Duration                `json:"bench_duration,omitempty"`
	SplitDuration             time.Duration                `json:"split_duration,omitempty"`
	OutputWeightBytes         int64                        `json:"output_weight_bytes,omitempty"`
	ReloadSkipped             bool                         `json:"reload_skipped,omitempty"`
	SplitOutput               string                       `json:"split_output,omitempty"`
	CPUFFNMemory              *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory,omitempty"`
	CPUFFNMemoryEstimate      *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory_estimate,omitempty"`
	CPUFFNMemoryEstimateError string                       `json:"cpu_ffn_memory_estimate_error,omitempty"`
	Slice                     *inference.ModelSlicePlan    `json:"slice,omitempty"`
	Placement                 *mlx.ModelSliceInspection    `json:"placement,omitempty"`
	Bench                     *bench.Report                `json:"bench,omitempty"`
	Error                     string                       `json:"error,omitempty"`
}

type sliceSmokeSplitResult struct {
	Output               string
	Duration             time.Duration
	CPUFFNMemory         *mlx.CPUSplitFFNMemoryReport
	CPUFFNMemoryEstimate *mlx.CPUSplitFFNMemoryReport
}

type tuneProfileReport struct {
	Version     int                       `json:"version"`
	ProfilePath string                    `json:"profile_path"`
	ModelPath   string                    `json:"model_path,omitempty"`
	Workload    inference.TuningWorkload  `json:"workload,omitempty"`
	MachineHash string                    `json:"machine_hash,omitempty"`
	CandidateID string                    `json:"candidate_id,omitempty"`
	Runtime     inference.RuntimeIdentity `json:"runtime"`
	Load        tuneProfileLoadSettings   `json:"load"`
	Score       inference.TuningScore     `json:"score"`
	Profile     *inference.TuningProfile  `json:"profile,omitempty"`
}

type tuneProfileLoadSettings struct {
	ContextLength         int    `json:"context_length,omitempty"`
	ParallelSlots         int    `json:"parallel_slots,omitempty"`
	PromptCache           bool   `json:"prompt_cache,omitempty"`
	PromptCacheMinTokens  int    `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy           string `json:"cache_policy,omitempty"`
	CacheMode             string `json:"cache_mode,omitempty"`
	KVCacheStorageDType   string `json:"kv_cache_storage_dtype,omitempty"`
	PagedKVPageSize       int    `json:"paged_kv_page_size,omitempty"`
	PagedKVPrealloc       bool   `json:"paged_kv_prealloc,omitempty"`
	FixedSlidingCacheSize int    `json:"fixed_sliding_cache_size,omitempty"`
	BatchSize             int    `json:"batch_size,omitempty"`
	PrefillChunkSize      int    `json:"prefill_chunk_size,omitempty"`
	ExpectedQuantization  int    `json:"expected_quantization,omitempty"`
	MemoryLimitBytes      uint64 `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes       uint64 `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes       uint64 `json:"wired_limit_bytes,omitempty"`
	AdapterPath           string `json:"adapter_path,omitempty"`
}

type replacePlanReport struct {
	Version            int                           `json:"version"`
	CurrentProfilePath string                        `json:"current_profile_path,omitempty"`
	NextProfilePath    string                        `json:"next_profile_path,omitempty"`
	Request            inference.ModelReplaceRequest `json:"request"`
	Plan               inference.ModelReplacePlan    `json:"plan"`
}

type profileSelectCriteria struct {
	MachineHash string                   `json:"machine_hash,omitempty"`
	ModelPath   string                   `json:"model_path,omitempty"`
	Workload    inference.TuningWorkload `json:"workload,omitempty"`
}

type profileListOptions struct {
	IncludeProfile  bool `json:"include_profile,omitempty"`
	BestPerWorkload bool `json:"best_per_workload,omitempty"`
}

type profileSelectReport struct {
	Version         int                       `json:"version"`
	ProfileDir      string                    `json:"profile_dir"`
	ProfilePath     string                    `json:"profile_path"`
	MachineHash     string                    `json:"machine_hash,omitempty"`
	ModelPath       string                    `json:"model_path,omitempty"`
	Workload        inference.TuningWorkload  `json:"workload,omitempty"`
	MatchedProfiles int                       `json:"matched_profiles"`
	CandidateID     string                    `json:"candidate_id,omitempty"`
	Runtime         inference.RuntimeIdentity `json:"runtime"`
	Load            tuneProfileLoadSettings   `json:"load"`
	Score           inference.TuningScore     `json:"score"`
	Profile         *inference.TuningProfile  `json:"profile,omitempty"`
	Warnings        []string                  `json:"warnings,omitempty"`
}

type profileListReport struct {
	Version      int                      `json:"version"`
	ProfileDir   string                   `json:"profile_dir"`
	MachineHash  string                   `json:"machine_hash,omitempty"`
	ModelPath    string                   `json:"model_path,omitempty"`
	Workload     inference.TuningWorkload `json:"workload,omitempty"`
	ProfileCount int                      `json:"profile_count"`
	Profiles     []tuneProfileReport      `json:"profiles,omitempty"`
	Warnings     []string                 `json:"warnings,omitempty"`
}

type stateRampFoldMarker struct {
	StorePath  string `json:"store_path,omitempty"`
	IndexURI   string `json:"index_uri,omitempty"`
	EntryURI   string `json:"entry_uri,omitempty"`
	BundleURI  string `json:"bundle_uri,omitempty"`
	TokenCount int    `json:"token_count,omitempty"`
}

func runDiscoverCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("discover"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON machine discovery report")
	modelDir := fs.String("model-dir", "", "model directory to scan without loading weights")
	includeModels := fs.Bool("include-models", false, "include discovered model packs")
	includeCandidates := fs.Bool("include-candidates", false, "include first-pass tuning candidates for discovered models")
	maxModels := fs.Int("max-models", 0, "maximum discovered models to report")
	probeDevice := fs.Bool("probe-device", false, "probe native Metal device facts")
	workload := fs.String("workload", "", "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s discover [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Report what MLX runtime + GPU device is available, and (optionally)\n")
		core.WriteString(stderr, "scan a directory for model packs without loading their weights. The\n")
		core.WriteString(stderr, "go-to first command on a new machine — answers \"do I have everything\n")
		core.WriteString(stderr, "I need to run inference here?\"\n")
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
		core.WriteString(stderr, core.Sprintf("  %s discover\n", name))
		core.WriteString(stderr, core.Sprintf("    # runtime + device only — quickest possible check\n"))
		core.WriteString(stderr, core.Sprintf("  %s discover -model-dir ~/models -include-models\n", name))
		core.WriteString(stderr, core.Sprintf("    # also list model packs found under the directory\n"))
		core.WriteString(stderr, core.Sprintf("  %s discover -probe-device -json\n", name))
		core.WriteString(stderr, core.Sprintf("    # detailed Metal device facts as JSON (memory, capabilities)\n"))
		core.WriteString(stderr, core.Sprintf("  %s discover -model-dir ~/models -include-candidates -workload chat\n", name))
		core.WriteString(stderr, core.Sprintf("    # add first-pass tuning candidates for each model under a workload\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s discover: unexpected positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s discover: %v", cliName(), err)
		return 2
	}
	cfg := mlx.LocalDiscoveryConfig{
		Workloads:         workloads,
		MaxModels:         *maxModels,
		IncludeModels:     *includeModels,
		IncludeCandidates: *includeCandidates,
	}
	if core.Trim(*modelDir) != "" {
		cfg.ModelDirs = []string{*modelDir}
	}
	if *probeDevice {
		cfg.Device = runGetDeviceInfo()
	}
	report, err := runDiscoverLocalRuntime(ctx, cfg)
	if err != nil {
		core.Print(stderr, "%s discover: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s discover: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printDiscoverySummary(stdout, report)
	return 0
}

func printDiscoverySummary(stdout io.Writer, report inference.MachineDiscoveryReport) {
	core.WriteString(stdout, core.Sprintf("runtime discovery: %s\n", report.Runtime.Backend))
	core.WriteString(stdout, core.Sprintf("  available: %t, device: %s\n", report.Available, report.Device.Architecture))
	core.WriteString(stdout, core.Sprintf("  memory: %d bytes, working set: %d bytes\n", report.Device.MemorySize, report.Device.MaxRecommendedWorkingSetSize))
	core.WriteString(stdout, core.Sprintf("  capabilities: %d, cache modes: %d\n", len(report.Capabilities), len(report.CacheModes)))
	core.WriteString(stdout, core.Sprintf("  models: %d, candidates: %d\n", len(report.Models), len(report.Candidates)))
}

func runFFNEstimateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ffn-estimate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON CPU FFN memory estimate")
	cpuFFNCache := fs.Int("cpu-ffn-cache", 0, "max CPU FFN layers to cache; 0 caches all, negative disables cache")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ffn-estimate [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Estimate the CPU FFN cache memory footprint for a model without\n")
		core.WriteString(stderr, "loading its weights. Reads the model config + safetensors index\n")
		core.WriteString(stderr, "and projects memory based on the requested CPU FFN cache layer\n")
		core.WriteString(stderr, "count. Cheap pre-flight check — answers \"will this fit?\" before\n")
		core.WriteString(stderr, "spending real GPU/RAM on a load attempt.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s ffn-estimate ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default: cache all FFN layers, full memory projection\n"))
		core.WriteString(stderr, core.Sprintf("  %s ffn-estimate -cpu-ffn-cache 8 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # cap CPU FFN cache at 8 layers (smaller footprint, more recompute)\n"))
		core.WriteString(stderr, core.Sprintf("  %s ffn-estimate -json -cpu-ffn-cache -1 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # disable cache entirely; JSON output for memory-budgeting scripts\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s ffn-estimate: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	report := &cpuFFNMemoryEstimateReport{
		Version:     1,
		SourcePath:  fs.Arg(0),
		CPUFFNCache: *cpuFFNCache,
	}
	estimate, err := runCPUFFNMemoryEstimate(ctx, report.SourcePath, report.CPUFFNCache)
	report.CPUFFNMemoryEstimate = estimate
	if err != nil {
		report.Error = err.Error()
	}
	return finishCPUFFNMemoryEstimateReport(report, jsonOut, stdout, stderr)
}

func finishCPUFFNMemoryEstimateReport(report *cpuFFNMemoryEstimateReport, jsonOut *bool, stdout, stderr io.Writer) int {
	if jsonOut != nil && *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s ffn-estimate: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if report.Error != "" {
			return 1
		}
		return 0
	}
	if report.Error != "" {
		core.Print(stderr, "%s ffn-estimate: %s", cliName(), report.Error)
		return 1
	}
	printCPUFFNMemoryEstimateSummary(stdout, report)
	return 0
}

func printCPUFFNMemoryEstimateSummary(stdout io.Writer, report *cpuFFNMemoryEstimateReport) {
	if report == nil || report.CPUFFNMemoryEstimate == nil {
		return
	}
	mem := report.CPUFFNMemoryEstimate
	core.WriteString(stdout, core.Sprintf("cpu ffn estimate: %s\n", report.SourcePath))
	core.WriteString(stdout, core.Sprintf("  cache layers: %d, total layers: %d, loaded layers: %d\n", report.CPUFFNCache, mem.TotalLayers, mem.LoadedLayers))
	core.WriteString(stdout, core.Sprintf("  peak resident: %d bytes, resident: %d bytes\n", mem.PeakResidentBytes, mem.ResidentBytes))
	core.WriteString(stdout, core.Sprintf("  dense equivalent: %d bytes, saved: %d bytes\n", mem.DenseEquivalentBytes, mem.SavedBytes))
	core.WriteString(stdout, core.Sprintf("  loads: %d, evictions: %d\n", mem.LayerLoads, mem.EvictedLayers))
}

func runTunePlanCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune-plan"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON tuning plan")
	workload := fs.String("workload", "", "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	maxCandidates := fs.Int("max-candidates", 0, "maximum candidates to return")
	splitFFNCaches := fs.String("split-ffn-caches", "", "comma-separated CPU FFN cache layer counts to rank; 0 caches all, negative disables cache")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-plan [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Step 1 of the auto-tune workflow. Generates a set of candidate\n")
		core.WriteString(stderr, "load configurations (context length, batch size, cache mode,\n")
		core.WriteString(stderr, "split-FFN cache layer count) for a model under a target workload —\n")
		core.WriteString(stderr, "chat / coding / long_context / agent_state / throughput / low_latency.\n")
		core.WriteString(stderr, "Output is the candidate list; tune-run executes them.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s tune-plan ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default workload (chat), all candidates\n"))
		core.WriteString(stderr, core.Sprintf("  %s tune-plan -workload coding -max-candidates 5 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # coding workload, top 5 candidates only\n"))
		core.WriteString(stderr, core.Sprintf("  %s tune-plan -workload long_context -split-ffn-caches 0,8,16 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # also rank split-FFN cache layer counts (0=all cached, 8/16=partial)\n"))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Next: pipe the model + workload into `tune-run` to actually measure candidates.\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-plan: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 2
	}
	caches, err := cliSplitFFNCacheLayers(*splitFFNCaches)
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 2
	}
	plan, err := runPlanLocalTuning(ctx, inference.TuningPlanRequest{
		Model:     inference.ModelIdentity{Path: fs.Arg(0)},
		Workloads: workloads,
		Budget:    inference.TuningBudget{MaxCandidates: *maxCandidates},
	})
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 1
	}
	if len(caches) > 0 {
		plan = appendSplitFFNTuningCandidates(ctx, plan, fs.Arg(0), caches)
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(plan, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s tune-plan: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printTunePlanSummary(stdout, plan)
	return 0
}

func printTunePlanSummary(stdout io.Writer, plan inference.TuningPlan) {
	core.WriteString(stdout, core.Sprintf("tuning plan: %s\n", plan.Model.Path))
	core.WriteString(stdout, core.Sprintf("  runtime: %s/%s, cache: %s\n", plan.Runtime.Backend, plan.Runtime.Device, plan.Runtime.CacheMode))
	core.WriteString(stdout, core.Sprintf("  workloads: %d, candidates: %d\n", len(plan.Workloads), len(plan.Candidates)))
	for _, candidate := range plan.Candidates {
		core.WriteString(stdout, core.Sprintf("  candidate: %s ctx=%d batch=%d cache=%s\n", candidate.ID, candidate.ContextLength, candidate.BatchSize, candidate.CacheMode))
	}
}

func runTuneProfileCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON profile load settings")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-profile [flags] <profile-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Read a saved tuning profile JSON (the output of `tune-run` or\n")
		core.WriteString(stderr, "`profile-select`) and print the reusable load settings it encodes:\n")
		core.WriteString(stderr, "model path, context length, batch size, cache mode, runtime backend.\n")
		core.WriteString(stderr, "Mostly for verification before piping the same profile into bench / serve.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s tune-profile ~/profiles/lemer-lite-m3ultra-chat.json\n", name))
		core.WriteString(stderr, core.Sprintf("    # human-readable settings table\n"))
		core.WriteString(stderr, core.Sprintf("  %s tune-profile -json ~/profiles/lemer-lite-m3ultra-chat.json\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON output for scripting (model path, ctx, batch, cache)\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-profile: expected exactly one profile path\n", cliName()))
		fs.Usage()
		return 2
	}
	report, err := readTuneProfileReport(fs.Arg(0))
	if err != nil {
		core.Print(stderr, "%s tune-profile: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s tune-profile: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printTuneProfileSummary(stdout, report)
	return 0
}

func readTuneProfileReport(path string) (tuneProfileReport, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return tuneProfileReport{}, core.Errorf("read profile: %v", read.Value)
	}
	var profile inference.TuningProfile
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		return tuneProfileReport{}, core.Errorf("decode profile: %v", result.Value)
	}
	candidate := profile.Candidate
	modelPath := candidate.Model.Path
	if modelPath == "" {
		modelPath = profile.Key.Model.Path
	}
	workload := candidate.Workload
	if workload == "" {
		workload = profile.Key.Workload
	}
	runtime := candidate.Runtime
	if runtime.Backend == "" {
		runtime = profile.Key.Runtime
	}
	return tuneProfileReport{
		Version:     1,
		ProfilePath: path,
		ModelPath:   modelPath,
		Workload:    workload,
		MachineHash: profile.Key.MachineHash,
		CandidateID: candidate.ID,
		Runtime:     runtime,
		Load:        tuneProfileLoadSettingsFromCandidate(candidate),
		Score:       profile.Score,
		Profile:     &profile,
	}, nil
}

func tuneProfileLoadSettingsFromCandidate(candidate inference.TuningCandidate) tuneProfileLoadSettings {
	return tuneProfileLoadSettings{
		ContextLength:        candidate.ContextLength,
		ParallelSlots:        candidate.ParallelSlots,
		PromptCache:          candidate.PromptCache,
		PromptCacheMinTokens: candidate.PromptCacheMinTokens,
		CachePolicy:          candidate.CachePolicy,
		CacheMode:            candidate.CacheMode,
		BatchSize:            candidate.BatchSize,
		PrefillChunkSize:     candidate.PrefillChunkSize,
		ExpectedQuantization: candidate.ExpectedQuantization,
		MemoryLimitBytes:     candidate.MemoryLimitBytes,
		CacheLimitBytes:      candidate.CacheLimitBytes,
		WiredLimitBytes:      candidate.WiredLimitBytes,
		AdapterPath:          candidate.Adapter.Path,
	}
}

func printTuneProfileSummary(stdout io.Writer, report tuneProfileReport) {
	core.WriteString(stdout, core.Sprintf("tuning profile: %s\n", report.ProfilePath))
	core.WriteString(stdout, core.Sprintf("  model: %s, workload: %s\n", report.ModelPath, report.Workload))
	core.WriteString(stdout, core.Sprintf("  candidate: %s, score: %.2f\n", report.CandidateID, report.Score.Score))
	core.WriteString(stdout, core.Sprintf("  load: ctx=%d batch=%d cache=%s prompt-cache=%t\n", report.Load.ContextLength, report.Load.BatchSize, report.Load.CacheMode, report.Load.PromptCache))
}

func runProfileListCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("profile-list"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON profile list")
	machineHash := fs.String("machine-hash", "", "machine hash to match")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash before listing")
	includeProfile := fs.Bool("include-profile", false, "include full nested tuning profile JSON in each row")
	bestPerWorkload := fs.Bool("best-per-workload", false, "list only the best matching profile for each workload")
	workload := fs.String("workload", "", "workload to match: chat, coding, long_context, agent_state, throughput, or low_latency")
	modelPath := fs.String("model-path", "", "model path to match")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s profile-list [flags] <profile-dir>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "List the tuning profiles saved under a directory, optionally\n")
		core.WriteString(stderr, "filtered by machine hash, model path, or workload. Use\n")
		core.WriteString(stderr, "-current-machine to auto-discover this machine's stable hash.\n")
		core.WriteString(stderr, "Pairs with `profile-select` (best match) and `tune-run` (saves\n")
		core.WriteString(stderr, "the profiles being listed).\n")
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
		core.WriteString(stderr, core.Sprintf("  %s profile-list ~/profiles\n", name))
		core.WriteString(stderr, core.Sprintf("    # everything in the directory\n"))
		core.WriteString(stderr, core.Sprintf("  %s profile-list -current-machine -model-path ~/models/lemer-lite ~/profiles\n", name))
		core.WriteString(stderr, core.Sprintf("    # profiles for this machine + this model\n"))
		core.WriteString(stderr, core.Sprintf("  %s profile-list -best-per-workload -current-machine ~/profiles\n", name))
		core.WriteString(stderr, core.Sprintf("    # one row per workload — the best score wins\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s profile-list: expected exactly one profile directory\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s profile-list: %v", cliName(), err)
		return 2
	}
	criteria := profileSelectCriteria{
		MachineHash: core.Trim(*machineHash),
		ModelPath:   core.Trim(*modelPath),
	}
	if *currentMachine {
		currentHash, err := currentMachineProfileHash(ctx)
		if err != nil {
			core.Print(stderr, "%s profile-list: %v", cliName(), err)
			return 1
		}
		criteria.MachineHash = currentHash
	}
	if len(workloads) > 0 {
		criteria.Workload = workloads[0]
	}
	report := listTuningProfiles(fs.Arg(0), criteria, profileListOptions{IncludeProfile: *includeProfile, BestPerWorkload: *bestPerWorkload})
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s profile-list: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProfileListSummary(stdout, report)
	return 0
}

func runProfileSelectCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("profile-select"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON selected profile")
	machineHash := fs.String("machine-hash", "", "machine hash to match")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash before matching")
	workload := fs.String("workload", "", "workload to match: chat, coding, long_context, agent_state, throughput, or low_latency")
	modelPath := fs.String("model-path", "", "model path to match")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s profile-select [flags] <profile-dir>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Pick the highest-scored saved tuning profile matching a machine\n")
		core.WriteString(stderr, "hash + model + workload. Returns the profile JSON (or the file\n")
		core.WriteString(stderr, "path with -path-only) — feed it to `bench -profile <path>` or\n")
		core.WriteString(stderr, "`serve --profile <path>` to load the model with those settings.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s profile-select -current-machine -workload chat -model-path ~/models/lemer-lite ~/profiles\n", name))
		core.WriteString(stderr, core.Sprintf("    # best chat profile for this machine + model\n"))
		core.WriteString(stderr, core.Sprintf("  %s profile-select -current-machine -workload long_context ~/profiles\n", name))
		core.WriteString(stderr, core.Sprintf("    # best long-context profile for this machine, any model\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s profile-select: expected exactly one profile directory\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s profile-select: %v", cliName(), err)
		return 2
	}
	criteria := profileSelectCriteria{
		MachineHash: core.Trim(*machineHash),
		ModelPath:   core.Trim(*modelPath),
	}
	if *currentMachine {
		currentHash, err := currentMachineProfileHash(ctx)
		if err != nil {
			core.Print(stderr, "%s profile-select: %v", cliName(), err)
			return 1
		}
		criteria.MachineHash = currentHash
	}
	if len(workloads) > 0 {
		criteria.Workload = workloads[0]
	}
	report, err := selectTuningProfile(fs.Arg(0), criteria)
	if err != nil {
		core.Print(stderr, "%s profile-select: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s profile-select: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProfileSelectSummary(stdout, report)
	return 0
}

func currentMachineProfileHash(ctx context.Context) (string, error) {
	report, err := runDiscoverLocalRuntime(ctx, mlx.LocalDiscoveryConfig{Device: runGetDeviceInfo()})
	if err != nil {
		return "", err
	}
	if report.Labels != nil && report.Labels["machine_hash"] != "" {
		return report.Labels["machine_hash"], nil
	}
	if report.Device.Labels != nil && report.Device.Labels["machine_hash"] != "" {
		return report.Device.Labels["machine_hash"], nil
	}
	return "", core.NewError("current machine hash unavailable")
}

func listTuningProfiles(profileDir string, criteria profileSelectCriteria, opts profileListOptions) profileListReport {
	paths := core.PathGlob(core.PathJoin(profileDir, "*.json"))
	core.SliceSort(paths)
	profiles := []tuneProfileReport{}
	warnings := []string{}
	for _, path := range paths {
		report, err := readTuneProfileReport(path)
		if err != nil {
			warnings = append(warnings, core.Sprintf("%s: %v", path, err))
			continue
		}
		if !profileMatchesCriteria(report, criteria) {
			continue
		}
		profiles = append(profiles, report)
	}
	sortTuneProfileReports(profiles)
	if opts.BestPerWorkload {
		profiles = bestTuneProfilesPerWorkload(profiles)
	}
	if !opts.IncludeProfile {
		for i := range profiles {
			profiles[i].Profile = nil
		}
	}
	return profileListReport{
		Version:      1,
		ProfileDir:   profileDir,
		MachineHash:  criteria.MachineHash,
		ModelPath:    criteria.ModelPath,
		Workload:     criteria.Workload,
		ProfileCount: len(profiles),
		Profiles:     profiles,
		Warnings:     warnings,
	}
}

func selectTuningProfile(profileDir string, criteria profileSelectCriteria) (profileSelectReport, error) {
	paths := core.PathGlob(core.PathJoin(profileDir, "*.json"))
	core.SliceSort(paths)
	var best tuneProfileReport
	bestPath := ""
	matched := 0
	warnings := []string{}
	for _, path := range paths {
		report, err := readTuneProfileReport(path)
		if err != nil {
			warnings = append(warnings, core.Sprintf("%s: %v", path, err))
			continue
		}
		if !profileMatchesCriteria(report, criteria) {
			continue
		}
		matched++
		if bestPath == "" || profileReportLess(best, bestPath, report, path) {
			best = report
			bestPath = path
		}
	}
	if bestPath == "" {
		return profileSelectReport{}, core.NewError("no matching tuning profiles")
	}
	return profileSelectReport{
		Version:         1,
		ProfileDir:      profileDir,
		ProfilePath:     bestPath,
		MachineHash:     best.MachineHash,
		ModelPath:       best.ModelPath,
		Workload:        best.Workload,
		MatchedProfiles: matched,
		CandidateID:     best.CandidateID,
		Runtime:         best.Runtime,
		Load:            best.Load,
		Score:           best.Score,
		Profile:         best.Profile,
		Warnings:        warnings,
	}, nil
}

func profileMatchesCriteria(report tuneProfileReport, criteria profileSelectCriteria) bool {
	if criteria.MachineHash != "" && report.MachineHash != criteria.MachineHash {
		return false
	}
	if criteria.ModelPath != "" && report.ModelPath != criteria.ModelPath {
		return false
	}
	if criteria.Workload != "" && report.Workload != criteria.Workload {
		return false
	}
	return true
}

func profileReportLess(best tuneProfileReport, bestPath string, candidate tuneProfileReport, candidatePath string) bool {
	if candidate.Score.Score != best.Score.Score {
		return candidate.Score.Score > best.Score.Score
	}
	if candidate.ProfileCreatedAtUnix() != best.ProfileCreatedAtUnix() {
		return candidate.ProfileCreatedAtUnix() > best.ProfileCreatedAtUnix()
	}
	return candidatePath < bestPath
}

func (report tuneProfileReport) ProfileCreatedAtUnix() int64 {
	if report.Profile == nil {
		return 0
	}
	return report.Profile.CreatedAtUnix
}

func sortTuneProfileReports(profiles []tuneProfileReport) {
	for i := 1; i < len(profiles); i++ {
		for j := i; j > 0 && profileReportLess(profiles[j-1], profiles[j-1].ProfilePath, profiles[j], profiles[j].ProfilePath); j-- {
			profiles[j-1], profiles[j] = profiles[j], profiles[j-1]
		}
	}
}

func bestTuneProfilesPerWorkload(profiles []tuneProfileReport) []tuneProfileReport {
	if len(profiles) == 0 {
		return nil
	}
	seen := map[inference.TuningWorkload]bool{}
	best := make([]tuneProfileReport, 0, len(profiles))
	for _, profile := range profiles {
		if seen[profile.Workload] {
			continue
		}
		seen[profile.Workload] = true
		best = append(best, profile)
	}
	return best
}

func printProfileListSummary(stdout io.Writer, report profileListReport) {
	core.WriteString(stdout, core.Sprintf("profile store: %s\n", report.ProfileDir))
	core.WriteString(stdout, core.Sprintf("  profiles: %d\n", report.ProfileCount))
	for _, profile := range report.Profiles {
		core.WriteString(stdout, core.Sprintf("  profile: %s model=%s workload=%s machine=%s score=%.2f\n", profile.ProfilePath, profile.ModelPath, profile.Workload, profile.MachineHash, profile.Score.Score))
	}
}

func printProfileSelectSummary(stdout io.Writer, report profileSelectReport) {
	core.WriteString(stdout, core.Sprintf("selected profile: %s\n", report.ProfilePath))
	core.WriteString(stdout, core.Sprintf("  model: %s, workload: %s, machine: %s\n", report.ModelPath, report.Workload, report.MachineHash))
	core.WriteString(stdout, core.Sprintf("  candidate: %s, score: %.2f, matches: %d\n", report.CandidateID, report.Score.Score, report.MatchedProfiles))
}

func runReplacePlanCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("replace-plan"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON model replace plan")
	currentProfile := fs.String("current-profile", "", "current saved tuning profile")
	nextProfile := fs.String("next-profile", "", "next saved tuning profile")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s replace-plan [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Given two saved tuning profiles (current + next), compute the\n")
		core.WriteString(stderr, "State-handling plan for a hot model swap — what KV cache state\n")
		core.WriteString(stderr, "must be invalidated vs preserved, whether the new cfg requires\n")
		core.WriteString(stderr, "a full reload or can stream into the existing process. Used by\n")
		core.WriteString(stderr, "lthn-mlx serve / lthn-desktop before applying a profile change\n")
		core.WriteString(stderr, "to a live session.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s replace-plan -current-profile ~/profiles/chat.json -next-profile ~/profiles/long_context.json\n", name))
		core.WriteString(stderr, core.Sprintf("    # plan the transition from chat → long_context, human-readable\n"))
		core.WriteString(stderr, core.Sprintf("  %s replace-plan -json -current-profile ~/profiles/v1.json -next-profile ~/profiles/v2.json\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON output (for serve-side automation)\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || core.Trim(*currentProfile) == "" || core.Trim(*nextProfile) == "" {
		core.WriteString(stderr, core.Sprintf("%s replace-plan: -current-profile and -next-profile are required\n", cliName()))
		fs.Usage()
		return 2
	}
	current, err := readTuneProfileReport(*currentProfile)
	if err != nil {
		core.Print(stderr, "%s replace-plan: current profile: %v", cliName(), err)
		return 1
	}
	next, err := readTuneProfileReport(*nextProfile)
	if err != nil {
		core.Print(stderr, "%s replace-plan: next profile: %v", cliName(), err)
		return 1
	}
	if current.Profile == nil || next.Profile == nil {
		core.Print(stderr, "%s replace-plan: profile payload missing", cliName())
		return 1
	}
	req := replaceRequestFromTuneProfiles(*current.Profile, *next.Profile)
	report := replacePlanReport{
		Version:            1,
		CurrentProfilePath: *currentProfile,
		NextProfilePath:    *nextProfile,
		Request:            req,
		Plan:               inference.PlanModelReplace(req),
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s replace-plan: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printReplacePlanSummary(stdout, report)
	return 0
}

func replaceRequestFromTuneProfiles(current, next inference.TuningProfile) inference.ModelReplaceRequest {
	return inference.ModelReplaceRequest{
		CurrentModel:   modelIdentityFromProfile(current),
		NextModel:      modelIdentityFromProfile(next),
		CurrentRuntime: runtimeIdentityFromProfile(current),
		NextRuntime:    runtimeIdentityFromProfile(next),
		CurrentAdapter: adapterIdentityFromProfile(current),
		NextAdapter:    adapterIdentityFromProfile(next),
	}
}

func modelIdentityFromProfile(profile inference.TuningProfile) inference.ModelIdentity {
	identity := profile.Key.Model
	candidate := profile.Candidate.Model
	if candidate.Path != "" {
		identity.Path = candidate.Path
	}
	if candidate.Hash != "" {
		identity.Hash = candidate.Hash
	}
	if candidate.Architecture != "" {
		identity.Architecture = candidate.Architecture
	}
	if candidate.QuantBits != 0 {
		identity.QuantBits = candidate.QuantBits
	}
	if candidate.QuantGroup != 0 {
		identity.QuantGroup = candidate.QuantGroup
	}
	if candidate.QuantType != "" {
		identity.QuantType = candidate.QuantType
	}
	if candidate.ContextLength != 0 {
		identity.ContextLength = candidate.ContextLength
	}
	if candidate.NumLayers != 0 {
		identity.NumLayers = candidate.NumLayers
	}
	if candidate.HiddenSize != 0 {
		identity.HiddenSize = candidate.HiddenSize
	}
	if candidate.VocabSize != 0 {
		identity.VocabSize = candidate.VocabSize
	}
	return identity
}

func runtimeIdentityFromProfile(profile inference.TuningProfile) inference.RuntimeIdentity {
	identity := profile.Key.Runtime
	candidate := profile.Candidate.Runtime
	if candidate.Backend != "" {
		identity.Backend = candidate.Backend
	}
	if candidate.Device != "" {
		identity.Device = candidate.Device
	}
	if candidate.CacheMode != "" {
		identity.CacheMode = candidate.CacheMode
	}
	if candidate.NativeRuntime {
		identity.NativeRuntime = candidate.NativeRuntime
	}
	if len(candidate.Labels) > 0 {
		identity.Labels = candidate.Labels
	}
	return identity
}

func adapterIdentityFromProfile(profile inference.TuningProfile) inference.AdapterIdentity {
	identity := profile.Key.Adapter
	candidate := profile.Candidate.Adapter
	if candidate.Path != "" {
		identity.Path = candidate.Path
	}
	if candidate.Hash != "" {
		identity.Hash = candidate.Hash
	}
	if candidate.Format != "" {
		identity.Format = candidate.Format
	}
	if candidate.Rank != 0 {
		identity.Rank = candidate.Rank
	}
	if candidate.Alpha != 0 {
		identity.Alpha = candidate.Alpha
	}
	return identity
}

func printReplacePlanSummary(stdout io.Writer, report replacePlanReport) {
	core.WriteString(stdout, core.Sprintf("replace plan: %s\n", report.Plan.Action))
	core.WriteString(stdout, core.Sprintf("  compatible: %t\n", report.Plan.Compatible))
	for _, reason := range report.Plan.Reasons {
		core.WriteString(stdout, core.Sprintf("  reason: %s\n", reason))
	}
}

func runTuneRunCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	defaultBench := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("tune-run"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonlOut := fs.Bool("jsonl", false, "stream JSONL tuning events")
	workload := fs.String("workload", string(inference.TuningWorkloadChat), "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	maxCandidates := fs.Int("max-candidates", 0, "maximum candidates to run")
	splitFFNCaches := fs.String("split-ffn-caches", "", "comma-separated CPU FFN cache layer counts to rank and test")
	profileOutput := fs.String("profile-output", "", "write the selected tuning profile JSON to this path")
	profileDir := fs.String("profile-dir", "", "write the selected tuning profile JSON into this directory")
	machineHash := fs.String("machine-hash", "", "stable machine/profile key supplied by the caller")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash for profile output")
	prompt := fs.String("prompt", defaultBench.Prompt, "smoke prompt for candidate measurements")
	maxTokens := fs.Int("max-tokens", defaultBench.MaxTokens, "generated tokens per candidate measurement")
	runs := fs.Int("runs", defaultBench.Runs, "measurement runs per candidate")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-run [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Step 2 of the auto-tune workflow. Builds the candidate plan (same\n")
		core.WriteString(stderr, "shape as tune-plan), runs each candidate against the model with a\n")
		core.WriteString(stderr, "real generation pass, and records the per-candidate score. The best\n")
		core.WriteString(stderr, "candidate per workload is saved to a profile JSON that bench / serve\n")
		core.WriteString(stderr, "can later consume via -profile / --profile.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Streams JSONL events with -jsonl (one event per candidate measurement).\n")
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
		core.WriteString(stderr, core.Sprintf("  %s tune-run -workload chat -current-machine -profile-dir ~/profiles ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # tune for chat on this machine, save best to ~/profiles/<hash>.json\n"))
		core.WriteString(stderr, core.Sprintf("  %s tune-run -workload long_context -max-tokens 256 -runs 3 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # long-context workload, 256-token measurements averaged over 3 runs\n"))
		core.WriteString(stderr, core.Sprintf("  %s tune-run -jsonl -workload coding ~/models/lemer-lite | tee tune.jsonl\n", name))
		core.WriteString(stderr, core.Sprintf("    # stream events for offline analysis\n"))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Next: `profile-list` to see what landed, `profile-select` to pick the best.\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-run: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 2
	}
	if len(workloads) == 0 {
		workloads = []inference.TuningWorkload{inference.TuningWorkloadChat}
	}
	caches, err := cliSplitFFNCacheLayers(*splitFFNCaches)
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 2
	}

	modelPath := fs.Arg(0)
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
		core.Print(stderr, "%s tune-run: plan: %v", cliName(), err)
		return 1
	}
	if len(caches) > 0 {
		plan = appendSplitFFNTuningCandidates(ctx, plan, modelPath, caches)
	}
	candidates := cliLimitTuningCandidates(plan.Candidates, *maxCandidates)
	if len(candidates) == 0 {
		core.Print(stderr, "%s tune-run: no tuning candidates", cliName())
		return 1
	}

	benchCfg := defaultBench
	benchCfg.Model = core.PathBase(modelPath)
	benchCfg.ModelPath = modelPath
	benchCfg.Prompt = *prompt
	benchCfg.CachePrompt = *prompt
	benchCfg.MaxTokens = *maxTokens
	benchCfg.Runs = *runs

	var emitErr error
	results, err := runLocalTuning(ctx, mlx.LocalTuningRunConfig{
		ModelPath:  modelPath,
		Workload:   workloads[0],
		Candidates: candidates,
		Bench:      benchCfg,
		Emit: func(event inference.TuningEvent) bool {
			if !*jsonlOut {
				return true
			}
			if emitErr != nil {
				return false
			}
			emitErr = writeTuningEventJSONL(stdout, event)
			return emitErr == nil
		},
	})
	if emitErr != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), emitErr)
		return 1
	}
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 1
	}
	profileOutputPath := core.Trim(*profileOutput)
	profileDirPath := core.Trim(*profileDir)
	if profileOutputPath != "" && profileDirPath != "" {
		core.Print(stderr, "%s tune-run: use only one of -profile-output or -profile-dir", cliName())
		return 2
	}
	if profileOutputPath != "" || profileDirPath != "" {
		selected, ok := cliSelectTuningResult(results)
		if !ok {
			core.Print(stderr, "%s tune-run: no successful tuning result to persist", cliName())
			return 1
		}
		profileMachineHash := core.Trim(*machineHash)
		if *currentMachine {
			profileMachineHash, err = currentMachineProfileHash(ctx)
			if err != nil {
				core.Print(stderr, "%s tune-run: %v", cliName(), err)
				return 1
			}
		}
		selectionLabels := cliTuningSelectionLabels(results, selected)
		profile := cliBuildTuningProfile(plan, modelPath, profileMachineHash, workloads[0], selected, selectionLabels, time.Now())
		if profileOutputPath == "" {
			profileOutputPath = cliTuningProfilePath(profileDirPath, profile)
		}
		if err := writeTuningProfile(profileOutputPath, profile); err != nil {
			core.Print(stderr, "%s tune-run: %v", cliName(), err)
			return 1
		}
		if *jsonlOut {
			selectedCopy := selected
			eventLabels := cliCloneStringLabels(selectionLabels)
			eventLabels["profile_output"] = profileOutputPath
			eventLabels["machine_hash"] = profileMachineHash
			if err := writeTuningEventJSONL(stdout, inference.TuningEvent{
				Kind:      inference.TuningEventSelected,
				Candidate: selected.Candidate,
				Result:    &selectedCopy,
				Labels:    eventLabels,
			}); err != nil {
				core.Print(stderr, "%s tune-run: %v", cliName(), err)
				return 1
			}
		}
	}
	if *jsonlOut {
		return 0
	}
	printTuneRunSummary(stdout, modelPath, results)
	return 0
}

func cliTuningProfilePath(profileDir string, profile inference.TuningProfile) string {
	modelName := core.PathBase(profile.Key.Model.Path)
	if modelName == "" {
		modelName = profile.Candidate.Model.Architecture
	}
	if modelName == "" {
		modelName = profile.Key.Model.Architecture
	}
	machineHash := profile.Key.MachineHash
	if parts := core.SplitN(machineHash, ":", 2); len(parts) == 2 {
		machineHash = parts[1]
	}
	name := core.Sprintf("%s-%s-%s-%s.json",
		cliProfileFilePart(string(profile.Key.Workload), "workload", 32),
		cliProfileFilePart(machineHash, "machine", 12),
		cliProfileFilePart(modelName, "model", 48),
		cliProfileFilePart(profile.Candidate.ID, "candidate", 48),
	)
	return core.PathJoin(profileDir, name)
}

func cliProfileFilePart(value, fallback string, maxLen int) string {
	value = core.Lower(core.Trim(value))
	builder := core.NewBuilder()
	lastDash := false
	for i := 0; i < len(value); i++ {
		b := value[i]
		if (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') {
			builder.WriteByte(b)
			lastDash = false
			continue
		}
		if builder.Len() > 0 && !lastDash {
			builder.WriteByte('-')
			lastDash = true
		}
	}
	part := trimProfileFileDashes(builder.String())
	if part == "" {
		part = fallback
	}
	if maxLen > 0 && len(part) > maxLen {
		part = trimProfileFileDashes(part[:maxLen])
	}
	if part == "" {
		return fallback
	}
	return part
}

func trimProfileFileDashes(value string) string {
	for len(value) > 0 && value[len(value)-1] == '-' {
		value = value[:len(value)-1]
	}
	return value
}

func cliSelectTuningResult(results []inference.TuningResult) (inference.TuningResult, bool) {
	var best inference.TuningResult
	found := false
	for _, result := range results {
		if result.Error != "" {
			continue
		}
		if !found || result.Score.Score > best.Score.Score {
			best = result
			found = true
		}
	}
	return best, found
}

func cliTuningSelectionLabels(results []inference.TuningResult, selected inference.TuningResult) map[string]string {
	labels := map[string]string{
		"source":           "lthn-mlx tune-run",
		"selection_policy": "highest_successful_score",
		"selection_reason": "selected highest successful score from measured tuning candidates",
		"selected_score":   core.Sprintf("%.6f", selected.Score.Score),
	}
	if selected.Candidate.ID != "" {
		labels["selected_candidate_id"] = selected.Candidate.ID
	}
	if selected.Measurements.DecodeTokensPerSec > 0 {
		labels["selected_decode_tokens_per_sec"] = core.Sprintf("%.6f", selected.Measurements.DecodeTokensPerSec)
	}
	if selected.Measurements.LoadMilliseconds > 0 {
		labels["selected_load_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.LoadMilliseconds)
	}
	if selected.Measurements.FirstTokenMilliseconds > 0 {
		labels["selected_first_token_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.FirstTokenMilliseconds)
	}
	if selected.Measurements.KVRestoreMilliseconds > 0 {
		labels["selected_restore_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.KVRestoreMilliseconds)
	}
	if selected.Measurements.PeakMemoryBytes > 0 {
		labels["selected_peak_memory_bytes"] = core.Sprintf("%d", selected.Measurements.PeakMemoryBytes)
	}
	if selected.Measurements.CorrectnessSmokeResult != "" {
		labels["selected_correctness_smoke_result"] = selected.Measurements.CorrectnessSmokeResult
	}
	if selected.Measurements.CorrectnessSmokeChecks > 0 {
		labels["selected_correctness_smoke_checks"] = core.Sprintf("%d", selected.Measurements.CorrectnessSmokeChecks)
	}
	successful := 0
	failed := 0
	var runnerUp inference.TuningResult
	hasRunnerUp := false
	for _, result := range results {
		if result.Error != "" {
			failed++
			continue
		}
		successful++
		if result.Candidate.ID == selected.Candidate.ID && result.Score.Score == selected.Score.Score {
			continue
		}
		if !hasRunnerUp || result.Score.Score > runnerUp.Score.Score {
			runnerUp = result
			hasRunnerUp = true
		}
	}
	labels["successful_candidates"] = core.Sprintf("%d", successful)
	labels["failed_candidates"] = core.Sprintf("%d", failed)
	if hasRunnerUp {
		if runnerUp.Candidate.ID != "" {
			labels["runner_up_candidate_id"] = runnerUp.Candidate.ID
		}
		labels["runner_up_score"] = core.Sprintf("%.6f", runnerUp.Score.Score)
		labels["selection_score_delta"] = core.Sprintf("%.6f", selected.Score.Score-runnerUp.Score.Score)
	}
	return labels
}

func cliBuildTuningProfile(plan inference.TuningPlan, modelPath, machineHash string, workload inference.TuningWorkload, result inference.TuningResult, labels map[string]string, createdAt time.Time) inference.TuningProfile {
	candidate := result.Candidate
	if candidate.Model.Path == "" && plan.Model.Path != "" {
		candidate.Model = plan.Model
	}
	if candidate.Model.Path == "" {
		candidate.Model.Path = modelPath
	}
	if candidate.Runtime.Backend == "" {
		candidate.Runtime = plan.Runtime
	}
	if candidate.Adapter.Path == "" && plan.Adapter.Path != "" {
		candidate.Adapter = plan.Adapter
	}
	if candidate.Workload == "" {
		candidate.Workload = workload
	}
	score := result.Score
	if score.Workload == "" {
		score.Workload = workload
	}
	profileLabels := cliCloneStringLabels(labels)
	if profileLabels == nil {
		profileLabels = map[string]string{}
	}
	if profileLabels["source"] == "" {
		profileLabels["source"] = "lthn-mlx tune-run"
	}
	return inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: machineHash,
			Runtime:     candidate.Runtime,
			Model:       candidate.Model,
			Adapter:     candidate.Adapter,
			Workload:    workload,
		},
		Candidate:     candidate,
		Measurements:  result.Measurements,
		Score:         score,
		CreatedAtUnix: createdAt.Unix(),
		Labels:        profileLabels,
	}
}

func writeTuningProfile(path string, profile inference.TuningProfile) error {
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		return core.NewError("marshal tuning profile failed")
	}
	if result := core.MkdirAll(core.PathDir(path), 0o755); !result.OK {
		return core.Errorf("create profile directory: %v", result.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.Errorf("write tuning profile: %v", result.Value)
	}
	return nil
}

func cliLimitTuningCandidates(candidates []inference.TuningCandidate, maxCandidates int) []inference.TuningCandidate {
	if maxCandidates > 0 && len(candidates) > maxCandidates {
		return append([]inference.TuningCandidate(nil), candidates[:maxCandidates]...)
	}
	return append([]inference.TuningCandidate(nil), candidates...)
}

func writeTuningEventJSONL(stdout io.Writer, event inference.TuningEvent) error {
	data := core.JSONMarshal(event)
	if !data.OK {
		return core.NewError("marshal tuning event failed")
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return nil
}

func printTuneRunSummary(stdout io.Writer, modelPath string, results []inference.TuningResult) {
	core.WriteString(stdout, core.Sprintf("tuning run: %s\n", modelPath))
	core.WriteString(stdout, core.Sprintf("  results: %d\n", len(results)))
	for _, result := range results {
		if result.Error != "" {
			core.WriteString(stdout, core.Sprintf("  candidate: %s error=%q\n", result.Candidate.ID, result.Error))
			continue
		}
		core.WriteString(stdout, core.Sprintf(
			"  candidate: %s score=%.2f decode=%.1f tok/s peak=%d MB\n",
			result.Candidate.ID,
			result.Score.Score,
			result.Measurements.DecodeTokensPerSec,
			result.Measurements.PeakMemoryBytes/1024/1024,
		))
	}
}

func cliTuningWorkloads(value string) ([]inference.TuningWorkload, error) {
	value = core.Trim(value)
	if value == "" {
		return nil, nil
	}
	workload := inference.TuningWorkload(value)
	if !cliValidTuningWorkload(workload) {
		return nil, core.Errorf("unsupported workload %q", value)
	}
	return []inference.TuningWorkload{workload}, nil
}

func cliValidTuningWorkload(workload inference.TuningWorkload) bool {
	switch workload {
	case inference.TuningWorkloadChat,
		inference.TuningWorkloadCoding,
		inference.TuningWorkloadLongContext,
		inference.TuningWorkloadAgentState,
		inference.TuningWorkloadThroughput,
		inference.TuningWorkloadLowLatency:
		return true
	default:
		return false
	}
}

func runSliceSmokeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	defaultBench := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("slice-smoke"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON smoke report")
	preset := fs.String("preset", string(inference.ModelSlicePresetClient), "slice preset to materialise before reload")
	output := fs.String("output", "", "output directory for the materialised slice")
	prompt := fs.String("prompt", "Write one short sentence about local inference.", "tiny reload smoke prompt")
	maxTokens := fs.Int("max-tokens", 1, "generated tokens for the smoke pass")
	runs := fs.Int("runs", 1, "generation runs for the smoke pass")
	contextLen := fs.Int("context", 0, "override context length when loading the slice")
	device := fs.String("device", "", "execution device: gpu or cpu")
	split := fs.Bool("split", false, "run split executor for client slices instead of skipping reload")
	cpuFFNCache := fs.Int("cpu-ffn-cache", 0, "max CPU FFN layers to cache during split smoke; 0 caches all, negative disables cache")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s slice-smoke [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s slice-smoke: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*output) == "" {
		core.WriteString(stderr, core.Sprintf("%s slice-smoke: -output is required\n", cliName()))
		fs.Usage()
		return 2
	}

	source := fs.Arg(0)
	report := &sliceSmokeReport{
		Version:    1,
		SourcePath: source,
		OutputPath: *output,
		Preset:     inference.ModelSlicePreset(*preset),
	}
	sliceStart := time.Now()
	plan, err := mlx.SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePreset(*preset),
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: *output,
	})
	report.SliceDuration = time.Since(sliceStart)
	report.Slice = plan
	report.OutputWeightBytes = fileSize(core.PathJoin(*output, "model.safetensors"))
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	placement, err := mlx.InspectModelSlice(*output)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	report.Placement = &placement
	if placement.RequiresSplitPlacement {
		estimate, estimateErr := runSliceSmokeEstimateCPUFFNMemory(ctx, source, *cpuFFNCache)
		report.CPUFFNMemoryEstimate = estimate
		if estimateErr != nil {
			report.CPUFFNMemoryEstimateError = estimateErr.Error()
		}
		if !*split {
			report.ReloadSkipped = true
			return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
		}
		result, err := runSliceSmokeSplitGenerate(ctx, *output, *prompt, *maxTokens, *contextLen, *device, *cpuFFNCache)
		report.SplitDuration = result.Duration
		report.SplitOutput = result.Output
		report.CPUFFNMemory = result.CPUFFNMemory
		report.CPUFFNMemoryEstimate = result.CPUFFNMemoryEstimate
		if err != nil {
			report.Error = err.Error()
		}
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}

	loadOptions := []mlx.LoadOption{}
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	loadStart := time.Now()
	loaded, err := loadBenchModel(*output, loadOptions...)
	report.LoadDuration = time.Since(loadStart)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	if loaded != nil {
		defer loaded.Close()
	}

	cfg := defaultBench
	cfg.Model = core.PathBase(*output)
	cfg.ModelPath = *output
	cfg.Prompt = *prompt
	cfg.CachePrompt = ""
	cfg.MaxTokens = *maxTokens
	cfg.Runs = *runs
	cfg.IncludePromptCache = false
	cfg.IncludeKVRestore = false
	cfg.IncludeStateBundleRoundTrip = false
	cfg.IncludeProbeOverhead = false
	benchStart := time.Now()
	report.Bench, err = runBenchReport(ctx, loaded, cfg)
	report.BenchDuration = time.Since(benchStart)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
}

func finishSliceSmokeReport(report *sliceSmokeReport, jsonOut *bool, stdout, stderr io.Writer) int {
	if jsonOut != nil && *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s slice-smoke: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if report.Error != "" {
			return 1
		}
		return 0
	}
	if report.Error != "" {
		core.Print(stderr, "%s slice-smoke: %s", cliName(), report.Error)
		return 1
	}
	printSliceSmokeSummary(stdout, report)
	return 0
}

func printSliceSmokeSummary(stdout io.Writer, report *sliceSmokeReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("slice smoke: %s\n", report.OutputPath))
	core.WriteString(stdout, core.Sprintf("  slice: %s, load: %s, bench: %s\n", report.SliceDuration, report.LoadDuration, report.BenchDuration))
	core.WriteString(stdout, core.Sprintf("  output weight bytes: %d\n", report.OutputWeightBytes))
	if report.Bench != nil {
		core.WriteString(stdout, core.Sprintf("  decode: %.1f tok/s, peak memory: %d MB\n", report.Bench.Generation.DecodeTokensPerSec, report.Bench.Generation.PeakMemoryBytes/1024/1024))
	}
	if report.SplitDuration > 0 {
		core.WriteString(stdout, core.Sprintf("  split: %s, output: %q\n", report.SplitDuration, report.SplitOutput))
	}
	if report.CPUFFNMemory != nil {
		mem := report.CPUFFNMemory
		core.WriteString(stdout, core.Sprintf("  cpu ffn: resident %d bytes, dense equivalent %d bytes, saved %d bytes\n", mem.ResidentBytes, mem.DenseEquivalentBytes, mem.SavedBytes))
	}
	if report.CPUFFNMemoryEstimate != nil {
		mem := report.CPUFFNMemoryEstimate
		core.WriteString(stdout, core.Sprintf("  cpu ffn estimate: peak %d bytes, resident %d bytes, loads %d, evictions %d\n", mem.PeakResidentBytes, mem.ResidentBytes, mem.LayerLoads, mem.EvictedLayers))
	}
}

var runCPUFFNMemoryEstimate = func(ctx context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
	report, err := mlx.EstimateCPUSplitFFNMemory(ctx, sourcePath, mlx.WithCPUSplitFFNMaxCachedLayers(cpuFFNCache))
	if err != nil {
		return nil, err
	}
	return &report, nil
}

var runSliceSmokeEstimateCPUFFNMemory = runCPUFFNMemoryEstimate

var runDiscoverLocalRuntime = mlx.DiscoverLocalRuntime

var runPlanLocalTuning = mlx.PlanLocalTuning

var runLocalTuning = mlx.RunLocalTuning

var runGetDeviceInfo = mlx.GetDeviceInfo

var runSliceSmokeSplitGenerate = func(ctx context.Context, slicePath, prompt string, maxTokens, contextLen int, device string, cpuFFNCache int) (sliceSmokeSplitResult, error) {
	loadOptions := []mlx.LoadOption{}
	if contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(contextLen))
	}
	if device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(device))
	}
	start := time.Now()
	executor, err := mlx.LoadSplitExecutor(
		ctx,
		slicePath,
		mlx.WithNativeSplitLocalRuntime(loadOptions...),
		mlx.WithCPUSplitFFNExecutor(mlx.WithCPUSplitFFNMaxCachedLayers(cpuFFNCache)),
	)
	if err != nil {
		return sliceSmokeSplitResult{Duration: time.Since(start)}, err
	}
	estimate, err := executor.CPUSplitFFNMemoryEstimate(ctx)
	if err != nil {
		return sliceSmokeSplitResult{Duration: time.Since(start)}, err
	}
	text, err := executor.Generate(ctx, prompt, mlx.GenerateConfig{MaxTokens: maxTokens, Temperature: 0})
	return sliceSmokeSplitResult{
		Output:               text,
		Duration:             time.Since(start),
		CPUFFNMemory:         executor.CPUSplitFFNMemoryReport(),
		CPUFFNMemoryEstimate: estimate,
	}, err
}

func fileSize(path string) int64 {
	stat := core.Stat(path)
	if !stat.OK {
		return 0
	}
	return stat.Value.(core.FsFileInfo).Size()
}

func runSliceCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("slice"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON slice plan")
	preset := fs.String("preset", string(inference.ModelSlicePresetClient), "slice preset: client, attention, embed, server, browse, router, expert_server, full")
	output := fs.String("output", "", "output directory for the materialised slice")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s slice [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s slice: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*output) == "" {
		core.WriteString(stderr, core.Sprintf("%s slice: -output is required\n", cliName()))
		fs.Usage()
		return 2
	}

	plan, err := mlx.SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePreset(*preset),
		Model:      inference.ModelIdentity{Path: fs.Arg(0)},
		OutputPath: *output,
	})
	if err != nil {
		core.Print(stderr, "%s slice: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(plan, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s slice: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printSliceSummary(stdout, plan)
	return 0
}

func printSliceSummary(stdout io.Writer, plan *inference.ModelSlicePlan) {
	if plan == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("model slice: %s\n", plan.OutputPath))
	core.WriteString(stdout, core.Sprintf("  preset: %s, components: %d\n", plan.Preset, len(plan.Components)))
	if plan.Labels != nil {
		core.WriteString(stdout, core.Sprintf("  tensors: %s, selected bytes: %s / %s\n", plan.Labels["tensor_count"], plan.Labels["selected_tensor_bytes"], plan.Labels["source_tensor_bytes"]))
		if plan.Labels["retained_tensor_ratio"] != "" {
			core.WriteString(stdout, core.Sprintf("  retained tensor ratio: %s\n", plan.Labels["retained_tensor_ratio"]))
		}
	}
}

var (
	loadBenchModel                    = mlx.LoadModel
	loadSpeculativePair               = mlx.LoadSpeculativePair
	runBenchReport                    = mlx.RunFastEvalBench
	runBenchReportWithDraft           = mlx.RunFastEvalBenchWithDraft
	runBenchReportWithSpeculativePair = mlx.RunFastEvalBenchWithSpeculativePair
)

func printUsage(w io.Writer) {
	name := cliName()
	core.WriteString(w, core.Sprintf("Usage: %s <command> [flags]\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, "Run inference\n")
	core.WriteString(w, "  menubar             tray-only macOS app — start/stop serve from the menu bar\n")
	core.WriteString(w, "  serve               host OpenAI/Anthropic/Ollama HTTP API for a loaded model\n")
	core.WriteString(w, "  bench               single-shot eval against a model (latency + cache + state)\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Inspect what is installed\n")
	core.WriteString(w, "  discover            report local MLX runtime + optional model candidates\n")
	core.WriteString(w, "  pack                validate a local native model pack\n")
	core.WriteString(w, "  official-gemma4-locks  print official Google Gemma 4 E2B source locks\n")
	core.WriteString(w, "  official-gemma4-12b-verify  verify official Google Gemma 4 12B Unified pack metadata\n")
	core.WriteString(w, "  auto-round         print native AutoRound quantization profile defaults\n")
	core.WriteString(w, "  ssd-recipes         print native Simple Self-Distillation recipe defaults\n")
	core.WriteString(w, "  ssd-eval            prepare a native Simple Self-Distillation eval plan\n")
	core.WriteString(w, "  memory-pretrain-build  build native hierarchical-memory pretraining artifacts\n")
	core.WriteString(w, "  official-gemma4-pair-verify  verify official Google Gemma 4 E2B target+assistant pair metadata\n")
	core.WriteString(w, "  official-gemma4-control-compare  compare official Google Gemma 4 E2B target metadata with archived q4 control\n")
	core.WriteString(w, "  official-gemma4-verify  verify an official Google Gemma 4 E2B snapshot lock\n")
	core.WriteString(w, "  production-architectures  print native-runtime architecture matrix\n")
	core.WriteString(w, "  production-quantization  select q8/q6/q4 Gemma 4 E2B app tier for this machine\n")
	core.WriteString(w, "  production-turboquant  print explicit TurboQuant KV-cache promotion policy\n")
	core.WriteString(w, "  production-turboquant-compare  compare TurboQuant driver-profile reports against cache-mode anchors\n")
	core.WriteString(w, "  production-mtp-compare  compare target-only and MTP driver-profile reports\n")
	core.WriteString(w, "  production-mtp-turboquant-compare  combine MTP and TurboQuant promotion reports\n")
	core.WriteString(w, "  ffn-estimate        estimate split CPU FFN memory without loading the model\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Transform a model\n")
	core.WriteString(w, "  slice               materialise a local model slice for split/reload tests\n")
	core.WriteString(w, "  slice-smoke         materialise + reload + benchmark a slice in one pass\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Tune for this machine\n")
	core.WriteString(w, "  auto-tune           one-shot: plan + run + save best profile to standard dir\n")
	core.WriteString(w, "  tune-plan           plan tuning candidates for a model\n")
	core.WriteString(w, "  tune-run            run + stream tuning candidate measurements\n")
	core.WriteString(w, "  tune-profile        read a saved tuning profile + print load settings\n")
	core.WriteString(w, "  profile-list        list saved tuning profiles by machine/model/workload\n")
	core.WriteString(w, "  profile-select      pick the best saved profile for this machine\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Measure timings\n")
	core.WriteString(w, "  driver-profile      measure load + first-token + decode timings for one question\n")
	core.WriteString(w, "  chapter-profile     measure generated chapter timings across a long prompt\n")
	core.WriteString(w, "  state-ramp-profile  measure warm retained-state growth across append/generate turns\n")
	core.WriteString(w, "  state-wake-profile  wake an existing State index + measure one continuation turn\n")
	core.WriteString(w, "  replace-plan        plan state handling for a profile/model reload\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "State container ops\n")
	core.WriteString(w, "  state-pack          pack a State marker + binary log into a Trix .kv container\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Examples\n")
	core.WriteString(w, core.Sprintf("  %s discover                                  # what runtime + models you have\n", name))
	core.WriteString(w, core.Sprintf("  %s serve --model ~/models/lemer-lite         # OpenAI HTTP on :11434\n", name))
	core.WriteString(w, core.Sprintf("  %s bench -max-tokens 32 ~/models/lemer-lite  # one-shot latency check\n", name))
	core.WriteString(w, core.Sprintf("  %s pack ~/models/lemer-lite                  # validate a model on disk\n", name))
	core.WriteString(w, core.Sprintf("  %s tune-plan ~/models/lemer-lite             # plan a per-machine tune run\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, core.Sprintf("Run \"%s <command> -h\" for command-specific flags.\n", name))
}
