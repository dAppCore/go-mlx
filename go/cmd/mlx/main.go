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
	case "menubar":
		return runMenubarCommand(ctx, args[1:], stdout, stderr)
	case "discover":
		return runDiscoverCommand(ctx, args[1:], stdout, stderr)
	case "pack":
		return runPackCommand(ctx, args[1:], stdout, stderr)
	case "ssd-recipes":
		return runSSDRecipesCommand(args[1:], stdout, stderr)
	case "ssd-eval":
		return runSSDEvalCommand(args[1:], stdout, stderr)
	case "memory-pretrain-build":
		return runMemoryPretrainBuildCommand(ctx, args[1:], stdout, stderr)
	case "serve":
		return runServeCommand(ctx, args[1:], stdout, stderr)
	case "generate":
		return runGenerateCommand(ctx, args[1:], stdout, stderr)
	case "sft":
		return runSFTCommand(ctx, args[1:], stdout, stderr)
	case "tune":
		return runTuneCommand(ctx, args[1:], stdout, stderr)
	case "diffuse":
		return runDiffuseCommand(ctx, args[1:], stdout, stderr)
	case "audio":
		return runAudioCommand(ctx, args[1:], stdout, stderr)
	case "vision":
		return runVisionCommand(ctx, args[1:], stdout, stderr)
	case "ebook":
		return runEbookCommand(ctx, args[1:], stdout, stderr)
	case "slice":
		return runSliceCommand(ctx, args[1:], stdout, stderr)
	case "state-pack":
		return runStatePackCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s: unknown command %q", cliName(), args[0])
		printUsage(stderr)
		return 2
	}
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
	if *probeDevice {
		annotateMetallib(&report)
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
	if report.Labels["metallib_kernel"] != "" {
		core.WriteString(stdout, core.Sprintf("  metallib: %s (%s) kernel=%s\n",
			report.Labels["metallib_source"], report.Labels["metallib_path"], report.Labels["metallib_kernel"]))
	}
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

var runCPUFFNMemoryEstimate = func(ctx context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
	report, err := mlx.EstimateCPUSplitFFNMemory(ctx, sourcePath, mlx.WithCPUSplitFFNMaxCachedLayers(cpuFFNCache))
	if err != nil {
		return nil, err
	}
	return &report, nil
}

var runDiscoverLocalRuntime = mlx.DiscoverLocalRuntime

var runGetDeviceInfo = mlx.GetDeviceInfo

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

func printUsage(w io.Writer) {
	name := cliName()
	core.WriteString(w, core.Sprintf("Usage: %s <command> [flags]\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, "Run inference\n")
	core.WriteString(w, "  menubar             tray-only macOS app — start/stop serve from the menu bar\n")
	core.WriteString(w, "  serve               host OpenAI/Anthropic/Ollama HTTP API for a loaded model\n")
	core.WriteString(w, "  generate            one-shot generate + decode tok/s (no serve; like-for-like bench)\n")
	core.WriteString(w, "  diffuse             block-diffusion decode (DiffusionGemma checkpoints)\n")
	core.WriteString(w, "  audio               answer a prompt about a WAV clip (Gemma 4 E2B/E4B audio tower)\n")
	core.WriteString(w, "  vision              answer a prompt about images / video frames (vision tower)\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Inspect what is installed\n")
	core.WriteString(w, "  discover            report local MLX runtime + optional model candidates\n")
	core.WriteString(w, "  pack                validate a local native model pack\n")
	core.WriteString(w, "  ssd-recipes         print native Simple Self-Distillation recipe defaults\n")
	core.WriteString(w, "  ssd-eval            prepare a native Simple Self-Distillation eval plan\n")
	core.WriteString(w, "  memory-pretrain-build  build native hierarchical-memory pretraining artifacts\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Tune a machine + model pairing\n")
	core.WriteString(w, "  tune                measure AR vs MTP draft blocks, persist the winner for serve\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Transform a model\n")
	core.WriteString(w, "  slice               materialise a local model slice for split/reload tests\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "State container ops\n")
	core.WriteString(w, "  state-pack          pack a State marker + binary log into a Trix .kv container\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Examples\n")
	core.WriteString(w, core.Sprintf("  %s discover                                  # what runtime + models you have\n", name))
	core.WriteString(w, core.Sprintf("  %s serve --model ~/models/lemer-lite         # OpenAI HTTP on :36911\n", name))
	core.WriteString(w, core.Sprintf("  %s pack ~/models/lemer-lite                  # validate a model on disk\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, core.Sprintf("Run \"%s <command> -h\" for command-specific flags.\n", name))
}
