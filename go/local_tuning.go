// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

// LocalDiscoveryConfig controls the cheap machine/model discovery path used by
// setup UIs before any optional autotune run.
type LocalDiscoveryConfig struct {
	ModelDirs         []string
	Workloads         []inference.TuningWorkload
	MaxModels         int
	IncludeModels     bool
	IncludeCandidates bool
	Device            DeviceInfo
	Labels            map[string]string
}

// LocalTuningRunConfig controls an opt-in tuning pass. Each candidate is
// loaded, measured, emitted, and closed independently so UIs can stream
// progress and stop early.
type LocalTuningRunConfig struct {
	ModelPath  string
	Workload   inference.TuningWorkload
	Candidates []inference.TuningCandidate
	Bench      bench.Config
	Emit       func(inference.TuningEvent) bool
}

var (
	loadTuningModel = LoadModel
	runTuningBench  = RunFastEvalBench
)

const tuningMachineHashLabel = "machine_hash"

func (backend *metalbackend) DiscoverMachine(ctx context.Context, req inference.MachineDiscoveryRequest) (*inference.MachineDiscoveryReport, error) {
	report, err := DiscoverLocalRuntime(ctx, LocalDiscoveryConfig{
		ModelDirs:         append([]string(nil), req.ModelDirs...),
		Workloads:         append([]inference.TuningWorkload(nil), req.Workloads...),
		MaxModels:         req.MaxModels,
		IncludeModels:     req.IncludeModels,
		IncludeCandidates: req.IncludeCandidates,
		Labels:            cloneTuningLabels(req.Labels),
	})
	if err != nil {
		return nil, err
	}
	return &report, nil
}

func (backend *metalbackend) PlanTuning(ctx context.Context, req inference.TuningPlanRequest) (*inference.TuningPlan, error) {
	plan, err := PlanLocalTuning(ctx, req)
	if err != nil {
		return nil, err
	}
	return &plan, nil
}

// DiscoverLocalRuntime returns the MLX runtime/device report and, when asked,
// discovered models plus first-pass tuning candidates. It is metadata-first and
// does not load model weights.
func DiscoverLocalRuntime(ctx context.Context, cfg LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.MachineDiscoveryReport{}, err
	}
	device := cfg.Device
	if device.MemorySize == 0 && device.MaxRecommendedWorkingSetSize == 0 && device.Architecture == "" {
		device = safeRuntimeDeviceInfo()
	}
	machineHash := tuningMachineHash(device)
	deviceInfo := tuningDeviceInfo(device)
	deviceInfo.Labels = withTuningMachineHash(deviceInfo.Labels, machineHash)
	workloads := tuningWorkloadsOrDefault(cfg.Workloads)
	caps := metalCapabilityReport(inference.ModelIdentity{}, inference.AdapterIdentity{}, Available())
	report := inference.MachineDiscoveryReport{
		Runtime:      caps.Runtime,
		Device:       deviceInfo,
		Available:    caps.Available,
		Capabilities: append([]inference.Capability(nil), caps.Capabilities...),
		CacheModes:   append([]string(nil), caps.CacheModes...),
		Workloads:    workloads,
		Labels:       withTuningMachineHash(cfg.Labels, machineHash),
	}
	if len(report.Runtime.Labels) == 0 {
		report.Runtime.Labels = nil
	}
	if !cfg.IncludeModels && len(cfg.ModelDirs) == 0 {
		return report, nil
	}

	maxModels := cfg.MaxModels
	for _, dir := range cfg.ModelDirs {
		for discovered := range inference.Discover(dir) {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			report.Models = append(report.Models, discovered)
			if cfg.IncludeCandidates {
				modelIdentity := discoveredModelIdentity(discovered)
				if inspected, err := model.Inspect(discovered.Path, mp.WithPackRequireChatTemplate(false)); err == nil {
					modelIdentity = modelPackIdentity(inspected, modelIdentity)
				}
				plan, err := PlanLocalTuning(ctx, inference.TuningPlanRequest{
					Runtime:   report.Runtime,
					Device:    report.Device,
					Model:     modelIdentity,
					Workloads: workloads,
					Budget:    inference.TuningBudget{MaxCandidates: 2},
				})
				if err != nil {
					report.Warnings = append(report.Warnings, err.Error())
				} else {
					report.Candidates = append(report.Candidates, plan.Candidates...)
				}
			}
			if maxModels > 0 && len(report.Models) >= maxModels {
				return report, nil
			}
		}
	}
	return report, nil
}

// PlanLocalTuning turns measured MLX device facts and model metadata into a
// small candidate set suitable for optional smoke benchmarking.
func PlanLocalTuning(ctx context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.TuningPlan{}, err
	}
	device := tuningRequestDevice(req.Device)
	modelIdentity := req.Model
	var pack *mp.ModelPack
	if req.Model.Path != "" {
		if inspected, err := model.Inspect(req.Model.Path, mp.WithPackRequireChatTemplate(false)); err == nil {
			pack = &inspected
			modelIdentity = modelPackIdentity(inspected, modelIdentity)
		}
	}
	modelInfo := tuningModelInfo(modelIdentity)
	memoryPlan := PlanMemory(MemoryPlanInput{
		Device:    device,
		Pack:      pack,
		ModelInfo: &modelInfo,
	})
	runtime := req.Runtime
	if runtime.Backend == "" {
		runtime.Backend = "metal"
	}
	if runtime.Device == "" {
		runtime.Device = device.Architecture
	}
	if runtime.CacheMode == "" {
		runtime.CacheMode = string(memoryPlan.CacheMode)
	}
	runtime, runtimeWarning := tuningRuntimeForArchitecture(runtime, modelIdentity.Architecture)

	workloads := tuningWorkloadsOrDefault(req.Workloads)
	plan := inference.TuningPlan{
		Runtime:     runtime,
		Device:      tuningDeviceInfo(device),
		Model:       modelIdentity,
		Adapter:     req.Adapter,
		Workloads:   workloads,
		Recommended: map[inference.TuningWorkload]string{},
		Labels:      cloneTuningLabels(req.Labels),
	}
	if runtimeWarning != "" {
		plan.Warnings = append(plan.Warnings, runtimeWarning)
	}
	maxCandidates := req.Budget.MaxCandidates
	for _, workload := range workloads {
		candidate := tuningCandidateForWorkload(workload, modelIdentity, req.Adapter, runtime, memoryPlan)
		plan.Candidates = append(plan.Candidates, candidate)
		if plan.Recommended[workload] == "" {
			plan.Recommended[workload] = candidate.ID
		}
		if maxCandidates > 0 && len(plan.Candidates) >= maxCandidates {
			break
		}
	}
	if len(plan.Recommended) == 0 {
		plan.Recommended = nil
	}
	return plan, nil
}

func tuningRuntimeForArchitecture(runtime inference.RuntimeIdentity, architecture string) (inference.RuntimeIdentity, string) {
	p, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return runtime, ""
	}
	runtime.NativeRuntime = p.NativeRuntime
	if runtime.Labels == nil {
		runtime.Labels = map[string]string{}
	} else {
		runtime.Labels = cloneTuningLabels(runtime.Labels)
	}
	runtime.Labels["architecture"] = p.ID
	runtime.Labels["native_runtime"] = boolLabel(p.NativeRuntime)
	if p.NativeRuntime {
		return runtime, ""
	}
	runtime.Backend = "mlx_lm"
	runtime.Labels["fallback_backend"] = "mlx_lm"
	return runtime, "architecture " + p.ID + " is metadata-only in native go-mlx; using mlx_lm fallback for tuning candidates"
}

// TuningCandidateLoadOptions converts a selected candidate into LoadModel
// options. This is the fast path a UI uses after selecting or persisting a
// tuning profile.
func TuningCandidateLoadOptions(candidate inference.TuningCandidate) []LoadOption {
	opts := []LoadOption{
		WithAutoMemoryPlan(false),
		WithPromptCache(candidate.PromptCache),
	}
	if candidate.ContextLength > 0 {
		opts = append(opts, WithContextLength(candidate.ContextLength))
	}
	if candidate.ParallelSlots > 0 {
		opts = append(opts, WithParallelSlots(candidate.ParallelSlots))
	}
	if candidate.PromptCacheMinTokens > 0 {
		opts = append(opts, WithPromptCacheMinTokens(candidate.PromptCacheMinTokens))
	}
	if candidate.CachePolicy != "" {
		opts = append(opts, WithCachePolicy(memory.KVCachePolicy(candidate.CachePolicy)))
	}
	if candidate.CacheMode != "" {
		opts = append(opts, WithKVCacheMode(memory.KVCacheMode(candidate.CacheMode)))
	}
	if candidate.BatchSize > 0 {
		opts = append(opts, WithBatchSize(candidate.BatchSize))
	}
	if candidate.PrefillChunkSize > 0 {
		opts = append(opts, WithPrefillChunkSize(candidate.PrefillChunkSize))
	}
	if candidate.ExpectedQuantization > 0 {
		opts = append(opts, WithExpectedQuantization(candidate.ExpectedQuantization))
	}
	if candidate.MemoryLimitBytes > 0 || candidate.CacheLimitBytes > 0 || candidate.WiredLimitBytes > 0 {
		opts = append(opts, WithAllocatorLimits(candidate.MemoryLimitBytes, candidate.CacheLimitBytes, candidate.WiredLimitBytes))
	}
	if candidate.Adapter.Path != "" {
		opts = append(opts, WithAdapterPath(candidate.Adapter.Path))
	}
	return opts
}

// RunLocalTuning loads and measures candidates one at a time, emitting a start
// and result event for each candidate. Candidate failures are returned as
// result entries so the UI can keep going.
func RunLocalTuning(ctx context.Context, cfg LocalTuningRunConfig) ([]inference.TuningResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(cfg.Candidates) == 0 {
		return nil, core.NewError("mlx: local tuning requires at least one candidate")
	}
	workload := cfg.Workload
	if workload == "" {
		workload = cfg.Candidates[0].Workload
	}
	if workload == "" {
		workload = inference.TuningWorkloadChat
	}
	benchCfg := normalizeLocalTuningBench(cfg.Bench)
	results := make([]inference.TuningResult, 0, len(cfg.Candidates))
	for _, candidate := range cfg.Candidates {
		if err := ctx.Err(); err != nil {
			return results, err
		}
		if !emitTuningEvent(cfg.Emit, inference.TuningEvent{Kind: inference.TuningEventCandidate, Candidate: candidate}) {
			return results, nil
		}
		result := runLocalTuningCandidate(ctx, cfg.ModelPath, workload, candidate, benchCfg)
		results = append(results, result)
		if !emitTuningEvent(cfg.Emit, inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: candidate, Result: &result}) {
			return results, nil
		}
	}
	return results, nil
}

func runLocalTuningCandidate(ctx context.Context, modelPath string, workload inference.TuningWorkload, candidate inference.TuningCandidate, benchCfg bench.Config) (result inference.TuningResult) {
	path := candidate.Model.Path
	if path == "" {
		path = modelPath
	}
	result = inference.TuningResult{Candidate: candidate}
	if path == "" {
		result.Error = "model path is required"
		return result
	}
	loadStart := time.Now()
	modelHandle, err := loadTuningModel(path, TuningCandidateLoadOptions(candidate)...)
	loadDuration := time.Since(loadStart)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	defer func() {
		if closeErr := modelHandle.Close(); closeErr != nil && result.Error == "" {
			result.Error = closeErr.Error()
		}
	}()
	benchCfg.ModelPath = path
	if benchCfg.Model == "" {
		benchCfg.Model = candidate.Model.ID
	}
	report, err := runTuningBench(ctx, modelHandle, benchCfg)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	result.Measurements = tuningMeasurementsFromBench(report)
	result.Measurements.LoadMilliseconds = durationMilliseconds(loadDuration)
	result.Score = inference.ScoreTuningMeasurements(workload, result.Measurements)
	return result
}

func normalizeLocalTuningBench(cfg bench.Config) bench.Config {
	if cfg.Prompt == "" {
		cfg.Prompt = "Write one precise sentence about local inference."
	}
	if cfg.CachePrompt == "" {
		cfg.CachePrompt = cfg.Prompt
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 16
	}
	if cfg.Runs <= 0 {
		cfg.Runs = 1
	}
	return cfg
}

func tuningMeasurementsFromBench(report *bench.Report) inference.TuningMeasurements {
	if report == nil {
		return inference.TuningMeasurements{}
	}
	return inference.TuningMeasurements{
		PromptTokens:            report.Generation.PromptTokens,
		GeneratedTokens:         report.Generation.GeneratedTokens,
		FirstTokenMilliseconds:  durationMilliseconds(report.Generation.FirstTokenDuration),
		PrefillTokensPerSec:     report.Generation.PrefillTokensPerSec,
		DecodeTokensPerSec:      report.Generation.DecodeTokensPerSec,
		PromptCacheHitRate:      report.PromptCache.HitRate,
		KVRestoreMilliseconds:   durationMilliseconds(report.KVRestore.Duration),
		StateBundleMilliseconds: durationMilliseconds(report.StateBundle.Duration),
		TotalMilliseconds:       durationMilliseconds(report.Generation.TotalDuration),
		PeakMemoryBytes:         report.Generation.PeakMemoryBytes,
		ActiveMemoryBytes:       report.Generation.ActiveMemoryBytes,
		CorrectnessSmokeResult:  tuningCorrectnessSmokeResult(report.Quality),
		CorrectnessSmokeChecks:  len(report.Quality.Checks),
	}
}

func tuningCorrectnessSmokeResult(report bench.QualityReport) string {
	if len(report.Checks) == 0 {
		return ""
	}
	for _, check := range report.Checks {
		if !check.Pass {
			return "failed"
		}
	}
	return "passed"
}

func durationMilliseconds(d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(d) / float64(time.Millisecond)
}

func emitTuningEvent(emit func(inference.TuningEvent) bool, event inference.TuningEvent) bool {
	if emit == nil {
		return true
	}
	return emit(event)
}

func tuningCandidateForWorkload(workload inference.TuningWorkload, modelIdentity inference.ModelIdentity, adapter inference.AdapterIdentity, runtime inference.RuntimeIdentity, plan memory.Plan) inference.TuningCandidate {
	candidate := inference.TuningCandidate{
		Workload:             workload,
		Model:                modelIdentity,
		Adapter:              adapter,
		Runtime:              runtime,
		ContextLength:        plan.ContextLength,
		ParallelSlots:        maxPositive(plan.ParallelSlots, 1),
		PromptCache:          plan.PromptCache,
		PromptCacheMinTokens: plan.PromptCacheMinTokens,
		CachePolicy:          string(plan.CachePolicy),
		CacheMode:            string(plan.CacheMode),
		BatchSize:            maxPositive(plan.BatchSize, 1),
		PrefillChunkSize:     maxPositive(plan.PrefillChunkSize, 512),
		ExpectedQuantization: plan.PreferredQuantization,
		MemoryLimitBytes:     plan.MemoryLimitBytes,
		CacheLimitBytes:      plan.CacheLimitBytes,
		WiredLimitBytes:      plan.WiredLimitBytes,
		Reasons:              append([]string(nil), plan.Notes...),
		Labels:               map[string]string{"machine_class": string(plan.MachineClass)},
	}
	switch workload {
	case inference.TuningWorkloadLowLatency:
		candidate.ContextLength = minPositive(candidate.ContextLength, 32768)
		candidate.BatchSize = 1
		candidate.ParallelSlots = 1
		candidate.PrefillChunkSize = minPositive(candidate.PrefillChunkSize, 1024)
		candidate.Reasons = append(candidate.Reasons, "latency profile favours small batches and short prefill chunks")
	case inference.TuningWorkloadThroughput:
		candidate.BatchSize = maxPositive(candidate.BatchSize, 4)
		candidate.Reasons = append(candidate.Reasons, "throughput profile favours larger batches where memory permits")
	case inference.TuningWorkloadLongContext:
		candidate.PromptCache = true
		candidate.CachePolicy = string(memory.KVCacheFull)
		candidate.Reasons = append(candidate.Reasons, "long-context profile favours full cache retention")
	case inference.TuningWorkloadAgentState:
		candidate.PromptCache = true
		candidate.Labels["state_restore"] = "candidate"
		candidate.Reasons = append(candidate.Reasons, "agent-state profile measures prompt-cache and state restore")
	}
	candidate.ID = inference.CandidateID(workload, candidate.CacheMode, candidate.ContextLength, candidate.BatchSize)
	if len(candidate.Reasons) == 0 {
		candidate.Reasons = nil
	}
	return candidate
}

func tuningRequestDevice(device inference.MachineDeviceInfo) DeviceInfo {
	if device.MemorySize == 0 && device.MaxRecommendedWorkingSetSize == 0 && device.Architecture == "" {
		return safeRuntimeDeviceInfo()
	}
	return DeviceInfo{
		Name:                         device.Name,
		Architecture:                 device.Architecture,
		MaxBufferLength:              device.MaxBufferLength,
		MaxRecommendedWorkingSetSize: device.MaxRecommendedWorkingSetSize,
		MemorySize:                   device.MemorySize,
	}
}

func tuningDeviceInfo(device DeviceInfo) inference.MachineDeviceInfo {
	return inference.MachineDeviceInfo{
		Name:                         device.Name,
		Architecture:                 device.Architecture,
		MaxBufferLength:              device.MaxBufferLength,
		MaxRecommendedWorkingSetSize: device.MaxRecommendedWorkingSetSize,
		MemorySize:                   device.MemorySize,
	}
}

func tuningMachineHash(device DeviceInfo) string {
	if device.Name == "" &&
		device.Architecture == "" &&
		device.MaxBufferLength == 0 &&
		device.MaxRecommendedWorkingSetSize == 0 &&
		device.MemorySize == 0 {
		return ""
	}
	identity := inference.MachineDeviceInfo{
		Name:                         device.Name,
		Architecture:                 device.Architecture,
		MaxBufferLength:              device.MaxBufferLength,
		MaxRecommendedWorkingSetSize: device.MaxRecommendedWorkingSetSize,
		MemorySize:                   device.MemorySize,
	}
	data := core.JSONMarshal(identity)
	if !data.OK {
		return ""
	}
	return "sha256:" + core.SHA256Hex(data.Value.([]byte))
}

func tuningModelInfo(identity inference.ModelIdentity) ModelInfo {
	return ModelInfo{
		Architecture:  identity.Architecture,
		VocabSize:     identity.VocabSize,
		NumLayers:     identity.NumLayers,
		HiddenSize:    identity.HiddenSize,
		QuantBits:     identity.QuantBits,
		QuantGroup:    identity.QuantGroup,
		ContextLength: identity.ContextLength,
	}
}

func discoveredModelIdentity(model inference.DiscoveredModel) inference.ModelIdentity {
	return inference.ModelIdentity{
		Path:         model.Path,
		Architecture: model.ModelType,
		QuantBits:    model.QuantBits,
		QuantGroup:   model.QuantGroup,
		QuantType:    model.QuantType,
	}
}

func modelPackIdentity(pack mp.ModelPack, fallback inference.ModelIdentity) inference.ModelIdentity {
	identity := fallback
	if identity.Path == "" {
		identity.Path = pack.Path
	}
	if identity.Architecture == "" {
		identity.Architecture = pack.Architecture
	}
	if identity.QuantBits == 0 {
		identity.QuantBits = pack.QuantBits
	}
	if identity.QuantGroup == 0 {
		identity.QuantGroup = pack.QuantGroup
	}
	if identity.QuantType == "" {
		identity.QuantType = pack.QuantType
	}
	if identity.ContextLength == 0 {
		identity.ContextLength = pack.ContextLength
	}
	if identity.NumLayers == 0 {
		identity.NumLayers = pack.NumLayers
	}
	if identity.HiddenSize == 0 {
		identity.HiddenSize = pack.HiddenSize
	}
	if identity.VocabSize == 0 {
		identity.VocabSize = pack.VocabSize
	}
	return identity
}

func tuningWorkloadsOrDefault(workloads []inference.TuningWorkload) []inference.TuningWorkload {
	if len(workloads) == 0 {
		return inference.DefaultTuningWorkloads()
	}
	return append([]inference.TuningWorkload(nil), workloads...)
}

func cloneTuningLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func withTuningMachineHash(labels map[string]string, machineHash string) map[string]string {
	out := cloneTuningLabels(labels)
	if machineHash == "" {
		return out
	}
	if out == nil {
		out = map[string]string{}
	}
	out[tuningMachineHashLabel] = machineHash
	return out
}

func boolLabel(value bool) string {
	if value {
		return "true"
	}
	return "false"
}
