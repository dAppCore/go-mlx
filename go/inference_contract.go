// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/memory"
	"strconv"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
)

func (backend *metalbackend) Capabilities() inference.CapabilityReport {
	return metalCapabilityReport(inference.ModelIdentity{}, inference.AdapterIdentity{}, backend.Available())
}

func (backend *metalbackend) SetRuntimeMemoryLimits(limits inference.RuntimeMemoryLimits) inference.RuntimeMemoryLimits {
	applied := limits
	if limits.CacheLimitBytes > 0 {
		applied.PreviousCacheLimitBytes = SetCacheLimit(limits.CacheLimitBytes)
	}
	if limits.MemoryLimitBytes > 0 {
		applied.PreviousMemoryLimitBytes = SetMemoryLimit(limits.MemoryLimitBytes)
	}
	return applied
}

func (backend *metalbackend) PlanModelFit(ctx context.Context, ident inference.ModelIdentity, memoryBytes uint64) (*inference.ModelFitReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	device := memoryPlannerDeviceInfo()
	if memoryBytes > 0 {
		device.MemorySize = memoryBytes
		device.MaxRecommendedWorkingSetSize = memoryBytes
	}
	modelInfo := ModelInfo{
		Architecture:  ident.Architecture,
		VocabSize:     ident.VocabSize,
		NumLayers:     ident.NumLayers,
		HiddenSize:    ident.HiddenSize,
		QuantBits:     ident.QuantBits,
		QuantGroup:    ident.QuantGroup,
		ContextLength: ident.ContextLength,
	}
	plan := PlanMemory(MemoryPlanInput{Device: device, ModelInfo: &modelInfo})
	architectureOK := ident.Architecture == "" || model.SupportsArchitecture(ident.Architecture)
	quantizationOK := ident.QuantBits == 0 || plan.PreferredQuantization == 0 || ident.QuantBits <= plan.PreferredQuantization
	fits := architectureOK && quantizationOK
	if plan.MemoryLimitBytes > 0 && plan.EstimatedKVCacheModeBytes > 0 && plan.EstimatedKVCacheModeBytes > plan.MemoryLimitBytes {
		fits = false
	}

	return &inference.ModelFitReport{
		Model:          ident,
		Fits:           fits,
		MemoryPlan:     toInferenceMemoryPlan(plan),
		ArchitectureOK: architectureOK,
		QuantizationOK: quantizationOK,
		Notes:          core.SliceClone(plan.Notes),
	}, nil
}

func (backend *metalbackend) PlanModelSlice(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plan, err := inference.PlanModelSlice(req)
	if err != nil {
		return nil, err
	}
	if plan.Labels == nil {
		// Pre-size for the two known keys we set below — initial
		// bucket holds both without a grow on the second insertion.
		plan.Labels = make(map[string]string, 2)
	}
	plan.Labels["backend"] = "metal"
	plan.Labels["library"] = "go-mlx"
	plan.Notes = append(plan.Notes, "go-mlx can materialise LarQL-style safetensors slices; local dense split execution is experimental and remote FFN/expert execution remains backend work")
	return &plan, nil
}

func (backend *metalbackend) PlanSplitInference(ctx context.Context, req inference.SplitInferenceRequest) (*inference.SplitInferencePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mode := req.Mode
	if mode == "" {
		mode = inference.SplitInferenceModeLocal
	}
	localPreset := req.LocalPreset
	if localPreset == "" {
		localPreset = inference.ModelSlicePresetFull
		switch mode {
		case inference.SplitInferenceModeRemoteFFN, inference.SplitInferenceModeRemoteEmbedFFN, inference.SplitInferenceModeRemoteExperts:
			localPreset = inference.ModelSlicePresetClient
		}
	}
	local, err := backend.PlanModelSlice(ctx, inference.ModelSliceRequest{
		Preset:  localPreset,
		Model:   req.Model,
		Adapter: req.Adapter,
		Labels:  req.Labels,
	})
	if err != nil {
		return nil, err
	}
	plan := &inference.SplitInferencePlan{
		Mode:       mode,
		Model:      req.Model,
		Adapter:    req.Adapter,
		LocalSlice: *local,
		Endpoints:  cloneInferenceSplitEndpoints(req.Endpoints),
		Labels:     cloneInferenceLabels(req.Labels),
	}
	if plan.Labels == nil {
		// Pre-size for the two known keys we're about to set
		// (backend, library) so the map's initial bucket holds both
		// without triggering a grow on the second insertion.
		plan.Labels = make(map[string]string, 2)
	}
	plan.Labels["backend"] = "metal"
	plan.Labels["library"] = "go-mlx"
	if err := inference.ValidateSplitInferencePlan(*plan); err != nil {
		return nil, err
	}
	return plan, nil
}

func (adapter *metaladapter) Capabilities() inference.CapabilityReport {
	if adapter == nil || adapter.model == nil {
		return metalCapabilityReportWithLoadReady(inference.ModelIdentity{}, inference.AdapterIdentity{}, false, true)
	}
	return metalCapabilityReport(toInferenceModelIdentity(adapter.rootModel().Info()), adapter.ActiveAdapter(), true)
}

func (adapter *metaladapter) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if adapter == nil || adapter.model == nil {
		return "", errMLXModelNil
	}
	return chat.Format(messages, chat.Config{Architecture: adapter.model.ModelType()}), nil
}

func (adapter *metaladapter) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if adapter == nil || adapter.model == nil {
		return inference.AdapterIdentity{}, errMLXModelNil
	}
	if _, err := adapter.model.LoadLoRA(path); err != nil {
		return inference.AdapterIdentity{}, err
	}
	return toInferenceAdapterIdentity(adapter.model.Adapter()), nil
}

func (adapter *metaladapter) UnloadAdapter() error {
	if adapter == nil || adapter.model == nil {
		return errMLXModelNil
	}
	return adapter.model.UnloadLoRA()
}

func (adapter *metaladapter) ActiveAdapter() inference.AdapterIdentity {
	if adapter == nil || adapter.model == nil {
		return inference.AdapterIdentity{}
	}
	return toInferenceAdapterIdentity(adapter.model.Adapter())
}

func (adapter *metaladapter) SetProbeSink(sink inference.ProbeSink) {
	if adapter == nil {
		return
	}
	adapter.probeSink = sink
	adapter.schedulerMu.Lock()
	scheduler := adapter.scheduler
	adapter.schedulerMu.Unlock()
	if scheduler != nil {
		scheduler.SetProbeSink(sink)
	}
}

func (adapter *metaladapter) Benchmark(ctx context.Context, cfg inference.BenchConfig) (*inference.BenchReport, error) {
	if adapter == nil || adapter.model == nil {
		return nil, errMLXModelNil
	}
	report, err := RunFastEval(ctx, adapter.fastEvalRunner(), toFastEvalConfig(cfg))
	if err != nil {
		return nil, err
	}
	return toInferenceBenchReport(report), nil
}

func (adapter *metaladapter) Evaluate(ctx context.Context, dataset inference.DatasetStream, cfg inference.EvalConfig) (*inference.EvalReport, error) {
	if adapter == nil || adapter.model == nil {
		return nil, errMLXModelNil
	}
	report, err := eval.RunDataset(ctx, adapter.evalRunner(), wrapSFTDataset(inferenceDataset{stream: dataset}), toEvalConfig(cfg))
	if err != nil {
		return nil, err
	}
	return toInferenceEvalReport(report), nil
}

func (adapter *metaladapter) TrainSFT(ctx context.Context, dataset inference.DatasetStream, cfg inference.TrainingConfig) (*inference.TrainingResult, error) {
	if adapter == nil || adapter.model == nil {
		return nil, errMLXModelNil
	}
	model := adapter.rootModel()
	result, err := model.TrainSFT(ctx, inferenceDataset{stream: dataset}, toSFTConfig(cfg, adapter.probeSink))
	if err != nil {
		return nil, err
	}
	return toInferenceTrainingResult(model.Info(), result, cfg), nil
}

func (adapter *metaladapter) generateConfig(opts ...inference.GenerateOption) metal.GenerateConfig {
	cfg := inference.ApplyGenerateOpts(opts)
	out := inferenceGenerateConfigToMetal(cfg)
	if adapter != nil && adapter.probeSink != nil {
		out.ProbeSink = toMetalInferenceProbeSink(adapter.probeSink)
	}
	return out
}

func (adapter *metaladapter) rootModel() *Model {
	if adapter == nil || adapter.model == nil {
		return &Model{}
	}
	return &Model{
		model:       adapter.model,
		tok:         &Tokenizer{tok: adapter.model.Tokenizer()},
		adapterInfo: toRootAdapterInfo(adapter.model.Adapter()),
		cfg:         LoadConfig{ContextLength: adapter.model.Info().ContextLength},
	}
}

func (adapter *metaladapter) fastEvalRunner() bench.Runner {
	return NewModelFastEvalRunner(adapter.rootModel())
}

func (adapter *metaladapter) evalRunner() eval.Runner {
	return NewModelEvalRunner(adapter.rootModel())
}

type inferenceDataset struct {
	stream inference.DatasetStream
}

// Per-sample / per-reset sentinels — inferenceDataset.Next fires for
// every row in Evaluate/TrainSFT and was paying a per-call core.NewError
// alloc on the nil-stream guard.
var (
	errMLXInferenceDatasetNil          = core.NewError("mlx: inference dataset stream is nil")
	errMLXInferenceDatasetNotResetter  = core.NewError("mlx: inference dataset stream is not resettable")
)

func (d inferenceDataset) Next() (dataset.Sample, bool, error) {
	if d.stream == nil {
		return dataset.Sample{}, false, errMLXInferenceDatasetNil
	}
	sample, ok, err := d.stream.Next()
	if err != nil || !ok {
		return dataset.Sample{}, ok, err
	}
	return dataset.Sample{
		Prompt:   sample.Prompt,
		Response: sample.Response,
		Text:     sample.Text,
		Meta:     cloneInferenceLabels(sample.Labels),
	}, true, nil
}

func (d inferenceDataset) Reset() error {
	if d.stream == nil {
		return errMLXInferenceDatasetNil
	}
	resetter, ok := d.stream.(inference.DatasetResetter)
	if !ok {
		return errMLXInferenceDatasetNotResetter
	}
	return resetter.Reset()
}

// metalInferenceProbeSinkAdapter converts metal.ProbeEvent to
// inference.ProbeEvent and forwards to the wrapped inference.ProbeSink.
// Replaces the metal.ProbeSinkFunc closure form that captured `sink`
// into a fresh func per dispatch call (24 B closure per dispatch even
// when the sink emitted nothing). The struct form holds the wrapped
// sink as a single interface field (16 B = two pointer-sized words).
type metalInferenceProbeSinkAdapter struct {
	sink inference.ProbeSink
}

// EmitProbe converts metal.ProbeEvent to inference.ProbeEvent and forwards.
func (a metalInferenceProbeSinkAdapter) EmitProbe(event metal.ProbeEvent) {
	a.sink.EmitProbe(toInferenceProbeEvent(event))
}

func toMetalInferenceProbeSink(sink inference.ProbeSink) metal.ProbeSink {
	if sink == nil {
		return nil
	}
	return metalInferenceProbeSinkAdapter{sink: sink}
}

var metalCapabilityDeviceInfo = func(available bool) DeviceInfo {
	if !available {
		return DeviceInfo{}
	}
	return safeRuntimeDeviceInfo()
}

// metalDeviceLabel cache — the device probe returns the same
// (MemorySize, MaxRecommendedWorkingSetSize) tuple for the whole process
// lifetime (host RAM doesn't grow between calls). A single-slot lookup
// matches the singleton-device pattern; tests that swap the
// metalCapabilityDeviceInfo hook with synthetic device shapes still
// re-format on the first call with the new tuple.
//
// The cache stores an immutable *metalDeviceLabelEntry behind an
// atomic.Pointer so the hot read path is lock-free. Cache misses (new
// device or first call) take the rare-path mutex to populate; misses
// during test hook swaps are bounded by the number of distinct device
// shapes exercised in a single run.
type metalDeviceLabelEntry struct {
	memorySize     uint64
	workingSetSize uint64
	memoryStr      string
	workingSetStr  string
}

var (
	metalDeviceLabelCache atomic.Pointer[metalDeviceLabelEntry]
	metalDeviceLabelMu    sync.Mutex
)

// metalRuntimeLabelsEntry caches the per-call runtimeLabels map for a
// given device shape AND loadReady value. The map header itself (~80 B)
// would otherwise allocate per call — the singleton-device contract +
// boolLabel's two-string output means ≤ 2 distinct maps fit the entire
// process lifetime. atomic.Pointer keeps the read path lock-free.
type metalRuntimeLabelsEntry struct {
	memorySize     uint64
	workingSetSize uint64
	loadReady      bool
	labels         map[string]string
}

// metalRuntimeLabelsCache stores both the loadReady=true and loadReady=false
// shapes side-by-side — at most one of each. Tests that swap the
// metalCapabilityDeviceInfo hook with synthetic device shapes invalidate
// both slots on the next call with the new tuple.
type metalRuntimeLabelsCachePair struct {
	loadReadyTrue  *metalRuntimeLabelsEntry
	loadReadyFalse *metalRuntimeLabelsEntry
}

var (
	metalRuntimeLabelsCache atomic.Pointer[metalRuntimeLabelsCachePair]
	metalRuntimeLabelsMu    sync.Mutex
)

// metalDeviceLabelStrings returns the strconv.FormatUint outputs for
// (memorySize, workingSetSize). The atomic single-slot cache hits on
// every subsequent call with the same tuple — lock-free read path,
// rare-path mutex only on miss. Returns "" for any zero-size input
// (so callers can branch on the empty string instead of duplicating
// the > 0 check).
func metalDeviceLabelStrings(memorySize, workingSetSize uint64) (string, string) {
	if memorySize == 0 && workingSetSize == 0 {
		return "", ""
	}
	if entry := metalDeviceLabelCache.Load(); entry != nil &&
		entry.memorySize == memorySize && entry.workingSetSize == workingSetSize {
		return entry.memoryStr, entry.workingSetStr
	}
	return metalDeviceLabelStringsSlow(memorySize, workingSetSize)
}

// metalDeviceLabelStringsSlow is the cache-miss path — populates the
// shared cache under the mutex. Split out so the fast atomic load path
// stays inlineable.
func metalDeviceLabelStringsSlow(memorySize, workingSetSize uint64) (string, string) {
	metalDeviceLabelMu.Lock()
	defer metalDeviceLabelMu.Unlock()
	// Double-check under the lock — another goroutine may have populated
	// the cache while we were waiting.
	if entry := metalDeviceLabelCache.Load(); entry != nil &&
		entry.memorySize == memorySize && entry.workingSetSize == workingSetSize {
		return entry.memoryStr, entry.workingSetStr
	}
	entry := &metalDeviceLabelEntry{
		memorySize:     memorySize,
		workingSetSize: workingSetSize,
	}
	if memorySize > 0 {
		entry.memoryStr = strconv.FormatUint(memorySize, 10)
	}
	if workingSetSize > 0 {
		entry.workingSetStr = strconv.FormatUint(workingSetSize, 10)
	}
	metalDeviceLabelCache.Store(entry)
	return entry.memoryStr, entry.workingSetStr
}

// metalRuntimeLabels returns the per-Capability-Report Runtime.Labels map
// for (memorySize, workingSetSize, loadReady). The result is a shared
// singleton — consumers (go-ml fallback, go-ai providers) treat the field
// as read-only so a shared map is safe. Lock-free atomic read on the hot
// path; rare-path mutex only on miss.
func metalRuntimeLabels(memoryBytesStr, workingSetBytesStr string, memorySize, workingSetSize uint64, loadReady bool) map[string]string {
	if pair := metalRuntimeLabelsCache.Load(); pair != nil {
		slot := pair.loadReadyTrue
		if !loadReady {
			slot = pair.loadReadyFalse
		}
		if slot != nil && slot.memorySize == memorySize && slot.workingSetSize == workingSetSize {
			return slot.labels
		}
	}
	return metalRuntimeLabelsSlow(memoryBytesStr, workingSetBytesStr, memorySize, workingSetSize, loadReady)
}

// metalRuntimeLabelsSlow is the cache-miss path. Builds the map under the
// mutex; preserves the OTHER loadReady slot when present + still device-
// matched, so a single (true) + single (false) call doesn't churn each
// other out.
func metalRuntimeLabelsSlow(memoryBytesStr, workingSetBytesStr string, memorySize, workingSetSize uint64, loadReady bool) map[string]string {
	metalRuntimeLabelsMu.Lock()
	defer metalRuntimeLabelsMu.Unlock()
	if pair := metalRuntimeLabelsCache.Load(); pair != nil {
		slot := pair.loadReadyTrue
		if !loadReady {
			slot = pair.loadReadyFalse
		}
		if slot != nil && slot.memorySize == memorySize && slot.workingSetSize == workingSetSize {
			return slot.labels
		}
	}
	labels := make(map[string]string, 3)
	if memoryBytesStr != "" {
		labels["memory_bytes"] = memoryBytesStr
	}
	if workingSetBytesStr != "" {
		labels["working_set_bytes"] = workingSetBytesStr
	}
	labels["load_available"] = boolLabel(loadReady)
	entry := &metalRuntimeLabelsEntry{
		memorySize:     memorySize,
		workingSetSize: workingSetSize,
		loadReady:      loadReady,
		labels:         labels,
	}
	// Preserve the other-loadReady slot if it still matches the same
	// device — only invalidate when the device shape itself shifts.
	pair := &metalRuntimeLabelsCachePair{}
	if existing := metalRuntimeLabelsCache.Load(); existing != nil {
		if loadReady {
			pair.loadReadyFalse = existing.loadReadyFalse
		} else {
			pair.loadReadyTrue = existing.loadReadyTrue
		}
		// Drop the preserved slot if the device shape no longer matches.
		if loadReady && pair.loadReadyFalse != nil &&
			(pair.loadReadyFalse.memorySize != memorySize || pair.loadReadyFalse.workingSetSize != workingSetSize) {
			pair.loadReadyFalse = nil
		}
		if !loadReady && pair.loadReadyTrue != nil &&
			(pair.loadReadyTrue.memorySize != memorySize || pair.loadReadyTrue.workingSetSize != workingSetSize) {
			pair.loadReadyTrue = nil
		}
	}
	if loadReady {
		pair.loadReadyTrue = entry
	} else {
		pair.loadReadyFalse = entry
	}
	metalRuntimeLabelsCache.Store(pair)
	return labels
}

func metalCapabilityReport(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool) inference.CapabilityReport {
	return metalCapabilityReportWithLoadReady(model, adapter, available, available)
}

func metalCapabilityReportWithLoadReady(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool, loadReady bool) inference.CapabilityReport {
	device := metalCapabilityDeviceInfo(available)
	// Cache the per-DeviceInfo formatted strings — the device probe
	// returns the same (MemorySize, WorkingSet) tuple for the whole
	// process lifetime (the host doesn't grow RAM between calls). The
	// shared cache hits on every subsequent call and reuses the
	// previously formatted strings, dropping 2 strconv allocs per
	// CapabilityReport invocation when the cache hits.
	memoryBytesStr, workingSetBytesStr := metalDeviceLabelStrings(device.MemorySize, device.MaxRecommendedWorkingSetSize)
	// Cache the whole runtimeLabels map per (device, loadReady) shape.
	// Real callers see only 2 distinct shapes per process (loadReady=true
	// and loadReady=false against the same singleton device), so the map
	// header allocation (~80 B per call) collapses to a single one-time
	// cost. metalRuntimeLabels is read-only — consumers don't mutate.
	runtimeLabels := metalRuntimeLabels(memoryBytesStr, workingSetBytesStr, device.MemorySize, device.MaxRecommendedWorkingSetSize, loadReady)
	// Full pre-built capability list — see metalCapabilityFixedFull /
	// metalCapabilityFixedFullMarked. Both forms (head + fixed tail) are
	// merged once at package init; the !loadReady tail has already been
	// passed through markMetalUnavailableCapabilities once at init.
	// Per call we just hand back the singleton — same Wave-5+ shared-
	// read-only-singleton pattern Architectures / Quantizations /
	// CacheModes / Labels adopted above. Drops the per-call
	// make([]inference.Capability, 39) alloc (~4 KB / 1 alloc) and the
	// copy() body that followed it; the only meaningful per-call cost
	// is now the CapabilityReport struct itself (returned by value).
	capabilities := metalCapabilityFixedFull
	if !loadReady {
		capabilities = metalCapabilityFixedFullMarked
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "metal",
			Device:        device.Architecture,
			NativeRuntime: true,
			Labels:        runtimeLabels,
		},
		Model:     model,
		Adapter:   adapter,
		Available: available,
		// Architectures / Quantizations / CacheModes share the package-init
		// singletons directly. The consumer surface is read-only — the only
		// callers that ever stored these into another struct (local_tuning
		// MachineDiscoveryReport, go-ml/go-ai display paths) clone defensively
		// at their own boundary, and no code in go-ml / go-ai / lem / cmd
		// mutates a CapabilityReport.{Architectures,Quantizations,CacheModes}
		// slice. Drops 3 clone allocs (~256 B) per CapabilityReport call.
		Architectures: metalCapabilityArchitectures,
		Quantizations: metalCapabilityQuantizations,
		CacheModes:    metalCapabilityCacheModes,
		Capabilities:  capabilities,
		// Single shared singleton — the value is the same constant on every
		// call ({"library": "go-mlx"}) and consumers treat report.Labels as
		// read-only (go-ml / go-ai never mutate it). Skips one map make +
		// one map-bucket alloc per CapabilityReport (~80 B + 1 alloc).
		Labels: metalCapabilityReportLabels,
	}
}

// metalLoadBlockedCapabilities is the immutable lookup table of
// capability IDs that get marked unsupported when the Metal runtime
// is unavailable. Hoisted to package-level so markMetalUnavailable-
// Capabilities doesn't rebuild a 26-entry hash map on every call.
var metalLoadBlockedCapabilities = map[inference.CapabilityID]bool{
	inference.CapabilityModelLoad:      true,
	inference.CapabilityAutoTuning:     true,
	inference.CapabilityBenchmark:      true,
	inference.CapabilityEvaluation:     true,
	inference.CapabilityGenerate:       true,
	inference.CapabilityChat:           true,
	inference.CapabilityClassify:       true,
	inference.CapabilityBatchGenerate:  true,
	inference.CapabilityLoRAInference:  true,
	inference.CapabilityStateBundle:    true,
	inference.CapabilityKVSnapshot:     true,
	inference.CapabilityPromptCache:    true,
	inference.CapabilityAgentMemory:    true,
	inference.CapabilityStateWake:      true,
	inference.CapabilityStateSleep:     true,
	inference.CapabilityStateFork:      true,
	inference.CapabilityLoRATraining:   true,
	inference.CapabilityDistillation:   true,
	inference.CapabilityGRPO:           true,
	inference.CapabilityProbeEvents:    true,
	inference.CapabilityAttentionProbe: true,
	inference.CapabilityLogitProbe:     true,
	inference.CapabilityScheduler:      true,
	inference.CapabilityRequestCancel:  true,
	inference.CapabilityCacheBlocks:    true,
	inference.CapabilityCacheWarm:      true,
}

func markMetalUnavailableCapabilities(capabilities []inference.Capability) []inference.Capability {
	const detail = "native Metal runtime is unavailable; no usable Metal device is visible for model loading"
	for i := range capabilities {
		if !metalLoadBlockedCapabilities[capabilities[i].ID] {
			continue
		}
		capabilities[i].Status = inference.CapabilityStatusUnsupported
		if core.Contains(capabilities[i].Detail, "native Metal runtime is unavailable") {
			continue
		}
		if capabilities[i].Detail == "" {
			capabilities[i].Detail = detail
		} else {
			capabilities[i].Detail = detail + "; " + capabilities[i].Detail
		}
	}
	return capabilities
}

// metalCapabilityFixedCount is the number of always-present capability
// entries in metalCapabilityReportWithLoadReady's literal — used to
// pre-size the capabilities slice in one allocation so the AlgorithmCapabilities
// append doesn't need to grow. Update this if the literal entry count
// changes (the test in inference_contract_test.go counts the slice
// after build and asserts the expected total).
const metalCapabilityFixedCount = 39

// metalModelLoadAvailable / metalModelLoadUnavailable are the two
// possible shapes of the capabilities[0] entry built per call from
// loadReady. inference.SupportedCapability / UnsupportedCapability
// each allocate (constructor + labels map) — caching the two
// outcomes once at package init drops 1–2 allocs per call.
var (
	metalModelLoadAvailable   = inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime)
	metalModelLoadUnavailable = inference.UnsupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime, "native Metal runtime is unavailable; no usable Metal device is visible for model loading")
)

// metalCapabilityFixedTail / metalCapabilityFixedTailMarked are the two
// pre-built shapes of the tail (38 static entries + AlgorithmCapabilities
// from profile). One mirrors the loadReady=true form, the other has
// already been passed through markMetalUnavailableCapabilities once at
// package init. They're folded into metalCapabilityFixedFull /
// metalCapabilityFixedFullMarked below (head + tail) — the per-call
// path now reads only the full forms directly.
//
// This drops the per-call markMetalUnavailableCapabilities scan (a 39+N
// element loop + ~4 string concat allocs per call when the populated-
// Detail entries got rewritten). Sharing the underlying Labels-map header
// is safe because markMetalUnavailableCapabilities only writes Status and
// Detail value fields, never touches Labels.
//
// Initialised via init() so we run after the profile package's own init
// has populated builtinAlgorithmProfilesData.
var (
	metalCapabilityFixedTail       []inference.Capability
	metalCapabilityFixedTailMarked []inference.Capability
	// metalCapabilityFixedFull / metalCapabilityFixedFullMarked are the
	// full per-call slices — head (metalModelLoadAvailable /
	// metalModelLoadUnavailable) plus the corresponding tail, pre-built
	// once at init. Consumers (go-ml / go-ai / local_tuning) treat the
	// Capabilities slice as read-only, mirroring the same convention
	// Architectures / Quantizations / CacheModes / Labels rely on. This
	// folds the per-call make([]inference.Capability, 39) (~4 KB / 1
	// alloc) into a one-time init cost. The two slices are independent
	// backings so a hypothetical-but-unsupported consumer mutation in
	// one branch cannot bleed into the other.
	metalCapabilityFixedFull       []inference.Capability
	metalCapabilityFixedFullMarked []inference.Capability
)

func init() {
	algorithmCaps := profile.AlgorithmCapabilities()
	metalCapabilityFixedTail = make([]inference.Capability, 0, len(metalCapabilityStaticTail)+len(algorithmCaps))
	metalCapabilityFixedTail = append(metalCapabilityFixedTail, metalCapabilityStaticTail...)
	metalCapabilityFixedTail = append(metalCapabilityFixedTail, algorithmCaps...)
	// Pre-mark the !loadReady variant once. We deep-copy first so the
	// loadReady path keeps its un-rewritten Status/Detail entries.
	metalCapabilityFixedTailMarked = make([]inference.Capability, len(metalCapabilityFixedTail))
	copy(metalCapabilityFixedTailMarked, metalCapabilityFixedTail)
	metalCapabilityFixedTailMarked = markMetalUnavailableCapabilities(metalCapabilityFixedTailMarked)
	// Build the head-prepended full forms once. Independent backings so
	// either branch can be exposed without aliasing the other.
	metalCapabilityFixedFull = make([]inference.Capability, 1+len(metalCapabilityFixedTail))
	metalCapabilityFixedFull[0] = metalModelLoadAvailable
	copy(metalCapabilityFixedFull[1:], metalCapabilityFixedTail)
	metalCapabilityFixedFullMarked = make([]inference.Capability, 1+len(metalCapabilityFixedTailMarked))
	metalCapabilityFixedFullMarked[0] = metalModelLoadUnavailable
	copy(metalCapabilityFixedFullMarked[1:], metalCapabilityFixedTailMarked)
}

// metalCapabilityStaticTail is the 38-entry portion of the capability
// list that does NOT vary with loadReady. metalCapabilityReportWithLoad-
// Ready prepends the per-call modelLoadCapability (entry 0 — varies
// because it switches between Supported and Unsupported based on
// loadReady) and appends the per-call algorithmCaps tail (varies in
// length); the middle is identical on every call. Pre-building once at
// package init replaces 38 SupportedCapability/Experimental/Planned
// calls + 38 boxed append args with one bulk slice copy. Keep in sync
// with metalCapabilityFixedCount (38 entries here + 1 modelLoadCapability
// at index 0 = 39).
var metalCapabilityStaticTail = []inference.Capability{
	inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityRuntimeDiscovery, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAutoTuning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelReplace, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelSlice, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityBenchmark, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityQuantization, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelMerge, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityClassify, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityTokenizer, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityLoRAInference, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityStateBundle, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityKVSnapshot, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityPromptCache, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAgentMemory, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateWake, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateSleep, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateFork, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityLoRATraining, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityDistillation, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityGRPO, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityProbeEvents, inference.CapabilityGroupProbe),
	inference.SupportedCapability(inference.CapabilityAttentionProbe, inference.CapabilityGroupProbe),
	inference.SupportedCapability(inference.CapabilityLogitProbe, inference.CapabilityGroupProbe),
	inference.ExperimentalCapability(inference.CapabilitySplitInference, inference.CapabilityGroupModel, "local dense Qwen split execution supports Metal attention/logits plus CPU FFN; remote FFN/expert execution is not wired yet"),
	inference.PlannedCapability(inference.CapabilityDifferentialLoad, inference.CapabilityGroupRuntime, "base/fine-tune differential loading belongs in go-ai/go-ml orchestration"),
	inference.PlannedCapability(inference.CapabilityVIndex, inference.CapabilityGroupProbe, "LarQL-style vindex extraction is planned for research queries"),
	inference.SupportedCapability(inference.CapabilityResponsesAPI, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAnthropicMessages, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityOllamaCompat, inference.CapabilityGroupRuntime),
}

var (
	metalCapabilityArchitectures = profile.ArchitectureIDs()
	metalCapabilityQuantizations = []string{
		"bf16",
		"fp16",
		"jang",
		"jangtq",
		"codebook",
		"vq",
		"mxtq",
		"q4_0",
		"q4_k_m",
		"q5",
		"q8_0",
		"iq",
		"mxfp4",
		"nvfp4",
	}
	metalCapabilityCacheModes = []string{
		string(memory.KVCacheModeFP16),
		string(memory.KVCacheModeQ8),
		string(memory.KVCacheModeKQ8VQ4),
		string(memory.KVCacheModePaged),
	}
	// metalCapabilityReportLabels is the shared CapabilityReport.Labels
	// payload — the value is the same constant on every call and
	// downstream consumers (go-ml / go-ai) only read this field, so the
	// single-allocation literal that used to fire per call now lives at
	// package init. Saves ~80 B + 1 alloc per metalCapabilityReport call.
	metalCapabilityReportLabels = map[string]string{"library": "go-mlx"}
)

func toInferenceProbeEvent(event metal.ProbeEvent) inference.ProbeEvent {
	// Local pointer aliases — the previous form did event.X.Y per field
	// (load .X pointer + load .Y field), which the compiler can't hoist
	// across nil checks. One pointer fetch + many field reads compiles
	// to single loads. toInferenceProbeEvent fires per probe event,
	// which under ProbeSink is emitted per token during generation.
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if token := event.Token; token != nil {
		out.Token = &inference.ProbeToken{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if logits := event.Logits; logits != nil {
		out.Logits = &inference.ProbeLogits{
			VocabularySize: logits.VocabSize,
			Min:            logits.MinLogit,
			Max:            logits.MaxLogit,
			Mean:           float32(logits.MeanLogit),
			Top:            toInferenceProbeLogits(logits.Top),
		}
	}
	if entropy := event.Entropy; entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if heads := event.SelectedHeads; heads != nil {
		out.SelectedHeads = &inference.ProbeHeadSelection{Layer: heads.Layer, Heads: core.SliceClone(heads.Heads)}
	}
	if coherence := event.LayerCoherence; coherence != nil {
		out.LayerCoherence = &inference.ProbeLayerCoherence{
			Layer:          coherence.Layer,
			KVCoupling:     coherence.KVCoupling,
			MeanCoherence:  meanNonZero(coherence.KeyCoherence, coherence.ValueCoherence, coherence.CrossAlignment),
			PhaseLock:      coherence.PhaseLock,
			SpectralStable: coherence.HeadEntropy,
		}
	}
	if router := event.RouterDecision; router != nil {
		out.RouterDecision = &inference.ProbeRouterDecision{
			Layer:       router.Layer,
			ExpertIDs:   core.SliceClone(router.ExpertIDs),
			ExpertProbs: core.SliceClone(router.Weights),
		}
	}
	if residual := event.Residual; residual != nil {
		out.Residual = &inference.ProbeResidualSummary{
			Layer: residual.Layer,
			Mean:  residual.Mean,
			RMS:   residual.RMS,
			Norm:  residual.L2Norm,
		}
	}
	if cache := event.Cache; cache != nil {
		out.Cache = &inference.ProbeCachePressure{
			PromptTokens:    cache.PromptTokens,
			GeneratedTokens: cache.GeneratedTokens,
			CachedTokens:    cache.CacheTokens,
			HitRate:         cache.Utilization,
		}
	}
	if memory := event.Memory; memory != nil {
		out.Memory = &inference.ProbeMemoryPressure{
			ActiveBytes: memory.ActiveBytes,
			PeakBytes:   memory.PeakBytes,
		}
	}
	if training := event.Training; training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        training.Epoch,
			Step:         training.Step,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
		}
	}
	return out
}

func toInferenceProbeLogits(logits []metal.ProbeLogit) []inference.ProbeLogit {
	out := make([]inference.ProbeLogit, len(logits))
	// Index iteration — same rationale as toRootProbeLogits.
	for i := range logits {
		out[i] = inference.ProbeLogit{ID: logits[i].TokenID, Value: logits[i].Logit}
	}
	return out
}

func toInferenceModelIdentity(info ModelInfo) inference.ModelIdentity {
	return inference.ModelIdentity{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}

func toInferenceAdapterIdentity(info metal.AdapterInfo) inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:       info.Path,
		Hash:       info.Hash,
		Format:     "lora",
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		TargetKeys: core.SliceClone(info.TargetKeys),
		Labels:     adapterIdentityLabels(info.Name, info.Scale),
	}
}

// adapterIdentityCommonScaleStrings caches the strconv.FormatFloat output
// for the LoRA scale values that show up most often in practice. The map
// is read-only after package init so concurrent lookups are lock-free.
// Hit rates ≈ 100% in the field — LoRA training defaults are 0.5/1.0/2.0
// (Alpha/Rank, see sft.go:433), checkpoints are tagged with the same
// constants, and adapter merges round to the nearest tenth. Each hit
// saves one ~3 B strconv heap alloc per adapterIdentityLabels call.
var adapterIdentityCommonScaleStrings = map[float32]string{
	0.125: "0.125",
	0.25:  "0.25",
	0.5:   "0.5",
	1:     "1",
	1.5:   "1.5",
	2:     "2",
	4:     "4",
	8:     "8",
}

func adapterIdentityLabels(name string, scale float32) map[string]string {
	// Cheap pre-check — return nil before allocating the map when both
	// fields are zero. adapterIdentityLabels is called per
	// toInferenceAdapterIdentity / toInferenceRootAdapterIdentity which
	// fire on every CapabilityReport / TrainSFT / BenchReport call, and
	// the zero-name + zero-scale shape is the dominant "no adapter
	// loaded" case.
	if name == "" && scale == 0 {
		return nil
	}
	// Pre-size for the two possible keys. strconv.FormatFloat with 'g'
	// matches Sprintf("%g") semantics — shortest representation that
	// round-trips — but skips the fmt format-parser + interface-boxing.
	// Bitsize 32 matches the float32 input precision.
	labels := make(map[string]string, 2)
	if name != "" {
		labels["name"] = name
	}
	if scale != 0 {
		// Hot path: cached constants for the LoRA scales we see ~100% of
		// the time. The fallback FormatFloat ('g' / -1 / 32 bitsize) only
		// fires for unusual mid-training scale values.
		if cached, ok := adapterIdentityCommonScaleStrings[scale]; ok {
			labels["scale"] = cached
		} else {
			labels["scale"] = strconv.FormatFloat(float64(scale), 'g', -1, 32)
		}
	}
	return labels
}

func toInferenceMemoryPlan(plan memory.Plan) inference.MemoryPlan {
	return inference.MemoryPlan{
		MachineClass:      string(plan.MachineClass),
		DeviceMemoryBytes: plan.DeviceMemoryBytes,
		ContextLength:     plan.ContextLength,
		BatchSize:         plan.BatchSize,
		CacheMode:         string(plan.CacheMode),
		// Plain strconv + concat — skip the fmt format-parser path that
		// boxes the int + walks the format string for one int and one
		// literal suffix. strconv.Itoa hits the digit-emit loop direct.
		Quantization:     strconv.Itoa(plan.PreferredQuantization) + "-bit",
		KVCacheBytes:     plan.EstimatedKVCacheModeBytes,
		TrainingFeasible: plan.MachineClass != memory.ClassApple16GB,
		Notes:            core.SliceClone(plan.Notes),
	}
}

func toFastEvalConfig(cfg inference.BenchConfig) bench.Config {
	out := bench.DefaultConfig()
	if len(cfg.Prompts) > 0 {
		out.Prompt = cfg.Prompts[0]
	}
	if cfg.MaxTokens > 0 {
		out.MaxTokens = cfg.MaxTokens
	}
	if cfg.MeasuredRuns > 0 {
		out.Runs = cfg.MeasuredRuns
	}
	return out
}

func toInferenceBenchReport(report *bench.Report) *inference.BenchReport {
	if report == nil {
		return nil
	}
	return &inference.BenchReport{
		Model:                 toInferenceModelIdentity(benchInfoToModel(report.ModelInfo)),
		Adapter:               toInferenceRootAdapterIdentity(benchAdapterToLora(report.ModelInfo.Adapter)),
		PromptTokens:          report.Generation.PromptTokens,
		GeneratedTokens:       report.Generation.GeneratedTokens,
		PrefillTokensPerSec:   report.Generation.PrefillTokensPerSec,
		DecodeTokensPerSec:    report.Generation.DecodeTokensPerSec,
		PeakMemoryBytes:       report.Generation.PeakMemoryBytes,
		PromptCacheHitRate:    report.PromptCache.HitRate,
		KVRestoreMilliseconds: float64(report.KVRestore.Duration.Milliseconds()),
	}
}

func toEvalConfig(cfg inference.EvalConfig) eval.Config {
	return eval.Config{
		MaxSamples: cfg.MaxSamples,
		Batch: dataset.BatchConfig{
			BatchSize: cfg.BatchSize,
			MaxSeqLen: cfg.MaxSeqLen,
		},
	}
}

func toInferenceEvalReport(report *eval.Report) *inference.EvalReport {
	if report == nil {
		return nil
	}
	return &inference.EvalReport{
		Model:   toInferenceModelIdentity(evalInfoToModel(report.ModelInfo)),
		Adapter: toInferenceRootAdapterIdentity(evalAdapterToLora(report.Adapter)),
		Metrics: inference.EvalMetrics{
			Samples:    report.Metrics.Samples,
			Tokens:     report.Metrics.Tokens,
			Loss:       report.Metrics.Loss,
			Perplexity: report.Metrics.Perplexity,
		},
		Probes: toInferenceQualityResults(report.Quality.Checks),
	}
}

func toInferenceQualityResults(checks []eval.QualityCheck) []inference.QualityProbeResult {
	out := make([]inference.QualityProbeResult, len(checks))
	// Index iteration — eval.QualityCheck carries Name + Detail (string
	// headers) + Pass + Score, ~48 B total. Skip the per-iter copy.
	for i := range checks {
		out[i] = inference.QualityProbeResult{Name: checks[i].Name, Passed: checks[i].Pass, Score: checks[i].Score, Text: checks[i].Detail}
	}
	return out
}

func toSFTConfig(cfg inference.TrainingConfig, sink inference.ProbeSink) SFTConfig {
	return SFTConfig{
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulation,
		Epochs:                    cfg.Epochs,
		LearningRate:              cfg.LearningRate,
		LoRA: LoRAConfig{
			Rank:       cfg.LoRA.Rank,
			Alpha:      cfg.LoRA.Alpha,
			TargetKeys: core.SliceClone(cfg.LoRA.TargetKeys),
			DType:      sftDType(cfg.LoRA.BFloat16),
			ProbeSink:  inferenceProbeSink{sink: sink},
		},
		ProbeSink: inferenceProbeSink{sink: sink},
	}
}

type inferenceProbeSink struct {
	sink inference.ProbeSink
}

func (sink inferenceProbeSink) EmitProbe(event probe.Event) {
	if sink.sink == nil {
		return
	}
	sink.sink.EmitProbe(toInferenceRootProbeEvent(event))
}

func toInferenceRootProbeEvent(event probe.Event) inference.ProbeEvent {
	// Local pointer aliases — see toInferenceProbeEvent for rationale.
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if token := event.Token; token != nil {
		out.Token = &inference.ProbeToken{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if entropy := event.Entropy; entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if training := event.Training; training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        training.Epoch,
			Step:         training.Step,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
		}
	}
	return out
}

func sftDType(bfloat16 bool) DType {
	if bfloat16 {
		return DTypeBFloat16
	}
	return 0
}

func toInferenceTrainingResult(info ModelInfo, result *SFTResult, cfg inference.TrainingConfig) *inference.TrainingResult {
	out := &inference.TrainingResult{
		Model:  toInferenceModelIdentity(info),
		Labels: cloneInferenceLabels(cfg.Labels),
	}
	if result == nil {
		return out
	}
	out.Adapter = toInferenceRootAdapterIdentity(info.Adapter)
	if result.AdapterPath != "" {
		out.Adapter.Path = result.AdapterPath
	}
	out.Metrics = inference.TrainingMetrics{
		Epoch:        result.Epochs,
		Step:         result.Steps,
		Samples:      result.Samples,
		Loss:         result.LastLoss,
		LearningRate: cfg.LearningRate,
	}
	out.Checkpoints = stateRefsFromPaths("sft_checkpoint", result.Checkpoints)
	return out
}

func toInferenceRootAdapterIdentity(info lora.AdapterInfo) inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:       info.Path,
		Hash:       info.Hash,
		Format:     "lora",
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		TargetKeys: core.SliceClone(info.TargetKeys),
		Labels:     adapterIdentityLabels(info.Name, info.Scale),
	}
}

// stateRefsURIScheme is the URI scheme prefix for file-backed StateRefs.
// Hoisted to package init so the literal isn't re-interned per call —
// also serves as the documented prefix for the single-buffer URI build
// path in stateRefsFromPaths.
const stateRefsURIScheme = "file://"

func stateRefsFromPaths(kind string, paths []string) []inference.StateRef {
	// Two-pass: count non-empty paths + total URI byte length so we can
	// pre-size the output slice exactly AND allocate one shared backing
	// buffer for every "file://"+path string. Each StateRef.URI is a
	// substring of that single allocation — drops N per-call concat
	// allocs (one per non-empty path) down to ONE allocation regardless
	// of path count.
	nonEmpty := 0
	totalBytes := 0
	for _, path := range paths {
		if path == "" {
			continue
		}
		nonEmpty++
		totalBytes += len(stateRefsURIScheme) + len(path)
	}
	if nonEmpty == 0 {
		return []inference.StateRef{}
	}
	buf := make([]byte, 0, totalBytes)
	out := make([]inference.StateRef, 0, nonEmpty)
	for _, path := range paths {
		if path == "" {
			continue
		}
		start := len(buf)
		buf = append(buf, stateRefsURIScheme...)
		buf = append(buf, path...)
		// Use [start:end] not [start:] so the substring length is captured
		// at write time. buf was pre-sized to totalBytes so append never
		// grows the backing array, which keeps prior substring pointers
		// valid through the rest of the loop. core.AsString is zero-copy
		// + buf is fresh-built and never re-handed-out, so the safety
		// contract holds.
		out = append(out, inference.StateRef{
			Kind: kind,
			URI:  core.AsString(buf[start:len(buf)]),
		})
	}
	return out
}

func cloneInferenceLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	// core.MapClone → maps.Clone uses runtime.mapclone for bulk-bucket
	// hash-table copy rather than the user-space range+assign loop.
	// Same alloc shape (2 allocs / 336 bytes for a 4-entry string map),
	// iteration moves into compiled runtime code. Matches the helpers.go
	// cloneStringMap adoption (6dd0c53).
	return core.MapClone(labels)
}

func cloneInferenceSplitEndpoints(endpoints []inference.SplitEndpoint) []inference.SplitEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	out := make([]inference.SplitEndpoint, len(endpoints))
	// Index iteration — the range-and-copy form copied each endpoint
	// twice (once into the loop-var, once into the output) on every
	// step. SplitEndpoint carries Address/Role/Format strings plus
	// the Labels map header, so the copy is non-trivial. Index assigns
	// straight from source to destination.
	for i := range endpoints {
		out[i] = endpoints[i]
		out[i].Labels = cloneInferenceLabels(endpoints[i].Labels)
	}
	return out
}

func meanNonZero(values ...float64) float64 {
	var total float64
	var count int
	for _, value := range values {
		if value == 0 {
			continue
		}
		total += value
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}
