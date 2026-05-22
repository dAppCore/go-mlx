// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/memory"
	"strconv"

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

func metalCapabilityReport(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool) inference.CapabilityReport {
	return metalCapabilityReportWithLoadReady(model, adapter, available, available)
}

func metalCapabilityReportWithLoadReady(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool, loadReady bool) inference.CapabilityReport {
	device := metalCapabilityDeviceInfo(available)
	// Pre-size for the three possible runtime labels (memory, working
	// set, load_available). Drop the fmt-format-parser path in favour
	// of strconv.FormatUint — same value, no interface-boxing of the
	// uint64 arg + no fmt format-machinery overhead.
	//
	// The original len()==0 guard that nil'd the map was dead code —
	// load_available is always set, so len ≥ 1 every call.
	runtimeLabels := make(map[string]string, 3)
	if device.MemorySize > 0 {
		runtimeLabels["memory_bytes"] = strconv.FormatUint(device.MemorySize, 10)
	}
	if device.MaxRecommendedWorkingSetSize > 0 {
		runtimeLabels["working_set_bytes"] = strconv.FormatUint(device.MaxRecommendedWorkingSetSize, 10)
	}
	runtimeLabels["load_available"] = boolLabel(loadReady)
	// Pre-built static tails — see metalCapabilityFixedTail (loadReady=true)
	// and metalCapabilityFixedTailMarked (loadReady=false, already passed
	// through markMetalUnavailableCapabilities once at package init). The
	// 38 static entries plus the (deterministic over a fixed model
	// architecture) AlgorithmCapabilities slice are merged once at package
	// init; the markMetalUnavailable pass is also done once for the
	// !loadReady form. Per call we issue ONE make() at the final size and
	// ONE copy() instead of three successive appends + the per-call
	// AlgorithmCapabilities() + the per-call markMetalUnavailableCapabilities
	// scan (which itself allocated 4 strings per call from the populated-
	// Detail concat path).
	source := metalCapabilityFixedTail
	head := metalModelLoadAvailable
	if !loadReady {
		source = metalCapabilityFixedTailMarked
		head = metalModelLoadUnavailable
	}
	capabilities := make([]inference.Capability, 1+len(source))
	capabilities[0] = head
	copy(capabilities[1:], source)
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "metal",
			Device:        device.Architecture,
			NativeRuntime: true,
			Labels:        runtimeLabels,
		},
		Model:         model,
		Adapter:       adapter,
		Available:     available,
		Architectures: core.SliceClone(metalCapabilityArchitectures),
		Quantizations: core.SliceClone(metalCapabilityQuantizations),
		CacheModes:    core.SliceClone(metalCapabilityCacheModes),
		Capabilities:  capabilities,
		Labels:        map[string]string{"library": "go-mlx"},
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
// package init. Per call we just pick the right one and copy.
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
		labels["scale"] = strconv.FormatFloat(float64(scale), 'g', -1, 32)
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

func stateRefsFromPaths(kind string, paths []string) []inference.StateRef {
	out := make([]inference.StateRef, 0, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		out = append(out, inference.StateRef{Kind: kind, URI: "file://" + path})
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
