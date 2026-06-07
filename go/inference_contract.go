// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/pkg/metal"
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
	// Quantisation never gates fit: a model's precision is descriptive, not a
	// ceiling. Whether a model fits is a bytes question (weights + KV vs the
	// memory budget), assessed below — not a bits comparison against a
	// machine-class preference.
	quantizationOK := true
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
	return chat.Format(messages, metalAdapterChatConfig(adapter.model.Info(), adapter.model.ModelType())), nil
}

func metalAdapterChatConfig(info metal.ModelInfo, modelType string) chat.Config {
	architecture := info.Architecture
	if architecture == "" {
		architecture = modelType
	}
	return modelChatConfigForArchitecture(architecture, info.NumHeads)
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

func (adapter *metaladapter) evalRunner() eval.Runner {
	return NewModelEvalRunner(adapter.rootModel())
}

func (adapter *metaladapter) ApplyLoRA(config inference.LoRAConfig) inference.Adapter {
	return adapter.model.ApplyLoRA(toMetalInferenceLoRAConfig(config))
}

func toMetalInferenceLoRAConfig(config inference.LoRAConfig) metal.LoRAConfig {
	mcfg := metal.LoRAConfig{
		Rank:  config.Rank,
		Alpha: config.Alpha,
	}
	if len(config.TargetKeys) > 0 {
		mcfg.TargetKeys = core.SliceClone(config.TargetKeys)
	}
	if config.BFloat16 {
		mcfg.DType = metal.DTypeBFloat16
	}
	return mcfg
}

func (adapter *metaladapter) Encode(text string) []int32 {
	return adapter.model.Encode(text)
}

func (adapter *metaladapter) Decode(tokenIDs []int32) string {
	return adapter.model.Decode(tokenIDs)
}

func (adapter *metaladapter) NumLayers() int {
	return adapter.model.NumLayers()
}

func (adapter *metaladapter) InternalModel() metal.InternalModel {
	return adapter.model.Internal()
}

type inferenceDataset struct {
	stream inference.DatasetStream
}

// Per-sample / per-reset sentinels — inferenceDataset.Next fires for
// every row in Evaluate/TrainSFT and was paying a per-call core.NewError
// alloc on the nil-stream guard.
var (
	errMLXInferenceDatasetNil         = core.NewError("mlx: inference dataset stream is nil")
	errMLXInferenceDatasetNotResetter = core.NewError("mlx: inference dataset stream is not resettable")
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
