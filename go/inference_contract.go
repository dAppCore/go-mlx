// SPDX-Licence-Identifier: EUPL-1.2


package mlx

import (
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/memory"
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/probe"
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
		Notes:          append([]string(nil), plan.Notes...),
	}, nil
}

func (adapter *metaladapter) Capabilities() inference.CapabilityReport {
	if adapter == nil || adapter.model == nil {
		return metalCapabilityReport(inference.ModelIdentity{}, inference.AdapterIdentity{}, false)
	}
	return metalCapabilityReport(toInferenceModelIdentity(adapter.rootModel().Info()), adapter.ActiveAdapter(), true)
}

func (adapter *metaladapter) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if adapter == nil || adapter.model == nil {
		return "", core.NewError("mlx: model is nil")
	}
	return chat.Format(messages, chat.Config{Architecture: adapter.model.ModelType()}), nil
}

func (adapter *metaladapter) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if adapter == nil || adapter.model == nil {
		return inference.AdapterIdentity{}, core.NewError("mlx: model is nil")
	}
	if _, err := adapter.model.LoadLoRA(path); err != nil {
		return inference.AdapterIdentity{}, err
	}
	return toInferenceAdapterIdentity(adapter.model.Adapter()), nil
}

func (adapter *metaladapter) UnloadAdapter() error {
	if adapter == nil || adapter.model == nil {
		return core.NewError("mlx: model is nil")
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
		return nil, core.NewError("mlx: model is nil")
	}
	report, err := RunFastEval(ctx, adapter.fastEvalRunner(), toFastEvalConfig(cfg))
	if err != nil {
		return nil, err
	}
	return toInferenceBenchReport(report), nil
}

func (adapter *metaladapter) Evaluate(ctx context.Context, dataset inference.DatasetStream, cfg inference.EvalConfig) (*inference.EvalReport, error) {
	if adapter == nil || adapter.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	report, err := eval.RunDataset(ctx, adapter.evalRunner(), wrapSFTDataset(inferenceDataset{stream: dataset}), toEvalConfig(cfg))
	if err != nil {
		return nil, err
	}
	return toInferenceEvalReport(report), nil
}

func (adapter *metaladapter) TrainSFT(ctx context.Context, dataset inference.DatasetStream, cfg inference.TrainingConfig) (*inference.TrainingResult, error) {
	if adapter == nil || adapter.model == nil {
		return nil, core.NewError("mlx: model is nil")
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

func (d inferenceDataset) Next() (dataset.Sample, bool, error) {
	if d.stream == nil {
		return dataset.Sample{}, false, core.NewError("mlx: inference dataset stream is nil")
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
		return core.NewError("mlx: inference dataset stream is nil")
	}
	resetter, ok := d.stream.(inference.DatasetResetter)
	if !ok {
		return core.NewError("mlx: inference dataset stream is not resettable")
	}
	return resetter.Reset()
}

func toMetalInferenceProbeSink(sink inference.ProbeSink) metal.ProbeSink {
	if sink == nil {
		return nil
	}
	return metal.ProbeSinkFunc(func(event metal.ProbeEvent) {
		sink.EmitProbe(toInferenceProbeEvent(event))
	})
}

var metalCapabilityDeviceInfo = func(available bool) DeviceInfo {
	if !available {
		return DeviceInfo{}
	}
	return safeRuntimeDeviceInfo()
}

func metalCapabilityReport(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool) inference.CapabilityReport {
	device := metalCapabilityDeviceInfo(available)
	runtimeLabels := map[string]string{}
	if device.MemorySize > 0 {
		runtimeLabels["memory_bytes"] = core.Sprintf("%d", device.MemorySize)
	}
	if device.MaxRecommendedWorkingSetSize > 0 {
		runtimeLabels["working_set_bytes"] = core.Sprintf("%d", device.MaxRecommendedWorkingSetSize)
	}
	if len(runtimeLabels) == 0 {
		runtimeLabels = nil
	}
	capabilities := []inference.Capability{
		inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime),
		inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
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
		inference.SupportedCapability(inference.CapabilityResponsesAPI, inference.CapabilityGroupRuntime),
		inference.SupportedCapability(inference.CapabilityAnthropicMessages, inference.CapabilityGroupRuntime),
		inference.SupportedCapability(inference.CapabilityOllamaCompat, inference.CapabilityGroupRuntime),
	}
	capabilities = append(capabilities, profile.AlgorithmCapabilities()...)
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
		Architectures: append([]string(nil), metalCapabilityArchitectures...),
		Quantizations: append([]string(nil), metalCapabilityQuantizations...),
		CacheModes:    append([]string(nil), metalCapabilityCacheModes...),
		Capabilities:  capabilities,
		Labels:        map[string]string{"library": "go-mlx"},
	}
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
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if event.Token != nil {
		out.Token = &inference.ProbeToken{
			ID:              event.Token.ID,
			Text:            event.Token.Text,
			PromptTokens:    event.Token.PromptTokens,
			GeneratedTokens: event.Token.GeneratedTokens,
		}
	}
	if event.Logits != nil {
		out.Logits = &inference.ProbeLogits{
			VocabularySize: event.Logits.VocabSize,
			Min:            event.Logits.MinLogit,
			Max:            event.Logits.MaxLogit,
			Mean:           float32(event.Logits.MeanLogit),
			Top:            toInferenceProbeLogits(event.Logits.Top),
		}
	}
	if event.Entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: event.Entropy.Value, Unit: event.Entropy.Unit}
	}
	if event.SelectedHeads != nil {
		out.SelectedHeads = &inference.ProbeHeadSelection{Layer: event.SelectedHeads.Layer, Heads: append([]int(nil), event.SelectedHeads.Heads...)}
	}
	if event.LayerCoherence != nil {
		out.LayerCoherence = &inference.ProbeLayerCoherence{
			Layer:          event.LayerCoherence.Layer,
			KVCoupling:     event.LayerCoherence.KVCoupling,
			MeanCoherence:  meanNonZero(event.LayerCoherence.KeyCoherence, event.LayerCoherence.ValueCoherence, event.LayerCoherence.CrossAlignment),
			PhaseLock:      event.LayerCoherence.PhaseLock,
			SpectralStable: event.LayerCoherence.HeadEntropy,
		}
	}
	if event.RouterDecision != nil {
		out.RouterDecision = &inference.ProbeRouterDecision{
			Layer:       event.RouterDecision.Layer,
			ExpertIDs:   append([]int(nil), event.RouterDecision.ExpertIDs...),
			ExpertProbs: append([]float32(nil), event.RouterDecision.Weights...),
		}
	}
	if event.Residual != nil {
		out.Residual = &inference.ProbeResidualSummary{
			Layer: event.Residual.Layer,
			Mean:  event.Residual.Mean,
			RMS:   event.Residual.RMS,
			Norm:  event.Residual.L2Norm,
		}
	}
	if event.Cache != nil {
		out.Cache = &inference.ProbeCachePressure{
			PromptTokens:    event.Cache.PromptTokens,
			GeneratedTokens: event.Cache.GeneratedTokens,
			CachedTokens:    event.Cache.CacheTokens,
			HitRate:         event.Cache.Utilization,
		}
	}
	if event.Memory != nil {
		out.Memory = &inference.ProbeMemoryPressure{
			ActiveBytes: event.Memory.ActiveBytes,
			PeakBytes:   event.Memory.PeakBytes,
		}
	}
	if event.Training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        event.Training.Epoch,
			Step:         event.Training.Step,
			Loss:         event.Training.Loss,
			LearningRate: event.Training.LearningRate,
		}
	}
	return out
}

func toInferenceProbeLogits(logits []metal.ProbeLogit) []inference.ProbeLogit {
	out := make([]inference.ProbeLogit, len(logits))
	for i, logit := range logits {
		out[i] = inference.ProbeLogit{ID: logit.TokenID, Value: logit.Logit}
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
		TargetKeys: append([]string(nil), info.TargetKeys...),
		Labels:     adapterIdentityLabels(info.Name, info.Scale),
	}
}

func adapterIdentityLabels(name string, scale float32) map[string]string {
	labels := map[string]string{}
	if name != "" {
		labels["name"] = name
	}
	if scale != 0 {
		labels["scale"] = core.Sprintf("%g", scale)
	}
	if len(labels) == 0 {
		return nil
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
		Quantization:      core.Sprintf("%d-bit", plan.PreferredQuantization),
		KVCacheBytes:      plan.EstimatedKVCacheModeBytes,
		TrainingFeasible:  plan.MachineClass != memory.ClassApple16GB,
		Notes:             append([]string(nil), plan.Notes...),
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
	for i, check := range checks {
		out[i] = inference.QualityProbeResult{Name: check.Name, Passed: check.Pass, Score: check.Score, Text: check.Detail}
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
			TargetKeys: append([]string(nil), cfg.LoRA.TargetKeys...),
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
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if event.Token != nil {
		out.Token = &inference.ProbeToken{
			ID:              event.Token.ID,
			Text:            event.Token.Text,
			PromptTokens:    event.Token.PromptTokens,
			GeneratedTokens: event.Token.GeneratedTokens,
		}
	}
	if event.Entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: event.Entropy.Value, Unit: event.Entropy.Unit}
	}
	if event.Training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        event.Training.Epoch,
			Step:         event.Training.Step,
			Loss:         event.Training.Loss,
			LearningRate: event.Training.LearningRate,
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
		TargetKeys: append([]string(nil), info.TargetKeys...),
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
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
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
