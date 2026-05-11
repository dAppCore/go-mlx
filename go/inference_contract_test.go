// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/profile"
)

func TestInferenceContract_MetalAdapterImplementsSharedInterfaces_Good(t *testing.T) {
	target := "metaladapter TokenizerModel AdapterModel ProbeableModel BenchableModel Evaluator SFTTrainer CapabilityReporter SchedulerModel CacheService"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.TokenizerModel = (*metaladapter)(nil)
	var _ inference.AdapterModel = (*metaladapter)(nil)
	var _ inference.ProbeableModel = (*metaladapter)(nil)
	var _ inference.BenchableModel = (*metaladapter)(nil)
	var _ inference.Evaluator = (*metaladapter)(nil)
	var _ inference.SFTTrainer = (*metaladapter)(nil)
	var _ inference.CapabilityReporter = (*metaladapter)(nil)
	var _ inference.ReasoningParser = (*metaladapter)(nil)
	var _ inference.ToolParser = (*metaladapter)(nil)
	var _ inference.SchedulerModel = (*metaladapter)(nil)
	var _ inference.CancellableModel = (*metaladapter)(nil)
	var _ inference.CacheService = (*metaladapter)(nil)
	var _ inference.AgentMemorySession = (*ModelSession)(nil)
	var _ inference.AgentMemoryForker = (*Model)(nil)
}

func TestInferenceContract_MetalBackendImplementsFitPlanner_Good(t *testing.T) {
	target := "metalbackend ModelFitPlanner CapabilityReporter"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.ModelFitPlanner = (*metalbackend)(nil)
	var _ inference.CapabilityReporter = (*metalbackend)(nil)
	var _ inference.RuntimeMemoryLimiter = (*metalbackend)(nil)
}

func TestInferenceContract_MetalBackendRuntimeMemoryLimits_UglyZero(t *testing.T) {
	got := (&metalbackend{}).SetRuntimeMemoryLimits(inference.RuntimeMemoryLimits{})

	if got != (inference.RuntimeMemoryLimits{}) {
		t.Fatalf("SetRuntimeMemoryLimits zero = %+v, want zero response", got)
	}
}

func TestInferenceContract_MetalBackendCapabilities_Good(t *testing.T) {
	report := (&metalbackend{}).Capabilities()

	if report.Runtime.Backend != "metal" || !report.Runtime.NativeRuntime {
		t.Fatalf("runtime = %+v, want native metal", report.Runtime)
	}
	if !report.Supports(inference.CapabilityModelLoad) || !report.Supports(inference.CapabilityMemoryPlanning) {
		t.Fatalf("capabilities = %+v, want load and memory planning", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityLoRATraining) || !report.Supports(inference.CapabilityGRPO) {
		t.Fatalf("capabilities = %+v, want training features", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityProbeEvents) || !report.Supports(inference.CapabilityAttentionProbe) {
		t.Fatalf("capabilities = %+v, want probe features", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityReasoningParse) || !report.Supports(inference.CapabilityToolParse) || !report.Supports(inference.CapabilityJANGTQ) {
		t.Fatalf("capabilities = %+v, want reasoning/tool/JANGTQ groundwork", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityScheduler) || !report.Supports(inference.CapabilityRequestCancel) {
		t.Fatalf("capabilities = %+v, want scheduler/request cancel support", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityCacheBlocks) || !report.Supports(inference.CapabilityCacheWarm) {
		t.Fatalf("capabilities = %+v, want block cache support", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityAgentMemory) || !report.Supports(inference.CapabilityStateWake) || !report.Supports(inference.CapabilityStateSleep) || !report.Supports(inference.CapabilityStateFork) {
		t.Fatalf("capabilities = %+v, want agent memory wake/sleep/fork support", report.CapabilityIDs())
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityResponsesAPI,
		inference.CapabilityAnthropicMessages,
		inference.CapabilityOllamaCompat,
	} {
		capability, ok := report.Capability(id)
		if !ok || capability.Status != inference.CapabilityStatusSupported {
			t.Fatalf("capability %q = %+v ok=%v, want supported wire compatibility", id, capability, ok)
		}
	}
	if report.Supports(inference.CapabilityCacheDisk) {
		t.Fatalf("capabilities = %+v, disk cache should be planned, not supported", report.CapabilityIDs())
	}
	if len(report.Architectures) == 0 || len(report.Quantizations) == 0 || len(report.CacheModes) == 0 {
		t.Fatalf("report = %+v, want architecture/quant/cache metadata", report)
	}
	for _, architecture := range []string{"minimax_m2", "mistral", "mixtral", "phi", "deepseek", "gpt_oss", "bert"} {
		if !stringSliceContains(report.Architectures, architecture) {
			t.Fatalf("architectures = %v, want metadata-only target %q", report.Architectures, architecture)
		}
	}
	for _, quantization := range []string{"jang", "jangtq", "mxtq"} {
		if !stringSliceContains(report.Quantizations, quantization) {
			t.Fatalf("quantizations = %v, want %q", report.Quantizations, quantization)
		}
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
	} {
		capability, ok := report.Capability(id)
		if !ok {
			t.Fatalf("capability %q missing from report", id)
		}
		if capability.Labels["runtime_status"] == "" {
			t.Fatalf("capability %q labels = %+v, want runtime_status", id, capability.Labels)
		}
	}
	if cap, _ := report.Capability(inference.CapabilityMoERouting); cap.Labels["runtime_status"] != string(profile.AlgorithmRuntimeMetadataOnly) {
		t.Fatalf("moe routing capability = %+v, want metadata-only runtime status", cap)
	}
	if cap, _ := report.Capability(inference.CapabilitySpeculativeDecode); cap.Labels["runtime_status"] != string(profile.AlgorithmRuntimeExperimental) {
		t.Fatalf("speculative capability = %+v, want experimental runtime status", cap)
	}
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestInferenceContract_MetalBackendCapabilities_Good_UsesSafeDeviceInfoHook(t *testing.T) {
	previous := metalCapabilityDeviceInfo
	called := false
	metalCapabilityDeviceInfo = func(available bool) DeviceInfo {
		called = true
		return DeviceInfo{Architecture: "test-metal", MemorySize: 16 * MemoryGiB}
	}
	t.Cleanup(func() { metalCapabilityDeviceInfo = previous })

	report := (&metalbackend{}).Capabilities()

	if !called {
		t.Fatal("metalCapabilityDeviceInfo was not called")
	}
	if report.Runtime.Device != "test-metal" {
		t.Fatalf("device = %q, want test-metal", report.Runtime.Device)
	}
	if report.Runtime.Labels["memory_bytes"] == "" {
		t.Fatalf("labels = %+v, want memory_bytes", report.Runtime.Labels)
	}
}

func TestInferenceContract_MetalAdapterCapabilities_UglyNilModel(t *testing.T) {
	report := (&metaladapter{}).Capabilities()

	if report.Available {
		t.Fatalf("Available = true, want false for nil loaded model")
	}
	if !report.Supports(inference.CapabilityGenerate) || !report.Supports(inference.CapabilityLoRAInference) {
		t.Fatalf("capabilities = %+v, want model feature surface even before load", report.CapabilityIDs())
	}
	if report.Adapter.Path != "" {
		t.Fatalf("adapter = %+v, want empty adapter identity", report.Adapter)
	}
}

func TestInferenceContract_MetalAdapterNilGuards_Bad(t *testing.T) {
	var adapter *metaladapter
	if _, err := adapter.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("expected nil model chat template error")
	}
	if _, err := adapter.LoadAdapter("adapter"); err == nil {
		t.Fatal("expected nil model load adapter error")
	}
	if err := adapter.UnloadAdapter(); err == nil {
		t.Fatal("expected nil model unload adapter error")
	}
	if active := adapter.ActiveAdapter(); active.Path != "" || active.Hash != "" {
		t.Fatalf("ActiveAdapter(nil) = %+v, want zero identity", active)
	}
	if _, err := adapter.Benchmark(context.Background(), inference.BenchConfig{}); err == nil {
		t.Fatal("expected nil model benchmark error")
	}
	if _, err := adapter.Evaluate(context.Background(), nil, inference.EvalConfig{}); err == nil {
		t.Fatal("expected nil model eval error")
	}
	if _, err := adapter.TrainSFT(context.Background(), nil, inference.TrainingConfig{}); err == nil {
		t.Fatal("expected nil model SFT error")
	}
	cfg := adapter.generateConfig(inference.WithMaxTokens(7), inference.WithTemperature(0.5))
	if cfg.MaxTokens != 7 || cfg.Temperature != 0.5 {
		t.Fatalf("generateConfig(nil) = %+v, want forwarded options", cfg)
	}
	if root := adapter.rootModel(); root == nil || root.model != nil {
		t.Fatalf("rootModel(nil) = %+v, want empty root model", root)
	}
	if runner := adapter.fastEvalRunner(); runner.Generate == nil {
		t.Fatalf("fastEvalRunner(nil) = %+v, want runner wrappers", runner)
	}
	if runner := adapter.evalRunner(); runner.EvaluateBatch == nil {
		t.Fatalf("evalRunner(nil) = %+v, want eval wrappers", runner)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Good(t *testing.T) {
	report, err := (&metalbackend{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "qwen3",
		QuantBits:     4,
		ContextLength: 32768,
		NumLayers:     28,
		HiddenSize:    2048,
	}, 16*MemoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.ArchitectureOK || !report.QuantizationOK {
		t.Fatalf("PlanModelFit report = %+v, want supported qwen3/q4", report)
	}
	if report.MemoryPlan.ContextLength == 0 || report.MemoryPlan.CacheMode == "" {
		t.Fatalf("MemoryPlan = %+v, want context/cache recommendation", report.MemoryPlan)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Bad(t *testing.T) {
	report, err := (&metalbackend{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture: "unknown-transformer",
		QuantBits:    16,
	}, 8*MemoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || report.ArchitectureOK || report.QuantizationOK {
		t.Fatalf("PlanModelFit report = %+v, want unsupported architecture and quantization", report)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	report, err := (&metalbackend{}).PlanModelFit(ctx, inference.ModelIdentity{Architecture: "qwen3"}, 0)

	if err == nil {
		t.Fatalf("PlanModelFit cancelled error = nil, report=%+v", report)
	}
}

func TestInferenceContract_MetalAdapterSetProbeSink_Good(t *testing.T) {
	adapter := &metaladapter{}
	var got inference.ProbeEvent
	adapter.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		got = event
	}))

	toMetalInferenceProbeSink(adapter.probeSink).EmitProbe(metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Token: &metal.ProbeToken{ID: 7, Text: "ok", PromptTokens: 3, GeneratedTokens: 1},
	})

	if got.Kind != inference.ProbeEventToken || got.Token == nil || got.Token.Text != "ok" {
		t.Fatalf("probe event = %+v, want token event", got)
	}
}

func TestInferenceContract_ToInferenceProbeEvent_Ugly(t *testing.T) {
	got := toInferenceProbeEvent(metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Logits: &metal.ProbeLogits{
			VocabSize: 11,
			MinLogit:  -1.5,
			MaxLogit:  2.5,
			MeanLogit: 0.25,
			Top:       []metal.ProbeLogit{{TokenID: 4, Logit: 2.5}},
		},
	})

	if got.Logits == nil || got.Logits.VocabularySize != 11 || got.Logits.Top[0].ID != 4 {
		t.Fatalf("logits event = %+v, want compact logits", got)
	}
}

func TestInferenceContract_DatasetAdapterAndConversionHelpers_Good(t *testing.T) {
	stream := &inferenceContractDatasetStream{
		samples: []inference.DatasetSample{{
			Prompt:   "p",
			Response: "r",
			Text:     "t",
			Labels:   map[string]string{"source": "unit"},
		}},
	}
	dataset := inferenceDataset{stream: stream}
	sample, ok, err := dataset.Next()
	if err != nil || !ok {
		t.Fatalf("Next() = %+v/%v/%v, want one sample", sample, ok, err)
	}
	if sample.Prompt != "p" || sample.Meta["source"] != "unit" {
		t.Fatalf("sample = %+v, want mapped prompt/meta", sample)
	}
	sample.Meta["source"] = "changed"
	if stream.samples[0].Labels["source"] != "unit" {
		t.Fatalf("dataset adapter leaked labels mutation: %+v", stream.samples[0].Labels)
	}
	if err := dataset.Reset(); err != nil || stream.resetCalls != 1 {
		t.Fatalf("Reset() = %v calls=%d, want one reset", err, stream.resetCalls)
	}
	if _, _, err := (inferenceDataset{}).Next(); err == nil {
		t.Fatal("Next(nil stream) error = nil")
	}
	if err := (inferenceDataset{}).Reset(); err == nil {
		t.Fatal("Reset(nil stream) error = nil")
	}
	if err := (inferenceDataset{stream: inferenceContractOneShotStream{}}).Reset(); err == nil {
		t.Fatal("Reset(non-resettable stream) error = nil")
	}

	model := toInferenceModelIdentity(ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     10,
		NumLayers:     2,
		HiddenSize:    8,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 128,
	})
	if model.Architecture != "qwen3" || model.QuantBits != 4 || model.ContextLength != 128 {
		t.Fatalf("model identity = %+v", model)
	}
	adapter := toInferenceAdapterIdentity(metal.AdapterInfo{
		Name: "demo", Path: "/tmp/a", Hash: "abc", Rank: 8, Alpha: 16, Scale: 0.5, TargetKeys: []string{"q_proj"},
	})
	if adapter.Format != "lora" || adapter.Labels["name"] != "demo" || adapter.Labels["scale"] != "0.5" {
		t.Fatalf("adapter identity = %+v", adapter)
	}
	if labels := adapterIdentityLabels("", 0); labels != nil {
		t.Fatalf("empty adapter labels = %+v, want nil", labels)
	}

	fastCfg := toFastEvalConfig(inference.BenchConfig{Prompts: []string{"bench"}, MaxTokens: 9, MeasuredRuns: 3})
	if fastCfg.Prompt != "bench" || fastCfg.MaxTokens != 9 || fastCfg.Runs != 3 {
		t.Fatalf("fast eval config = %+v", fastCfg)
	}
	bench := toInferenceBenchReport(&FastEvalReport{
		ModelInfo: ModelInfo{Architecture: "qwen3", Adapter: LoRAAdapterInfo{Name: "root"}},
		Generation: FastEvalGenerationSummary{
			PromptTokens:        4,
			GeneratedTokens:     5,
			PrefillTokensPerSec: 10,
			DecodeTokensPerSec:  20,
			PeakMemoryBytes:     30,
		},
		PromptCache: FastEvalPromptCacheReport{HitRate: 0.25},
		KVRestore:   FastEvalLatencyReport{Duration: 12 * time.Millisecond},
	})
	if bench == nil || bench.Model.Architecture != "qwen3" || bench.KVRestoreMilliseconds != 12 {
		t.Fatalf("bench report = %+v", bench)
	}
	if toInferenceBenchReport(nil) != nil {
		t.Fatal("toInferenceBenchReport(nil) != nil")
	}

	evalCfg := toEvalConfig(inference.EvalConfig{MaxSamples: 2, BatchSize: 3, MaxSeqLen: 4})
	if evalCfg.MaxSamples != 2 || evalCfg.Batch.BatchSize != 3 || evalCfg.Batch.MaxSeqLen != 4 {
		t.Fatalf("eval config = %+v", evalCfg)
	}
	eval := toInferenceEvalReport(&EvalReport{
		ModelInfo: ModelInfo{Architecture: "qwen3"},
		Adapter:   LoRAAdapterInfo{Name: "eval"},
		Metrics:   EvalMetrics{Samples: 1, Tokens: 2, Loss: 0.3, Perplexity: 1.4},
		Quality:   EvalQualityReport{Checks: []EvalQualityCheck{{Name: "q", Pass: true, Score: 0.9, Detail: "ok"}}},
	})
	if eval == nil || eval.Metrics.Samples != 1 || len(eval.Probes) != 1 || !eval.Probes[0].Passed {
		t.Fatalf("eval report = %+v", eval)
	}
	if toInferenceEvalReport(nil) != nil {
		t.Fatal("toInferenceEvalReport(nil) != nil")
	}

	trainingCfg := inference.TrainingConfig{
		Epochs:               2,
		BatchSize:            3,
		GradientAccumulation: 4,
		LearningRate:         0.01,
		LoRA:                 inference.LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{"v_proj"}, BFloat16: true},
		Labels:               map[string]string{"run": "unit"},
	}
	sftCfg := toSFTConfig(trainingCfg, nil)
	if sftCfg.LoRA.DType != DTypeBFloat16 || sftCfg.LoRA.TargetKeys[0] != "v_proj" || sftCfg.GradientAccumulationSteps != 4 {
		t.Fatalf("SFT config = %+v", sftCfg)
	}
	training := toInferenceTrainingResult(ModelInfo{
		Architecture: "qwen3",
		Adapter:      LoRAAdapterInfo{Name: "train", Path: "/tmp/original", Rank: 8},
	}, &SFTResult{
		Epochs:      2,
		Steps:       5,
		Samples:     7,
		LastLoss:    0.2,
		Checkpoints: []string{"", "/tmp/ckpt"},
		AdapterPath: "/tmp/final",
	}, trainingCfg)
	if training.Metrics.Step != 5 || training.Adapter.Path != "/tmp/final" || len(training.Checkpoints) != 1 || training.Checkpoints[0].URI != "file:///tmp/ckpt" {
		t.Fatalf("training result = %+v", training)
	}
	if toInferenceTrainingResult(ModelInfo{Architecture: "qwen3"}, nil, inference.TrainingConfig{}).Model.Architecture != "qwen3" {
		t.Fatal("nil training result did not preserve model identity")
	}

	if meanNonZero(0, 2, 4) != 3 || meanNonZero(0, 0) != 0 {
		t.Fatal("meanNonZero returned unexpected value")
	}
}

func TestInferenceContract_RootProbeSink_Good(t *testing.T) {
	var got inference.ProbeEvent
	sink := inferenceProbeSink{sink: inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		got = event
	})}
	sink.EmitProbe(ProbeEvent{
		Kind:  ProbeEventToken,
		Phase: ProbePhaseDecode,
		Step:  3,
		Meta:  map[string]string{"k": "v"},
		Token: &ProbeToken{ID: 8, Text: "tok", PromptTokens: 1, GeneratedTokens: 2},
		Entropy: &ProbeEntropy{
			Value: 0.7,
			Unit:  "nats",
		},
		Training: &ProbeTraining{
			Epoch:        1,
			Step:         3,
			Loss:         0.4,
			LearningRate: 0.01,
		},
	})
	if got.Token == nil || got.Token.Text != "tok" || got.Entropy == nil || got.Training == nil || got.Labels["k"] != "v" {
		t.Fatalf("root probe event = %+v, want token/entropy/training", got)
	}
	inferenceProbeSink{}.EmitProbe(ProbeEvent{Kind: ProbeEventToken})
}

type inferenceContractDatasetStream struct {
	samples    []inference.DatasetSample
	index      int
	resetCalls int
}

func (stream *inferenceContractDatasetStream) Next() (inference.DatasetSample, bool, error) {
	if stream.index >= len(stream.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := stream.samples[stream.index]
	stream.index++
	return sample, true, nil
}

func (stream *inferenceContractDatasetStream) Reset() error {
	stream.resetCalls++
	stream.index = 0
	return nil
}

type inferenceContractOneShotStream struct{}

func (inferenceContractOneShotStream) Next() (inference.DatasetSample, bool, error) {
	return inference.DatasetSample{}, false, nil
}
