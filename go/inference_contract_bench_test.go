// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for inference_contract.go — the shared-inference façade
// boundary. Per AX-11 — these are the type-shuffling helpers that run
// on every call across the inference.Capability* / Bench* / Eval* /
// Probe surfaces. CapabilityReport() fires per CapabilityReporter
// query (once per agent dispatch, per fleet sync, per fit-plan check);
// the toInference* mappers fire per BenchReport / EvalReport / probe
// event, so allocation budget for those flows runs through here.
//
// Run:    go test -bench='BenchmarkInferenceContract' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/inference/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/inference/probe"
)

// Sinks defeat compiler DCE.
var (
	icBenchSinkReport         inference.CapabilityReport
	icBenchSinkProbeEvent     inference.ProbeEvent
	icBenchSinkRootProbeEvent inference.ProbeEvent
	icBenchSinkLabels         map[string]string
	icBenchSinkAdapterID      inference.AdapterIdentity
	icBenchSinkModelID        inference.ModelIdentity
	icBenchSinkMemPlan        inference.MemoryPlan
	icBenchSinkEvalCfg        eval.Config
	icBenchSinkEvalReport     *inference.EvalReport
	icBenchSinkTrainingResult *inference.TrainingResult
	icBenchSinkSFTConfig      SFTConfig
	icBenchSinkSFTDType       DType
	icBenchSinkProbeLogits    []inference.ProbeLogit
	icBenchSinkQuality        []inference.QualityProbeResult
	icBenchSinkSplitEndpoints []inference.SplitEndpoint
	icBenchSinkStateRefs      []inference.StateRef
	icBenchSinkFloat          float64
	icBenchSinkCapabilities   []inference.Capability
)

// --- metalCapabilityReport ---
// `available=false` skips the safeRuntimeDeviceInfo() path entirely
// (metalCapabilityDeviceInfo returns zero on !available) so this bench
// measures the pure report-shape work — the capability slice copy +
// label map population that runs every CapabilityReporter call.

func BenchmarkInferenceContract_MetalCapabilityReport_Unavailable(b *testing.B) {
	model := inference.ModelIdentity{Architecture: "qwen3"}
	adapter := inference.AdapterIdentity{Format: "lora"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkReport = metalCapabilityReport(model, adapter, false)
	}
}

// `available=true` runs the full report path including the
// safeRuntimeDeviceInfo() host probe. Sets the package-level hook so
// we don't actually touch cgo here — replicating the same pattern
// inference_contract_test.go uses for the *UsesSafeDeviceInfoHook*
// test.
func BenchmarkInferenceContract_MetalCapabilityReport_Available(b *testing.B) {
	prev := metalCapabilityDeviceInfo
	metalCapabilityDeviceInfo = func(available bool) DeviceInfo {
		return DeviceInfo{
			Architecture:                 "apple9",
			MaxBufferLength:              16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
			MemorySize:                   96 * memory.GiB,
		}
	}
	b.Cleanup(func() { metalCapabilityDeviceInfo = prev })
	model := inference.ModelIdentity{Architecture: "qwen3", NumLayers: 28}
	adapter := inference.AdapterIdentity{Format: "lora"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkReport = metalCapabilityReport(model, adapter, true)
	}
}

// --- markMetalUnavailableCapabilities ---
// Internal pass that rewrites the capability slice when Metal is
// unavailable. Fires once per CapabilityReporter call with
// loadReady=false, hits ~30 capability entries.

func BenchmarkInferenceContract_MarkMetalUnavailableCapabilities(b *testing.B) {
	template := metalCapabilityReport(inference.ModelIdentity{}, inference.AdapterIdentity{}, true)
	original := template.Capabilities
	caps := make([]inference.Capability, len(original))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(caps, original)
		icBenchSinkCapabilities = markMetalUnavailableCapabilities(caps)
	}
}

// --- toInferenceProbeEvent ---
// Per probe.Event → inference.ProbeEvent conversion. Fires for every
// probe emitted during generation/training. Two shapes — minimal
// (just kind+phase) and rich (logits + cache + memory).

func BenchmarkInferenceContract_ToInferenceProbeEvent_Minimal(b *testing.B) {
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Step:  3,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkProbeEvent = toInferenceProbeEvent(event)
	}
}

func BenchmarkInferenceContract_ToInferenceProbeEvent_Full(b *testing.B) {
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Step:  5,
		Token: &metal.ProbeToken{ID: 7, Text: "answer", PromptTokens: 16, GeneratedTokens: 3},
		Logits: &metal.ProbeLogits{
			VocabSize: 151936,
			MaxLogit:  4.5,
			MinLogit:  -3.2,
			MeanLogit: 0.05,
			Top: []metal.ProbeLogit{
				{TokenID: 7, Logit: 4.5},
				{TokenID: 9, Logit: 4.2},
				{TokenID: 11, Logit: 3.9},
				{TokenID: 13, Logit: 3.7},
				{TokenID: 15, Logit: 3.5},
			},
		},
		Entropy: &metal.ProbeEntropy{Value: 1.2, Unit: "nats"},
		Cache: &metal.ProbeCachePressure{
			PromptTokens:    256,
			GeneratedTokens: 12,
			CacheTokens:     268,
			Utilization:     0.72,
		},
		Memory: &metal.ProbeMemoryPressure{ActiveBytes: 4 << 30, PeakBytes: 6 << 30},
		Meta:   map[string]string{"prompt_id": "abc", "step": "5"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkProbeEvent = toInferenceProbeEvent(event)
	}
}

// --- toInferenceProbeLogits ---
// Top-K logit slice copy. Top-K varies by sampler config; bench
// representative K=10.

func BenchmarkInferenceContract_ToInferenceProbeLogits_10(b *testing.B) {
	logits := make([]metal.ProbeLogit, 10)
	for i := range logits {
		logits[i] = metal.ProbeLogit{TokenID: int32(i + 1), Logit: float32(5 - i)}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkProbeLogits = toInferenceProbeLogits(logits)
	}
}

// --- toInferenceModelIdentity ---
// Per-info conversion at every CapabilityReport call.

func BenchmarkInferenceContract_ToInferenceModelIdentity(b *testing.B) {
	info := ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 40960,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkModelID = toInferenceModelIdentity(info)
	}
}

// --- toInferenceAdapterIdentity ---

func BenchmarkInferenceContract_ToInferenceAdapterIdentity(b *testing.B) {
	info := metal.AdapterInfo{
		Name:       "demo",
		Path:       "/tmp/adapter",
		Hash:       "0xabc",
		Rank:       8,
		Alpha:      16,
		Scale:      0.5,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkAdapterID = toInferenceAdapterIdentity(info)
	}
}

// --- adapterIdentityLabels ---

func BenchmarkInferenceContract_AdapterIdentityLabels_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkLabels = adapterIdentityLabels("", 0)
	}
}

func BenchmarkInferenceContract_AdapterIdentityLabels_Populated(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkLabels = adapterIdentityLabels("demo", 0.5)
	}
}

// --- toInferenceMemoryPlan ---

func BenchmarkInferenceContract_ToInferenceMemoryPlan(b *testing.B) {
	plan := memory.Plan{
		MachineClass:              memory.ClassApple96GB,
		DeviceMemoryBytes:         96 * memory.GiB,
		ContextLength:             131072,
		BatchSize:                 4,
		CacheMode:                 memory.KVCacheModePaged,
		EstimatedKVCacheModeBytes: 4 << 30,
		Notes:                     []string{"note1", "note2", "note3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkMemPlan = toInferenceMemoryPlan(plan)
	}
}

// --- toEvalConfig ---

func BenchmarkInferenceContract_ToEvalConfig(b *testing.B) {
	cfg := inference.EvalConfig{MaxSamples: 50, BatchSize: 4, MaxSeqLen: 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkEvalCfg = toEvalConfig(cfg)
	}
}

// --- toInferenceEvalReport ---

func BenchmarkInferenceContract_ToInferenceEvalReport(b *testing.B) {
	rpt := &eval.Report{
		ModelInfo: eval.Info{Architecture: "qwen3", NumLayers: 28},
		Adapter:   eval.AdapterInfo{Name: "demo", Rank: 8},
		Metrics:   eval.Metrics{Samples: 50, Tokens: 25600, Loss: 0.3, Perplexity: 1.4},
		Quality: eval.QualityReport{
			Checks: []eval.QualityCheck{
				{Name: "exact_match", Pass: true, Score: 0.92, Detail: "ok"},
				{Name: "format", Pass: true, Score: 1.0, Detail: ""},
				{Name: "safety", Pass: true, Score: 0.99, Detail: "passed"},
			},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkEvalReport = toInferenceEvalReport(rpt)
	}
}

// --- toInferenceQualityResults ---

func BenchmarkInferenceContract_ToInferenceQualityResults(b *testing.B) {
	checks := []eval.QualityCheck{
		{Name: "exact_match", Pass: true, Score: 0.9, Detail: "ok"},
		{Name: "format", Pass: false, Score: 0.5, Detail: "drift"},
		{Name: "safety", Pass: true, Score: 1.0, Detail: ""},
		{Name: "rouge", Pass: true, Score: 0.7, Detail: "good"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkQuality = toInferenceQualityResults(checks)
	}
}

// --- toSFTConfig ---

func BenchmarkInferenceContract_ToSFTConfig(b *testing.B) {
	cfg := inference.TrainingConfig{
		Epochs:               2,
		BatchSize:            4,
		GradientAccumulation: 8,
		LearningRate:         3e-4,
		LoRA: inference.LoRAConfig{
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
			BFloat16:   true,
		},
		Labels: map[string]string{"run": "unit", "kind": "sft"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkSFTConfig = toSFTConfig(cfg, nil)
	}
}

// --- sftDType ---

func BenchmarkInferenceContract_SFTDType_True(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkSFTDType = sftDType(true)
	}
}

func BenchmarkInferenceContract_SFTDType_False(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkSFTDType = sftDType(false)
	}
}

// --- toInferenceTrainingResult ---

func BenchmarkInferenceContract_ToInferenceTrainingResult(b *testing.B) {
	info := ModelInfo{
		Architecture: "qwen3",
		Adapter:      lora.AdapterInfo{Name: "demo", Path: "/tmp/orig", Rank: 8},
	}
	result := &SFTResult{
		Epochs:      2,
		Steps:       100,
		Samples:     200,
		LastLoss:    0.25,
		Checkpoints: []string{"/tmp/ckpt1", "", "/tmp/ckpt2", "/tmp/ckpt3"},
		AdapterPath: "/tmp/final",
	}
	cfg := inference.TrainingConfig{
		LearningRate: 3e-4,
		Labels:       map[string]string{"run": "unit"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkTrainingResult = toInferenceTrainingResult(info, result, cfg)
	}
}

// --- toInferenceRootAdapterIdentity ---

func BenchmarkInferenceContract_ToInferenceRootAdapterIdentity(b *testing.B) {
	info := lora.AdapterInfo{
		Path:       "/tmp/adapter",
		Hash:       "0xabc",
		Rank:       8,
		Alpha:      16,
		Scale:      1.0,
		Name:       "demo",
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkAdapterID = toInferenceRootAdapterIdentity(info)
	}
}

// --- stateRefsFromPaths ---

func BenchmarkInferenceContract_StateRefsFromPaths(b *testing.B) {
	paths := []string{"/tmp/ckpt1", "", "/tmp/ckpt2", "/tmp/ckpt3", "/tmp/ckpt4"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkStateRefs = stateRefsFromPaths("sft_checkpoint", paths)
	}
}

// --- cloneInferenceLabels ---

func BenchmarkInferenceContract_CloneInferenceLabels_Empty(b *testing.B) {
	var labels map[string]string
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkLabels = cloneInferenceLabels(labels)
	}
}

func BenchmarkInferenceContract_CloneInferenceLabels_Typical(b *testing.B) {
	labels := map[string]string{
		"backend": "metal",
		"library": "go-mlx",
		"run_id":  "abc-123",
		"prompt":  "demo",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkLabels = cloneInferenceLabels(labels)
	}
}

// --- cloneInferenceSplitEndpoints ---

func BenchmarkInferenceContract_CloneInferenceSplitEndpoints(b *testing.B) {
	endpoints := []inference.SplitEndpoint{
		{Labels: map[string]string{"role": "ffn"}},
		{Labels: map[string]string{"role": "experts"}},
		{Labels: map[string]string{"role": "embed"}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkSplitEndpoints = cloneInferenceSplitEndpoints(endpoints)
	}
}

// --- meanNonZero ---

func BenchmarkInferenceContract_MeanNonZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkFloat = meanNonZero(0.0, 0.7, 0.0, 0.9, 0.85, 0.0)
	}
}

// --- toInferenceRootProbeEvent ---
// The root-package probe sink path — wraps a probe.Event coming from
// lora/sft/grpo training back to inference.ProbeEvent.

func BenchmarkInferenceContract_ToInferenceRootProbeEvent_Training(b *testing.B) {
	event := probe.Event{
		Kind:    probe.KindTraining,
		Phase:   probe.PhaseTraining,
		Step:    100,
		Token:   &probe.Token{ID: 7, Text: "tok", PromptTokens: 16, GeneratedTokens: 3},
		Entropy: &probe.Entropy{Value: 1.2, Unit: "nats"},
		Training: &probe.Training{
			Epoch:        1,
			Step:         100,
			Loss:         0.4,
			LearningRate: 3e-4,
		},
		Meta: map[string]string{"run": "unit", "step": "100"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icBenchSinkRootProbeEvent = toInferenceRootProbeEvent(event)
	}
}
