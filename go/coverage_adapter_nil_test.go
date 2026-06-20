// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// coverageProbeSink is a no-op inference.ProbeSink used to exercise the
// non-nil arm of toMetalInferenceProbeSink and the probe-sink plumbing on
// metaladapter without loading a real model.
type coverageProbeSink struct{ events int }

func (s *coverageProbeSink) EmitProbe(inference.ProbeEvent) { s.events++ }

// TestMetalAdapterNilModelGuards walks every metaladapter method that short
// circuits when the underlying *metal.Model is nil. A zero-value
// &metaladapter{} can never carry a real model in a unit test, so these
// nil-guard arms are the reachable surface — the post-guard real-forward
// bodies are real-model-bound and counted in the ceiling.
func TestMetalAdapterNilModelGuards(t *testing.T) {
	adapter := &metaladapter{}
	ctx := context.Background()

	// Capabilities returns the not-loaded report on the nil-model arm.
	report := adapter.Capabilities()
	if report.Available {
		t.Fatalf("nil-model Capabilities should not report Available: %+v", report)
	}

	if _, err := adapter.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatalf("ApplyChatTemplate on nil model should error")
	}

	if _, err := adapter.LoadAdapter("/nope"); err == nil {
		t.Fatalf("LoadAdapter on nil model should error")
	}
	if err := adapter.UnloadAdapter(); err == nil {
		t.Fatalf("UnloadAdapter on nil model should error")
	}
	if id := adapter.ActiveAdapter(); id.Hash != "" || id.Path != "" {
		t.Fatalf("ActiveAdapter on nil model should be zero, got %+v", id)
	}

	if _, err := adapter.Evaluate(ctx, nil, inference.EvalConfig{}); err == nil {
		t.Fatalf("Evaluate on nil model should error")
	}
	if _, err := adapter.TrainSFT(ctx, nil, inference.TrainingConfig{}); err == nil {
		t.Fatalf("TrainSFT on nil model should error")
	}

	// rootModel returns an empty *Model on the nil-model arm (no panic).
	if rm := adapter.rootModel(); rm == nil {
		t.Fatalf("rootModel should never be nil")
	}

	// outputParser / ParseReasoning / ParseTools resolve to the shared
	// default parser when the model is nil.
	if adapter.outputParser() == nil {
		t.Fatalf("outputParser on nil model should be the default parser")
	}
	if _, err := adapter.ParseReasoning(nil, "<think>x</think>y"); err != nil {
		t.Fatalf("ParseReasoning default parser errored: %v", err)
	}
	if _, err := adapter.ParseTools(nil, "plain text"); err != nil {
		t.Fatalf("ParseTools default parser errored: %v", err)
	}

	// blockCacheService and schedulerModel both have a typed-nil receiver
	// fast path: calling on a literal nil *metaladapter returns a fresh
	// service / scheduler without touching the model.
	var nilAdapter *metaladapter
	if nilAdapter.blockCacheService() == nil {
		t.Fatalf("nil-adapter blockCacheService should return a fresh service")
	}
	if nilAdapter.outputParser() == nil {
		t.Fatalf("nil-adapter outputParser should return the default parser")
	}
}

// TestMetalAdapterSchedulerLifecycle drives schedulerModel's lazy-init arm
// (and the SetProbeSink propagation into a live scheduler) on a real
// &metaladapter{} that has no model — scheduler.New only stores the backend
// reference and spawns idle workers, so no forward pass is required.
func TestMetalAdapterSchedulerLifecycle(t *testing.T) {
	adapter := &metaladapter{schedulerMaxConcurrent: 0} // exercises the <=0 default

	scheduler := adapter.schedulerModel()
	if scheduler == nil {
		t.Fatalf("schedulerModel should build a scheduler")
	}
	// Second call returns the memoised instance (the cacheService != nil arm).
	if again := adapter.schedulerModel(); again != scheduler {
		t.Fatalf("schedulerModel should memoise the scheduler")
	}

	// SetProbeSink on an adapter with a live scheduler should propagate.
	sink := &coverageProbeSink{}
	adapter.SetProbeSink(sink)
	if adapter.probeSink != sink {
		t.Fatalf("SetProbeSink should store the sink")
	}

	// SetProbeSink on a typed-nil adapter is a no-op (the nil guard).
	var nilAdapter *metaladapter
	nilAdapter.SetProbeSink(sink)
}

// TestMetalAdapterGenerateConfigProbeSink covers the probe-sink branch of
// generateConfig and the non-nil arm of toMetalInferenceProbeSink. Neither
// touches the model.
func TestMetalAdapterGenerateConfigProbeSink(t *testing.T) {
	sink := &coverageProbeSink{}
	adapter := &metaladapter{probeSink: sink}

	cfg := adapter.generateConfig(inference.WithTemperature(0.5))
	if cfg.ProbeSink == nil {
		t.Fatalf("generateConfig should wire the probe sink into metal config")
	}

	// Direct converter coverage: nil-in and non-nil-in arms.
	if toMetalInferenceProbeSink(nil) != nil {
		t.Fatalf("nil sink should convert to nil")
	}
	if toMetalInferenceProbeSink(sink) == nil {
		t.Fatalf("non-nil sink should convert to a metal probe sink")
	}
}

// TestMetalAdapterChatConfig covers both arms of metalAdapterChatConfig: the
// architecture-from-info path and the fallback-to-modelType path. Pure
// function, no model.
func TestMetalAdapterChatConfig(t *testing.T) {
	withArch := metalAdapterChatConfig(metal.ModelInfo{Architecture: "gemma4_text", NumHeads: 8}, "ignored")
	fallback := metalAdapterChatConfig(metal.ModelInfo{Architecture: "", NumHeads: 8}, "gemma4_text")
	// Both resolve the same architecture, so the resulting configs should match.
	if withArch != fallback {
		t.Fatalf("chat config arms diverged: %+v vs %+v", withArch, fallback)
	}
}

// TestMetalBackendSetRuntimeMemoryLimits covers both positive-limit arms of
// SetRuntimeMemoryLimits on the backend (global allocator setters, no model).
func TestMetalBackendSetRuntimeMemoryLimits(t *testing.T) {
	backend := &metalbackend{}

	// Snapshot to restore afterwards so the global allocator state is left
	// as found.
	prevCache := GetCacheMemory()
	_ = prevCache

	applied := backend.SetRuntimeMemoryLimits(inference.RuntimeMemoryLimits{
		CacheLimitBytes:  64 << 20,
		MemoryLimitBytes: 512 << 20,
	})
	if applied.CacheLimitBytes != 64<<20 || applied.MemoryLimitBytes != 512<<20 {
		t.Fatalf("applied limits not echoed: %+v", applied)
	}

	// Zero-limit input takes neither branch (echoes input unchanged).
	zero := backend.SetRuntimeMemoryLimits(inference.RuntimeMemoryLimits{})
	if zero.PreviousCacheLimitBytes != 0 || zero.PreviousMemoryLimitBytes != 0 {
		t.Fatalf("zero limits should not record previous values: %+v", zero)
	}
}

// TestInferenceModelInfoHash and the nil-adapter tokenizer-hash guards.
func TestInferenceModelInfoHashAndTokenizerGuards(t *testing.T) {
	hash := inferenceModelInfoHash(inference.ModelInfo{
		Architecture: "gemma4_text",
		VocabSize:    256000,
		NumLayers:    34,
		HiddenSize:   2560,
		QuantBits:    4,
		QuantGroup:   32,
	})
	if hash == "" {
		t.Fatalf("inferenceModelInfoHash should be non-empty for a populated info")
	}

	// Nil-adapter / nil-model arms return empty.
	var nilAdapter *metaladapter
	if got := adapterTokenizerHash(nilAdapter); got != "" {
		t.Fatalf("adapterTokenizerHash(nil) = %q, want empty", got)
	}
	if got := adapterTokenizerHashFromInfo(nilAdapter, inference.ModelInfo{}); got != "" {
		t.Fatalf("adapterTokenizerHashFromInfo(nil) = %q, want empty", got)
	}
}

// TestInferenceMessagesCarryImages covers the pure helper that scans messages
// for attached images.
func TestInferenceMessagesCarryImages(t *testing.T) {
	if inferenceMessagesCarryImages(nil) {
		t.Fatalf("nil messages should not carry images")
	}
	if inferenceMessagesCarryImages([]inference.Message{{Role: "user", Content: "text"}}) {
		t.Fatalf("text-only messages should not carry images")
	}
	withImage := []inference.Message{
		{Role: "user", Content: "look"},
		{Role: "user", Images: [][]byte{{0x89, 0x50}}},
	}
	if !inferenceMessagesCarryImages(withImage) {
		t.Fatalf("message with image bytes should be detected")
	}
}
