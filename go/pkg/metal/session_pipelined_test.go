// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Coverage for the model-independent helpers of the pipelined one-ahead decode
// loop: the eligibility predicate, the pending-cache staging helpers, and the
// token-input shaper. The hot loop itself (runPipelinedDecodeLocked) needs a
// real decode through forwardLastTokenLogits on a fixed-cache compiled regime —
// it is exercised by the engine's model-level tests, not here.

// --- pipelineTokenInput: shape the lazy sampled token into a [1,1] int32 ---

func TestPipelineTokenInput_PassesThroughInt32Matrix(t *testing.T) {
	requireMetalRuntime(t)
	out := pipelineTokenInput(FromSingleInt32Matrix(42)) // already [1,1] int32
	defer Free(out)
	if out.NumDims() != 2 || out.Dim(0) != 1 || out.Dim(1) != 1 {
		t.Fatalf("shape = %v, want [1 1]", out.Shape())
	}
	if got := out.Int(); got != 42 {
		t.Fatalf("value = %d, want 42 passed through", got)
	}
}

func TestPipelineTokenInput_ReshapesAndCastsScalar(t *testing.T) {
	requireMetalRuntime(t)
	out := pipelineTokenInput(FromValues([]float32{5}, 1)) // [1] float32 → must become [1,1] int32
	defer Free(out)
	if out.NumDims() != 2 || out.Dim(0) != 1 || out.Dim(1) != 1 {
		t.Fatalf("shape = %v, want [1 1]", out.Shape())
	}
	if got := out.Int(); got != 5 {
		t.Fatalf("value = %d, want 5 (float cast to int32)", got)
	}
}

// --- pipelinedDecodeEligibleLocked: only the fully-functional gated fixed-cache
// regime runs one-ahead; any host-value control flow keeps the serial loop ---

func newPipelinedEligibleSession() *ModelSession {
	return &ModelSession{
		model:  &Model{tokenizer: NewForDecode(nil)},
		logits: Zeros([]int32{1, 1, 2}, DTypeFloat32),
		caches: []Cache{NewFixedKVCache(512)},
	}
}

func TestPipelinedDecodeEligible_GatedFixedRegime(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate(GatePipelinedDecode, true))
	t.Cleanup(SetRuntimeGate(GateCompiledLayerDecode, true))
	s := newPipelinedEligibleSession()
	defer s.resetState()
	if ok, reason := s.pipelinedDecodeEligibleLocked(GenerateConfig{MaxTokens: 4}); !ok {
		t.Fatalf("eligible = false (%q), want true for the gated fixed-cache regime", reason)
	}
}

func TestPipelinedDecodeEligible_RejectsHostControlFlow(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate(GatePipelinedDecode, true))
	t.Cleanup(SetRuntimeGate(GateCompiledLayerDecode, true))
	cases := []struct {
		name string
		cfg  GenerateConfig
	}{
		{"repeat penalty", GenerateConfig{RepeatPenalty: 1.1}},
		{"token suppression", GenerateConfig{SuppressTokens: []int32{7}}},
		{"min-tokens-before-stop window", GenerateConfig{MinTokensBeforeStop: 3}},
		{"thinking budget", GenerateConfig{ThinkingBudget: 8}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newPipelinedEligibleSession()
			defer s.resetState()
			if ok, reason := s.pipelinedDecodeEligibleLocked(tc.cfg); ok || reason == "" {
				t.Fatalf("eligible = %v (%q), want false with a reason for %s", ok, reason, tc.name)
			}
		})
	}
}

func TestPipelinedDecodeEligible_RejectsGateOff(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate(GatePipelinedDecode, false))
	s := newPipelinedEligibleSession()
	defer s.resetState()
	if ok, reason := s.pipelinedDecodeEligibleLocked(GenerateConfig{MaxTokens: 4}); ok {
		t.Fatalf("eligible = true (%q), want false when the pipelined gate is off", reason)
	}
}

func TestPipelinedDecodeEligible_RejectsMissingPrefillLogits(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate(GatePipelinedDecode, true))
	t.Cleanup(SetRuntimeGate(GateCompiledLayerDecode, true))
	s := &ModelSession{model: &Model{tokenizer: NewForDecode(nil)}, caches: []Cache{NewFixedKVCache(512)}}
	defer s.resetState()
	if ok, reason := s.pipelinedDecodeEligibleLocked(GenerateConfig{MaxTokens: 4}); ok {
		t.Fatalf("eligible = true (%q), want false with nil prefill logits", reason)
	}
}

// --- arm / commit / discard / violation: stage the speculated forward's K/V ---

func TestPipelinedPendingCaches_ArmCommitDiscardCleanRegime(t *testing.T) {
	requireMetalRuntime(t)
	s := &ModelSession{
		model:  &Model{tokenizer: NewForDecode(nil)},
		logits: Zeros([]int32{1, 1, 2}, DTypeFloat32),
		caches: []Cache{NewFixedKVCache(512), NewFixedKVCache(512)},
	}
	defer s.resetState()

	if s.pendingViolationLocked() {
		t.Fatal("fresh fixed caches report a pending violation")
	}
	s.armPendingCachesLocked()
	if s.pendingViolationLocked() {
		t.Fatal("arming clean caches must not raise a violation")
	}
	s.commitPendingCachesLocked()
	s.armPendingCachesLocked()
	s.discardPendingCachesLocked()
	if s.pendingViolationLocked() {
		t.Fatal("commit then discard of clean caches must not raise a violation")
	}
}
