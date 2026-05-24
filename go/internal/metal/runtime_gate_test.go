// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestRuntimeGate_SetRuntimeGate_Good(t *testing.T) {
	coverageTokens := "RuntimeGate SetRuntimeGate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restore := SetRuntimeGate("GO_MLX_TEST_RUNTIME_GATE", "1")
	t.Cleanup(restore)

	if got := RuntimeGateValue("GO_MLX_TEST_RUNTIME_GATE"); got != "1" {
		t.Fatalf("RuntimeGateValue() = %q, want 1", got)
	}
	if !RuntimeGateEnabled("GO_MLX_TEST_RUNTIME_GATE") {
		t.Fatal("RuntimeGateEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownGemma4AttentionOMatVec_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownGemma4AttentionOMatVec"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC", "0")
	t.Cleanup(restoreOff)
	if nativeGemma4AttentionOMatVecRuntimeEnabled() {
		t.Fatal("nativeGemma4AttentionOMatVecRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC", "1")
	t.Cleanup(restoreOn)
	if !nativeGemma4AttentionOMatVecRuntimeEnabled() {
		t.Fatal("nativeGemma4AttentionOMatVecRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownGenerationStream_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownGenerationStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "0")
	t.Cleanup(restoreOff)
	if generationStreamRuntimeEnabled() {
		t.Fatal("generationStreamRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "1")
	t.Cleanup(restoreOn)
	if !generationStreamRuntimeEnabled() {
		t.Fatal("generationStreamRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownGenerationClearCache_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownGenerationClearCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_GENERATION_CLEAR_CACHE", "0")
	t.Cleanup(restoreOff)
	if generationClearCacheRuntimeEnabled() {
		t.Fatal("generationClearCacheRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_GENERATION_CLEAR_CACHE", "1")
	t.Cleanup(restoreOn)
	if !generationClearCacheRuntimeEnabled() {
		t.Fatal("generationClearCacheRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownZeroCopyPagedRestore_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownZeroCopyPagedRestore"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Setenv("GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE", "")
	restoreDefault := SetRuntimeGate("GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE", "")
	t.Cleanup(restoreDefault)
	if !zeroCopyPagedRestoreRuntimeEnabled() {
		t.Fatal("zeroCopyPagedRestoreRuntimeEnabled() default = false, want true")
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE", "0")
	t.Cleanup(restoreOff)
	if zeroCopyPagedRestoreRuntimeEnabled() {
		t.Fatal("zeroCopyPagedRestoreRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE", "1")
	t.Cleanup(restoreOn)
	if !zeroCopyPagedRestoreRuntimeEnabled() {
		t.Fatal("zeroCopyPagedRestoreRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativePagedAttention_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownNativePagedAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "0")
	t.Cleanup(restoreOff)
	if nativePagedAttentionEnabled() {
		t.Fatal("nativePagedAttentionEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1")
	t.Cleanup(restoreOn)
	if !nativePagedAttentionEnabled() {
		t.Fatal("nativePagedAttentionEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownPagedFullKVMaterialize_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownPagedFullKVMaterialize"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE", "0")
	t.Cleanup(restoreOff)
	if pagedFullKVMaterializeEnabled() {
		t.Fatal("pagedFullKVMaterializeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE", "1")
	t.Cleanup(restoreOn)
	if !pagedFullKVMaterializeEnabled() {
		t.Fatal("pagedFullKVMaterializeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownPagedKVPrealloc_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownPagedKVPrealloc"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "0")
	t.Cleanup(restoreOff)
	if pagedKVPreallocRuntimeEnabled() {
		t.Fatal("pagedKVPreallocRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "1")
	t.Cleanup(restoreOn)
	if !pagedKVPreallocRuntimeEnabled() {
		t.Fatal("pagedKVPreallocRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownFixedGemma4SlidingCacheBound_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownFixedGemma4SlidingCacheBound"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "0")
	t.Cleanup(restoreOff)
	if fixedGemma4SlidingCacheBoundRuntimeEnabled() {
		t.Fatal("fixedGemma4SlidingCacheBoundRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Cleanup(restoreOn)
	if !fixedGemma4SlidingCacheBoundRuntimeEnabled() {
		t.Fatal("fixedGemma4SlidingCacheBoundRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativeFixedSlidingAttention_Good(t *testing.T) {
	coverageTokens := "RuntimeGate KnownNativeFixedSlidingAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "0")
	t.Cleanup(restoreOff)
	if nativeFixedSlidingAttentionRuntimeEnabled() {
		t.Fatal("nativeFixedSlidingAttentionRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "1")
	t.Cleanup(restoreOn)
	if !nativeFixedSlidingAttentionRuntimeEnabled() {
		t.Fatal("nativeFixedSlidingAttentionRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_RuntimeGateValue_Bad(t *testing.T) {
	coverageTokens := "RuntimeGate RuntimeGateValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	if got := RuntimeGateValue(""); got != "" {
		t.Fatalf("RuntimeGateValue(empty) = %q, want empty", got)
	}
}

func TestRuntimeGate_RuntimeGateEnabled_Ugly(t *testing.T) {
	coverageTokens := "RuntimeGate RuntimeGateEnabled"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Setenv("GO_MLX_TEST_RUNTIME_GATE_RESTORE", "1")
	restore := SetRuntimeGate("GO_MLX_TEST_RUNTIME_GATE_RESTORE", "0")
	if RuntimeGateEnabled("GO_MLX_TEST_RUNTIME_GATE_RESTORE") {
		t.Fatal("RuntimeGateEnabled() = true under disabled override, want false")
	}
	restore()
	if !RuntimeGateEnabled("GO_MLX_TEST_RUNTIME_GATE_RESTORE") {
		t.Fatal("RuntimeGateEnabled() = false after override restore, want env fallback")
	}
}
