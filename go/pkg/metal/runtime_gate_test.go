// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestRuntimeGate_SetRuntimeGate_Good(t *testing.T) {
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

func TestRuntimeGate_KnownNativeQ6BitstreamMatVec_Good(t *testing.T) {
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "0")
	t.Cleanup(restoreOff)
	if nativeQ6BitstreamMatVecRuntimeEnabled() {
		t.Fatal("nativeQ6BitstreamMatVecRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "1")
	t.Cleanup(restoreOn)
	if !nativeQ6BitstreamMatVecRuntimeEnabled() {
		t.Fatal("nativeQ6BitstreamMatVecRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownGenerationStream_Good(t *testing.T) {
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

func TestRuntimeGate_KnownAsyncDecodePrefetch_Good(t *testing.T) {
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "0")
	t.Cleanup(restoreOff)
	if asyncDecodePrefetchRuntimeEnabled() {
		t.Fatal("asyncDecodePrefetchRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "1")
	t.Cleanup(restoreOn)
	if !asyncDecodePrefetchRuntimeEnabled() {
		t.Fatal("asyncDecodePrefetchRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativePagedAttention_Good(t *testing.T) {
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "0")
	t.Cleanup(restoreOff)
	if NativePagedAttentionEnabled() {
		t.Fatal("NativePagedAttentionEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1")
	t.Cleanup(restoreOn)
	if !NativePagedAttentionEnabled() {
		t.Fatal("NativePagedAttentionEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownFixedGemma4SlidingCacheBound_Good(t *testing.T) {
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

func TestRuntimeGate_FixedGemma4ZeroOverrideWins_Good(t *testing.T) {
	oldCache := enableFixedGemma4Cache
	oldSliding := enableFixedGemma4SlidingCacheBound
	oldShared := enableFixedGemma4SharedMask
	oldNativeSliding := enableNativeFixedSlidingAttention
	enableFixedGemma4Cache = true
	enableFixedGemma4SlidingCacheBound = true
	enableFixedGemma4SharedMask = true
	enableNativeFixedSlidingAttention = true
	t.Cleanup(func() {
		enableFixedGemma4Cache = oldCache
		enableFixedGemma4SlidingCacheBound = oldSliding
		enableFixedGemma4SharedMask = oldShared
		enableNativeFixedSlidingAttention = oldNativeSliding
	})
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "0"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "0"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK", "0"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "0"))

	if fixedGemma4CacheEnabled() {
		t.Fatal("fixedGemma4CacheEnabled() = true, want runtime 0 to override package env")
	}
	if fixedGemma4SlidingCacheBoundEnabled() {
		t.Fatal("fixedGemma4SlidingCacheBoundEnabled() = true, want runtime 0 to override package env")
	}
	if FixedGemma4SharedMaskEnabled() {
		t.Fatal("FixedGemma4SharedMaskEnabled() = true, want runtime 0 to override package env")
	}
	if NativeFixedSlidingAttentionEnabled() {
		t.Fatal("NativeFixedSlidingAttentionEnabled() = true, want runtime 0 to override package env")
	}
}

func TestRuntimeGate_FixedGemma4AmbientEnvIgnored_Good(t *testing.T) {
	gates := []string{
		"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
		"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
	}
	for _, gate := range gates {
		restore := SetRuntimeGate(gate, "")
		t.Cleanup(restore)
		t.Setenv(gate, "1")
		if got := RuntimeGateValue(gate); got != "" {
			t.Fatalf("RuntimeGateValue(%s) = %q from ambient env, want empty", gate, got)
		}
	}

	if fixedGemma4CacheEnabled() {
		t.Fatal("fixedGemma4CacheEnabled() = true from ambient env, want explicit runtime override only")
	}
	if fixedGemma4SlidingCacheBoundEnabled() {
		t.Fatal("fixedGemma4SlidingCacheBoundEnabled() = true from ambient env, want explicit runtime override only")
	}
	if FixedGemma4SharedMaskEnabled() {
		t.Fatal("FixedGemma4SharedMaskEnabled() = true from ambient env, want explicit runtime override only")
	}
	if NativeFixedSlidingAttentionEnabled() {
		t.Fatal("NativeFixedSlidingAttentionEnabled() = true from ambient env, want explicit runtime override only")
	}
	if NativeGemma4FixedOwnerAttentionEnabled() {
		t.Fatal("NativeGemma4FixedOwnerAttentionEnabled() = true from ambient env, want explicit runtime override only")
	}
	if NativeGemma4FixedOwnerAttentionResidualEnabled() {
		t.Fatal("NativeGemma4FixedOwnerAttentionResidualEnabled() = true from ambient env, want explicit runtime override only")
	}
}

func TestRuntimeGate_KnownNativeFixedSlidingAttention_Good(t *testing.T) {
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
	if got := RuntimeGateValue(""); got != "" {
		t.Fatalf("RuntimeGateValue(empty) = %q, want empty", got)
	}
}

func TestRuntimeGate_RuntimeGateEnabled_Ugly(t *testing.T) {
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
