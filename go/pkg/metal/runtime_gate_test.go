// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestRuntimeGate_SetEnabledRestore_Good(t *testing.T) {
	// GatePagedDecodeFastConcat is not in the accepted default set, so it starts
	// off in a unit test (no model load). Set turns it on; restore reverts it.
	const gate = GatePagedDecodeFastConcat
	before := RuntimeGateEnabled(gate)

	restore := SetRuntimeGate(gate, true)
	if !RuntimeGateEnabled(gate) {
		t.Fatal("SetRuntimeGate(true) did not enable the gate")
	}

	restore()
	if RuntimeGateEnabled(gate) != before {
		t.Fatalf("restore() left gate = %v, want %v", RuntimeGateEnabled(gate), before)
	}
}

func TestRuntimeGate_KnownAttentionOMatVec_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateNativeAttentionOMatVec, false)
	t.Cleanup(restoreOff)
	if nativeAttentionOMatVecRuntimeEnabled() {
		t.Fatal("nativeAttentionOMatVecRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateNativeAttentionOMatVec, true)
	t.Cleanup(restoreOn)
	if !nativeAttentionOMatVecRuntimeEnabled() {
		t.Fatal("nativeAttentionOMatVecRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativeQ6BitstreamMatVec_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateNativeQ6BitstreamMatVec, false)
	t.Cleanup(restoreOff)
	if nativeQ6BitstreamMatVecRuntimeEnabled() {
		t.Fatal("nativeQ6BitstreamMatVecRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateNativeQ6BitstreamMatVec, true)
	t.Cleanup(restoreOn)
	if !nativeQ6BitstreamMatVecRuntimeEnabled() {
		t.Fatal("nativeQ6BitstreamMatVecRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownGenerationStream_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateGenerationStream, false)
	t.Cleanup(restoreOff)
	if generationStreamRuntimeEnabled() {
		t.Fatal("generationStreamRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateGenerationStream, true)
	t.Cleanup(restoreOn)
	if !generationStreamRuntimeEnabled() {
		t.Fatal("generationStreamRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownAsyncDecodePrefetch_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateAsyncDecodePrefetch, false)
	t.Cleanup(restoreOff)
	if asyncDecodePrefetchRuntimeEnabled() {
		t.Fatal("asyncDecodePrefetchRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateAsyncDecodePrefetch, true)
	t.Cleanup(restoreOn)
	if !asyncDecodePrefetchRuntimeEnabled() {
		t.Fatal("asyncDecodePrefetchRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativePagedAttention_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateNativePagedAttention, false)
	t.Cleanup(restoreOff)
	if NativePagedAttentionEnabled() {
		t.Fatal("NativePagedAttentionEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateNativePagedAttention, true)
	t.Cleanup(restoreOn)
	if !NativePagedAttentionEnabled() {
		t.Fatal("NativePagedAttentionEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownFixedSlidingCacheBound_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateFixedSlidingCacheBound, false)
	t.Cleanup(restoreOff)
	if fixedSlidingCacheBoundRuntimeEnabled() {
		t.Fatal("fixedSlidingCacheBoundRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateFixedSlidingCacheBound, true)
	t.Cleanup(restoreOn)
	if !fixedSlidingCacheBoundRuntimeEnabled() {
		t.Fatal("fixedSlidingCacheBoundRuntimeEnabled() = false, want true")
	}
}

func TestRuntimeGate_KnownNativeFixedSlidingAttention_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateNativeFixedSlidingAttention, false)
	t.Cleanup(restoreOff)
	if nativeFixedSlidingAttentionRuntimeEnabled() {
		t.Fatal("nativeFixedSlidingAttentionRuntimeEnabled() = true, want false")
	}
	restoreOn := SetRuntimeGate(GateNativeFixedSlidingAttention, true)
	t.Cleanup(restoreOn)
	if !nativeFixedSlidingAttentionRuntimeEnabled() {
		t.Fatal("nativeFixedSlidingAttentionRuntimeEnabled() = false, want true")
	}
}

// TestRuntimeGate_OutOfRange_Bad — a Gate outside [0, gateCount) must be inert:
// RuntimeGateEnabled reports false and SetRuntimeGate is a no-op that returns a
// safe restore, never panicking on the array bounds.
func TestRuntimeGate_OutOfRange_Bad(t *testing.T) {
	if RuntimeGateEnabled(Gate(-1)) {
		t.Fatal("RuntimeGateEnabled(-1) = true, want false")
	}
	if RuntimeGateEnabled(gateCount) {
		t.Fatal("RuntimeGateEnabled(gateCount) = true, want false")
	}
	restore := SetRuntimeGate(Gate(-1), true)
	restore()
	restore = SetRuntimeGate(gateCount, true)
	restore()
}

// TestRuntimeGate_AmbientEnvIgnored_Ugly — no gate is ever read from process
// env. Setting the legacy GO_MLX_ENABLE_* env names must not move any typed
// gate: the external-control surface (Cerberus DREAD) stays closed by
// construction, since the gate array has no Getenv path at all.
func TestRuntimeGate_AmbientEnvIgnored_Ugly(t *testing.T) {
	t.Setenv("GO_MLX_ENABLE_FIXED_SLIDING_CACHE", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "1")
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, false))
	t.Cleanup(SetRuntimeGate(GateNativeFixedSlidingAttention, false))

	if fixedSlidingCacheEnabled() {
		t.Fatal("fixedSlidingCacheEnabled() = true from ambient env, want gates closed to env")
	}
	if NativeFixedSlidingAttentionEnabled() {
		t.Fatal("NativeFixedSlidingAttentionEnabled() = true from ambient env, want gates closed to env")
	}
}
