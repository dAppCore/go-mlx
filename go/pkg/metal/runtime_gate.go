// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "sync/atomic"

// Gate identifies an engine runtime fast-path toggle. Gates are typed Go
// identifiers, NOT env-var strings: a model's EngineFeatures declares which it
// turns on (the declared source of truth) and tests/diagnostics flip them via
// SetRuntimeGate. A gate is NEVER read from ambient process env — that would let
// any parent process steer the engine's compute paths, an external-control
// surface (Cerberus DREAD). Each gate is one bool: a feature is on or off.
type Gate int

const (
	GateDirectGreedyToken Gate = iota
	GateNativeMLPMatVec
	GateNativeLinearMatVec
	GateNativeQ6BitstreamMatVec
	GateNativeAttentionOMatVec
	GateGenerationStream
	GateAsyncDecodePrefetch
	GateFixedSlidingCache
	GateFixedSlidingCacheBound
	GateFixedSharedMask
	GateNativeFixedSlidingAttention
	GatePagedDecodeFastConcat
	GateNativePagedAttention
	GateCacheOnlyChunkPrefill
	GateSortedExpertPrefill
	GateGatherQMMReferenceTests
	gateCount
)

// runtimeGates is the live gate state — one atomic bool per Gate, indexed by the
// typed enum. Replaces the codex env-shaped string map + the
// refreshKnownRuntimeGate string-switch + the per-gate named atomics.
var runtimeGates [gateCount]atomic.Bool

// SetRuntimeGate turns a gate on or off and returns a restore func that reverts
// it to the value it held before this call. EngineFeatures.Apply uses it to
// install a model's declaration; tests/diagnostics use it for scoped overrides.
//
//	restore := metal.SetRuntimeGate(metal.GateNativeMLPMatVec, true)
//	defer restore()
func SetRuntimeGate(gate Gate, on bool) func() {
	if gate < 0 || gate >= gateCount {
		return func() {}
	}
	previous := runtimeGates[gate].Swap(on)
	return func() { runtimeGates[gate].Store(previous) }
}

// RuntimeGateEnabled reports whether a gate is currently on.
//
//	if metal.RuntimeGateEnabled(metal.GateCacheOnlyChunkPrefill) { … }
func RuntimeGateEnabled(gate Gate) bool {
	if gate < 0 || gate >= gateCount {
		return false
	}
	return runtimeGates[gate].Load()
}

// Per-gate accessors — the read API the engine compute paths call. Each is a
// typed read of the gate array; the names are unchanged from the pre-typed gate
// system so the consuming kernels did not move.
func PagedDecodeFastConcatEnabled() bool { return runtimeGates[GatePagedDecodeFastConcat].Load() }

func NativePagedAttentionEnabled() bool { return runtimeGates[GateNativePagedAttention].Load() }

func nativeMLPMatVecRuntimeEnabled() bool { return runtimeGates[GateNativeMLPMatVec].Load() }

func nativeLinearMatVecRuntimeEnabled() bool { return runtimeGates[GateNativeLinearMatVec].Load() }

func nativeQ6BitstreamMatVecRuntimeEnabled() bool {
	return runtimeGates[GateNativeQ6BitstreamMatVec].Load()
}

func fixedSlidingCacheRuntimeEnabled() bool { return runtimeGates[GateFixedSlidingCache].Load() }

func fixedSlidingCacheBoundRuntimeEnabled() bool {
	return runtimeGates[GateFixedSlidingCacheBound].Load()
}

func fixedSharedMaskRuntimeEnabled() bool { return runtimeGates[GateFixedSharedMask].Load() }

func nativeFixedSlidingAttentionRuntimeEnabled() bool {
	return runtimeGates[GateNativeFixedSlidingAttention].Load()
}

func directGreedyTokenRuntimeEnabled() bool { return runtimeGates[GateDirectGreedyToken].Load() }

func nativeAttentionOMatVecRuntimeEnabled() bool {
	return runtimeGates[GateNativeAttentionOMatVec].Load()
}

func generationStreamRuntimeEnabled() bool { return runtimeGates[GateGenerationStream].Load() }

func asyncDecodePrefetchRuntimeEnabled() bool { return runtimeGates[GateAsyncDecodePrefetch].Load() }
