// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"
	"sync/atomic"

	core "dappco.re/go"
)

var runtimeGateOverrides struct {
	sync.RWMutex
	values map[string]string
}

var (
	runtimeGatePagedDecodeFastConcat        atomic.Bool
	runtimeGateNativePagedAttention         atomic.Bool
	runtimeGateNativeMLPMatVec              atomic.Bool
	runtimeGateNativeLinearMatVec           atomic.Bool
	runtimeGateNativeQ6BitstreamMatVec      atomic.Bool
	runtimeGateFixedSlidingCache            atomic.Bool
	runtimeGateFixedSlidingCacheBound       atomic.Bool
	runtimeGateFixedSharedMask        atomic.Bool
	runtimeGateNativeFixedSlidingAttention  atomic.Bool
	runtimeGateDirectGreedyToken            atomic.Bool
	runtimeGateNativeAttentionOMatVec atomic.Bool
	runtimeGateGenerationStream             atomic.Bool
	runtimeGateAsyncDecodePrefetch          atomic.Bool
)

func SetRuntimeGate(name, value string) func() {
	name = core.Trim(name)
	value = core.Trim(value)
	if name == "" {
		return func() {}
	}

	runtimeGateOverrides.Lock()
	if runtimeGateOverrides.values == nil {
		runtimeGateOverrides.values = map[string]string{}
	}
	previous, hadPrevious := runtimeGateOverrides.values[name]
	if value == "" {
		delete(runtimeGateOverrides.values, name)
	} else {
		runtimeGateOverrides.values[name] = value
	}
	runtimeGateOverrides.Unlock()
	refreshKnownRuntimeGate(name)

	return func() {
		runtimeGateOverrides.Lock()
		if runtimeGateOverrides.values == nil {
			runtimeGateOverrides.values = map[string]string{}
		}
		if hadPrevious {
			runtimeGateOverrides.values[name] = previous
		} else {
			delete(runtimeGateOverrides.values, name)
		}
		runtimeGateOverrides.Unlock()
		refreshKnownRuntimeGate(name)
	}
}

// RuntimeGateValue returns a gate's value from the in-process override map only.
// It NEVER reads ambient process env: a gate that an env var could flip would let
// any parent process steer the engine's compute paths — an external-control
// surface. Gates are set solely by the model's EngineFeatures.Apply at load (the
// declared source of truth) or an explicit SetRuntimeGate (tests / diagnostics).
func RuntimeGateValue(name string) string {
	name = core.Trim(name)
	if name == "" {
		return ""
	}
	runtimeGateOverrides.RLock()
	defer runtimeGateOverrides.RUnlock()
	if value, ok := runtimeGateOverrides.values[name]; ok {
		return core.Trim(value)
	}
	return ""
}

func RuntimeGateEnabled(name string) bool {
	return RuntimeGateValue(name) == "1"
}

func refreshKnownRuntimeGate(name string) {
	enabled := RuntimeGateValue(name) == "1"
	switch name {
	case "GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT":
		runtimeGatePagedDecodeFastConcat.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION":
		runtimeGateNativePagedAttention.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_MLP_MATVEC":
		runtimeGateNativeMLPMatVec.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC":
		runtimeGateNativeLinearMatVec.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC":
		runtimeGateNativeQ6BitstreamMatVec.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_SLIDING_CACHE":
		runtimeGateFixedSlidingCache.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_SLIDING_CACHE_BOUND":
		runtimeGateFixedSlidingCacheBound.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_SHARED_MASK":
		runtimeGateFixedSharedMask.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION":
		runtimeGateNativeFixedSlidingAttention.Store(enabled)
	case "GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN":
		runtimeGateDirectGreedyToken.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_ATTENTION_O_MATVEC":
		runtimeGateNativeAttentionOMatVec.Store(enabled)
	case "GO_MLX_ENABLE_GENERATION_STREAM":
		runtimeGateGenerationStream.Store(enabled)
	case "GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH":
		runtimeGateAsyncDecodePrefetch.Store(enabled)
	}
}

func PagedDecodeFastConcatEnabled() bool { return runtimeGatePagedDecodeFastConcat.Load() }

func NativePagedAttentionEnabled() bool { return runtimeGateNativePagedAttention.Load() }

func nativeMLPMatVecRuntimeEnabled() bool { return runtimeGateNativeMLPMatVec.Load() }

func nativeLinearMatVecRuntimeEnabled() bool { return runtimeGateNativeLinearMatVec.Load() }

func nativeQ6BitstreamMatVecRuntimeEnabled() bool { return runtimeGateNativeQ6BitstreamMatVec.Load() }

func fixedSlidingCacheRuntimeEnabled() bool { return runtimeGateFixedSlidingCache.Load() }

func fixedSlidingCacheBoundRuntimeEnabled() bool {
	return runtimeGateFixedSlidingCacheBound.Load()
}

func fixedSharedMaskRuntimeEnabled() bool { return runtimeGateFixedSharedMask.Load() }

func nativeFixedSlidingAttentionRuntimeEnabled() bool {
	return runtimeGateNativeFixedSlidingAttention.Load()
}

func directGreedyTokenRuntimeEnabled() bool { return runtimeGateDirectGreedyToken.Load() }

func nativeAttentionOMatVecRuntimeEnabled() bool {
	return runtimeGateNativeAttentionOMatVec.Load()
}

func generationStreamRuntimeEnabled() bool { return runtimeGateGenerationStream.Load() }

func asyncDecodePrefetchRuntimeEnabled() bool { return runtimeGateAsyncDecodePrefetch.Load() }
