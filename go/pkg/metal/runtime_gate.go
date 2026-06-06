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
	runtimeGatePagedDecodeFastConcat                atomic.Bool
	runtimeGateNativePagedAttention                 atomic.Bool
	runtimeGateNativeMLPMatVec                      atomic.Bool
	runtimeGateNativeLinearMatVec                   atomic.Bool
	runtimeGateNativeQ6BitstreamMatVec              atomic.Bool
	runtimeGateNativeGemma4Layer                    atomic.Bool
	runtimeGateNativeGemma4MoELayer                 atomic.Bool
	runtimeGateCompiledGemma4Layer                  atomic.Bool
	runtimeGateFixedSlidingCache                    atomic.Bool
	runtimeGateFixedSlidingCacheBound               atomic.Bool
	runtimeGateFixedGemma4SharedMask                atomic.Bool
	runtimeGateNativeFixedSlidingAttention          atomic.Bool
	runtimeGateDirectGreedyToken                    atomic.Bool
	runtimeGateNativeGemma4FixedOwnerAttention      atomic.Bool
	runtimeGateNativeGemma4FixedOwnerAttentionResid atomic.Bool
	runtimeGateNativeGemma4AttentionOMatVec         atomic.Bool
	runtimeGateGenerationStream                     atomic.Bool
	runtimeGateAsyncDecodePrefetch                  atomic.Bool
)

func init() {
	refreshKnownRuntimeGates()
}

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

func RuntimeGateValue(name string) string {
	name = core.Trim(name)
	if name == "" {
		return ""
	}
	runtimeGateOverrides.RLock()
	if value, ok := runtimeGateOverrides.values[name]; ok {
		runtimeGateOverrides.RUnlock()
		return core.Trim(value)
	}
	runtimeGateOverrides.RUnlock()
	if runtimeGateIgnoresAmbientEnv(name) {
		return ""
	}
	return core.Trim(core.Env(name))
}

func runtimeGateIgnoresAmbientEnv(name string) bool {
	switch name {
	case "GO_MLX_ENABLE_FIXED_SLIDING_CACHE",
		"GO_MLX_ENABLE_FIXED_SLIDING_CACHE_BOUND",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
		"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL":
		return true
	default:
		return false
	}
}

func RuntimeGateEnabled(name string) bool {
	return RuntimeGateValue(name) == "1"
}

func refreshKnownRuntimeGates() {
	for _, name := range []string{
		"GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT",
		"GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_MLP_MATVEC",
		"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC",
		"GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER",
		"GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER",
		"GO_MLX_ENABLE_FIXED_SLIDING_CACHE",
		"GO_MLX_ENABLE_FIXED_SLIDING_CACHE_BOUND",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
		"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION",
		"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC",
		"GO_MLX_ENABLE_GENERATION_STREAM",
		"GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH",
	} {
		refreshKnownRuntimeGate(name)
	}
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
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER":
		runtimeGateNativeGemma4Layer.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER":
		runtimeGateNativeGemma4MoELayer.Store(enabled)
	case "GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER":
		runtimeGateCompiledGemma4Layer.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_SLIDING_CACHE":
		runtimeGateFixedSlidingCache.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_SLIDING_CACHE_BOUND":
		runtimeGateFixedSlidingCacheBound.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":
		runtimeGateFixedGemma4SharedMask.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION":
		runtimeGateNativeFixedSlidingAttention.Store(enabled)
	case "GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN":
		runtimeGateDirectGreedyToken.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION":
		runtimeGateNativeGemma4FixedOwnerAttention.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL":
		runtimeGateNativeGemma4FixedOwnerAttentionResid.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC":
		runtimeGateNativeGemma4AttentionOMatVec.Store(enabled)
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

func nativeGemma4LayerRuntimeEnabled() bool { return runtimeGateNativeGemma4Layer.Load() }

func nativeGemma4MoELayerRuntimeEnabled() bool { return runtimeGateNativeGemma4MoELayer.Load() }

func compiledGemma4LayerRuntimeEnabled() bool { return runtimeGateCompiledGemma4Layer.Load() }

func fixedSlidingCacheRuntimeEnabled() bool { return runtimeGateFixedSlidingCache.Load() }

func fixedSlidingCacheBoundRuntimeEnabled() bool {
	return runtimeGateFixedSlidingCacheBound.Load()
}

func fixedGemma4SharedMaskRuntimeEnabled() bool { return runtimeGateFixedGemma4SharedMask.Load() }

func nativeFixedSlidingAttentionRuntimeEnabled() bool {
	return runtimeGateNativeFixedSlidingAttention.Load()
}

func directGreedyTokenRuntimeEnabled() bool { return runtimeGateDirectGreedyToken.Load() }

func nativeGemma4FixedOwnerAttentionRuntimeEnabled() bool {
	return runtimeGateNativeGemma4FixedOwnerAttention.Load()
}

func nativeGemma4FixedOwnerAttentionResidualRuntimeEnabled() bool {
	return runtimeGateNativeGemma4FixedOwnerAttentionResid.Load()
}

func nativeGemma4AttentionOMatVecRuntimeEnabled() bool {
	return runtimeGateNativeGemma4AttentionOMatVec.Load()
}

func generationStreamRuntimeEnabled() bool { return runtimeGateGenerationStream.Load() }

func asyncDecodePrefetchRuntimeEnabled() bool { return runtimeGateAsyncDecodePrefetch.Load() }
