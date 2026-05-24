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
	runtimeGateExpertIDMatVec                       atomic.Bool
	runtimeGateExpertIDFusedActivation              atomic.Bool
	runtimeGateExpertIDUnrolledQ4                   atomic.Bool
	runtimeGateSortedExpertPrefill                  atomic.Bool
	runtimeGatePagedDecodeFastConcat                atomic.Bool
	runtimeGatePagedKVPrealloc                      atomic.Bool
	runtimeGateNativePagedAttention                 atomic.Bool
	runtimeGateNativeMLPMatVec                      atomic.Bool
	runtimeGateNativeLinearMatVec                   atomic.Bool
	runtimeGateNativeGemma4FFNResidual              atomic.Bool
	runtimeGateNativeGemma4RouterMatVec             atomic.Bool
	runtimeGateNativeGemma4RouterTopK               atomic.Bool
	runtimeGateNativeGemma4Layer                    atomic.Bool
	runtimeGateNativeGemma4MoELayer                 atomic.Bool
	runtimeGateNativeGemma4ModelGreedy              atomic.Bool
	runtimeGateCompiledGemma4Layer                  atomic.Bool
	runtimeGateFixedGemma4Cache                     atomic.Bool
	runtimeGateFixedGemma4SlidingCacheBound         atomic.Bool
	runtimeGateFixedGemma4SharedMask                atomic.Bool
	runtimeGateNativeFixedSlidingAttention          atomic.Bool
	runtimeGateDirectGreedyToken                    atomic.Bool
	runtimeGateNativeGemma4FixedOwnerAttention      atomic.Bool
	runtimeGateNativeGemma4FixedOwnerAttentionResid atomic.Bool
	runtimeGateNativeGemma4AttentionOMatVec         atomic.Bool
	runtimeGateNativeGemma4ResidualNorm             atomic.Bool
	runtimeGateGenerationStream                     atomic.Bool
	runtimeGateAsyncDecodePrefetch                  atomic.Bool
	runtimeGateGenerationClearCache                 atomic.Bool
	runtimeGateZeroCopyPagedRestore                 atomic.Bool
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
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY",
		"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND",
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
		"GO_MLX_ENABLE_EXPERT_ID_MATVEC",
		"GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION",
		"GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4",
		"GO_MLX_ENABLE_SORTED_EXPERT_PREFILL",
		"GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT",
		"GO_MLX_ENABLE_PAGED_KV_PREALLOC",
		"GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_MLP_MATVEC",
		"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY",
		"GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER",
		"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
		"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION",
		"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM",
		"GO_MLX_ENABLE_GENERATION_STREAM",
		"GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH",
		"GO_MLX_ENABLE_GENERATION_CLEAR_CACHE",
		"GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE",
	} {
		refreshKnownRuntimeGate(name)
	}
}

func refreshKnownRuntimeGate(name string) {
	enabled := RuntimeGateValue(name) == "1"
	switch name {
	case "GO_MLX_ENABLE_EXPERT_ID_MATVEC":
		runtimeGateExpertIDMatVec.Store(enabled)
	case "GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION":
		runtimeGateExpertIDFusedActivation.Store(enabled)
	case "GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4":
		runtimeGateExpertIDUnrolledQ4.Store(enabled)
	case "GO_MLX_ENABLE_SORTED_EXPERT_PREFILL":
		runtimeGateSortedExpertPrefill.Store(enabled)
	case "GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT":
		runtimeGatePagedDecodeFastConcat.Store(enabled)
	case "GO_MLX_ENABLE_PAGED_KV_PREALLOC":
		runtimeGatePagedKVPrealloc.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION":
		runtimeGateNativePagedAttention.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_MLP_MATVEC":
		runtimeGateNativeMLPMatVec.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC":
		runtimeGateNativeLinearMatVec.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL":
		runtimeGateNativeGemma4FFNResidual.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC":
		runtimeGateNativeGemma4RouterMatVec.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK":
		runtimeGateNativeGemma4RouterTopK.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER":
		runtimeGateNativeGemma4Layer.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER":
		runtimeGateNativeGemma4MoELayer.Store(enabled)
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY":
		runtimeGateNativeGemma4ModelGreedy.Store(enabled)
	case "GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER":
		runtimeGateCompiledGemma4Layer.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":
		runtimeGateFixedGemma4Cache.Store(enabled)
	case "GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":
		runtimeGateFixedGemma4SlidingCacheBound.Store(enabled)
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
	case "GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM":
		runtimeGateNativeGemma4ResidualNorm.Store(enabled)
	case "GO_MLX_ENABLE_GENERATION_STREAM":
		runtimeGateGenerationStream.Store(enabled)
	case "GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH":
		runtimeGateAsyncDecodePrefetch.Store(enabled)
	case "GO_MLX_ENABLE_GENERATION_CLEAR_CACHE":
		runtimeGateGenerationClearCache.Store(enabled)
	case "GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE":
		// The retained State path is streaming-first. Keep the legacy
		// coalescing path available for regression comparison with an
		// explicit 0, but do not require an enable flag for the production
		// zero-copy restore.
		runtimeGateZeroCopyPagedRestore.Store(RuntimeGateValue(name) != "0")
	}
}

func expertIDMatVecEnabled() bool { return runtimeGateExpertIDMatVec.Load() }

func expertIDFusedActivationEnabled() bool { return runtimeGateExpertIDFusedActivation.Load() }

func expertIDUnrolledQ4RuntimeEnabled() bool { return runtimeGateExpertIDUnrolledQ4.Load() }

func sortedExpertPrefillEnabled() bool { return runtimeGateSortedExpertPrefill.Load() }

func pagedDecodeFastConcatEnabled() bool { return runtimeGatePagedDecodeFastConcat.Load() }

func pagedKVPreallocRuntimeEnabled() bool { return runtimeGatePagedKVPrealloc.Load() }

func nativePagedAttentionEnabled() bool { return runtimeGateNativePagedAttention.Load() }

func nativeMLPMatVecRuntimeEnabled() bool { return runtimeGateNativeMLPMatVec.Load() }

func nativeLinearMatVecRuntimeEnabled() bool { return runtimeGateNativeLinearMatVec.Load() }

func nativeGemma4FFNResidualRuntimeEnabled() bool { return runtimeGateNativeGemma4FFNResidual.Load() }

func nativeGemma4RouterMatVecRuntimeEnabled() bool { return runtimeGateNativeGemma4RouterMatVec.Load() }

func nativeGemma4RouterTopKRuntimeEnabled() bool { return runtimeGateNativeGemma4RouterTopK.Load() }

func nativeGemma4LayerRuntimeEnabled() bool { return runtimeGateNativeGemma4Layer.Load() }

func nativeGemma4MoELayerRuntimeEnabled() bool { return runtimeGateNativeGemma4MoELayer.Load() }

func nativeGemma4ModelGreedyRuntimeEnabled() bool { return runtimeGateNativeGemma4ModelGreedy.Load() }

func compiledGemma4LayerRuntimeEnabled() bool { return runtimeGateCompiledGemma4Layer.Load() }

func fixedGemma4CacheRuntimeEnabled() bool { return runtimeGateFixedGemma4Cache.Load() }

func fixedGemma4SlidingCacheBoundRuntimeEnabled() bool {
	return runtimeGateFixedGemma4SlidingCacheBound.Load()
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

func nativeGemma4ResidualNormRuntimeEnabled() bool { return runtimeGateNativeGemma4ResidualNorm.Load() }

func generationStreamRuntimeEnabled() bool { return runtimeGateGenerationStream.Load() }

func asyncDecodePrefetchRuntimeEnabled() bool { return runtimeGateAsyncDecodePrefetch.Load() }

func generationClearCacheRuntimeEnabled() bool {
	return runtimeGateGenerationClearCache.Load()
}

func zeroCopyPagedRestoreRuntimeEnabled() bool {
	return runtimeGateZeroCopyPagedRestore.Load()
}
