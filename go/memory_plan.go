// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/model/minimax/m2"
)

// MemoryGiB is the number of bytes in a gibibyte.
const MemoryGiB = memory.GiB

// Legacy aliases — the canonical memory planner lives at
// dappco.re/go/mlx/memory/. mlx-root callers keep their existing
// Memory* + KVCache* + ExpertResidency* surface via these aliases.
type (
	MemoryClass   = memory.Class
	KVCachePolicy = memory.KVCachePolicy
	KVCacheMode   = memory.KVCacheMode
	MemoryPlan    = memory.Plan
)

// Memory class constants forwarded from the memory package.
const (
	MemoryClassUnknown    = memory.ClassUnknown
	MemoryClassApple16GB  = memory.ClassApple16GB
	MemoryClassApple24GB  = memory.ClassApple24GB
	MemoryClassApple32GB  = memory.ClassApple32GB
	MemoryClassApple64GB  = memory.ClassApple64GB
	MemoryClassApple96GB  = memory.ClassApple96GB
	MemoryClassApple128GB = memory.ClassApple128GB
)

// KV cache policy constants forwarded from the memory package.
const (
	KVCacheDefault  = memory.KVCacheDefault
	KVCacheRotating = memory.KVCacheRotating
	KVCacheFull     = memory.KVCacheFull
)

// KV cache mode constants forwarded from the memory package.
const (
	KVCacheModeDefault = memory.KVCacheModeDefault
	KVCacheModeFP16    = memory.KVCacheModeFP16
	KVCacheModeQ8      = memory.KVCacheModeQ8
	KVCacheModeKQ8VQ4  = memory.KVCacheModeKQ8VQ4
	KVCacheModePaged   = memory.KVCacheModePaged
)

// MemoryPlanInput supplies measured hardware and optional model metadata.
// Carries mlx-shaped DeviceInfo + ModelInfo at the boundary; PlanMemory
// converts to memory.Input before delegating.
type MemoryPlanInput struct {
	Device    DeviceInfo
	Pack      *mp.ModelPack
	ModelInfo *ModelInfo
}

// PlanMemory chooses opinionated local inference settings from measured
// memory. Calls the generic planner, then layers MiniMax-M2-specific
// expert-residency and forward-skeleton hints on top.
//
//	plan := mlx.PlanMemory(mlx.MemoryPlanInput{Device: dev, Pack: &pack})
func PlanMemory(input MemoryPlanInput) MemoryPlan {
	plan := memory.NewPlan(memory.Input{
		Device:    deviceInfoToMemory(input.Device),
		Pack:      input.Pack,
		ModelInfo: modelInfoPtrToMemory(input.ModelInfo),
	})
	if input.Pack != nil {
		if skel, _ := input.Pack.MiniMaxM2LayerSkeleton.(*m2.LayerForwardSkeleton); skel != nil {
			plan.ModelForwardSkeletonValidated = true
			plan.ModelForwardSkeletonBytes = skel.EstimatedBytes()
			plan.Notes = append(plan.Notes, "MiniMax M2 first-layer tensor skeleton validated from safetensors metadata")
		}
		if mm, _ := input.Pack.MiniMaxM2.(*m2.TensorPlan); mm != nil {
			plan.ExpertResidency = m2.PlanResidency(*mm, plan, nil)
			plan.Notes = append(plan.Notes, "MiniMax M2 lazy expert residency enabled by memory planner")
		}
	}
	return plan
}

func deviceInfoToMemory(info DeviceInfo) memory.DeviceInfo {
	return memory.DeviceInfo{
		Architecture:                 info.Architecture,
		MaxBufferLength:              info.MaxBufferLength,
		MaxRecommendedWorkingSetSize: info.MaxRecommendedWorkingSetSize,
		MemorySize:                   info.MemorySize,
	}
}

func modelInfoPtrToMemory(info *ModelInfo) *memory.ModelInfo {
	if info == nil {
		return nil
	}
	return &memory.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}

// minPositive returns the smaller of a and b, treating non-positive as
// "unset" (the other operand wins). Retained as a private mlx-root
// helper for callers (small_model_smoke.go) that referenced the old
// in-package name.
func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

// maxPositive returns the larger of a and b. Retained as a private
// mlx-root helper for callers (small_model_smoke.go) that referenced
// the old in-package name.
func maxPositive(a, b int) int {
	if a > b {
		return a
	}
	return b
}

var memoryPlannerDeviceInfo = safeRuntimeDeviceInfo

func applyMemoryPlanToLoadConfig(modelPath string, cfg LoadConfig) LoadConfig {
	var plan MemoryPlan
	if cfg.MemoryPlan != nil {
		plan = *cfg.MemoryPlan
	} else if cfg.AutoMemoryPlan {
		var pack *mp.ModelPack
		if inspected, err := InspectModelPack(modelPath, mp.WithPackRequireChatTemplate(false)); err == nil {
			pack = &inspected
		}
		plan = PlanMemory(MemoryPlanInput{
			Device: memoryPlannerDeviceInfo(),
			Pack:   pack,
		})
	} else {
		return cfg
	}

	cfg.MemoryPlan = &plan
	if plan.ContextLength > 0 && (cfg.ContextLength == 0 || cfg.ContextLength == DefaultLocalContextLength) {
		cfg.ContextLength = plan.ContextLength
	}
	if plan.ParallelSlots > 0 && (cfg.ParallelSlots == 0 || cfg.ParallelSlots == DefaultLocalParallelSlots) {
		cfg.ParallelSlots = plan.ParallelSlots
	}
	if !plan.PromptCache {
		cfg.PromptCache = false
	} else if plan.PromptCacheMinTokens > 0 && (cfg.PromptCacheMinTokens == 0 || cfg.PromptCacheMinTokens == DefaultPromptCacheMinTokens) {
		cfg.PromptCacheMinTokens = plan.PromptCacheMinTokens
	}
	if cfg.CachePolicy == "" {
		cfg.CachePolicy = plan.CachePolicy
	}
	if cfg.CacheMode == "" {
		cfg.CacheMode = plan.CacheMode
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = plan.BatchSize
	}
	if cfg.PrefillChunkSize == 0 {
		cfg.PrefillChunkSize = plan.PrefillChunkSize
	}
	if cfg.ExpectedQuantization == 0 {
		cfg.ExpectedQuantization = plan.PreferredQuantization
	}
	if cfg.MemoryLimitBytes == 0 {
		cfg.MemoryLimitBytes = plan.MemoryLimitBytes
	}
	if cfg.CacheLimitBytes == 0 {
		cfg.CacheLimitBytes = plan.CacheLimitBytes
	}
	if cfg.WiredLimitBytes == 0 {
		cfg.WiredLimitBytes = plan.WiredLimitBytes
	}
	return cfg
}
