// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/model/minimax/m2"
	mp "dappco.re/go/mlx/pack"
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
func PlanMemory(input MemoryPlanInput) memory.Plan {
	plan := memory.NewPlan(memory.Input{
		Device:    deviceInfoToMemory(input.Device),
		Pack:      input.Pack,
		ModelInfo: modelInfoPtrToMemory(input.ModelInfo),
	})
	if input.Pack == nil {
		return plan
	}
	skel, _ := input.Pack.MiniMaxM2LayerSkeleton.(*m2.LayerForwardSkeleton)
	mm, _ := input.Pack.MiniMaxM2.(*m2.TensorPlan)
	if skel == nil && mm == nil {
		return plan
	}
	// At least one M2 note will be appended below; grow Notes once now
	// so each append lands in spare capacity instead of triggering a
	// per-append heap copy (NewPlan returns Notes sized at its own len).
	extra := 0
	if skel != nil {
		extra++
	}
	if mm != nil {
		extra++
	}
	if cap(plan.Notes)-len(plan.Notes) < extra {
		grown := make([]string, len(plan.Notes), len(plan.Notes)+extra)
		copy(grown, plan.Notes)
		plan.Notes = grown
	}
	if skel != nil {
		plan.ModelForwardSkeletonValidated = true
		plan.ModelForwardSkeletonBytes = skel.EstimatedBytes()
		plan.Notes = append(plan.Notes, "MiniMax M2 first-layer tensor skeleton validated from safetensors metadata")
	}
	if mm != nil {
		plan.ExpertResidency = m2.PlanResidency(*mm, plan, nil)
		plan.Notes = append(plan.Notes, "MiniMax M2 lazy expert residency enabled by memory planner")
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
		NumKVHeads:    info.NumKVHeads,
		HeadDim:       info.HeadDim,
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
	// Caller-supplied plan path is the typical inference re-entry: the
	// model was loaded once, the plan was persisted, and every later
	// call reuses it. Read directly through the pointer instead of
	// dereferencing into a stack value (memory.Plan is ~300B with
	// embedded ExpertResidencyPlan, so the value-copy was a measurable
	// per-call overhead on the LoadModel hot path).
	var plan *memory.Plan
	autoInspectedPack := false
	switch {
	case cfg.MemoryPlan != nil:
		plan = cfg.MemoryPlan
	case cfg.AutoMemoryPlan:
		var pack *mp.ModelPack
		if inspected, err := model.Inspect(modelPath, mp.WithPackRequireChatTemplate(false)); err == nil {
			pack = &inspected
			autoInspectedPack = true
		}
		built := PlanMemory(MemoryPlanInput{
			Device: memoryPlannerDeviceInfo(),
			Pack:   pack,
		})
		// Only when WE built the plan does cfg.MemoryPlan need an
		// updated pointer; the caller-supplied case already has it.
		cfg.MemoryPlan = &built
		plan = &built
	default:
		return cfg
	}
	if plan.ContextLength > 0 && !cfg.contextLengthExplicit && cfg.ContextLength == 0 {
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
		switch {
		case plan.ModelQuantization > 0:
			cfg.ExpectedQuantization = plan.ModelQuantization
		case autoInspectedPack:
			// Source snapshots such as google/gemma-4-E2B-it are dense
			// BF16/FP16 packs, not q6 derivatives. Keep q6 in the
			// selectable policy, but do not report it as the expected
			// quantisation for an inspected unquantised model.
		default:
			cfg.ExpectedQuantization = plan.PreferredQuantization
		}
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
