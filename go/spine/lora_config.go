// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// LoRAConfig specifies which layers to apply LoRA to and with what parameters.
type LoRAConfig struct {
	Rank                 int
	Alpha                float32
	Scale                float32
	TargetKeys           []string
	TargetLayers         []string
	Lambda               float32
	DType                metal.DType
	AllowExtendedTargets bool
	ProbeSink            probe.Sink
}

// DefaultLoRAConfig returns the standard LoRA configuration for LLM fine-tuning.
//
//	config := spine.DefaultLoRAConfig() // rank=8, alpha=16, targets=[q_proj, v_proj]
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfigFromMetal(metal.DefaultLoRAConfig())
}

// ToMetalLoRAConfig shuffles a LoRAConfig into the metal package equivalent.
func ToMetalLoRAConfig(cfg LoRAConfig) metal.LoRAConfig {
	// Build the metal-side struct without the SliceClone calls inline —
	// callers commonly leave TargetKeys/TargetLayers nil so the empty
	// branch skips the slices.Clone generic dispatch and only the
	// populated path pays the defensive copy.
	out := metal.LoRAConfig{
		Rank:                 cfg.Rank,
		Alpha:                cfg.Alpha,
		Scale:                cfg.Scale,
		Lambda:               cfg.Lambda,
		DType:                cfg.DType,
		AllowExtendedTargets: cfg.AllowExtendedTargets,
		ProbeSink:            ToMetalProbeSink(cfg.ProbeSink),
	}
	if len(cfg.TargetKeys) > 0 {
		out.TargetKeys = core.SliceClone(cfg.TargetKeys)
	}
	if len(cfg.TargetLayers) > 0 {
		out.TargetLayers = core.SliceClone(cfg.TargetLayers)
	}
	return out
}

// LoRAConfigFromMetal shuffles a metal LoRA config back into the spine form.
// The metal-side ProbeSink is not carried back — callers re-attach their own.
func LoRAConfigFromMetal(cfg metal.LoRAConfig) LoRAConfig {
	// Mirror ToMetalLoRAConfig: guard each SliceClone behind a len>0
	// check so the no-overrides branch (the typical adapter shape)
	// pays only a nil-comparison instead of slices.Clone's generic
	// dispatch.
	out := LoRAConfig{
		Rank:                 cfg.Rank,
		Alpha:                cfg.Alpha,
		Scale:                cfg.Scale,
		Lambda:               cfg.Lambda,
		DType:                cfg.DType,
		AllowExtendedTargets: cfg.AllowExtendedTargets,
	}
	if len(cfg.TargetKeys) > 0 {
		out.TargetKeys = core.SliceClone(cfg.TargetKeys)
	}
	if len(cfg.TargetLayers) > 0 {
		out.TargetLayers = core.SliceClone(cfg.TargetLayers)
	}
	return out
}
