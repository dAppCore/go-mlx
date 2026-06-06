// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/profile"
)

// gemma4Architecture is the registry key for the Gemma-4 target family. The
// loader-neutral LoRA target policy (default set, safe-target rule, canonical
// paths) lives in the profile registry; this package holds only the metal-typed
// normalisation that consumes it.
const gemma4Architecture = "gemma4"

// NormalizeLoRA applies the base LoRA defaults, then the Gemma-4 target policy
// from the architecture registry: an unspecified target set falls back to the
// safe q/v/o default, and extended router/per-layer targets are filtered out
// unless explicitly opted in via AllowExtendedTargets. Named NormalizeLoRA
// rather than NormalizeLoRAConfig to avoid colliding with metal's base entry
// point under the dot-import in decode_kernels_test.go.
func NormalizeLoRA(cfg metal.LoRAConfig) metal.LoRAConfig {
	explicitTargets := len(cfg.TargetKeys) > 0 || len(cfg.TargetLayers) > 0
	cfg = metal.NormalizeLoRAConfig(cfg)
	if !explicitTargets {
		cfg.TargetKeys = profile.DefaultLoRATargets(gemma4Architecture)
		cfg.TargetLayers = append([]string(nil), cfg.TargetKeys...)
	}
	if cfg.AllowExtendedTargets {
		return cfg
	}
	targets := make([]string, 0, len(cfg.TargetKeys))
	skipped := make([]string, 0)
	for _, target := range cfg.TargetKeys {
		if profile.SafeLoRATarget(gemma4Architecture, target) {
			targets = append(targets, target)
			continue
		}
		skipped = append(skipped, target)
	}
	if len(skipped) > 0 {
		core.Warn("gemma4 lora: skipping extended targets without opt-in",
			"targets", skipped,
			"set", "AllowExtendedTargets",
		)
	}
	cfg.TargetKeys = targets
	cfg.TargetLayers = append([]string(nil), targets...)
	return cfg
}
