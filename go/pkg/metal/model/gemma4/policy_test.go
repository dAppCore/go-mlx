// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4_test

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	"dappco.re/go/mlx/profile"
)

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestGemma4Policy_NormalizeLoRA_Good(t *testing.T) {
	// No explicit targets -> the safe q/v/o default.
	out := gemma4.NormalizeLoRA(metal.LoRAConfig{})
	if !equalStrings(out.TargetKeys, []string{"q_proj", "v_proj", "o_proj"}) {
		t.Fatalf("NormalizeLoRA(empty).TargetKeys = %v, want [q_proj v_proj o_proj]", out.TargetKeys)
	}

	// Extended target filtered out without opt-in.
	out = gemma4.NormalizeLoRA(metal.LoRAConfig{TargetKeys: []string{"q_proj", "router.proj"}})
	if !equalStrings(out.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("NormalizeLoRA(extended, no opt-in).TargetKeys = %v, want [q_proj]", out.TargetKeys)
	}

	// Extended target kept with explicit opt-in.
	out = gemma4.NormalizeLoRA(metal.LoRAConfig{TargetKeys: []string{"q_proj", "router.proj"}, AllowExtendedTargets: true})
	if !equalStrings(out.TargetKeys, []string{"q_proj", "router.proj"}) {
		t.Fatalf("NormalizeLoRA(extended, opt-in).TargetKeys = %v, want [q_proj router.proj]", out.TargetKeys)
	}

	// Explicit standard attention + MLP targets are all safe and preserved.
	standard := []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
	out = gemma4.NormalizeLoRA(metal.LoRAConfig{TargetKeys: standard})
	if !equalStrings(out.TargetKeys, standard) {
		t.Fatalf("NormalizeLoRA(standard+mlp).TargetKeys = %v, want all kept", out.TargetKeys)
	}

	// Targets supplied via TargetLayers flow through the same safe policy.
	out = gemma4.NormalizeLoRA(metal.LoRAConfig{TargetLayers: []string{"gate_proj", "per_layer_projection"}})
	if !equalStrings(out.TargetKeys, []string{"gate_proj"}) {
		t.Fatalf("NormalizeLoRA(TargetLayers extended).TargetKeys = %v, want [gate_proj]", out.TargetKeys)
	}
}

// TestGemma4Policy_RegistryWiring_Good proves the generic accessors resolve the
// Gemma-4 LoRA policy NormalizeLoRA depends on: the loader-neutral data lives in
// the registry, the model package consumes it through profile — no Gemma-4
// knowledge in the engine.
func TestGemma4Policy_RegistryWiring_Good(t *testing.T) {
	for _, architecture := range []string{"gemma4", "gemma4_text", "gemma4_unified"} {
		if !equalStrings(profile.DefaultLoRATargets(architecture), []string{"q_proj", "v_proj", "o_proj"}) {
			t.Fatalf("profile.DefaultLoRATargets(%q) = %v, want [q_proj v_proj o_proj]", architecture, profile.DefaultLoRATargets(architecture))
		}
		if path, ok := profile.LoRATargetPath(architecture, "q_proj"); !ok || path != "self_attn.q_proj" {
			t.Fatalf("profile.LoRATargetPath(%q, q_proj) = %q, %v; want self_attn.q_proj, true", architecture, path, ok)
		}
		if !profile.SafeLoRATarget(architecture, "q_proj") {
			t.Fatalf("profile.SafeLoRATarget(%q, q_proj) = false, want true", architecture)
		}
		if profile.SafeLoRATarget(architecture, "router.proj") {
			t.Fatalf("profile.SafeLoRATarget(%q, router.proj) = true, want false (extended)", architecture)
		}
	}
}
