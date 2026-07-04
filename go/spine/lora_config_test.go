// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/inference/probe"
)

// Tests for the LoRAConfig <-> metal.LoRAConfig shufflers. Field copy is
// tautological, so the behaviour worth asserting is: the defensive
// SliceClone on TargetKeys/TargetLayers (mutating the source must not
// reach the copy), the len>0 guard (nil slices come out nil, not cloned
// empties), and that LoRAConfigFromMetal deliberately drops the metal
// ProbeSink (the documented one-way contract).

// loraSpyProbeSink is a do-nothing probe.Sink used only as a non-nil
// pointer marker on the spine side.
type loraSpyProbeSink struct{}

func (loraSpyProbeSink) EmitProbe(probe.Event) {}

// --- DefaultLoRAConfig ---

func TestLoraConfig_DefaultLoRAConfig_Good(t *testing.T) {
	// Documented contract from the doc comment: rank=8, alpha=16, the two
	// standard projection targets. Sourced from metal.DefaultLoRAConfig.
	cfg := DefaultLoRAConfig()
	if cfg.Rank != 8 || cfg.Alpha != 16 {
		t.Fatalf("Default = rank %d alpha %v, want 8/16", cfg.Rank, cfg.Alpha)
	}
	if len(cfg.TargetKeys) != 2 || cfg.TargetKeys[0] != "q_proj" || cfg.TargetKeys[1] != "v_proj" {
		t.Fatalf("TargetKeys = %v, want [q_proj v_proj]", cfg.TargetKeys)
	}
	if len(cfg.TargetLayers) != 2 || cfg.TargetLayers[0] != "q_proj" {
		t.Fatalf("TargetLayers = %v, want [q_proj v_proj]", cfg.TargetLayers)
	}
}

// --- ToMetalLoRAConfig ---

func TestLoraConfig_ToMetalLoRAConfig_Good(t *testing.T) {
	// Populated slices must round into the metal struct, the ProbeSink
	// must be wrapped (non-nil in → non-nil out), and the clone must be
	// independent: mutating the source slice does not reach the copy.
	src := LoRAConfig{
		Rank:                 4,
		Alpha:                8,
		Scale:                1.5,
		Lambda:               0.1,
		DType:                metal.DTypeFloat32,
		AllowExtendedTargets: true,
		TargetKeys:           []string{"q_proj"},
		TargetLayers:         []string{"v_proj"},
		ProbeSink:            loraSpyProbeSink{},
	}
	got := ToMetalLoRAConfig(src)
	if got.Rank != 4 || got.Alpha != 8 || got.Scale != 1.5 || got.Lambda != 0.1 {
		t.Fatalf("metal cfg scalars = %+v, want rank4/alpha8/scale1.5/lambda0.1", got)
	}
	if !got.AllowExtendedTargets || got.DType != metal.DTypeFloat32 {
		t.Fatalf("AllowExtendedTargets=%v DType=%v, want true/float32", got.AllowExtendedTargets, got.DType)
	}
	if got.ProbeSink == nil {
		t.Fatal("non-nil ProbeSink dropped by ToMetalLoRAConfig")
	}
	if len(got.TargetKeys) != 1 || got.TargetKeys[0] != "q_proj" {
		t.Fatalf("TargetKeys = %v, want [q_proj]", got.TargetKeys)
	}
	// Clone independence: mutate the source, the copy must not move.
	src.TargetKeys[0] = "MUTATED"
	if got.TargetKeys[0] != "q_proj" {
		t.Fatal("ToMetalLoRAConfig did not defensively clone TargetKeys")
	}
}

func TestLoraConfig_ToMetalLoRAConfig_Bad(t *testing.T) {
	// Nil slices + nil sink → the len>0 guard skips the clone, so the
	// outputs stay nil (not zero-length non-nil), and no metal sink is
	// manufactured.
	got := ToMetalLoRAConfig(LoRAConfig{Rank: 2})
	if got.TargetKeys != nil || got.TargetLayers != nil {
		t.Fatalf("nil-in produced non-nil slices: keys=%v layers=%v", got.TargetKeys, got.TargetLayers)
	}
	if got.ProbeSink != nil {
		t.Fatal("nil ProbeSink produced a non-nil metal sink")
	}
}

func TestLoraConfig_ToMetalLoRAConfig_Ugly(t *testing.T) {
	// Only one of the two slices populated — the guard must clone the
	// populated one and leave the empty one nil (independent branches).
	got := ToMetalLoRAConfig(LoRAConfig{TargetLayers: []string{"o_proj"}})
	if got.TargetKeys != nil {
		t.Fatalf("TargetKeys = %v, want nil (empty branch)", got.TargetKeys)
	}
	if len(got.TargetLayers) != 1 || got.TargetLayers[0] != "o_proj" {
		t.Fatalf("TargetLayers = %v, want [o_proj]", got.TargetLayers)
	}
}

// --- LoRAConfigFromMetal ---

func TestLoraConfig_LoRAConfigFromMetal_Good(t *testing.T) {
	// Round the metal struct back, clone independence on the populated
	// slice, and — the documented one-way contract — the metal-side
	// ProbeSink is NOT carried back.
	src := metal.LoRAConfig{
		Rank:                 6,
		Alpha:                12,
		Scale:                2,
		Lambda:               0.2,
		DType:                metal.DTypeFloat32,
		AllowExtendedTargets: true,
		TargetKeys:           []string{"k_proj"},
		ProbeSink:            metalProbeSinkAdapter{sink: loraSpyProbeSink{}},
	}
	got := LoRAConfigFromMetal(src)
	if got.Rank != 6 || got.Alpha != 12 || got.Lambda != 0.2 {
		t.Fatalf("spine cfg = %+v, want rank6/alpha12/lambda0.2", got)
	}
	if got.ProbeSink != nil {
		t.Fatal("LoRAConfigFromMetal carried the metal ProbeSink back; doc says it must not")
	}
	if len(got.TargetKeys) != 1 || got.TargetKeys[0] != "k_proj" {
		t.Fatalf("TargetKeys = %v, want [k_proj]", got.TargetKeys)
	}
	src.TargetKeys[0] = "MUTATED"
	if got.TargetKeys[0] != "k_proj" {
		t.Fatal("LoRAConfigFromMetal did not defensively clone TargetKeys")
	}
}

func TestLoraConfig_LoRAConfigFromMetal_Bad(t *testing.T) {
	// Nil slices → nil out via the len>0 guard.
	got := LoRAConfigFromMetal(metal.LoRAConfig{Rank: 1})
	if got.TargetKeys != nil || got.TargetLayers != nil {
		t.Fatalf("nil-in produced non-nil slices: keys=%v layers=%v", got.TargetKeys, got.TargetLayers)
	}
}
