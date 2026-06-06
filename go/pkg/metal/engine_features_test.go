// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The accepted fast-path set, as runtime-gate names. This is the contract
// DefaultEngineFeatures() must reproduce — the typed replacement for the
// loose defaultGemma4FastRuntimeGates string list in package mlx.
var acceptedEngineGateNames = []string{
	"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN",
	"GO_MLX_ENABLE_NATIVE_MLP_MATVEC",
	"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC",
	"GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC",
	"GO_MLX_ENABLE_NATIVE_ATTENTION_O_MATVEC",
	"GO_MLX_ENABLE_GENERATION_STREAM",
	"GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH",
}

func TestDefaultEngineFeatures_GateValues_MatchesAcceptedSet(t *testing.T) {
	got := DefaultEngineFeatures().GateValues()
	if len(got) != len(acceptedEngineGateNames) {
		t.Fatalf("DefaultEngineFeatures().GateValues() has %d entries, want %d: %v",
			len(got), len(acceptedEngineGateNames), got)
	}
	for _, name := range acceptedEngineGateNames {
		if got[name] != "1" {
			t.Errorf("gate %s = %q, want %q", name, got[name], "1")
		}
	}
}

func TestEngineFeatures_GateValues_OmitsDisabled(t *testing.T) {
	// A bare declaration turns nothing on — GateValues must be empty, so a
	// model that selects nothing applies no gates (no accidental defaults).
	if got := (EngineFeatures{}).GateValues(); len(got) != 0 {
		t.Fatalf("zero EngineFeatures produced gates %v, want none", got)
	}
	// One field on → exactly that gate.
	got := (EngineFeatures{NativeMLPMatVec: true}).GateValues()
	if len(got) != 1 || got["GO_MLX_ENABLE_NATIVE_MLP_MATVEC"] != "1" {
		t.Fatalf("single-feature GateValues = %v, want only NATIVE_MLP_MATVEC", got)
	}
}

func TestEngineFeatures_GateNames_StableOrderAcceptedSet(t *testing.T) {
	got := DefaultEngineFeatures().GateNames()
	if len(got) != len(acceptedEngineGateNames) {
		t.Fatalf("GateNames() len = %d, want %d: %v", len(got), len(acceptedEngineGateNames), got)
	}
	for i := range acceptedEngineGateNames {
		if got[i] != acceptedEngineGateNames[i] {
			t.Errorf("GateNames()[%d] = %q, want %q", i, got[i], acceptedEngineGateNames[i])
		}
	}
	// Fresh slice each call — mutating the result must not leak into the next.
	got[0] = "mutated"
	if next := DefaultEngineFeatures().GateNames(); next[0] == "mutated" {
		t.Fatalf("GateNames() leaked a shared slice: %v", next)
	}
}

type fakeEngineFeaturesModel struct{ ef EngineFeatures }

func (f fakeEngineFeaturesModel) EngineFeatures() EngineFeatures { return f.ef }

func TestEngineFeaturesFor_UsesModelDeclaration(t *testing.T) {
	want := EngineFeatures{NativeMLPMatVec: true, GenerationStream: true}
	if got := EngineFeaturesFor(fakeEngineFeaturesModel{want}); got != want {
		t.Fatalf("EngineFeaturesFor(declaring model) = %+v, want %+v", got, want)
	}
}

func TestEngineFeaturesFor_FallsBackToDefault(t *testing.T) {
	if got := EngineFeaturesFor(struct{}{}); got != DefaultEngineFeatures() {
		t.Fatalf("EngineFeaturesFor(non-declaring) = %+v, want default", got)
	}
}

func TestEngineFeatures_Apply_EnablesThenRestores(t *testing.T) {
	const gate = "GO_MLX_ENABLE_GENERATION_STREAM"
	before := RuntimeGateEnabled(gate)

	restore := (EngineFeatures{GenerationStream: true}).Apply()
	if !RuntimeGateEnabled(gate) {
		t.Fatalf("Apply() did not enable %s", gate)
	}

	restore()
	if RuntimeGateEnabled(gate) != before {
		t.Fatalf("restore() left %s = %v, want %v", gate, RuntimeGateEnabled(gate), before)
	}
}
