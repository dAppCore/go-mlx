// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The accepted fast-path set, as typed runtime Gates. This is the contract
// DefaultEngineFeatures() must reproduce — the typed replacement for the loose
// defaultGemma4FastRuntimeGates string list that used to live in package mlx.
var acceptedEngineGates = []Gate{
	GateDirectGreedyToken,
	GateNativeMLPMatVec,
	GateNativeLinearMatVec,
	GateNativeQ6BitstreamMatVec,
	GateNativeAttentionOMatVec,
	GateGenerationStream,
	GateAsyncDecodePrefetch,
}

func TestDefaultEngineFeatures_EnabledGates_MatchesAcceptedSet(t *testing.T) {
	got := DefaultEngineFeatures().EnabledGates()
	if len(got) != len(acceptedEngineGates) {
		t.Fatalf("DefaultEngineFeatures().EnabledGates() has %d entries, want %d: %v",
			len(got), len(acceptedEngineGates), got)
	}
	for i, want := range acceptedEngineGates {
		if got[i] != want {
			t.Errorf("EnabledGates()[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestEngineFeatures_EnabledGates_OmitsDisabled(t *testing.T) {
	// A bare declaration turns nothing on — EnabledGates must be empty, so a
	// model that selects nothing applies no gates (no accidental defaults).
	if got := (EngineFeatures{}).EnabledGates(); len(got) != 0 {
		t.Fatalf("zero EngineFeatures produced gates %v, want none", got)
	}
	// One field on → exactly that gate.
	got := (EngineFeatures{NativeMLPMatVec: true}).EnabledGates()
	if len(got) != 1 || got[0] != GateNativeMLPMatVec {
		t.Fatalf("single-feature EnabledGates = %v, want only GateNativeMLPMatVec", got)
	}
}

func TestEngineFeatures_EnabledGates_StableOrderFreshSlice(t *testing.T) {
	got := DefaultEngineFeatures().EnabledGates()
	if len(got) != len(acceptedEngineGates) {
		t.Fatalf("EnabledGates() len = %d, want %d: %v", len(got), len(acceptedEngineGates), got)
	}
	// Fresh slice each call — mutating the result must not leak into the next.
	got[0] = GateAsyncDecodePrefetch
	if next := DefaultEngineFeatures().EnabledGates(); next[0] != GateDirectGreedyToken {
		t.Fatalf("EnabledGates() leaked a shared slice: %v", next)
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
	const gate = GateGenerationStream
	before := RuntimeGateEnabled(gate)

	restore := (EngineFeatures{GenerationStream: true}).Apply()
	if !RuntimeGateEnabled(gate) {
		t.Fatalf("Apply() did not enable GateGenerationStream")
	}

	restore()
	if RuntimeGateEnabled(gate) != before {
		t.Fatalf("restore() left gate = %v, want %v", RuntimeGateEnabled(gate), before)
	}
}
