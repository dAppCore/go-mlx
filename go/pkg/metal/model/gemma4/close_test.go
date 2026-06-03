// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin closeGemma4's nil/partial-layer cleanup. They moved here from
// package metal's close_test.go with the Gemma4Model type. The metal-resident
// per-architecture close helpers (closeGemma/closeQwen3) keep their own
// nil-safety test in metal's close_test.go.

func TestClose_CloseGemma4_NilModel_Ugly(t *testing.T) {
	coverageTokens := "CloseGemma4 NilModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeGemma4(nil) panicked: %v", recovered)
		}
	}()
	closeGemma4(nil)
}

// TestClose_CloseGemma4_PartialLayers_Ugly guards Mantis #1829: when a Metal
// op panics mid-build, m.Layers is allocated to full length but only partly
// populated, leaving nil layer entries. Cleanup must skip them rather than
// nil-deref layer.compiledNativeOwnerDecode and bury the original failure.
func TestClose_CloseGemma4_PartialLayers_Ugly(t *testing.T) {
	coverageTokens := "CloseGemma4 PartialLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeGemma4 with nil layer panicked: %v", recovered)
		}
	}()

	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	metal.Materialize(embedW, normW)

	m := &Gemma4Model{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Norm:        &metal.RMSNormModule{Weight: normW},
		// Pre-allocated like LoadGemma4 does, but only the first slot is
		// nil — modelling a build that panicked before populating layer 0.
		Layers: make([]*Gemma4DecoderLayer, 3),
	}

	closeGemma4(m)

	if embedW.Valid() {
		t.Error("embed weight should be freed despite nil layers")
	}
	if normW.Valid() {
		t.Error("norm weight should be freed despite nil layers")
	}
}
