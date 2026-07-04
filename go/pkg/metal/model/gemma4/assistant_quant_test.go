// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestGemma4Assistant_LinearInputMatches_Good covers the quant-aware input-dim
// check that lets QAT (quantized) drafters load: a 4-bit weight packs its input
// dim into uint32 words, so the stored dim is the logical dim divided by the
// pack factor (10752 -> 1344). A bf16 weight stores the logical dim verbatim.
func TestGemma4Assistant_LinearInputMatches_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime")
	}

	// bf16 (no scales): only an exact match is accepted.
	bf16 := &metal.Linear{}
	if !gemma4AssistantLinearInputMatches(bf16, 10752, 10752) {
		t.Errorf("bf16 exact dim should match")
	}
	if gemma4AssistantLinearInputMatches(bf16, 1344, 10752) {
		t.Errorf("bf16 packed-looking dim must NOT match (no quantization)")
	}

	// q4 legacy packing: packedIn = inDim / (32/bits) = 10752 / 8 = 1344.
	scales := metal.FromValue(float32(1))
	defer metal.Free(scales)
	q4 := &metal.Linear{Scales: scales, Bits: 4}
	if !gemma4AssistantLinearInputMatches(q4, 1344, 10752) {
		t.Errorf("q4 packed input dim 1344 should match logical 10752")
	}
	if !gemma4AssistantLinearInputMatches(q4, 10752, 10752) {
		t.Errorf("q4 already-unpacked dim should still match")
	}
	if gemma4AssistantLinearInputMatches(q4, 1000, 10752) {
		t.Errorf("q4 wrong dim 1000 must not match 10752")
	}

	// q4 bitstream packing: packedIn = (inDim*bits + 31) / 32 for a non-
	// pack-factor-divisible dim (1025*4+31)/32 = 129.
	if !gemma4AssistantLinearInputMatches(q4, 129, 1025) {
		t.Errorf("q4 bitstream-packed input dim 129 should match logical 1025")
	}
}
