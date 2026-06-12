// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestParseDiffusionConfigExtras_IntAndListEOS_Good(t *testing.T) {
	canvas, eos := parseDiffusionConfigExtras([]byte(`{"canvas_length":256,"eos_token_id":1}`))
	if canvas != 256 || len(eos) != 1 || eos[0] != 1 {
		t.Fatalf("int eos: canvas=%d eos=%v, want 256/[1]", canvas, eos)
	}
	canvas, eos = parseDiffusionConfigExtras([]byte(`{"canvas_length":64,"eos_token_id":[1,106]}`))
	if canvas != 64 || len(eos) != 2 || eos[0] != 1 || eos[1] != 106 {
		t.Fatalf("list eos: canvas=%d eos=%v, want 64/[1 106]", canvas, eos)
	}
}

func TestParseDiffusionConfigExtras_MissingFields_Bad(t *testing.T) {
	canvas, eos := parseDiffusionConfigExtras([]byte(`{}`))
	if canvas != 0 || eos != nil {
		t.Fatalf("empty config: canvas=%d eos=%v, want zero values", canvas, eos)
	}
	canvas, eos = parseDiffusionConfigExtras([]byte(`not json`))
	if canvas != 0 || eos != nil {
		t.Fatalf("invalid json: canvas=%d eos=%v, want zero values", canvas, eos)
	}
}

func TestExtractDiffusionEncoderScalars_BothNameForms_Good(t *testing.T) {
	bare := metal.FromValues([]float32{1}, 1)
	suffixed := metal.FromValues([]float32{2}, 1)
	defer metal.Free(bare, suffixed)
	raw := map[string]*metal.Array{
		"model.encoder.language_model.layers.0.layer_scalar":        bare,
		"model.encoder.language_model.layers.1.layer_scalar.weight": suffixed,
	}

	scalars := extractDiffusionEncoderScalars(raw, 2)

	if len(scalars) != 2 || scalars[0] != bare || scalars[1] != suffixed {
		t.Fatalf("scalars = %v, want both name forms extracted in layer order", scalars)
	}
	if len(raw) != 0 {
		t.Fatalf("raw map = %v, want extracted entries deleted (trunk sanitize must not free them)", raw)
	}
}

func TestExtractDiffusionEncoderScalars_PartialSet_Ugly(t *testing.T) {
	only := metal.FromValues([]float32{3}, 1)
	defer metal.Free(only)
	raw := map[string]*metal.Array{
		"model.encoder.language_model.layers.1.layer_scalar": only,
	}

	scalars := extractDiffusionEncoderScalars(raw, 3)

	// The loader fail-louds on a count mismatch; the extractor itself
	// compacts to the present entries.
	if len(scalars) != 1 || scalars[0] != only {
		t.Fatalf("scalars = %v, want the single present scalar compacted", scalars)
	}
}
