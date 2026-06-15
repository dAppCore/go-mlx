// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "testing"

// TestWeights_canonicalGemma4WeightName_Good drives the package-default
// canonicaliser (gemma4Architecture) over the two outcomes the loader's weight
// sanitiser branches on: a wrapped text tensor canonicalises to its model.
// rooted name with skip=false, and a non-text helper tensor (vision/audio
// tower) reports skip=true with an empty canonical name. The defaulted wrapper
// (canonicalGemma4WeightName) is exercised here; canonicalGemma4WeightNameAs is
// already covered through the explicit-architecture callers.
func TestWeights_canonicalGemma4WeightName_Good(t *testing.T) {
	cases := []struct {
		name      string
		input     string
		want      string
		wantSkip  bool
		wantEmpty bool
	}{
		{
			name:     "wrapped text tensor re-rooted under model.",
			input:    "model.language_model.model.layers.0.self_attn.q_proj.weight",
			want:     "model.layers.0.self_attn.q_proj.weight",
			wantSkip: false,
		},
		{
			name:     "bare text tensor takes the model. root",
			input:    "embed_tokens.weight",
			want:     "model.embed_tokens.weight",
			wantSkip: false,
		},
		{
			name:      "vision tower tensor is skipped",
			input:     "vision_tower.encoder.layers.0.mlp.fc1.weight",
			wantSkip:  true,
			wantEmpty: true,
		},
		{
			name:      "audio tower tensor is skipped",
			input:     "audio_tower.layers.0.conv.weight",
			wantSkip:  true,
			wantEmpty: true,
		},
		{
			name:      "rotary-emb helper substring is skipped",
			input:     "model.layers.0.self_attn.rotary_emb.inv_freq",
			wantSkip:  true,
			wantEmpty: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, skip := canonicalGemma4WeightName(tc.input)
			if skip != tc.wantSkip {
				t.Fatalf("canonicalGemma4WeightName(%q) skip = %v, want %v", tc.input, skip, tc.wantSkip)
			}
			if tc.wantEmpty {
				if got != "" {
					t.Fatalf("canonicalGemma4WeightName(%q) = %q, want empty on skip", tc.input, got)
				}
				return
			}
			if got != tc.want {
				t.Fatalf("canonicalGemma4WeightName(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
