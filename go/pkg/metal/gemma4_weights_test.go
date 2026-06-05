// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestGemma4CanonicalWeightName_GoodBad(t *testing.T) {
	tests := map[string]struct {
		name string
		want string
		ok   bool
	}{
		"prefixed text layer": {
			name: "language_model.model.layers.0.self_attn.q_proj.weight",
			want: "model.layers.0.self_attn.q_proj.weight",
			ok:   true,
		},
		"repeated wrapper": {
			name: "model.language_model.model.model.layers.1.mlp.down_proj.scales",
			want: "model.layers.1.mlp.down_proj.scales",
			ok:   true,
		},
		"already canonical": {
			name: "model.layers.2.self_attn.o_proj.weight",
			want: "model.layers.2.self_attn.o_proj.weight",
			ok:   true,
		},
		"input quant metadata skipped": {
			name: "language_model.model.layers.0.self_attn.q_proj.input_max",
			ok:   false,
		},
		"vision skipped": {
			name: "model.vision_tower.patch_embedding.weight",
			ok:   false,
		},
		"audio skipped": {
			name: "language_model.embed_audio.embedding_projection.weight",
			ok:   false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, ok := Gemma4CanonicalWeightName(tc.name)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("Gemma4CanonicalWeightName(%q) = %q, %v; want %q, %v", tc.name, got, ok, tc.want, tc.ok)
			}
		})
	}
}
