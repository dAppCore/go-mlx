// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

// classifyEquivalenceCases enumerates the tensor-name shapes covered by
// the projection-family classifier. Each shape exercises the byte-walk
// branches that distinguish q/k/v/o/out/up/down/gate as well as the
// reject paths (no leading '.', no anchor at all, mixed cases).
var classifyEquivalenceCases = []string{
	// Attention paths through the single-char discriminator.
	"model.layers.0.self_attn.q_proj.weight",
	"model.layers.5.self_attn.k_proj.weight",
	"model.layers.7.self_attn.v_proj.weight",
	"model.layers.12.self_attn.o_proj.weight",
	"model.layers.12.attn.q_proj.bias",
	// Attention via .out_proj.
	"model.layers.0.attn.out_proj.weight",
	"transformer.h.5.attn.out_proj.weight",
	// FFN via .up_proj. / .down_proj.
	"model.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight",
	// Gate via .gate_proj. and .gate.
	"model.layers.0.mlp.gate_proj.weight",
	"model.layers.0.gate.weight",
	// Reject paths — wrong leading byte or no leading '.'.
	"foo_proj.weight",
	"q_proj.weight",     // no leading "."
	"down_proj.weight",  // no leading "."
	"out_proj.weight",   // no leading "."
	"_proj.weight",      // anchor at start
	".x_proj.weight",    // unknown discriminator
	"model.embed_tokens.weight",
	"model.layers.0.input_layernorm.weight",
	"lm_head.weight",
	"router.weight",
	// Edge: anchor in the middle but not preceded by valid prefix.
	"foo_bar_proj.weight",
}

func TestModelSliceClassify_ProjectionFamilyEquivalence(t *testing.T) {
	for _, name := range classifyEquivalenceCases {
		fam := modelSliceProjectionFamily(name)

		// Cross-check projAttention against the legacy 5-projection chain.
		wantAttn := false
		if core.Contains(name, "_proj.") {
			wantAttn = modelSliceHasProjection(name, "q_proj") ||
				modelSliceHasProjection(name, "k_proj") ||
				modelSliceHasProjection(name, "v_proj") ||
				modelSliceHasProjection(name, "o_proj") ||
				modelSliceHasProjection(name, "out_proj")
		}
		gotAttn := fam&projAttention != 0
		if gotAttn != wantAttn {
			t.Errorf("name %q: projAttention=%v want %v", name, gotAttn, wantAttn)
		}

		// projFFN — up_proj or down_proj.
		wantFFN := false
		if core.Contains(name, "_proj.") {
			wantFFN = modelSliceHasProjection(name, "up_proj") ||
				modelSliceHasProjection(name, "down_proj")
		}
		gotFFN := fam&projFFN != 0
		if gotFFN != wantFFN {
			t.Errorf("name %q: projFFN=%v want %v", name, gotFFN, wantFFN)
		}

		// projGate — gate_proj.
		wantGate := modelSliceHasProjection(name, "gate_proj")
		gotGate := fam&projGate != 0
		if gotGate != wantGate {
			t.Errorf("name %q: projGate=%v want %v", name, gotGate, wantGate)
		}
	}
}

func TestModelSliceClassify_AttentionFFNGateEquivalence(t *testing.T) {
	for _, name := range classifyEquivalenceCases {
		// Recompute the previous-implementation result so each branch
		// stays pinned to the original semantics post-byte-walk swap.
		oldAttn := false
		if core.Contains(name, "self_attn") || core.Contains(name, "attention") || core.Contains(name, ".attn.") {
			oldAttn = true
		} else if core.Contains(name, "_proj.") {
			oldAttn = modelSliceHasProjection(name, "q_proj") ||
				modelSliceHasProjection(name, "k_proj") ||
				modelSliceHasProjection(name, "v_proj") ||
				modelSliceHasProjection(name, "o_proj") ||
				modelSliceHasProjection(name, "out_proj")
		}
		if got := modelSliceTensorIsAttention(name); got != oldAttn {
			t.Errorf("modelSliceTensorIsAttention(%q) = %v want %v", name, got, oldAttn)
		}

		oldFFN := false
		if core.Contains(name, ".mlp.") || core.Contains(name, "feed_forward") || core.Contains(name, "ffn") {
			oldFFN = true
		} else if core.Contains(name, "_proj.") {
			oldFFN = modelSliceHasProjection(name, "up_proj") ||
				modelSliceHasProjection(name, "down_proj")
		}
		if got := modelSliceTensorIsFFN(name); got != oldFFN {
			t.Errorf("modelSliceTensorIsFFN(%q) = %v want %v", name, got, oldFFN)
		}

		oldGate := modelSliceHasProjection(name, "gate_proj") || core.Contains(name, ".gate.")
		if got := modelSliceTensorIsGate(name); got != oldGate {
			t.Errorf("modelSliceTensorIsGate(%q) = %v want %v", name, got, oldGate)
		}
	}
}
