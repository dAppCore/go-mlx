// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for the pure-Go tensor-classification + plan helpers in
// model_slice.go. These operate purely on tensor names and *ModelSlicePlan
// metadata (no safetensors file content, no model load), so every mask arm
// and projection-family fallback is directly reachable.

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

func TestModelSliceIncludesTensorMask_EveryArm(t *testing.T) {
	cases := []struct {
		name   string
		mask   modelSliceInclusionMask
		tensor string
		want   bool
	}{
		{"all", modelSliceInclusionMask{all: true}, "anything", true},
		{"attn-substring", modelSliceInclusionMask{attention: true}, "model.layers.0.self_attn.q_proj.weight", true},
		{"attn-attndot", modelSliceInclusionMask{attention: true}, "model.layers.0.attn.k.weight", true},
		// Projection-family fallback for attention: "_proj." without "self_attn".
		{"attn-projfamily", modelSliceInclusionMask{attention: true}, "blocks.0.q_proj.weight", true},
		{"attn-miss", modelSliceInclusionMask{attention: true}, "model.embed_tokens.weight", false},
		{"ffn-mlp", modelSliceInclusionMask{ffn: true}, "model.layers.0.mlp.up_proj.weight", true},
		{"ffn-feedforward", modelSliceInclusionMask{ffn: true}, "h.0.feed_forward.w1.weight", true},
		{"ffn-projfamily", modelSliceInclusionMask{ffn: true}, "blocks.0.down_proj.weight", true},
		{"norms", modelSliceInclusionMask{norms: true}, "model.layers.0.input_layernorm.weight", true},
		{"gate-dotgate", modelSliceInclusionMask{gate: true}, "model.layers.0.gate.weight", true},
		{"experts", modelSliceInclusionMask{experts: true}, "model.layers.0.experts.0.w.weight", true},
		{"router", modelSliceInclusionMask{router: true}, "model.layers.0.router.weight", true},
		{"downmeta", modelSliceInclusionMask{downMeta: true}, "model.layers.0.down_meta.weight", true},
		{"embeddings", modelSliceInclusionMask{embeddings: true}, "model.embed_tokens.weight", true},
		{"lmhead", modelSliceInclusionMask{lmHead: true}, "lm_head.weight", true},
		{"empty-mask-miss", modelSliceInclusionMask{}, "model.embed_tokens.weight", false},
	}
	for _, tc := range cases {
		if got := modelSliceIncludesTensorMask(tc.mask, tc.tensor); got != tc.want {
			t.Errorf("%s: includes(%q) = %v, want %v", tc.name, tc.tensor, got, tc.want)
		}
	}
}

func TestBuildModelSliceInclusionMask(t *testing.T) {
	// ExtractLevelAll → all=true short-circuit.
	all := buildModelSliceInclusionMask(&inference.ModelSlicePlan{ExtractLevel: inference.ModelExtractLevelAll})
	if !all.all {
		t.Fatalf("ExtractLevelAll should set all=true")
	}

	// A plan listing each component flips the corresponding mask bit.
	plan := &inference.ModelSlicePlan{Components: []inference.ModelComponent{
		inference.ModelComponentEmbeddings,
		inference.ModelComponentNorms,
		inference.ModelComponentAttention,
		inference.ModelComponentFFN,
		inference.ModelComponentGate,
		inference.ModelComponentDownMeta,
		inference.ModelComponentRouter,
		inference.ModelComponentExperts,
		inference.ModelComponentLMHead,
	}}
	mask := buildModelSliceInclusionMask(plan)
	if !(mask.embeddings && mask.norms && mask.attention && mask.ffn && mask.gate && mask.downMeta && mask.router && mask.experts && mask.lmHead) {
		t.Fatalf("expected every component bit set: %+v", mask)
	}
}

func TestModelSliceStandalone(t *testing.T) {
	// ExtractLevelAll → standalone, no missing.
	ok, missing := modelSliceStandalone(&inference.ModelSlicePlan{ExtractLevel: inference.ModelExtractLevelAll})
	if !ok || missing != nil {
		t.Fatalf("ExtractLevelAll should be standalone with no missing, got ok=%v missing=%v", ok, missing)
	}

	// All four required components present → standalone.
	full := &inference.ModelSlicePlan{Components: []inference.ModelComponent{
		inference.ModelComponentEmbeddings,
		inference.ModelComponentAttention,
		inference.ModelComponentFFN,
		inference.ModelComponentLMHead,
	}}
	if ok, missing := modelSliceStandalone(full); !ok || missing != nil {
		t.Fatalf("full plan should be standalone, got ok=%v missing=%v", ok, missing)
	}

	// Missing FFN + LMHead → not standalone, both reported.
	partial := &inference.ModelSlicePlan{Components: []inference.ModelComponent{
		inference.ModelComponentEmbeddings,
		inference.ModelComponentAttention,
	}}
	ok, missing = modelSliceStandalone(partial)
	if ok {
		t.Fatalf("partial plan should not be standalone")
	}
	if len(missing) != 2 {
		t.Fatalf("expected 2 missing components, got %v", missing)
	}
}

func TestModelSliceLabelInt64(t *testing.T) {
	// Empty labels → 0.
	if got := modelSliceLabelInt64(nil, "k"); got != 0 {
		t.Fatalf("nil labels = %d, want 0", got)
	}
	labels := map[string]string{"present": "42", "empty": "", "bad": "notanumber"}
	if got := modelSliceLabelInt64(labels, "present"); got != 42 {
		t.Fatalf("present = %d, want 42", got)
	}
	// Empty value → 0.
	if got := modelSliceLabelInt64(labels, "empty"); got != 0 {
		t.Fatalf("empty value = %d, want 0", got)
	}
	// Parse failure → 0.
	if got := modelSliceLabelInt64(labels, "bad"); got != 0 {
		t.Fatalf("unparseable = %d, want 0", got)
	}
	// Missing key → 0.
	if got := modelSliceLabelInt64(labels, "absent"); got != 0 {
		t.Fatalf("absent key = %d, want 0", got)
	}
}

func TestModelSliceIncludesTensor_PlanWrapper(t *testing.T) {
	// The plan-level wrapper builds the mask then classifies.
	plan := inference.ModelSlicePlan{Components: []inference.ModelComponent{inference.ModelComponentAttention}}
	if !modelSliceIncludesTensor(plan, "model.layers.0.self_attn.q_proj.weight") {
		t.Fatalf("attention tensor should be included")
	}
	if modelSliceIncludesTensor(plan, "model.embed_tokens.weight") {
		t.Fatalf("embedding tensor should be excluded from attention-only plan")
	}
}
