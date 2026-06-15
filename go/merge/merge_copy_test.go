// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"testing"

	core "dappco.re/go"
)

// TestModelMerge_IsModelWeightMetadataCopySkip_Good covers the
// isModelWeightMetadataCopySkip predicate and its equalFold / containsFold
// helpers — including the ASCII case-folding legs (mixed-case provenance name,
// uppercase weight extensions) that the metadata-copy integration test does
// not reach.
func TestModelMerge_IsModelWeightMetadataCopySkip_Good(t *testing.T) {
	skip := []string{
		"adapter_provenance.json",
		"Adapter_Provenance.JSON",          // equalFold case-folding leg
		"model-00001-of-00002.safetensors", // containsFold suffix
		"model.SAFETENSORS",                // containsFold case-folding leg
		"weights.gguf",                     // gguf containsFold
		"model.safetensors.index.json",     // .safetensors substring inside name
	}
	for _, name := range skip {
		if !isModelWeightMetadataCopySkip(name) {
			t.Fatalf("isModelWeightMetadataCopySkip(%q) = false, want true", name)
		}
	}
	keep := []string{"config.json", "tokenizer.json", "special_tokens_map.txt", "generation_config.json"}
	for _, name := range keep {
		if isModelWeightMetadataCopySkip(name) {
			t.Fatalf("isModelWeightMetadataCopySkip(%q) = true, want false", name)
		}
	}
	// Length-mismatch short-circuit in equalFold (different lengths never match).
	if equalFold("abc", "abcd") {
		t.Fatal("equalFold(differing lengths) = true, want false")
	}
	// containsFold: empty substring is trivially contained; over-long is not.
	if !containsFold("anything", "") {
		t.Fatal("containsFold(_, \"\") = false, want true")
	}
	if containsFold("ab", "abc") {
		t.Fatal("containsFold(shorter than substr) = true, want false")
	}
}

// TestModelMerge_SamePath_Good covers samePath, which compares two paths after
// resolving each to absolute form.
func TestModelMerge_SamePath_Good(t *testing.T) {
	dir := t.TempDir()
	a := core.PathJoin(dir, "model")
	if !samePath(a, a) {
		t.Fatalf("samePath(%q, %q) = false, want true", a, a)
	}
	b := core.PathJoin(dir, "other")
	if samePath(a, b) {
		t.Fatalf("samePath(%q, %q) = true, want false", a, b)
	}
	// A relative path resolves to the same absolute target as its abs form.
	abs := core.PathAbs(a)
	if !abs.OK {
		t.Fatalf("PathAbs(%q): %v", a, abs.Value)
	}
	if !samePath(a, abs.Value.(string)) {
		t.Fatal("samePath(rel, abs) = false, want true for equivalent paths")
	}
}
