// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for the pure-Go (no-GPU, no-model) helpers in split_cpu_ffn.go:
// the weight/sidecar candidate fan-out, suffix trimming, string de-dup, the
// nested-config merge, and the quantization-hint appliers. These run entirely
// on slices and structs the test constructs directly.

package mlx

import (
	"slices"
	"testing"

	infjang "dappco.re/go/inference/quant/jang"
)

func TestCPUSplitWeightCandidates_BothPrefixArms(t *testing.T) {
	// "model." prefix arm: 5 candidates, language_model rewrites.
	withModel := cpuSplitWeightCandidates("model.layers.0.mlp.gate_proj.weight")
	if len(withModel) != 5 {
		t.Fatalf("model.-prefixed candidates = %d, want 5: %v", len(withModel), withModel)
	}
	if withModel[0] != "model.layers.0.mlp.gate_proj.weight" {
		t.Fatalf("first candidate should be the input name: %v", withModel[0])
	}
	if !slices.Contains(withModel, "language_model.model.layers.0.mlp.gate_proj.weight") {
		t.Fatalf("expected language_model.model. rewrite in %v", withModel)
	}

	// Non-prefix arm: 6 candidates.
	noModel := cpuSplitWeightCandidates("embed_tokens.weight")
	if len(noModel) != 6 {
		t.Fatalf("non-prefixed candidates = %d, want 6: %v", len(noModel), noModel)
	}
	if !slices.Contains(noModel, "model.embed_tokens.weight") {
		t.Fatalf("expected model. rewrite in %v", noModel)
	}
}

func TestCPUSplitTrimSuffixes(t *testing.T) {
	if got := cpuSplitTrimWeightSuffix("x.weight"); got != "x" {
		t.Fatalf("trim .weight = %q, want x", got)
	}
	if got := cpuSplitTrimWeightSuffix("x.bias"); got != "x.bias" {
		t.Fatalf("non-.weight should be unchanged, got %q", got)
	}
	if got := cpuSplitTrimPackedSuffix("x.packed"); got != "x" {
		t.Fatalf("trim .packed = %q, want x", got)
	}
	if got := cpuSplitTrimPackedSuffix("x.qweight"); got != "x" {
		t.Fatalf("trim .qweight = %q, want x", got)
	}
	if got := cpuSplitTrimPackedSuffix("x.weight"); got != "x.weight" {
		t.Fatalf("non-packed should be unchanged, got %q", got)
	}
}

func TestCPUSplitUniqueStrings_SkipsEmptyAndDupes(t *testing.T) {
	got := cpuSplitUniqueStrings([]string{"a", "", "b", "a", "", "c", "b"})
	want := []string{"a", "b", "c"}
	if !slices.Equal(got, want) {
		t.Fatalf("cpuSplitUniqueStrings = %v, want %v", got, want)
	}
}

func TestCPUSplitSidecarCandidates_TrimsPackedFound(t *testing.T) {
	// foundName carries a packed suffix → its trimmed form is added too.
	got := cpuSplitSidecarCandidates(
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.gate_proj.weight.packed",
		"scales",
	)
	if len(got) == 0 {
		t.Fatalf("expected sidecar candidates")
	}
	// The scales sidecar of the trimmed found name must appear.
	if !slices.Contains(got, "model.layers.0.mlp.gate_proj.weight.scales") {
		t.Fatalf("expected trimmed-found scales sidecar in %v", got)
	}
}

func TestMergeCPUSplitQwenConfig_FillsFromTop(t *testing.T) {
	top := cpuSplitQwenConfig{
		ModelType:        "qwen3",
		HiddenSize:       2048,
		IntermediateSize: 8192,
		NumHiddenLayers:  24,
		RMSNormEps:       1e-6,
		Quantization:     &cpuSplitQuantizationConfig{Bits: 4},
		QuantizationConfig: &cpuSplitQuantizationConfig{
			GroupSize: 32,
		},
	}

	// Empty text config → every field inherited from top.
	merged := mergeCPUSplitQwenConfig(top, cpuSplitQwenConfig{})
	if merged.ModelType != "qwen3" || merged.HiddenSize != 2048 || merged.IntermediateSize != 8192 {
		t.Fatalf("merge did not inherit scalars: %+v", merged)
	}
	if merged.NumHiddenLayers != 24 || merged.RMSNormEps != 1e-6 {
		t.Fatalf("merge did not inherit layers/eps: %+v", merged)
	}
	if merged.Quantization == nil || merged.QuantizationConfig == nil {
		t.Fatalf("merge did not inherit quantization pointers: %+v", merged)
	}

	// Text config that already sets fields keeps its own values.
	text := cpuSplitQwenConfig{ModelType: "text-only", HiddenSize: 1024}
	kept := mergeCPUSplitQwenConfig(top, text)
	if kept.ModelType != "text-only" || kept.HiddenSize != 1024 {
		t.Fatalf("merge overwrote set fields: %+v", kept)
	}
	if kept.IntermediateSize != 8192 {
		t.Fatalf("merge should still fill the unset IntermediateSize: %+v", kept)
	}
}

func TestCPUSplitQwenConfig_QuantizationHints(t *testing.T) {
	cfg := cpuSplitQwenConfig{
		Quantization:       &cpuSplitQuantizationConfig{GroupSize: 64, Bits: 4},
		QuantizationConfig: &cpuSplitQuantizationConfig{BitsDefault: 8},
	}
	cfg.applyQuantizationHints()
	if cfg.PackedGroupSize != 64 {
		t.Fatalf("PackedGroupSize = %d, want 64", cfg.PackedGroupSize)
	}
	if cfg.PackedBits != 4 {
		t.Fatalf("PackedBits = %d, want 4 (first positive of Quantization)", cfg.PackedBits)
	}

	// Nil hint is a no-op.
	var empty cpuSplitQwenConfig
	empty.applyQuantizationHint(nil)
	if empty.PackedGroupSize != 0 || empty.PackedBits != 0 {
		t.Fatalf("nil hint should not mutate: %+v", empty)
	}
}

func TestCPUSplitQwenConfig_ApplyJANGInfo(t *testing.T) {
	var cfg cpuSplitQwenConfig
	cfg.applyJANGInfo(nil) // nil guard, no-op
	if cfg.JANG != nil {
		t.Fatalf("nil JANG info should be a no-op")
	}

	info := &infjang.Info{GroupSize: 128, BitsDefault: 4}
	cfg.applyJANGInfo(info)
	if cfg.JANG != info {
		t.Fatalf("applyJANGInfo should store the info pointer")
	}
	if cfg.PackedGroupSize != 128 {
		t.Fatalf("PackedGroupSize = %d, want 128", cfg.PackedGroupSize)
	}
	if cfg.PackedBits != 4 {
		t.Fatalf("PackedBits = %d, want 4", cfg.PackedBits)
	}
}
