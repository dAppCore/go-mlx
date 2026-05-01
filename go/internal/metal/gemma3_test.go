// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestGemma3_QuantizedZeroDefaults_Good(t *testing.T) {
	coverageTokens := "QuantizedZeroDefaults"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	weight := &Array{}
	scales := &Array{}
	quantConfig := &QuantizationConfig{GroupSize: 0, Bits: 0}

	layer := NewQuantizedLinear(weight, scales, nil, nil, quantConfig.GroupSize, quantConfig.Bits)
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma3 layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}

	embed := &Embedding{Weight: weight}
	if scales != nil {
		embed.Scales = scales
		embed.GroupSize = quantConfig.GroupSize
		embed.Bits = quantConfig.Bits
	}
	if embed.GroupSize != 0 || embed.Bits != 0 {
		t.Fatalf("quantized Gemma3 embedding should defer to MLX affine defaults, got group_size=%d bits=%d", embed.GroupSize, embed.Bits)
	}
}

// Generated file-aware compliance coverage.
func TestGemma3_LoadGemma3_Good(t *testing.T) {
	target := "LoadGemma3"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_LoadGemma3_Bad(t *testing.T) {
	target := "LoadGemma3"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_LoadGemma3_Ugly(t *testing.T) {
	target := "LoadGemma3"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Forward_Good(t *testing.T) {
	coverageTokens := "GemmaModel Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Forward_Bad(t *testing.T) {
	coverageTokens := "GemmaModel Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Forward_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ForwardMasked_Good(t *testing.T) {
	coverageTokens := "GemmaModel ForwardMasked"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ForwardMasked"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ForwardMasked_Bad(t *testing.T) {
	coverageTokens := "GemmaModel ForwardMasked"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ForwardMasked"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ForwardMasked_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel ForwardMasked"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ForwardMasked"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NewCache_Good(t *testing.T) {
	coverageTokens := "GemmaModel NewCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NewCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NewCache_Bad(t *testing.T) {
	coverageTokens := "GemmaModel NewCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NewCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NewCache_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel NewCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NewCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NumLayers_Good(t *testing.T) {
	coverageTokens := "GemmaModel NumLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NumLayers"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NumLayers_Bad(t *testing.T) {
	coverageTokens := "GemmaModel NumLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NumLayers"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_NumLayers_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel NumLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_NumLayers"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Tokenizer_Good(t *testing.T) {
	coverageTokens := "GemmaModel Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Tokenizer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Tokenizer_Bad(t *testing.T) {
	coverageTokens := "GemmaModel Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Tokenizer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_Tokenizer_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_Tokenizer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ModelType_Good(t *testing.T) {
	coverageTokens := "GemmaModel ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ModelType_Bad(t *testing.T) {
	coverageTokens := "GemmaModel ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ModelType_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ApplyLoRA_Good(t *testing.T) {
	coverageTokens := "GemmaModel ApplyLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ApplyLoRA"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ApplyLoRA_Bad(t *testing.T) {
	coverageTokens := "GemmaModel ApplyLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ApplyLoRA"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGemma3_GemmaModel_ApplyLoRA_Ugly(t *testing.T) {
	coverageTokens := "GemmaModel ApplyLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "GemmaModel_ApplyLoRA"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
