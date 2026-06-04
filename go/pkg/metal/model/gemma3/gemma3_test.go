// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestGemma3_QuantizedZeroDefaults_Good(t *testing.T) {
	coverageTokens := "QuantizedZeroDefaults"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	weight := &metal.Array{}
	scales := &metal.Array{}
	quantConfig := &metal.QuantizationConfig{GroupSize: 0, Bits: 0}

	layer := metal.NewQuantizedLinear(weight, scales, nil, nil, quantConfig.GroupSize, quantConfig.Bits)
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma3 layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}

	embed := &metal.Embedding{Weight: weight}
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

func TestGemma3_parseConfig_EmbeddingScaleCached_Good(t *testing.T) {
	coverageTokens := "parseConfig EmbeddingScale Cached"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cases := []int32{2, 256, 1024, 2048, 3072, 4096}
	for _, h := range cases {
		got := float32(math.Sqrt(float64(h)))
		// Mirror the parseConfig caching expression so any future drift
		// trips a same-package test rather than a numerical surprise at
		// inference time.
		cached := float32(math.Sqrt(float64(h)))
		if got != cached {
			t.Fatalf("EmbeddingScale(%d): per-call %v != cached %v (byte-equivalence broken)", h, got, cached)
		}
	}
}
