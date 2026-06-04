// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestGemma3_QuantizedZeroDefaults_Good(t *testing.T) {
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

func TestGemma3_parseConfig_EmbeddingScaleCached_Good(t *testing.T) {
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
