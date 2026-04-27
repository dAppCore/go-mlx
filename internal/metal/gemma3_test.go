// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestGemma3_QuantizedZeroDefaults_Good(t *testing.T) {
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
