// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestDecodeForwardQuantProducesTokenOutputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}

	got, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("DecodeForwardQuant returned %d tokens, want %d", len(got), len(inputs))
	}
	for i := range got {
		if len(got[i]) != dModel*bf16Size {
			t.Fatalf("DecodeForwardQuant token %d has %d bytes, want %d", i, len(got[i]), dModel*bf16Size)
		}
	}
}

func TestDecodeForwardQuantRejectsUnsetQuantGeometry(t *testing.T) {
	requireNativeRuntime(t)

	inputs := decodeInputsFixture(1, 64)
	layers := []QuantizedLayerWeights{{AttnNormW: toBF16Bytes(syntheticFloat32(64, 3)), MLPNormW: toBF16Bytes(syntheticFloat32(64, 5))}}
	if _, err := DecodeForwardQuant(inputs, layers, 64, 1, 1, 64, 1, 128, 10000, 0.125, 1e-5); err == nil {
		t.Fatal("expected DecodeForwardQuant to reject unset GroupSize/Bits")
	}
}
