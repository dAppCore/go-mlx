// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestDecodeForwardICBQuantMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}

	want, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	got, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBQuant: %v", err)
	}
	for i := range want {
		eqBytes(t, "DecodeForwardICBQuant token", got[i], want[i])
	}
}
