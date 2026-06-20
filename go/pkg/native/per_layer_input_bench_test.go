// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkPerLayerInputsBF16(b *testing.B) {
	requireNativeRuntime(b)

	const vocabPLI, numLayers, pliDim, dModel = 8, 2, 32, 64
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	b.SetBytes(int64(len(embed) + len(projW) + len(hidden)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
