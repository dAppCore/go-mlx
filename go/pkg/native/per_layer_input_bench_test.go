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
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateBF16(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim = 64, 32
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	b.SetBytes(int64(len(hNext) + len(gateW) + len(projW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateQuant(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(b, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(b, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	b.SetBytes(int64(len(hNext) + len(gate.Packed) + len(proj.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
