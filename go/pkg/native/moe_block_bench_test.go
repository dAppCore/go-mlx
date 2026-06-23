// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoEBlockBF16Top2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantFusedGateUpTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	w.ExpGateUp = fusedGateUpQuantForBench(w.ExpGate, w.ExpUp, numExperts, expertDFF, dModel, groupSize, bits)
	w.ExpGate, w.ExpUp = QuantWeight{}, QuantWeight{}
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGateUp.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantViewBackedTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := withQuantViewsForBench(quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits))
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func fusedGateUpQuantForBench(gate, up QuantWeight, numExperts, expertDFF, dModel, groupSize, bits int) QuantWeight {
	gatePacked := expertDFF * dModel * bits / 8
	gateScale := expertDFF * (dModel / groupSize) * bf16Size
	fuse := func(a, b []byte, perExpert int) []byte {
		out := make([]byte, 0, len(a)+len(b))
		for e := 0; e < numExperts; e++ {
			start := e * perExpert
			end := start + perExpert
			out = append(out, a[start:end]...)
			out = append(out, b[start:end]...)
		}
		return out
	}
	return QuantWeight{
		Packed:    fuse(gate.Packed, up.Packed, gatePacked),
		Scales:    fuse(gate.Scales, up.Scales, gateScale),
		Biases:    fuse(gate.Biases, up.Biases, gateScale),
		GroupSize: groupSize,
		Bits:      bits,
	}
}

func withQuantViewsForBench(w MoEQuantLayerWeights) MoEQuantLayerWeights {
	view := func(q QuantWeight) QuantWeight {
		if len(q.Packed) == 0 {
			return q
		}
		q.packedView = bufView{buf: sharedBytes(q.Packed)}
		q.scalesView = bufView{buf: sharedBytes(q.Scales)}
		q.biasesView = bufView{buf: sharedBytes(q.Biases)}
		return q
	}
	w.LocalGate = view(w.LocalGate)
	w.LocalUp = view(w.LocalUp)
	w.LocalDown = view(w.LocalDown)
	w.Router = view(w.Router)
	w.ExpGate = view(w.ExpGate)
	w.ExpUp = view(w.ExpUp)
	w.ExpGateUp = view(w.ExpGateUp)
	w.ExpDown = view(w.ExpDown)
	w.preFFNormView = bufView{buf: sharedBytes(w.PreFFNormW)}
	w.preFFNorm2View = bufView{buf: sharedBytes(w.PreFFNorm2W)}
	w.postFFNorm1View = bufView{buf: sharedBytes(w.PostFFNorm1W)}
	w.postFFNorm2View = bufView{buf: sharedBytes(w.PostFFNorm2W)}
	w.postFFNormView = bufView{buf: sharedBytes(w.PostFFNormW)}
	w.routerNormView = bufView{buf: sharedBytes(w.RouterNormWScaled)}
	w.perExpertScaleView = bufView{buf: sharedBytes(w.PerExpertScale)}
	return w
}
