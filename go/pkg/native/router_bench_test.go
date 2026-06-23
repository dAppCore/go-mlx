// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkTopKByScoreTop2Of4096(b *testing.B) {
	const numExperts, topK = 4096, 2
	scores := make([]float32, numExperts)
	for i := range scores {
		scores[i] = float32((i*37)%1000) * 0.001
	}
	scores[17] = 9
	scores[4095] = 8
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		topKByScoreSink = topKByScore(scores, topK)
	}
}

func BenchmarkMoERouterTop2Of8(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel = 8, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	b.SetBytes(int64(len(x) + len(normW) + len(routerW)))
	if _, _, err := MoERouter(x, normW, routerW, scale, numExperts, topK, dModel, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := MoERouter(x, normW, routerW, scale, numExperts, topK, dModel, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoERouterHostSelectTop2Of8(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel = 8, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	b.SetBytes(int64(len(x) + len(normW) + len(routerW)))
	run := func() error {
		normed, err := RMSNormBF16(x, normW, 1, dModel, 1e-5)
		if err != nil {
			return err
		}
		scoresB, err := matVecBF16Resident(routerW, normed, numExperts, dModel)
		if err != nil {
			return err
		}
		_, _ = routerSelect(scoresB, scale, numExperts, topK)
		return nil
	}
	if err := run(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := run(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoERouterTop2Of4096(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel = 4096, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	b.SetBytes(int64(len(x) + len(normW) + len(routerW)))
	if _, _, err := MoERouter(x, normW, routerW, nil, numExperts, topK, dModel, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := MoERouter(x, normW, routerW, nil, numExperts, topK, dModel, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoERouterHostSelectTop2Of4096(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel = 4096, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	b.SetBytes(int64(len(x) + len(normW) + len(routerW)))
	run := func() error {
		normed, err := RMSNormBF16(x, normW, 1, dModel, 1e-5)
		if err != nil {
			return err
		}
		scoresB, err := matVecBF16Resident(routerW, normed, numExperts, dModel)
		if err != nil {
			return err
		}
		_, _ = routerSelect(scoresB, nil, numExperts, topK)
		return nil
	}
	if err := run(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := run(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoERouterQuantTop2Of8(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := quantWeightFixture(b, numExperts, dModel, groupSize, bits, 43)
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	b.SetBytes(int64(len(x) + len(normW) + len(routerW.Packed) + len(routerW.Scales) + len(routerW.Biases)))
	if _, _, err := MoERouterQuant(x, normW, routerW, scale, numExperts, topK, dModel, groupSize, bits, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := MoERouterQuant(x, normW, routerW, scale, numExperts, topK, dModel, groupSize, bits, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoERouterQuantHostSelectTop2Of8(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := quantWeightFixture(b, numExperts, dModel, groupSize, bits, 43)
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	b.SetBytes(int64(len(x) + len(normW) + len(routerW.Packed) + len(routerW.Scales) + len(routerW.Biases)))
	run := func() error {
		normed, err := RMSNormBF16(x, normW, 1, dModel, 1e-5)
		if err != nil {
			return err
		}
		scoresB, err := qmvBF16Resident(normed, routerW, numExperts, dModel, groupSize, bits)
		if err != nil {
			return err
		}
		_, _ = routerSelect(scoresB, scale, numExperts, topK)
		return nil
	}
	if err := run(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := run(); err != nil {
			b.Fatal(err)
		}
	}
}
