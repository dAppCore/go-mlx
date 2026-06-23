// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

func BenchmarkHeadEncoderSoftcapKernelRoute(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 64, 2048
	const eps, softCap = float32(1e-6), float32(30)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 5))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 7))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}
	h.initSoftcapBuffers()
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := h.encode(hidden, false); err != nil {
			b.Fatal(err)
		}
	}
}

func quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits int) (*headEncoder, []byte) {
	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*31 + 19) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.01 + float32((i%13)+1)*0.0015
		biasesF[i] = -0.05 + float32(i%17)*0.004
	}
	return &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 41))),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       1e-6,
	}, toBF16Bytes(syntheticFloat32(dModel, 43))
}

func bf16HeadEncoderBenchFixture(dModel, vocab int) (*headEncoder, []byte) {
	return &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 61))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 67))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       1e-6,
	}, toBF16Bytes(syntheticFloat32(dModel, 71))
}

func BenchmarkHeadEncoderBF16FullLogitsGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	if logits, err := h.encode(hidden, true); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderBF16DirectGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	if _, ok, err := h.greedy(hidden, nil); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined BF16 head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined BF16 head")
		}
	}
}

func BenchmarkHeadEncoderBF16FullLogitsSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderBF16DirectSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab = 512, 4096
	h, hidden := bf16HeadEncoderBenchFixture(dModel, vocab)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, ok, err := h.greedy(hidden, suppress); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined BF16 head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, suppress); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined BF16 head")
		}
	}
}

func BenchmarkHeadEncoderQuantFullLogitsGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	if logits, err := h.encode(hidden, true); err != nil {
		b.Fatal(err)
	} else if _, err := model.Greedy(logits, vocab); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := model.Greedy(logits, vocab); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderQuantDirectGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	if _, ok, err := h.greedy(hidden, nil); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined quant head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, nil); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined quant head")
		}
	}
}

func BenchmarkHeadEncoderQuantFullLogitsSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := h.encode(hidden, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := greedyBF16Suppressed(logits, vocab, suppress); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHeadEncoderQuantDirectSuppressedGreedySynthetic(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	logits, err := h.encode(hidden, true)
	if err != nil {
		b.Fatal(err)
	}
	top, err := model.Greedy(logits, vocab)
	if err != nil {
		b.Fatal(err)
	}
	suppress := []int32{top}
	if _, ok, err := h.greedy(hidden, suppress); err != nil {
		b.Fatal(err)
	} else if !ok {
		b.Fatal("direct greedy declined quant head")
	}
	b.SetBytes(int64(vocab * bf16Size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := h.greedy(hidden, suppress); err != nil {
			b.Fatal(err)
		} else if !ok {
			b.Fatal("direct greedy declined quant head")
		}
	}
}
