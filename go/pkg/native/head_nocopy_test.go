// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

func TestNewHeadEncoderNilShardBuffersFallsBack(t *testing.T) {
	h, err := newHeadEncoder(nil, nil, nil, nil, nil, 64, 128, 64, 4, 1e-5, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder nil shard buffers: %v", err)
	}
	if h != nil {
		t.Fatalf("newHeadEncoder nil shard buffers = %+v, want nil fallback", h)
	}
}

func TestNewHeadEncoderNilShardBuffersBuildsOwnedBF16Head(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 51))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 53))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 57))

	h, err := newHeadEncoder(nil, finalNorm, head, nil, nil, dModel, vocab, 0, 0, eps, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder owned bf16: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned bf16 returned nil; in-memory sessions would miss direct greedy")
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("owned bf16 head logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("owned bf16 full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("owned bf16 direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("owned bf16 direct greedy declined")
	}
	if got != want {
		t.Fatalf("owned bf16 direct greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestNewHeadEncoderNilShardBuffersBuildsOwnedQuantHead(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}

	h, err := newHeadEncoder(nil, finalNorm, packed, toBF16Bytes(scalesF), toBF16Bytes(biasesF), dModel, vocab, groupSize, bits, eps, 0, true)
	if err != nil {
		t.Fatalf("newHeadEncoder owned quant: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned quant returned nil; in-memory quant sessions would miss direct greedy")
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("owned quant head logits: %v", err)
	}
	want, err := LMHeadQuant(hidden, finalNorm, packed, toBF16Bytes(scalesF), toBF16Bytes(biasesF), dModel, vocab, groupSize, bits, eps, 0)
	if err != nil {
		t.Fatalf("LMHeadQuant reference: %v", err)
	}
	if !bytes.Equal(logits, want) {
		t.Fatalf("owned quant head logits diverged from LMHeadQuant")
	}
}

func TestOwnedQuantHeadDirectGreedyMatchesContractFixture(t *testing.T) {
	requireNativeRuntime(t)

	const gs, bits = 32, 4
	const maxLen, maxNew = 16, 6
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	head, err := newHeadEncoder(nil, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
	if err != nil {
		t.Fatalf("newHeadEncoder owned quant: %v", err)
	}
	if head == nil {
		t.Fatal("newHeadEncoder owned quant returned nil")
	}

	prompt := []int32{1, 5, 3}
	var hidden []byte
	for _, id := range prompt {
		if hidden, err = sess.stepID(id); err != nil {
			t.Fatalf("prefill stepID(%d): %v", id, err)
		}
	}
	for i := 0; i < maxNew; i++ {
		logits, err := head.encode(hidden, true)
		if err != nil {
			t.Fatalf("owned quant full logits at generated step %d: %v", i, err)
		}
		want, err := model.Greedy(logits, arch.Vocab)
		if err != nil {
			t.Fatalf("owned quant full-logits greedy at generated step %d: %v", i, err)
		}
		got, ok, err := head.greedy(hidden, nil)
		if err != nil {
			t.Fatalf("owned quant direct greedy at generated step %d: %v", i, err)
		}
		if !ok {
			t.Fatal("owned quant direct greedy declined contract fixture")
		}
		if got != want {
			t.Fatalf("owned quant direct greedy at generated step %d = %d, want resident qmv full-logits greedy %d", i, got, want)
		}
		if hidden, err = sess.stepID(want); err != nil {
			t.Fatalf("generated stepID(%d) at step %d: %v", want, i, err)
		}
	}
}

func TestHeadEncoderRejectsHiddenShapeMismatch(t *testing.T) {
	h := &headEncoder{dModel: 2, vocab: 2}
	if _, err := h.encode(toBF16Bytes([]float32{1}), false); err == nil {
		t.Fatal("expected headEncoder.encode to reject hidden shape mismatch")
	}
}

func TestHeadEncoderSoftcapUsesBF16Kernel(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 1, 8
	const eps, softCap = float32(1e-6), float32(30)
	hidden := toBF16Bytes([]float32{1})
	finalNorm := toBF16Bytes([]float32{1})
	head := toBF16Bytes([]float32{-120, -30, -3, -0.5, 0.5, 3, 30, 120})
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(head),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}

	raw, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder raw logits: %v", err)
	}
	scaled, err := MulBF16(raw, bf16ConstBytes(vocab, 1/softCap))
	if err != nil {
		t.Fatalf("scale logits: %v", err)
	}
	capped, err := TanhBF16(scaled)
	if err != nil {
		t.Fatalf("tanh logits: %v", err)
	}
	want, err := MulBF16(capped, bf16ConstBytes(vocab, softCap))
	if err != nil {
		t.Fatalf("restore logits: %v", err)
	}

	got, err := h.encode(hidden, false)
	if err != nil {
		t.Fatalf("headEncoder softcap logits: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("headEncoder softcap = %v, want BF16-kernel softcap %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestHeadEncoderSoftcapUsesScalarScaleBuffers(t *testing.T) {
	requireNativeRuntime(t)

	h := &headEncoder{vocab: 8192, softCap: 30}
	h.initSoftcapBuffers()
	if h.invSoftCapScale.buf == nil || h.softCapScale.buf == nil {
		t.Fatalf("softcap scalar buffers missing (inv=%v cap=%v)", h.invSoftCapScale.buf != nil, h.softCapScale.buf != nil)
	}
	if got := int(h.invSoftCapScale.buf.Length()); got != bf16Size {
		t.Fatalf("inverse softcap scale buffer length = %d, want scalar bf16 length %d", got, bf16Size)
	}
	if got := int(h.softCapScale.buf.Length()); got != bf16Size {
		t.Fatalf("softcap scale buffer length = %d, want scalar bf16 length %d", got, bf16Size)
	}
}

func TestHeadGreedyScratchKeepsPerTokenBuffersResident(t *testing.T) {
	requireNativeRuntime(t)

	s := newHeadGreedyScratch(3, 64, 17, true)
	if s.normed == nil {
		t.Fatal("greedy scratch did not retain the normed activation buffer")
	}
	if s.logits == nil {
		t.Fatal("quant greedy scratch did not retain the qmv logits buffer")
	}
	if got := int(s.normed.Length()); got != 64*bf16Size {
		t.Fatalf("normed scratch length = %d, want %d", got, 64*bf16Size)
	}
	if got := int(s.logits.Length()); got != 17*bf16Size {
		t.Fatalf("logits scratch length = %d, want %d", got, 17*bf16Size)
	}

	bf16 := newHeadGreedyScratch(3, 64, 17, false)
	if bf16.normed == nil {
		t.Fatal("BF16 greedy scratch did not retain the normed activation buffer")
	}
	if bf16.logits != nil {
		t.Fatal("BF16 greedy scratch allocated a quant logits buffer")
	}
}

func TestHeadEncoderQuantGreedyMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("headEncoder direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined quant head")
	}
	if got != want {
		t.Fatalf("headEncoder direct greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderQuantGreedySuppressesIDs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	first, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	want, err := greedyBF16Suppressed(logits, vocab, []int32{first})
	if err != nil {
		t.Fatalf("suppressed full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, []int32{first})
	if err != nil {
		t.Fatalf("headEncoder suppressed direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined quant head with suppression")
	}
	if got != want {
		t.Fatalf("headEncoder suppressed direct greedy = %d, want full-logits suppressed greedy %d", got, want)
	}
}

func TestHeadEncoderBF16GreedyMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 51))
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("headEncoder direct bf16 greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined BF16 head")
	}
	if got != want {
		t.Fatalf("headEncoder direct bf16 greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderBF16GreedySuppressesIDs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 1, 8
	const eps = float32(1e-6)
	hidden := toBF16Bytes([]float32{1})
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes([]float32{1})),
		weight:    copyView(toBF16Bytes([]float32{-4, -2, -1, 0, 1, 2, 4, 8})),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	first, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	if first != 7 {
		t.Fatalf("fixture top token = %d, want 7", first)
	}
	want, err := greedyBF16Suppressed(logits, vocab, []int32{first})
	if err != nil {
		t.Fatalf("suppressed full-logits greedy: %v", err)
	}
	if want != 6 {
		t.Fatalf("fixture suppressed token = %d, want 6", want)
	}
	got, ok, err := h.greedy(hidden, []int32{first})
	if err != nil {
		t.Fatalf("headEncoder suppressed direct bf16 greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined BF16 head with suppression")
	}
	if got != want {
		t.Fatalf("headEncoder suppressed direct bf16 greedy = %d, want full-logits suppressed greedy %d", got, want)
	}
}
