// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestPerLayerInputsGPUParity gates the on-GPU PLE: PerLayerInputsGPU (per-layer embed-gather + projection
// + norm + combine, all on the GPU from a token id) must reproduce the host PerLayerInputs. This is the
// gate the submit-ahead decode pipeline needs for e2b — the PLE tensor computed on-GPU so the next step
// can be submitted before the token is read back.
func TestPerLayerInputsGPUParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	plDim := numLayers * pliDim

	// 4-bit per-layer embedding table [vocabPLI × plDim], bf16 projection [plDim × dModel] + projNorm [pliDim].
	embedPacked := make([]byte, vocabPLI*plDim*embBits/8)
	for i := range embedPacked {
		embedPacked[i] = byte((i*131 + 17) % 256)
	}
	embedScales := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 11))
	embedBiases := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 13))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 7))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 19))
	emb := toBF16Bytes(syntheticFloat32(dModel, 23))

	for _, tok := range []int32{0, 5, 17, 31} {
		ref, err := PerLayerInputs(embedPacked, embedScales, embedBiases, projW, nil, nil, projNormW, tok, emb, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, 0, 0, eps, bufView{})
		if err != nil {
			t.Fatalf("tok %d: host PerLayerInputs: %v", tok, err)
		}
		got, err := PerLayerInputsGPU(tok, emb, embedPacked, embedScales, embedBiases, projW, projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
		if err != nil {
			t.Fatalf("tok %d: PerLayerInputsGPU: %v", tok, err)
		}
		if cos := cosineBF16(got, ref); cos < 0.9999 {
			t.Fatalf("tok %d: GPU PLE cosine=%.6f vs host PerLayerInputs", tok, cos)
		}
	}
	t.Logf("GPU PLE matches host PerLayerInputs")
}

// TestSessionNextInputsGPUParity gates the session wiring (not just the math): a PLE-enabled quant
// session's encNextInputsGPU must reproduce s.embed + s.perLayerInput for the SAME token, using the
// session's real resident weights/dims/scales. This is the seam the chained decode step appends to
// produce the next step's emb+pli on-GPU — a wiring slip (wrong scale, wrong weight, wrong dim) shows
// here before it ever reaches the decode loop.
func TestSessionNextInputsGPUParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture should have the per-layer-input tower")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired for an e2b-shaped PLE session")
	}

	for _, tok := range []int32{1, 5, 17, 31} {
		gotEmb, gotPli, ok, err := sess.nextInputsGPU(tok)
		if err != nil {
			t.Fatalf("tok %d: nextInputsGPU: %v", tok, err)
		}
		if !ok {
			t.Fatalf("tok %d: nextInputsGPU ok=false on a wired session", tok)
		}
		wantEmb, err := sess.embed(tok)
		if err != nil {
			t.Fatalf("tok %d: host embed: %v", tok, err)
		}
		wantPli, err := sess.perLayerInput(tok, wantEmb)
		if err != nil {
			t.Fatalf("tok %d: host perLayerInput: %v", tok, err)
		}
		if cos := cosineBF16(gotEmb, wantEmb); cos < 0.9999 {
			t.Fatalf("tok %d: GPU emb cosine=%.6f vs host s.embed", tok, cos)
		}
		if cos := cosineBF16(gotPli, wantPli); cos < 0.9999 {
			t.Fatalf("tok %d: GPU pli cosine=%.6f vs host s.perLayerInput", tok, cos)
		}
	}
	t.Logf("session GPU next-inputs (emb+pli) matches host s.embed + s.perLayerInput")
}
