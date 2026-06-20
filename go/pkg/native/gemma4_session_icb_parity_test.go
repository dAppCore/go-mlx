// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestGemma4QuantSessionICBParity proves the incremental ICB encode-bypass (Phase B) is
// byte-identical to the stepToken host-encode path: an eligible E2B-shaped PLE session records
// the arch ICB (state.icb != nil) and replays it per StepWithID; Generate through the ICB must
// equal Generate with the ICB force-disabled (the stepToken path), token-for-token over a
// multi-step prefill+decode. The synthetic model is uniform (no sliding, no MoE, simple rope) so
// it is ICB-eligible — the assertion that state.icb != nil pins that the ICB path is the one
// actually exercised.
func TestGemma4QuantSessionICBParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &g4.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	g, err := AssembleGemma4Quant(ts, arch, &g4.QuantConfig{GroupSize: gs, Bits: bits})
	if err != nil {
		t.Fatalf("AssembleGemma4Quant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	prompt := []int32{1, 5, 3, 2}

	// ICB path: the eligible session records + replays the recorded arch ICB.
	sessICB, err := NewGemma4QuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewGemma4QuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the uniform E2B-shaped session to be ICB-eligible (icb recorded) — the parity check is meaningless if the ICB path is not exercised")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	// stepToken path: a fresh identical session with the ICB force-disabled.
	sessHost, err := NewGemma4QuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewGemma4QuantSession (host): %v", err)
	}
	sessHost.state.icb = nil // force the stepToken host re-encode path
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — the incremental ICB replay is NOT byte-identical to stepToken", i, genICB[i], genHost[i])
		}
	}
}
