// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestArchQuantSessionICBParity proves the incremental ICB encode-bypass (Phase B) is
// byte-identical to the stepToken host-encode path: an eligible E2B-shaped PLE session records
// the arch ICB (state.icb != nil) and replays it per StepWithID; Generate through the ICB must
// equal Generate with the ICB force-disabled (the stepToken path), token-for-token over a
// multi-step prefill+decode. The synthetic model is uniform (no sliding, no MoE, simple rope) so
// it is ICB-eligible — the assertion that state.icb != nil pins that the ICB path is the one
// actually exercised.
func TestArchQuantSessionICBParity(t *testing.T) {
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
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := g4.Assemble(ts, arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	prompt := []int32{1, 5, 3, 2}

	// ICB path: the eligible session records + replays the recorded arch ICB.
	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the uniform E2B-shaped session to be ICB-eligible (icb recorded) — the parity check is meaningless if the ICB path is not exercised")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	// stepToken path: a fresh identical session with the ICB force-disabled.
	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
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

// TestArchQuantSessionICBParity_PerLayerRope exercises the NEW per-layer rope branches: a model
// with a sliding layer (rope theta 10000) + a global layer (theta 1000000) so localBase != base —
// the exact shape (sliding/global different theta) that gates real gemma4 E2B. The ICB must rope each
// layer on its own base (the recorder's ropeLocalBaseB vs ropeBaseB), matching the host stepToken
// pick token-for-token. If the per-layer rope were wrong, the bases would diverge and the tokens drift.
func TestArchQuantSessionICBParity_PerLayerRope(t *testing.T) {
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
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeLocalBase == arch.RopeBase {
		t.Fatalf("fixture must have localBase != base to exercise per-layer rope (both %v)", arch.RopeBase)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := g4.Assemble(ts, arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the per-layer-rope session to be ICB-eligible (icb recorded)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — per-layer rope (sliding localBase=%v vs base=%v) ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], arch.RopeLocalBase, arch.RopeBase)
		}
	}
}

// TestArchQuantSessionICBParity_PerLayerHeadDim exercises the per-layer HEAD DIM path: a sliding
// layer (head_dim 64) + a global layer (head_dim 128 via global_head_dim) — gemma4's real shape (E2B:
// 256 sliding / 512 global). The ICB sizes the KV cache + attention scratch per layer, picks the SDPA
// PSO + qmv dim buffers per hd, and must decode token-identical to stepToken.
func TestArchQuantSessionICBParity_PerLayerHeadDim(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, globalHeadDim, dFF, vocab = 256, 2, 1, 64, 128, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, GlobalHeadDim: globalHeadDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.GlobalHeadDim == arch.HeadDim {
		t.Fatalf("fixture must have globalHeadDim != headDim to exercise per-layer head dim (both %d)", arch.HeadDim)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := g4.Assemble(ts, arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the per-layer-head-dim session to be ICB-eligible (icb recorded)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — per-layer head dim (sliding %d / global %d) ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], arch.HeadDim, arch.GlobalHeadDim)
		}
	}
}
