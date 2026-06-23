// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestStepGreedyMatchesSerial gates the chained decode op: the next-token sequence from driving
// stepGreedyInPool (ICB replay + LM head + argmax in ONE command buffer) must equal the serial
// GenerateFromCache (greedy then stepID, two command buffers). Same greedy on the same hiddens, just
// fused into one submission — a bug in the chaining (wrong buffer, missed dependency) diverges the tokens.
func TestStepGreedyMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, N = 32, 8
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 256, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	prompt := []int32{1, 5, 3, 7}

	sA, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session A: %v", err)
	}
	if err := sA.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill A: %v", err)
	}
	serial, err := sA.GenerateFromCache(N, -1)
	_ = sA.Close()
	if err != nil {
		t.Fatalf("serial generate: %v", err)
	}

	sB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session B: %v", err)
	}
	if err := sB.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill B: %v", err)
	}
	combined := make([]int32, 0, N)
	withAutoreleasePool(func() {
		hidden := append([]byte(nil), sB.retainedHidden...)
		first, ok, gerr := sB.greedy(hidden, nil)
		if gerr != nil {
			t.Fatalf("first greedy: %v", gerr)
		}
		if !ok {
			t.Skip("head has no GPU argmax path on this fixture")
		}
		combined = append(combined, first)
		next := first
		for len(combined) < N {
			emb, eerr := sB.embed(next)
			if eerr != nil {
				t.Fatalf("embed: %v", eerr)
			}
			n2, _, sok, serr := sB.stepGreedyInPool(next, emb, nil)
			if serr != nil {
				t.Fatalf("stepGreedy: %v", serr)
			}
			if !sok {
				t.Fatal("stepGreedyInPool returned ok=false on a quant session with a GPU-argmax head")
			}
			combined = append(combined, n2)
			next = n2
		}
	})
	_ = sB.Close()

	if len(combined) != len(serial) {
		t.Fatalf("len %d vs %d", len(combined), len(serial))
	}
	for i := range serial {
		if combined[i] != serial[i] {
			t.Fatalf("tok%d: chained=%d vs serial=%d (chained=%v serial=%v)", i, combined[i], serial[i], combined, serial)
		}
	}
	t.Logf("chained stepGreedy matches serial: %v", serial)
}
