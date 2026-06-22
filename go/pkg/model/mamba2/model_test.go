// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

func mkModel(cfg BlockConfig, D, vocab, nLayers int) *MambaModel {
	layers := make([]MambaLayer, nLayers)
	for li := range layers {
		layers[li] = MambaLayer{Norm: syn(D, li*9+1), W: mkBlockWeights(cfg, D)}
	}
	return &MambaModel{
		Embed:  syn(vocab*D, 100),
		NormF:  syn(D, 101),
		LMHead: nil, // tied to Embed
		Layers: layers,
		Cfg:    cfg,
		D:      D,
		Vocab:  vocab,
	}
}

// TestMambaDecodeEqualsPrefill is the recurrent-decode correctness: stepping a sequence one token at a
// time through a fresh session (each step O(1), threading the per-layer conv + SSM state) produces hidden
// states BIT-EXACT to a single prefill pass over the whole sequence. This is the SSM analogue of the KV
// cache being byte-faithful — what makes streaming Mamba-2 decode correct.
func TestMambaDecodeEqualsPrefill(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const D, vocab, nLayers = 8, 32, 2
	m := mkModel(cfg, D, vocab, nLayers)
	tokens := []int32{1, 5, 9, 2, 7, 3}

	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}

	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode step %d: %v", t0, err)
		}
		for i := 0; i < D; i++ {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (recurrent decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("mamba2 recurrent decode == prefill bit-exact over %d tokens, %d layers", len(tokens), nLayers)
}

// TestMambaGenerate checks the greedy generate loop runs and is deterministic (same prompt → same
// tokens), exercising prefill + the per-token recurrent decode + the LM head.
func TestMambaGenerate(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	prompt := []int32{1, 2, 3}
	g1, err := NewSession(m).Generate(prompt, 6, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(g1) != 6 {
		t.Fatalf("generated %d tokens, want 6", len(g1))
	}
	g2, _ := NewSession(m).Generate(prompt, 6, -1)
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("non-deterministic generate at %d: %d != %d", i, g1[i], g2[i])
		}
	}
	t.Logf("mamba2 Generate: prefill→recurrent decode→head produced %v (deterministic)", g1)
}
