// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
)

// TestGemma4Backend_ContractParity gates the cgo (mlx-c) backend's token-loop
// contract impl against the model's OWN fused stack: the adapter's DecodeForward
// (precomputed embeddings injected into the proven stack via the embedHook) must
// yield byte-identical hidden states to forwardHidden run on the same tokens, and
// model.Generate over the adapter must produce deterministic in-range tokens. So a
// SECOND real backend (cgo) satisfies the same contract pkg/native (no-cgo) does —
// the contract is backend-agnostic, proven across two backends, not asserted.
func TestGemma4Backend_ContractParity(t *testing.T) {
	m := loadGemma4DenseTestModel(t)
	b, err := NewBackend(m)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	prompt := []int32{2, 3, 4}

	// reference hidden: the model's own stack on the real tokens (fresh cache).
	tokens := metal.FromValues(prompt, 1, len(prompt))
	caches := m.NewCache()
	hRef, _, _ := m.forwardHidden(tokens, nil, caches)
	metal.Materialize(hRef)
	refRaw := append([]byte(nil), hRef.RawBytes()...)
	metal.Free(tokens, hRef)
	metal.FreeCaches(caches)

	// the contract path: embed each token, run DecodeForward (the same embeddings
	// injected into the same stack over a fresh cache).
	embs := make([][]byte, len(prompt))
	for i, id := range prompt {
		e, eerr := b.Embed(id)
		if eerr != nil {
			t.Fatalf("Embed: %v", eerr)
		}
		embs[i] = e
	}
	hGot, err := b.DecodeForward(embs)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	if len(hGot) != len(prompt) {
		t.Fatalf("DecodeForward returned %d hidden states, want %d", len(hGot), len(prompt))
	}
	for tkn := 0; tkn < len(prompt); tkn++ {
		want := refRaw[tkn*b.embBytes : (tkn+1)*b.embBytes]
		if string(hGot[tkn]) != string(want) {
			t.Fatalf("contract hidden[%d] != stack hidden — the adapter diverged from forwardHidden", tkn)
		}
	}

	// model.Generate over the adapter (whole-seq TokenModel path): deterministic,
	// in-range tokens — the shared contract loop drives the cgo backend end to end.
	gen, err := model.Generate(b, prompt, 5, -1)
	if err != nil {
		t.Fatalf("model.Generate: %v", err)
	}
	if len(gen) != 5 {
		t.Fatalf("generated %d tokens, want 5", len(gen))
	}
	for i, id := range gen {
		if id < 0 || int(id) >= b.Vocab() {
			t.Fatalf("token %d = %d out of [0,%d)", i, id, b.Vocab())
		}
	}
	gen2, err := model.Generate(b, prompt, 5, -1)
	if err != nil {
		t.Fatalf("re-run: %v", err)
	}
	for i := range gen {
		if gen[i] != gen2[i] {
			t.Fatalf("non-deterministic at %d: %d vs %d", i, gen[i], gen2[i])
		}
	}
	t.Logf("cgo gemma4 satisfies the token-loop contract: hidden ≡ stack byte-for-byte; model.Generate = %v", gen)
}
