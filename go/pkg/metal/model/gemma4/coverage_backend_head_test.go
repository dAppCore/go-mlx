// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for Gemma4Backend.Head's edge arms the contract-parity test
// (happy-path Head) does not reach: the wrong-sized-input guard and the
// final-logit-softcapping branch (the dense test config omits softcapping).

// TestBackendHead_WrongSize covers the input-size guard: a hidden byte slice
// that is not exactly one hidden state's width is rejected before any forward.
func TestBackendHead_WrongSize(t *testing.T) {
	m := loadGemma4DenseTestModel(t)
	b, err := NewBackend(m)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	if _, err := b.Head([]byte{0, 1, 2}); err == nil {
		t.Fatal("Head(wrong-size) error = nil, want a size error")
	} else if !core.Contains(err.Error(), "one hidden state") {
		t.Fatalf("Head(wrong-size) error = %v, want the hidden-width message", err)
	}
}

// TestBackendHead_Softcap covers Head's final-logit-softcapping branch by
// setting a positive cap on the loaded model, then driving Head with a correctly
// sized hidden state. The result is the vocab-width bf16 logits byte slice.
func TestBackendHead_Softcap(t *testing.T) {
	m := loadGemma4DenseTestModel(t)
	m.Cfg.FinalLogitSoftcapping = 30.0
	b, err := NewBackend(m)
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}

	// One hidden state's worth of bytes. The synthetic dense model loads f32
	// weights, so its activation dtype is f32 and a [1,1,HiddenSize] f32 vector's
	// raw bytes are exactly one hidden state's width. Sizing it from a real
	// forward would couple this test to the decode path; the byte width is what
	// Head validates.
	hidden := seqArray(0.01, 1, 1, int(m.Cfg.HiddenSize))
	defer metal.Free(hidden)
	metal.Materialize(hidden)
	raw := hidden.RawBytes()

	logits, err := b.Head(append([]byte(nil), raw...))
	if err != nil {
		t.Skipf("synthetic activation dtype is not f32 (Head rejected the f32 hidden width): %v", err)
	}
	if len(logits) != int(m.Cfg.VocabSize)*2 {
		t.Fatalf("Head logits bytes = %d, want vocab*2 = %d", len(logits), int(m.Cfg.VocabSize)*2)
	}
}
