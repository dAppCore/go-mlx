// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// f32ToBF16Bytes writes v as the two bf16 bytes the seam uses (the high 16 bits
// of the f32, little-endian within the 16). Small integers are exact in bf16, so
// the counter model round-trips its token ids losslessly.
func f32ToBF16Bytes(v float32) (lo, hi byte) {
	h := uint16(math.Float32bits(v) >> 16)
	return byte(h), byte(h >> 8)
}

// counterModel is a deterministic fake TokenModel: it encodes a token id in
// hidden dim 0, its decode is the identity stack (so the last hidden carries the
// last input id), and its head emits a one-hot logit at (id+1) mod vocab. So
// greedy generation from [k] yields k+1, k+2, … — a sequence that ONLY stays a
// clean count if Generate re-embeds each generated token into the running
// sequence (a broken re-embed breaks the count at the first generated token).
type counterModel struct {
	vocab  int
	dModel int
}

func (m counterModel) Vocab() int { return m.vocab }

func (m counterModel) Embed(id int32) ([]byte, error) {
	emb := make([]byte, m.dModel*bf16Size)
	emb[0], emb[1] = f32ToBF16Bytes(float32(id)) // id in dim 0, rest zero
	return emb, nil
}

// DecodeForward is the identity stack: each output hidden equals its input
// embedding, so the last hidden carries the last token's id.
func (m counterModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return inputs, nil
}

func (m counterModel) Head(hidden []byte) ([]byte, error) {
	id := int(math.Round(float64(bf16ToF32(hidden[0], hidden[1]))))
	target := (id + 1) % m.vocab
	logits := make([]byte, m.vocab*bf16Size)
	logits[target*bf16Size], logits[target*bf16Size+1] = f32ToBF16Bytes(1) // one-hot at id+1
	return logits, nil
}

func idsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGenerate_CounterLoop(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}

	// greedy from [0] for 5 tokens → 1,2,3,4,5 (only correct if re-embed feeds
	// each generated token back into the next step).
	got, err := Generate(m, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{1, 2, 3, 4, 5}; !idsEqual(got, want) {
		t.Fatalf("greedy count = %v, want %v", got, want)
	}

	// eos stops the loop the moment that token is generated (3 reached on the
	// third step), so maxNew=10 still ends at [1,2,3].
	got, err = Generate(m, []int32{0}, 10, 3)
	if err != nil {
		t.Fatalf("Generate eos: %v", err)
	}
	if want := []int32{1, 2, 3}; !idsEqual(got, want) {
		t.Fatalf("eos count = %v, want %v", got, want)
	}

	// the prompt's LAST id drives the first generated token (prompt [5,9] → the
	// 9 leads, so 10,11,12).
	got, err = Generate(m, []int32{5, 9}, 3, -1)
	if err != nil {
		t.Fatalf("Generate multi-prompt: %v", err)
	}
	if want := []int32{10, 11, 12}; !idsEqual(got, want) {
		t.Fatalf("multi-prompt count = %v, want %v", got, want)
	}
}

func TestGenerate_Errors(t *testing.T) {
	m := counterModel{vocab: 8, dModel: 2}
	if _, err := Generate(nil, []int32{0}, 4, -1); err == nil {
		t.Fatal("nil model should error")
	}
	if _, err := Generate(m, nil, 4, -1); err == nil {
		t.Fatal("empty prompt should error")
	}
	if _, err := Generate(m, []int32{0}, 0, -1); err == nil {
		t.Fatal("maxNew <= 0 should error")
	}
}

func TestGenerateSampled_ZeroTempIsGreedy(t *testing.T) {
	m := counterModel{vocab: 16, dModel: 4}
	greedy, err := Generate(m, []int32{0}, 6, -1)
	if err != nil {
		t.Fatalf("greedy: %v", err)
	}
	// temperature 0 → the sampler falls back to greedy per token, so the
	// stochastic path reproduces the greedy sequence exactly.
	sampled, err := GenerateSampled(m, NewSampler(1), SampleParams{Temperature: 0}, []int32{0}, 6, -1)
	if err != nil {
		t.Fatalf("sampled: %v", err)
	}
	if !idsEqual(greedy, sampled) {
		t.Fatalf("zero-temp sampled %v != greedy %v", sampled, greedy)
	}
	if _, err := GenerateSampled(m, nil, SampleParams{}, []int32{0}, 4, -1); err == nil {
		t.Fatal("nil sampler should error")
	}
}

// counterStepper is the incremental decode of counterModel: the counter is
// memoryless (next = id+1), so the last token's embedding IS its hidden state —
// the identity step. It carries no cache because nothing depends on history.
type counterStepper struct{}

func (counterStepper) Step(emb []byte) ([]byte, error) { return emb, nil }

// sessionCounterModel is counterModel that ALSO offers a persistent-cache
// session — but whose whole-sequence DecodeForward ERRORS, so a passing
// generation proves Generate took the incremental SessionModel path.
type sessionCounterModel struct {
	counterModel
	opened *int
}

func (sessionCounterModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return nil, core.NewError("whole-seq path must not run when a session is available")
}

func (m sessionCounterModel) OpenSession() (DecodeStepper, error) {
	if m.opened != nil {
		*m.opened++
	}
	return counterStepper{}, nil
}

func TestGenerate_SessionPath(t *testing.T) {
	var _ SessionModel = sessionCounterModel{} // compile-time: it offers the incremental path

	opened := 0
	m := sessionCounterModel{counterModel: counterModel{vocab: 16, dModel: 4}, opened: &opened}

	// Generate must dispatch to the incremental session path — its DecodeForward
	// errors, so any produced token proves the whole-seq fallback was NOT used.
	got, err := Generate(m, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate (session path): %v", err)
	}
	if want := []int32{1, 2, 3, 4, 5}; !idsEqual(got, want) {
		t.Fatalf("session-path count = %v, want %v", got, want)
	}
	if opened != 1 {
		t.Fatalf("OpenSession called %d times, want exactly 1", opened)
	}

	// the incremental path is output-identical to the whole-seq fallback on the
	// equivalent session-less model.
	wholeSeq, err := Generate(counterModel{vocab: 16, dModel: 4}, []int32{0}, 5, -1)
	if err != nil {
		t.Fatalf("Generate (whole-seq): %v", err)
	}
	if !idsEqual(got, wholeSeq) {
		t.Fatalf("session %v != whole-seq %v", got, wholeSeq)
	}
}
