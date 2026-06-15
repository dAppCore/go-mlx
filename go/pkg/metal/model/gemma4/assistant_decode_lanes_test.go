// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Decode-step lane coverage reachable from hand-built tensors, without loading a
// model. These exercise the in-graph suppression mask, the assistant attention
// rotary (both the explicit-freqs and the base/dims arms), and the batched
// greedy-pick graph — seams the speculative-decode tests drive end to end but
// never on the exact inputs that isolate each branch.

// suppressLogitsContains reports whether the masked-id column of row 0 in a
// [B, L, vocab] logits readback is -inf and the unmasked columns are not, i.e.
// the suppression actually landed on the requested ids and left the rest alone.
func suppressLogitsContains(values []float32, vocab int, suppressed map[int]bool) (maskedAllNegInf, othersFinite bool) {
	maskedAllNegInf, othersFinite = true, true
	for col := 0; col < vocab && col < len(values); col++ {
		v := float64(values[col])
		if suppressed[col] {
			if !math.IsInf(v, -1) {
				maskedAllNegInf = false
			}
		} else if math.IsInf(v, -1) {
			othersFinite = false
		}
	}
	return maskedAllNegInf, othersFinite
}

// TestGemma4AssistantDecode_SuppressLogits_Good drives the real masking arm: a
// rank-3 [B, L, vocab] logits tensor and a non-empty id list produce a tensor
// where exactly the listed columns are forced to -inf across every position,
// in-graph (PutAlongAxis), leaving the other columns untouched.
func TestGemma4AssistantDecode_SuppressLogits_Good(t *testing.T) {
	requireMetalRuntime(t)

	const B, L, vocab = int32(1), int32(2), int32(6)
	logits := seqArray(0.1, int(B), int(L), int(vocab))
	defer metal.Free(logits)

	suppress := []int32{1, 4}
	out := gemma4AssistantSuppressLogits(logits, suppress)
	if out == nil || !out.Valid() {
		t.Fatal("gemma4AssistantSuppressLogits returned nil, want the masked logits")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval suppressed logits: %v", err)
	}
	assertShape(t, "suppressed logits", out, []int32{B, L, vocab})

	masked := map[int]bool{1: true, 4: true}
	maskedNegInf, othersFinite := suppressLogitsContains(out.Floats(), int(vocab), masked)
	if !maskedNegInf {
		t.Fatal("suppressed columns are not all -inf, want every listed id masked")
	}
	if !othersFinite {
		t.Fatal("an unmasked column became -inf, want only the listed ids masked")
	}
}

// TestGemma4AssistantDecode_SuppressLogits_EmptyIDs_Bad pins the no-op guard:
// an empty id list (nothing to suppress) returns a plain copy of the logits
// rather than building a degenerate scatter, so the caller can fold suppression
// uniformly whether or not any ids are present.
func TestGemma4AssistantDecode_SuppressLogits_EmptyIDs_Bad(t *testing.T) {
	requireMetalRuntime(t)

	const B, L, vocab = int32(1), int32(1), int32(4)
	logits := seqArray(0.2, int(B), int(L), int(vocab))
	defer metal.Free(logits)

	out := gemma4AssistantSuppressLogits(logits, nil)
	if out == nil || !out.Valid() {
		t.Fatal("gemma4AssistantSuppressLogits(empty ids) returned nil, want a copy")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval copy logits: %v", err)
	}
	assertShape(t, "copied logits", out, []int32{B, L, vocab})

	// A copy preserves every value — no column should be -inf.
	_, othersFinite := suppressLogitsContains(out.Floats(), int(vocab), map[int]bool{})
	if !othersFinite {
		t.Fatal("copy path produced a -inf column, want the values preserved verbatim")
	}
}

// TestGemma4AssistantDecode_SuppressLogits_WrongRank_Ugly pins the rank guard:
// a rank-2 logits tensor (not [B, L, vocab]) cannot be scattered along the
// vocab axis the masking assumes, so the function returns a copy rather than
// indexing a missing dimension.
func TestGemma4AssistantDecode_SuppressLogits_WrongRank_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	rank2 := seqArray(0.2, 2, 4) // not rank-3
	defer metal.Free(rank2)

	out := gemma4AssistantSuppressLogits(rank2, []int32{1})
	if out == nil || !out.Valid() {
		t.Fatal("gemma4AssistantSuppressLogits(rank-2) returned nil, want a copy")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval rank-2 copy: %v", err)
	}
	assertShape(t, "rank-2 copy", out, []int32{2, 4})
}

// TestGemma4AssistantDecode_ApplyRoPE_FreqsArm_Good drives the explicit-freqs
// arm of the assistant attention rotary: when RopeFreqs is present, applyRoPE
// routes through RoPEWithFreqs (base ignored, the precomputed inverse
// frequencies drive the rotation). The freqs come from gemma4ProportionalFreqs,
// the same builder the loader uses. Input is the [B, NHeads, L, HeadDim] shape
// forwardWithTargetKV hands in.
func TestGemma4AssistantDecode_ApplyRoPE_FreqsArm_Good(t *testing.T) {
	requireMetalRuntime(t)

	const B, NHeads, L, HeadDim = int32(1), int32(2), int32(3), int32(8)
	freqs := gemma4ProportionalFreqs(HeadDim, HeadDim, 10000.0, 1.0)
	if freqs == nil {
		t.Fatal("gemma4ProportionalFreqs returned nil fixture")
	}
	attn := &Gemma4AssistantAttention{HeadDim: HeadDim, RopeBase: 10000.0, RopeRotatedDim: HeadDim, RopeFreqs: freqs}
	defer metal.Free(attn.RopeFreqs)

	x := seqArray(0.04, int(B), int(NHeads), int(L), int(HeadDim))
	defer metal.Free(x)

	out := attn.applyRoPE(x, 0)
	if out == nil || !out.Valid() {
		t.Fatal("applyRoPE(freqs) returned nil, want the rotated tensor")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval rope-with-freqs: %v", err)
	}
	assertShape(t, "rope freqs", out, []int32{B, NHeads, L, HeadDim})
}

// TestGemma4AssistantDecode_ApplyRoPE_BaseArm_Good drives the no-freqs arm: with
// RopeFreqs nil, applyRoPE routes through metal.RoPE using the rotated dim and
// base. A non-zero offset exercises the position shift the decode step passes
// (the accepted-prefix length), distinguishing this from the prefill path.
func TestGemma4AssistantDecode_ApplyRoPE_BaseArm_Good(t *testing.T) {
	requireMetalRuntime(t)

	const B, NHeads, L, HeadDim = int32(1), int32(2), int32(3), int32(8)
	attn := &Gemma4AssistantAttention{HeadDim: HeadDim, RopeBase: 10000.0, RopeRotatedDim: HeadDim, RopeFreqs: nil}

	x := seqArray(0.04, int(B), int(NHeads), int(L), int(HeadDim))
	defer metal.Free(x)

	out := attn.applyRoPE(x, 5) // non-zero decode offset
	if out == nil || !out.Valid() {
		t.Fatal("applyRoPE(base) returned nil, want the rotated tensor")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval rope-with-base: %v", err)
	}
	assertShape(t, "rope base", out, []int32{B, NHeads, L, HeadDim})
}

// TestGemma4AssistantDecode_BatchedGreedyGraph_Suppress_Good drives the
// suppress arm of gemma4AssistantBatchedGreedyGraph: with a non-empty suppress
// list, both the incoming-position logits and the per-row verify logits are
// masked in-graph before the argmax picks, and the intermediate masked tensors
// are returned for the caller to free post-Eval. The masked argmax must avoid a
// suppressed id when an unmasked id outscores it.
func TestGemma4AssistantDecode_BatchedGreedyGraph_Suppress_Good(t *testing.T) {
	requireMetalRuntime(t)

	const vocab = int32(4)
	// Row peaks at column 3 (highest seq value); suppressing 3 must push the
	// argmax to the next-highest unmasked column (2).
	targetLogits := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.9}, 1, 1, int(vocab))
	allLogits := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.9}, 1, 1, int(vocab))
	defer metal.Free(targetLogits, allLogits)

	firstTok, rowToks, owned := gemma4AssistantBatchedGreedyGraph(targetLogits, allLogits, []int32{3})
	if firstTok == nil || rowToks == nil {
		t.Fatal("gemma4AssistantBatchedGreedyGraph(suppress) returned nil picks")
	}
	if len(owned) != 2 {
		t.Fatalf("suppress arm owned = %d intermediates, want 2 (incoming + all)", len(owned))
	}
	if err := metal.Eval(firstTok, rowToks); err != nil {
		metal.Free(firstTok, rowToks)
		metal.Free(owned...)
		t.Fatalf("Eval batched greedy picks: %v", err)
	}
	first, rows, err := gemma4AssistantReadGreedyTokens(firstTok, rowToks)
	metal.Free(firstTok, rowToks)
	metal.Free(owned...)
	if err != nil {
		t.Fatalf("read greedy tokens: %v", err)
	}
	if first == 3 {
		t.Fatalf("incoming argmax = %d, want a non-suppressed id (3 was masked)", first)
	}
	if len(rows) != 1 || rows[0] == 3 {
		t.Fatalf("row argmax = %v, want a single non-suppressed pick (3 was masked)", rows)
	}
}

// TestGemma4AssistantDecode_BatchedGreedyGraph_NoSuppress_Good pins the
// no-suppress arm for contrast: an empty suppress list takes the argmax of the
// raw logits with no intermediate tensors to free, so the peak column wins
// directly.
func TestGemma4AssistantDecode_BatchedGreedyGraph_NoSuppress_Good(t *testing.T) {
	requireMetalRuntime(t)

	const vocab = int32(4)
	targetLogits := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.9}, 1, 1, int(vocab))
	allLogits := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.9}, 1, 1, int(vocab))
	defer metal.Free(targetLogits, allLogits)

	firstTok, rowToks, owned := gemma4AssistantBatchedGreedyGraph(targetLogits, allLogits, nil)
	if firstTok == nil || rowToks == nil {
		t.Fatal("gemma4AssistantBatchedGreedyGraph(no suppress) returned nil picks")
	}
	if len(owned) != 0 {
		t.Fatalf("no-suppress arm owned = %d intermediates, want 0", len(owned))
	}
	if err := metal.Eval(firstTok, rowToks); err != nil {
		metal.Free(firstTok, rowToks)
		t.Fatalf("Eval batched greedy picks: %v", err)
	}
	first, rows, err := gemma4AssistantReadGreedyTokens(firstTok, rowToks)
	metal.Free(firstTok, rowToks)
	if err != nil {
		t.Fatalf("read greedy tokens: %v", err)
	}
	if first != 3 {
		t.Fatalf("incoming argmax = %d, want the peak column 3", first)
	}
	if len(rows) != 1 || rows[0] != 3 {
		t.Fatalf("row argmax = %v, want the peak column 3", rows)
	}
}
