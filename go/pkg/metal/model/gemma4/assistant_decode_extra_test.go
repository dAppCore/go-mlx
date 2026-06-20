// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Hand-built coverage for the assistant decode guard/error seams that the
// speculative-decode end-to-end tests never isolate: nil-receiver Close paths,
// the previous-hidden reshape arms, the per-layer/per-attention completeness
// guards, the cache-read guards, and the ordered-embedding candidate guards.
// None of these load a model — they drive the validation arms directly on
// hand-built structs and tensors, so they stay inside the AX-11 no-model rule.

// TestGemma4AssistantDecode_DraftStepResultClose_NilSafe pins the nil-receiver
// guard: Close on a nil *Gemma4AssistantDraftStepResult is a no-op, and Close on
// a populated one frees and nils every owned array.
func TestGemma4AssistantDecode_DraftStepResultClose_NilSafe(t *testing.T) {
	requireMetalRuntime(t)

	var nilResult *Gemma4AssistantDraftStepResult
	nilResult.Close() // must not panic

	result := &Gemma4AssistantDraftStepResult{
		Logits: seqArray(0.1, 1, 1, 4),
		Token:  metal.FromSingleInt32Matrix(2),
		Hidden: seqArray(0.2, 1, 1, 8),
	}
	result.Close()
	if result.Logits != nil || result.Token != nil || result.Hidden != nil {
		t.Fatalf("Close left non-nil arrays: logits=%v token=%v hidden=%v", result.Logits, result.Token, result.Hidden)
	}
}

// TestGemma4AssistantDecode_DraftBlockResultClose_NilSafe pins the nil-receiver
// guard for the block result and that Close clears the token slice + hidden.
func TestGemma4AssistantDecode_DraftBlockResultClose_NilSafe(t *testing.T) {
	requireMetalRuntime(t)

	var nilBlock *Gemma4AssistantDraftBlockResult
	nilBlock.Close() // must not panic

	block := &Gemma4AssistantDraftBlockResult{
		Tokens: []int32{1, 2, 3},
		Hidden: seqArray(0.3, 1, 1, 8),
	}
	block.Close()
	if block.Hidden != nil || block.Tokens != nil {
		t.Fatalf("Close left block state: hidden=%v tokens=%v", block.Hidden, block.Tokens)
	}
}

// TestGemma4AssistantDecode_VerifyResultClose_NilSafe pins the nil-receiver
// guard for the verify result and that Close clears every owned slice + array.
func TestGemma4AssistantDecode_VerifyResultClose_NilSafe(t *testing.T) {
	requireMetalRuntime(t)

	var nilVerify *Gemma4AssistantVerifyResult
	nilVerify.Close() // must not panic

	result := &Gemma4AssistantVerifyResult{
		DraftedTokens:  []int32{1},
		TargetTokens:   []int32{1},
		AcceptedTokens: []int32{1},
		RejectedTokens: []int32{2},
		Logits:         seqArray(0.1, 1, 1, 4),
		Hidden:         seqArray(0.2, 1, 1, 8),
	}
	result.Close()
	if result.Logits != nil || result.Hidden != nil || result.Caches != nil {
		t.Fatalf("Close left arrays/caches: logits=%v hidden=%v caches=%v", result.Logits, result.Hidden, result.Caches)
	}
	if result.DraftedTokens != nil || result.TargetTokens != nil || result.AcceptedTokens != nil || result.RejectedTokens != nil {
		t.Fatal("Close left token slices populated, want all cleared")
	}
}

// TestGemma4AssistantDecode_BackboneHidden_Reshapes drives all three accepted
// previous-hidden shapes through gemma4AssistantBackboneHidden: the already-3D
// [1,1,H] case is returned as-is (not owned), and the 2D [1,H] and 1D [H] cases
// are reshaped to [1,1,H] and reported as caller-owned so the decode path frees
// the reshape.
func TestGemma4AssistantDecode_BackboneHidden_Reshapes(t *testing.T) {
	requireMetalRuntime(t)

	const backbone = int32(8)

	// Rank-3 [1,1,H]: passthrough, not owned.
	h3 := seqArray(0.05, 1, 1, int(backbone))
	defer metal.Free(h3)
	got3, owned3, err := gemma4AssistantBackboneHidden(h3, backbone)
	if err != nil {
		t.Fatalf("backboneHidden rank-3: %v", err)
	}
	if owned3 {
		t.Fatal("rank-3 passthrough reported owned, want shared (not reshaped)")
	}
	if got3 != h3 {
		t.Fatal("rank-3 passthrough returned a different array, want the input")
	}

	// Rank-2 [1,H]: reshaped to [1,1,H], owned.
	h2 := seqArray(0.05, 1, int(backbone))
	defer metal.Free(h2)
	got2, owned2, err := gemma4AssistantBackboneHidden(h2, backbone)
	if err != nil {
		t.Fatalf("backboneHidden rank-2: %v", err)
	}
	if !owned2 {
		t.Fatal("rank-2 reshape reported not owned, want caller-owned")
	}
	defer metal.Free(got2)
	assertShape(t, "rank-2 reshape", got2, []int32{1, 1, backbone})

	// Rank-1 [H]: reshaped to [1,1,H], owned.
	h1 := seqArray(0.05, int(backbone))
	defer metal.Free(h1)
	got1, owned1, err := gemma4AssistantBackboneHidden(h1, backbone)
	if err != nil {
		t.Fatalf("backboneHidden rank-1: %v", err)
	}
	if !owned1 {
		t.Fatal("rank-1 reshape reported not owned, want caller-owned")
	}
	defer metal.Free(got1)
	assertShape(t, "rank-1 reshape", got1, []int32{1, 1, backbone})
}

// TestGemma4AssistantDecode_BackboneHidden_Bad covers the default arm: a hidden
// whose last dim is not the backbone size is rejected with a shape error.
func TestGemma4AssistantDecode_BackboneHidden_Bad(t *testing.T) {
	requireMetalRuntime(t)

	bad := seqArray(0.05, 1, 1, 7) // backbone is 8
	defer metal.Free(bad)
	if _, _, err := gemma4AssistantBackboneHidden(bad, 8); err == nil {
		t.Fatal("backboneHidden(bad) error = nil, want previous-hidden shape error")
	} else if !core.Contains(err.Error(), "previous hidden shape") {
		t.Fatalf("backboneHidden(bad) error = %v, want previous hidden shape", err)
	}
}

// TestGemma4AssistantDecode_CloneArray_Bad covers cloneGemma4AssistantArray's
// invalid-input guard: a nil array cannot be cloned and returns the hoisted
// clone error.
func TestGemma4AssistantDecode_CloneArray_Bad(t *testing.T) {
	if _, err := cloneGemma4AssistantArray(nil); err == nil {
		t.Fatal("cloneGemma4AssistantArray(nil) error = nil, want invalid-array error")
	} else if !core.Contains(err.Error(), "invalid array") {
		t.Fatalf("cloneGemma4AssistantArray(nil) error = %v, want invalid array", err)
	}
}

// TestGemma4AssistantDecode_CloneArray_Good covers the clone success path: a
// valid array is copied, evaluated, and detached into a standalone array whose
// values match the source.
func TestGemma4AssistantDecode_CloneArray_Good(t *testing.T) {
	requireMetalRuntime(t)

	src := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 4)
	defer metal.Free(src)
	cloned, err := cloneGemma4AssistantArray(src)
	if err != nil {
		t.Fatalf("cloneGemma4AssistantArray: %v", err)
	}
	defer metal.Free(cloned)
	if cloned == src {
		t.Fatal("clone returned the source array, want an independent copy")
	}
	assertShape(t, "clone", cloned, []int32{1, 1, 4})
	if got := cloned.Floats(); len(got) != 4 || got[0] != 1 || got[3] != 4 {
		t.Fatalf("clone values = %v, want [1 2 3 4]", got)
	}
}

// TestGemma4AssistantDecode_GreedyToken_ArgmaxPath covers the no-suppress arm of
// gemma4AssistantGreedyToken: with no suppress list it argmaxes the logits and
// reads back the peak id directly (no suppression-guard sampler).
func TestGemma4AssistantDecode_GreedyToken_ArgmaxPath(t *testing.T) {
	requireMetalRuntime(t)

	logits := metal.FromValues([]float32{0.1, 0.4, 9.0, 0.2}, 1, 1, 4)
	defer metal.Free(logits)
	got, err := gemma4AssistantGreedyToken(logits)
	if err != nil {
		t.Fatalf("gemma4AssistantGreedyToken(argmax): %v", err)
	}
	if got != 2 {
		t.Fatalf("greedy token = %d, want peak column 2", got)
	}
}

// TestGemma4AssistantDecode_ForwardWithTargetKV_Incomplete covers the
// attention-completeness guard: an attention struct missing its projections is
// rejected before any matmul.
func TestGemma4AssistantDecode_ForwardWithTargetKV_Incomplete(t *testing.T) {
	attn := &Gemma4AssistantAttention{} // no QProj/OProj/QNorm
	_, err := attn.forwardWithTargetKV(nil, sharedKV{}, 1, 1, &Gemma4TextConfig{})
	if err == nil {
		t.Fatal("forwardWithTargetKV(incomplete) error = nil, want incomplete-attention error")
	}
	if !core.Contains(err.Error(), "attention is incomplete") {
		t.Fatalf("forwardWithTargetKV(incomplete) error = %v, want attention incomplete", err)
	}
}

// TestGemma4AssistantDecode_ForwardWithTargetKV_MissingKV covers the missing-KV
// guard: a complete attention with an empty sharedKV (no keys/values/pages) is
// rejected because there is no target stream to attend over.
func TestGemma4AssistantDecode_ForwardWithTargetKV_MissingKV(t *testing.T) {
	requireMetalRuntime(t)

	attn := newTinyAssistantAttention()
	defer freeTinyAssistantAttention(attn)
	x := seqArray(0.05, 1, 1, 4)
	defer metal.Free(x)
	_, err := attn.forwardWithTargetKV(x, sharedKV{}, 1, 1, &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{RMSNormEps: 1e-6}})
	if err == nil {
		t.Fatal("forwardWithTargetKV(no KV) error = nil, want missing target K/V error")
	}
	if !core.Contains(err.Error(), "missing target K/V") {
		t.Fatalf("forwardWithTargetKV(no KV) error = %v, want missing target K/V", err)
	}
}

// TestGemma4AssistantDecode_ForwardDraftStep_Incomplete covers the layer
// completeness guard: a layer missing its attention/MLP is rejected.
func TestGemma4AssistantDecode_ForwardDraftStep_Incomplete(t *testing.T) {
	layer := &Gemma4AssistantLayer{} // no Attention/MLP
	_, err := layer.forwardDraftStep(nil, sharedKV{}, &Gemma4TextConfig{})
	if err == nil {
		t.Fatal("forwardDraftStep(incomplete) error = nil, want incomplete-layer error")
	}
	if !core.Contains(err.Error(), "layer is incomplete") {
		t.Fatalf("forwardDraftStep(incomplete) error = %v, want layer incomplete", err)
	}
}

// TestGemma4AssistantDecode_ForwardDraftStep_BadRank covers the input-rank guard:
// the draft step only accepts a rank-3 [batch, sequence, hidden] input, so a
// rank-2 input is rejected before the norm.
func TestGemma4AssistantDecode_ForwardDraftStep_BadRank(t *testing.T) {
	requireMetalRuntime(t)

	attn := newTinyAssistantAttention()
	defer freeTinyAssistantAttention(attn)
	layer := &Gemma4AssistantLayer{Attention: attn, MLP: &metal.MLP{}}
	rank2 := seqArray(0.05, 1, 4) // not rank-3
	defer metal.Free(rank2)
	_, err := layer.forwardDraftStep(rank2, sharedKV{}, &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{RMSNormEps: 1e-6}})
	if err == nil {
		t.Fatal("forwardDraftStep(rank-2) error = nil, want input-shape error")
	}
	if !core.Contains(err.Error(), "layer input shape") {
		t.Fatalf("forwardDraftStep(rank-2) error = %v, want layer input shape", err)
	}
}

// TestGemma4AssistantDecode_ForwardDraftStep_BadSeq covers the [1,1,hidden]-only
// guard: the draft step is a single-token step, so a multi-token sequence is
// rejected even though the rank is correct.
func TestGemma4AssistantDecode_ForwardDraftStep_BadSeq(t *testing.T) {
	requireMetalRuntime(t)

	attn := newTinyAssistantAttention()
	defer freeTinyAssistantAttention(attn)
	layer := &Gemma4AssistantLayer{Attention: attn, MLP: &metal.MLP{}}
	multiSeq := seqArray(0.05, 1, 2, 4) // L=2, not a single decode step
	defer metal.Free(multiSeq)
	_, err := layer.forwardDraftStep(multiSeq, sharedKV{}, &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{RMSNormEps: 1e-6}})
	if err == nil {
		t.Fatal("forwardDraftStep(L=2) error = nil, want single-step shape error")
	}
	if !core.Contains(err.Error(), "only supports [1 1 hidden]") {
		t.Fatalf("forwardDraftStep(L=2) error = %v, want single-step shape", err)
	}
}

// TestGemma4AssistantDecode_KVFromCache_EmptyCache covers gemma4AssistantKVFromCache's
// empty-cache guard: a nil cache (Len()<=0) is rejected with the empty-cache error.
func TestGemma4AssistantDecode_KVFromCache_EmptyCache(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	// A fresh, never-prefilled cache has Len()==0 — the empty guard fires.
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) == 0 {
		t.Skip("target produced no caches")
	}
	if _, err := gemma4AssistantKVFromCache(caches[0]); err == nil {
		t.Fatal("KVFromCache(empty) error = nil, want empty-cache error")
	}
}

// TestGemma4AssistantDecode_OrderedCandidates_Guards covers the four ordered-
// embedding candidate validation arms reachable without any GPU work: missing
// centroids, missing token ordering, an invalid vocab size, and an invalid
// top-K. Each returns its hoisted guard error before the centroid forward.
func TestGemma4AssistantDecode_OrderedCandidates_Guards(t *testing.T) {
	requireMetalRuntime(t)

	hidden := seqArray(0.05, 1, 1, 4)
	defer metal.Free(hidden)

	// Missing centroids.
	mNoCentroids := &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 10}}}
	if _, err := mNoCentroids.orderedEmbeddingCandidates(hidden); err == nil {
		t.Fatal("orderedCandidates(no centroids) error = nil, want centroids error")
	} else if !core.Contains(err.Error(), "centroids") {
		t.Fatalf("orderedCandidates(no centroids) error = %v, want centroids", err)
	}

	// Centroids present but no token ordering.
	mNoOrdering := &Gemma4AssistantModel{
		Cfg:             &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 10}},
		MaskedCentroids: metal.NewLinear(seqArray(0.05, 2, 4), nil),
	}
	defer metal.Free(mNoOrdering.MaskedCentroids.Weight)
	if _, err := mNoOrdering.orderedEmbeddingCandidates(hidden); err == nil {
		t.Fatal("orderedCandidates(no ordering) error = nil, want token_ordering error")
	} else if !core.Contains(err.Error(), "token_ordering") {
		t.Fatalf("orderedCandidates(no ordering) error = %v, want token_ordering", err)
	}

	// Ordering present but vocab size invalid.
	mBadVocab := &Gemma4AssistantModel{
		Cfg:             &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 0}},
		MaskedCentroids: metal.NewLinear(seqArray(0.05, 2, 4), nil),
		TokenOrdering:   metal.FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10),
	}
	defer metal.Free(mBadVocab.MaskedCentroids.Weight)
	defer metal.Free(mBadVocab.TokenOrdering)
	if _, err := mBadVocab.orderedEmbeddingCandidates(hidden); err == nil {
		t.Fatal("orderedCandidates(bad vocab) error = nil, want vocab_size error")
	} else if !core.Contains(err.Error(), "vocab_size") {
		t.Fatalf("orderedCandidates(bad vocab) error = %v, want vocab_size", err)
	}

	// Valid vocab/centroids but top-K out of range (topK > numCentroids).
	mBadTopK := &Gemma4AssistantModel{
		Cfg:                      &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 10}},
		MaskedCentroids:          metal.NewLinear(seqArray(0.05, 2, 4), nil),
		TokenOrdering:            metal.FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10),
		NumCentroids:             2,
		CentroidIntermediateTopK: 5, // > numCentroids
	}
	defer metal.Free(mBadTopK.MaskedCentroids.Weight)
	defer metal.Free(mBadTopK.TokenOrdering)
	if _, err := mBadTopK.orderedEmbeddingCandidates(hidden); err == nil {
		t.Fatal("orderedCandidates(bad topK) error = nil, want top_k error")
	} else if !core.Contains(err.Error(), "centroid_intermediate_top_k") {
		t.Fatalf("orderedCandidates(bad topK) error = %v, want centroid_intermediate_top_k", err)
	}
}

// TestGemma4AssistantDecode_SuppressOrderedSparse_AllNegativeIDs covers the
// suppressOrderedEmbeddingSparseLogits no-op-after-filter arm: a suppress list
// whose every id is negative (out of range) filters down to nothing, so the
// function returns the unmodified sparse logits with ownership false rather than
// building a degenerate equality mask.
func TestGemma4AssistantDecode_SuppressOrderedSparse_AllNegativeIDs(t *testing.T) {
	requireMetalRuntime(t)

	selected := metal.FromValues([]int32{0, 1, 2, 3}, 1, 4)
	sparse := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 4)
	defer metal.Free(selected, sparse)

	out, owned, err := suppressOrderedEmbeddingSparseLogits(selected, sparse, []int32{-1, -7})
	if err != nil {
		t.Fatalf("suppressOrderedEmbeddingSparseLogits(all-neg): %v", err)
	}
	if owned {
		t.Fatal("all-negative ids reported owned, want the source logits passed through")
	}
	if out != sparse {
		t.Fatal("all-negative ids returned a new array, want the unmodified sparse logits")
	}
}

// TestGemma4AssistantDecode_SuppressOrderedSparse_NoIDs covers the empty-list
// fast path: no suppress ids returns the sparse logits unmodified, ownership
// false (the caller must not free a borrowed array).
func TestGemma4AssistantDecode_SuppressOrderedSparse_NoIDs(t *testing.T) {
	requireMetalRuntime(t)

	selected := metal.FromValues([]int32{0, 1}, 1, 2)
	sparse := metal.FromValues([]float32{0.1, 0.2}, 1, 2)
	defer metal.Free(selected, sparse)

	out, owned, err := suppressOrderedEmbeddingSparseLogits(selected, sparse, nil)
	if err != nil {
		t.Fatalf("suppressOrderedEmbeddingSparseLogits(nil): %v", err)
	}
	if owned || out != sparse {
		t.Fatalf("nil suppress = owned %v out!=sparse %v, want passthrough", owned, out != sparse)
	}
}

// newTinyAssistantAttention builds a minimal complete Gemma4AssistantAttention
// (Q/O projections + Q norm) for the draft-step shape guards. It owns small
// hand-built weights; free via freeTinyAssistantAttention.
func newTinyAssistantAttention() *Gemma4AssistantAttention {
	return &Gemma4AssistantAttention{
		QProj:   metal.NewLinear(seqArray(0.1, 4, 4), nil),
		OProj:   metal.NewLinear(seqArray(0.1, 4, 4), nil),
		QNorm:   &metal.RMSNormModule{Weight: seqArray(0.1, 4)},
		HeadDim: 4,
		NHeads:  1,
		Scale:   0.5,
	}
}

func freeTinyAssistantAttention(attn *Gemma4AssistantAttention) {
	if attn == nil {
		return
	}
	if attn.QProj != nil {
		metal.Free(attn.QProj.Weight)
	}
	if attn.OProj != nil {
		metal.Free(attn.OProj.Weight)
	}
	if attn.QNorm != nil {
		metal.Free(attn.QNorm.Weight)
	}
}
