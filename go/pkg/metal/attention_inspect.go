// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"sync"

	core "dappco.re/go"
)

// InspectAttention runs a single prefill pass and returns post-RoPE K tensors.
// Result.Keys is indexed [layer][head], each slice is seq_len*head_dim float32.
//
//	result, err := m.InspectAttention(ctx, "What is kindness?")
//	fmt.Printf("layers=%d heads=%d seq=%d\n", result.NumLayers, result.NumHeads, result.SeqLen)
func (m *Model) InspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	if err := m.requireTextRuntime("Model.InspectAttention"); err != nil {
		return nil, err
	}
	var (
		result *AttentionResult
		err    error
	)
	release, slotErr := m.acquireSlot(ctx)
	if slotErr != nil {
		return nil, slotErr
	}
	defer release()
	if deviceErr := m.withDevice(func() {
		result, err = m.inspectAttention(ctx, prompt)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return result, err
}

func (m *Model) inspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	tokens := m.tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, core.E("Model.InspectAttention", "empty prompt after tokenisation", nil)
	}

	caches := m.newCaches()
	defer FreeCaches(caches)

	vInput := FromValues(tokens, len(tokens))
	input := Reshape2(vInput, 1, int32(len(tokens)))
	Free(vInput)
	logits := m.model.Forward(input, caches)
	defer Free(logits)
	Free(input)
	if err := Eval(logits); err != nil {
		return nil, core.E("Model.InspectAttention", "prefill", err)
	}
	detachEvalState(logits, caches)

	info := m.Info()
	seqLen := len(tokens)

	keys := make([][][]float32, info.NumLayers)
	cacheIndexByLayer := attentionCacheIndexByLayer(m.model, info.NumLayers, len(caches))
	cacheSnapshots := make(map[int]attentionCacheSnapshot, len(caches))
	var numHeads, headDim int

	for layerIdx, cacheIdx := range cacheIndexByLayer {
		if cacheIdx < 0 {
			continue
		}
		snapshot, ok := cacheSnapshots[cacheIdx]
		if !ok {
			var extracted bool
			snapshot, extracted = inspectAttentionCache(caches[cacheIdx], seqLen)
			if !extracted {
				continue
			}
			cacheSnapshots[cacheIdx] = snapshot
		}
		// A freshly extracted snapshot (ok == false) is single-owner —
		// inspectAttentionCache just allocated its per-head buffers for this
		// layer alone, so reference them directly. Only a snapshot REUSED from
		// the map (ok == true, a cache shared across layers) must clone to avoid
		// two layers aliasing the same backing arrays. Drops the redundant
		// per-head data copy on the common 1:1-mapped path (allocs/op + B/op).
		// Mirrors the snapshotKVCachesWithOptions single-owner-heads fix.
		if ok {
			keys[layerIdx] = cloneAttentionHeads(snapshot.Keys)
		} else {
			keys[layerIdx] = snapshot.Keys
		}
		if numHeads == 0 {
			numHeads = snapshot.NumHeads
		}
		if headDim == 0 {
			headDim = snapshot.HeadDim
		}
	}

	return &AttentionResult{
		NumLayers:     info.NumLayers,
		NumHeads:      numHeads,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		NumQueryHeads: attentionQueryHeads(m.model),
		Keys:          keys,
		Architecture:  info.Architecture,
	}, nil
}

type attentionCacheSnapshot struct {
	NumHeads int
	HeadDim  int
	Keys     [][]float32
}

func attentionCacheIndexByLayer(model InternalModel, numLayers, numCaches int) []int {
	if layouter, ok := model.(AttentionCacheLayouter); ok {
		return layouter.AttentionCacheLayout(numLayers, numCaches)
	}
	if planner, ok := model.(HybridAttentionCachePlanner); ok {
		return hybridAttentionCacheIndexByLayer(planner, numLayers, numCaches)
	}

	// Default: identity mapping (layer i → cache i), capped by cache count.
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	limit := min(numCaches, numLayers)
	for i := 0; i < limit; i++ {
		cacheIndexByLayer[i] = i
	}
	return cacheIndexByLayer
}

func hybridAttentionCacheIndexByLayer(model HybridAttentionCachePlanner, numLayers, numCaches int) []int {
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	plan, ok := model.HybridAttentionCachePlan()
	if !ok {
		return cacheIndexByLayer
	}
	for layerIdx := 0; layerIdx < numLayers && layerIdx < len(plan.CacheIndexByLayer); layerIdx++ {
		cacheIdx := plan.CacheIndexByLayer[layerIdx]
		if cacheIdx >= 0 && cacheIdx < numCaches {
			cacheIndexByLayer[layerIdx] = cacheIdx
		}
	}
	return cacheIndexByLayer
}

func inspectAttentionCache(cache Cache, seqLen int) (attentionCacheSnapshot, bool) {
	if cache == nil {
		return attentionCacheSnapshot{}, false
	}
	state, ownedState := CacheReadState(cache)
	defer Free(ownedState...)
	if len(state) < 1 {
		return attentionCacheSnapshot{}, false
	}
	kArray := state[0] // K tensor from cache: [B, H, L_alloc, D]
	shape := kArray.Shape()
	if len(shape) != 4 {
		return attentionCacheSnapshot{}, false
	}

	numHeads := int(shape[1])
	headDim := int(shape[3])
	validLen := min(cache.Len(), seqLen)
	if validLen <= 0 {
		return attentionCacheSnapshot{}, false
	}

	kSliced := Slice(kArray, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(validLen), shape[3]})
	if err := Eval(kSliced); err != nil {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}

	// W11-X / W11-AE: borrow an MLX-memory view rather than copying the full
	// [1, H, L, D] K-tensor into a fresh Go []float32 (Floats() does
	// make + per-element copy — ~16MB on a 32-head/1024-token/128-dim
	// cache).  Per-head slices are copied into independent buffers via
	// the loop below, so the borrowed view ends at function return.
	// W11-AE: kSliced was Eval'd above, so the fast-path skips the final
	// Materialize crossing when dtype + layout already match.
	flat, flatCleanup, err := materialiseFloat32ViewFast(kSliced)
	if err != nil {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}
	defer flatCleanup()
	if len(flat) == 0 {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}

	keys := make([][]float32, numHeads)
	stride := validLen * headDim
	for h := range numHeads {
		start := h * stride
		end := start + stride
		if end > len(flat) {
			break
		}
		head := make([]float32, stride)
		copy(head, flat[start:end])
		keys[h] = head
	}
	Free(kSliced)

	return attentionCacheSnapshot{
		NumHeads: numHeads,
		HeadDim:  headDim,
		Keys:     keys,
	}, true
}

func cloneAttentionHeads(src [][]float32) [][]float32 {
	if len(src) == 0 {
		return nil
	}
	cloned := make([][]float32, len(src))
	for i, head := range src {
		if len(head) == 0 {
			continue
		}
		buf := make([]float32, len(head))
		copy(buf, head)
		cloned[i] = buf
	}
	return cloned
}

// AttentionResult holds extracted K vectors from the KV cache.
type AttentionResult struct {
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	Keys          [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
	Queries       [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
	Architecture  string
}

func attentionQueryHeads(model InternalModel) int {
	if counter, ok := model.(QueryHeadCounter); ok {
		return counter.NumQueryHeads()
	}
	return 0
}

// repeatPenaltyScratch is a pooled []int32 buffer reused for history dedup
// inside applyRepeatPenalty.  Sampling fires once per emitted token, so
// recycling the dedup scratch eliminates the map+slice allocation pair on
// the per-token hot path.  Capacity grows as needed and stays in the pool.
var repeatPenaltyScratch = sync.Pool{
	New: func() any {
		buf := make([]int32, 0, 64)
		return &buf
	},
}
