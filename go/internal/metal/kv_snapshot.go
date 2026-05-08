// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"

	core "dappco.re/go"
)

const (
	// KVSnapshotVersion is the native KV snapshot schema version.
	KVSnapshotVersion = 3
)

// KVSnapshot is a CPU-readable copy of model key/value cache tensors.
type KVSnapshot struct {
	Version       int
	Architecture  string
	Tokens        []int32
	Generated     []int32
	TokenOffset   int
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	LogitShape    []int32
	Logits        []float32
	Layers        []KVLayerSnapshot
}

// KVLayerSnapshot contains cache tensors for a logical transformer layer.
type KVLayerSnapshot struct {
	Layer      int
	CacheIndex int
	Heads      []KVHeadSnapshot
}

// KVHeadSnapshot contains flattened key/value tensors for one KV head.
type KVHeadSnapshot struct {
	Key   []float32
	Value []float32
}

// CaptureKV runs one prefill pass and returns the resulting K/V cache tensors.
func (m *Model) CaptureKV(ctx context.Context, prompt string) (*KVSnapshot, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	release, slotErr := m.acquireSlot(ctx)
	if slotErr != nil {
		return nil, slotErr
	}
	defer release()

	var (
		result *KVSnapshot
		err    error
	)
	if deviceErr := m.withDevice(func() {
		result, err = m.captureKV(ctx, prompt)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return result, err
}

func (m *Model) captureKV(ctx context.Context, prompt string) (*KVSnapshot, error) {
	tokens := m.tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, core.E("Model.CaptureKV", "empty prompt after tokenisation", nil)
	}

	caches := m.newCaches()
	defer freeCaches(caches)

	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err != nil {
		return nil, core.E("Model.CaptureKV", "prefill", err)
	}
	defer Free(logits)

	return m.snapshotKVCaches(tokens, caches, logits)
}

func (m *Model) snapshotKVCaches(tokens []int32, caches []Cache, logits ...*Array) (*KVSnapshot, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	if len(tokens) == 0 {
		return nil, core.E("Model.CaptureKV", "empty token state", nil)
	}
	info := m.Info()
	seqLen := kvSnapshotSeqLen(tokens, caches)
	snapshotTokens := tokens
	if seqLen < len(snapshotTokens) {
		snapshotTokens = snapshotTokens[len(snapshotTokens)-seqLen:]
	}
	layers := make([]KVLayerSnapshot, info.NumLayers)
	cacheIndexByLayer := attentionCacheIndexByLayer(m.model, info.NumLayers, len(caches))
	cacheSnapshots := make(map[int]kvCacheSnapshot, len(caches))
	var numHeads, headDim int
	var logitShape []int32
	var logitValues []float32

	for layerIdx, cacheIdx := range cacheIndexByLayer {
		if cacheIdx < 0 {
			continue
		}
		snapshot, ok := cacheSnapshots[cacheIdx]
		if !ok {
			var extracted bool
			snapshot, extracted = inspectKVCache(caches[cacheIdx], seqLen)
			if !extracted {
				continue
			}
			cacheSnapshots[cacheIdx] = snapshot
		}
		layers[layerIdx] = KVLayerSnapshot{
			Layer:      layerIdx,
			CacheIndex: cacheIdx,
			Heads:      cloneKVSnapshotHeads(snapshot.Heads),
		}
		if numHeads == 0 {
			numHeads = snapshot.NumHeads
		}
		if headDim == 0 {
			headDim = snapshot.HeadDim
		}
	}
	if len(logits) > 0 && logits[0] != nil && logits[0].Valid() {
		logitShape = append([]int32(nil), logits[0].Shape()...)
		logitValues = logits[0].Floats()
	}

	return &KVSnapshot{
		Version:       KVSnapshotVersion,
		Architecture:  info.Architecture,
		Tokens:        append([]int32(nil), snapshotTokens...),
		TokenOffset:   len(tokens),
		NumLayers:     info.NumLayers,
		NumHeads:      numHeads,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		NumQueryHeads: attentionQueryHeads(m.model),
		LogitShape:    logitShape,
		Logits:        logitValues,
		Layers:        layers,
	}, nil
}

func kvSnapshotSeqLen(tokens []int32, caches []Cache) int {
	seqLen := len(tokens)
	var cacheLen int
	for _, cache := range caches {
		if cache == nil {
			continue
		}
		cacheLen = max(cacheLen, cache.Len())
	}
	if cacheLen > 0 && cacheLen < seqLen {
		return cacheLen
	}
	return seqLen
}

type kvCacheSnapshot struct {
	NumHeads int
	HeadDim  int
	Heads    []KVHeadSnapshot
}

func inspectKVCache(cache Cache, seqLen int) (kvCacheSnapshot, bool) {
	if cache == nil {
		return kvCacheSnapshot{}, false
	}
	state, ownedState := cacheReadState(cache)
	defer Free(ownedState...)
	if len(state) < 2 || !state[0].Valid() || !state[1].Valid() {
		return kvCacheSnapshot{}, false
	}

	kArray := state[0] // K tensor from cache: [B, H, L_alloc, D]
	vArray := state[1] // V tensor from cache: [B, H, L_alloc, D]
	kShape := kArray.Shape()
	vShape := vArray.Shape()
	if len(kShape) != 4 || len(vShape) != 4 || kShape[1] != vShape[1] {
		return kvCacheSnapshot{}, false
	}

	numHeads := int(kShape[1])
	headDim := int(kShape[3])
	valueHeadDim := int(vShape[3])
	validLen := min(cache.Len(), seqLen)
	if validLen <= 0 {
		return kvCacheSnapshot{}, false
	}

	kSliced := Slice(kArray, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(validLen), kShape[3]})
	vSliced := Slice(vArray, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(validLen), vShape[3]})
	if err := Eval(kSliced, vSliced); err != nil {
		Free(kSliced, vSliced)
		return kvCacheSnapshot{}, false
	}

	kFlat := kSliced.Floats()
	vFlat := vSliced.Floats()
	Free(kSliced, vSliced)

	heads := make([]KVHeadSnapshot, numHeads)
	keyStride := validLen * headDim
	valueStride := validLen * valueHeadDim
	for h := 0; h < numHeads; h++ {
		keyStart := h * keyStride
		keyEnd := keyStart + keyStride
		valueStart := h * valueStride
		valueEnd := valueStart + valueStride
		if keyEnd > len(kFlat) || valueEnd > len(vFlat) {
			break
		}
		heads[h] = KVHeadSnapshot{
			Key:   append([]float32(nil), kFlat[keyStart:keyEnd]...),
			Value: append([]float32(nil), vFlat[valueStart:valueEnd]...),
		}
	}

	return kvCacheSnapshot{
		NumHeads: numHeads,
		HeadDim:  headDim,
		Heads:    heads,
	}, true
}

func cloneKVSnapshotHeads(src []KVHeadSnapshot) []KVHeadSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]KVHeadSnapshot, len(src))
	for i, head := range src {
		cloned[i] = KVHeadSnapshot{
			Key:   append([]float32(nil), head.Key...),
			Value: append([]float32(nil), head.Value...),
		}
	}
	return cloned
}
