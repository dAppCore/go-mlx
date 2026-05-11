// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"

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

// KVSnapshotCaptureOptions controls native K/V capture.
type KVSnapshotCaptureOptions struct {
	// RawKVOnly captures native K/V dtype bytes without retaining float32
	// key/value slices.
	RawKVOnly bool
}

// KVLayerSnapshot contains cache tensors for a logical transformer layer.
type KVLayerSnapshot struct {
	Layer      int
	CacheIndex int
	Heads      []KVHeadSnapshot
}

// KVHeadSnapshot contains flattened key/value tensors for one KV head.
type KVHeadSnapshot struct {
	Key        []float32
	KeyDType   DType
	KeyBytes   []byte
	Value      []float32
	ValueDType DType
	ValueBytes []byte
}

// KVSnapshotBlock is one contiguous token range from a KV snapshot.
type KVSnapshotBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Snapshot   *KVSnapshot
}

// KVSnapshotBlockSource streams KV snapshot blocks without requiring callers to
// assemble a full CPU snapshot first.
type KVSnapshotBlockSource struct {
	TokenCount   int
	PrefixTokens int
	BlockCount   int
	Load         func(context.Context, int) (KVSnapshotBlock, error)
}

// CaptureKV runs one prefill pass and returns the resulting K/V cache tensors.
func (m *Model) CaptureKV(ctx context.Context, prompt string) (*KVSnapshot, error) {
	return m.CaptureKVWithOptions(ctx, prompt, KVSnapshotCaptureOptions{})
}

// CaptureKVWithOptions runs one prefill pass and returns the resulting K/V
// cache tensors with explicit capture options.
func (m *Model) CaptureKVWithOptions(ctx context.Context, prompt string, opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
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
		result, err = m.captureKVWithOptions(ctx, prompt, opts)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return result, err
}

// CaptureKVChunks runs one streaming prefill pass over bounded prompt chunks
// and returns the resulting K/V cache tensors.
func (m *Model) CaptureKVChunks(ctx context.Context, chunks iter.Seq[string]) (*KVSnapshot, error) {
	return m.CaptureKVChunksWithOptions(ctx, chunks, KVSnapshotCaptureOptions{})
}

// CaptureKVChunksWithOptions runs one streaming prefill pass over bounded
// prompt chunks and returns K/V cache tensors with explicit capture options.
func (m *Model) CaptureKVChunksWithOptions(ctx context.Context, chunks iter.Seq[string], opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
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
		result, err = m.captureKVChunksWithOptions(ctx, chunks, opts)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return result, err
}

func (m *Model) captureKV(ctx context.Context, prompt string) (*KVSnapshot, error) {
	return m.captureKVWithOptions(ctx, prompt, KVSnapshotCaptureOptions{})
}

func (m *Model) captureKVWithOptions(ctx context.Context, prompt string, opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
	tokens := m.tokenizer.Encode(prompt)
	return m.captureKVTokensWithOptions(ctx, tokens, opts)
}

func (m *Model) captureKVChunks(ctx context.Context, chunks iter.Seq[string]) (*KVSnapshot, error) {
	return m.captureKVChunksWithOptions(ctx, chunks, KVSnapshotCaptureOptions{})
}

func (m *Model) captureKVChunksWithOptions(ctx context.Context, chunks iter.Seq[string], opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
	caches := m.newPromptSnapshotCaches()
	defer freeCaches(caches)

	tokens, logits, err := m.prefillPromptChunks(ctx, chunks, caches)
	if err != nil {
		return nil, core.E("Model.CaptureKV", "prefill chunks", err)
	}
	defer Free(logits)

	return m.snapshotKVCachesWithOptions(tokens, caches, opts, logits)
}

func (m *Model) captureKVTokens(ctx context.Context, tokens []int32) (*KVSnapshot, error) {
	return m.captureKVTokensWithOptions(ctx, tokens, KVSnapshotCaptureOptions{})
}

func (m *Model) captureKVTokensWithOptions(ctx context.Context, tokens []int32, opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
	if len(tokens) == 0 {
		return nil, core.E("Model.CaptureKV", "empty prompt after tokenisation", nil)
	}

	caches := m.newPromptSnapshotCaches()
	defer freeCaches(caches)

	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err != nil {
		return nil, core.E("Model.CaptureKV", "prefill", err)
	}
	defer Free(logits)

	return m.snapshotKVCachesWithOptions(tokens, caches, opts, logits)
}

func (m *Model) snapshotKVCaches(tokens []int32, caches []Cache, logits ...*Array) (*KVSnapshot, error) {
	return m.snapshotKVCachesWithOptions(tokens, caches, KVSnapshotCaptureOptions{}, logits...)
}

func (m *Model) snapshotKVCachesWithOptions(tokens []int32, caches []Cache, opts KVSnapshotCaptureOptions, logits ...*Array) (*KVSnapshot, error) {
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
			snapshot, extracted = inspectKVCacheWithOptions(caches[cacheIdx], seqLen, opts)
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

func (m *Model) kvBlockBoundaries(blockSize, seqLen int, caches []Cache) []int {
	seen := map[int]bool{0: true, seqLen: true}
	for next := blockSize; next < seqLen; next += blockSize {
		seen[next] = true
	}
	for _, cache := range caches {
		if cache == nil {
			continue
		}
		windowLen := min(cache.Len(), seqLen)
		if windowLen <= 0 || windowLen >= seqLen {
			continue
		}
		seen[seqLen-windowLen] = true
	}
	boundaries := make([]int, 0, len(seen))
	for boundary := range seen {
		boundaries = append(boundaries, boundary)
	}
	core.SliceSort(boundaries)
	return boundaries
}

func (m *Model) snapshotKVCacheBlockWithOptions(tokens []int32, caches []Cache, baseOffset, start, end int, final bool, opts KVSnapshotCaptureOptions, logits *Array) (*KVSnapshot, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	if start < 0 || end <= start || end > len(tokens) {
		return nil, core.NewError("mlx: invalid KV snapshot block range")
	}
	info := m.Info()
	seqLen := len(tokens)
	layers := make([]KVLayerSnapshot, info.NumLayers)
	cacheIndexByLayer := attentionCacheIndexByLayer(m.model, info.NumLayers, len(caches))
	cacheSnapshots := make(map[int]kvCacheSnapshot, len(caches))
	var numHeads, headDim int

	for layerIdx, cacheIdx := range cacheIndexByLayer {
		if cacheIdx < 0 || cacheIdx >= len(caches) || caches[cacheIdx] == nil {
			continue
		}
		cacheWindowLen := min(caches[cacheIdx].Len(), seqLen)
		if cacheWindowLen <= 0 {
			continue
		}
		windowStart := seqLen - cacheWindowLen
		overlapStart := max(start, windowStart)
		overlapEnd := min(end, seqLen)
		layers[layerIdx] = KVLayerSnapshot{
			Layer:      layerIdx,
			CacheIndex: cacheIdx,
		}
		if overlapStart >= overlapEnd {
			continue
		}
		snapshot, ok := cacheSnapshots[cacheIdx]
		if !ok {
			var extracted bool
			snapshot, extracted = inspectKVCacheRangeWithOptions(caches[cacheIdx], overlapStart-windowStart, overlapEnd-windowStart, opts)
			if !extracted {
				continue
			}
			cacheSnapshots[cacheIdx] = snapshot
		}
		layers[layerIdx].Heads = cloneKVSnapshotHeads(snapshot.Heads)
		if numHeads == 0 {
			numHeads = snapshot.NumHeads
		}
		if headDim == 0 {
			headDim = snapshot.HeadDim
		}
	}

	var logitShape []int32
	var logitValues []float32
	if final && logits != nil && logits.Valid() {
		logitShape = append([]int32(nil), logits.Shape()...)
		logitValues = logits.Floats()
	}
	return &KVSnapshot{
		Version:       KVSnapshotVersion,
		Architecture:  info.Architecture,
		Tokens:        append([]int32(nil), tokens[start:end]...),
		TokenOffset:   baseOffset + end,
		NumLayers:     info.NumLayers,
		NumHeads:      numHeads,
		SeqLen:        end - start,
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
	return inspectKVCacheWithOptions(cache, seqLen, KVSnapshotCaptureOptions{})
}

func inspectKVCacheWithOptions(cache Cache, seqLen int, opts KVSnapshotCaptureOptions) (kvCacheSnapshot, bool) {
	return inspectKVCacheRangeWithOptions(cache, 0, min(cache.Len(), seqLen), opts)
}

func inspectKVCacheRangeWithOptions(cache Cache, start, end int, opts KVSnapshotCaptureOptions) (kvCacheSnapshot, bool) {
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
	validLen := cache.Len()
	if start < 0 || end <= start || end > validLen {
		return kvCacheSnapshot{}, false
	}

	kSliced := Slice(kArray, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(end), kShape[3]})
	vSliced := Slice(vArray, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(end), vShape[3]})
	if err := Eval(kSliced, vSliced); err != nil {
		Free(kSliced, vSliced)
		return kvCacheSnapshot{}, false
	}

	kDType := kSliced.Dtype()
	vDType := vSliced.Dtype()
	kRaw := kSliced.RawBytes()
	vRaw := vSliced.RawBytes()
	var kFlat, vFlat []float32
	if !opts.RawKVOnly {
		kFlat = kSliced.Floats()
		vFlat = vSliced.Floats()
	}
	Free(kSliced, vSliced)

	blockLen := end - start
	heads := make([]KVHeadSnapshot, numHeads)
	keyStride := blockLen * headDim
	valueStride := blockLen * valueHeadDim
	keyRawStride := keyStride * DTypeByteSize(kDType)
	valueRawStride := valueStride * DTypeByteSize(vDType)
	for h := 0; h < numHeads; h++ {
		keyStart := h * keyStride
		keyEnd := keyStart + keyStride
		valueStart := h * valueStride
		valueEnd := valueStart + valueStride
		if !opts.RawKVOnly && (keyEnd > len(kFlat) || valueEnd > len(vFlat)) {
			break
		}
		keyHeadDType, keyHeadBytes := kvSnapshotHeadRaw(kRaw, kDType, h*keyRawStride, keyRawStride)
		valueHeadDType, valueHeadBytes := kvSnapshotHeadRaw(vRaw, vDType, h*valueRawStride, valueRawStride)
		head := KVHeadSnapshot{
			KeyDType:   keyHeadDType,
			KeyBytes:   keyHeadBytes,
			ValueDType: valueHeadDType,
			ValueBytes: valueHeadBytes,
		}
		if !opts.RawKVOnly {
			head.Key = append([]float32(nil), kFlat[keyStart:keyEnd]...)
			head.Value = append([]float32(nil), vFlat[valueStart:valueEnd]...)
		}
		heads[h] = head
	}

	return kvCacheSnapshot{
		NumHeads: numHeads,
		HeadDim:  headDim,
		Heads:    heads,
	}, true
}

func kvSnapshotHeadRaw(raw []byte, dtype DType, start, count int) (DType, []byte) {
	if len(raw) == 0 || DTypeByteSize(dtype) <= 0 || count <= 0 {
		return 0, nil
	}
	end := start + count
	if start < 0 || end > len(raw) || start >= end {
		return 0, nil
	}
	return dtype, append([]byte(nil), raw[start:end]...)
}

func cloneKVSnapshotHeads(src []KVHeadSnapshot) []KVHeadSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]KVHeadSnapshot, len(src))
	for i, head := range src {
		cloned[i] = KVHeadSnapshot{
			Key:        append([]float32(nil), head.Key...),
			KeyDType:   head.KeyDType,
			KeyBytes:   append([]byte(nil), head.KeyBytes...),
			Value:      append([]float32(nil), head.Value...),
			ValueDType: head.ValueDType,
			ValueBytes: append([]byte(nil), head.ValueBytes...),
		}
	}
	return cloned
}
