// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"

	core "dappco.re/go"
	"dappco.re/go/mlx/kv"
)

const nativeKVSnapshotDTypeBF16 = "bfloat16"

// CaptureKV captures the session's current native K/V cache as a root KV
// snapshot. Native stores cache rows token-major; root KV snapshots store raw
// layer slabs as [1, heads, seq, head_dim], so capture transposes once at the
// API boundary and keeps the resident cache layout unchanged.
func (s *ArchSession) CaptureKV() (*kv.Snapshot, error) {
	return s.CaptureKVWithOptions(kv.CaptureOptions{})
}

// CaptureKVWithOptions captures native K/V as root kv.Snapshot data without
// depending on pkg/metal. RawKVOnly preserves the fast native BF16 slab path;
// the default path also derives per-head float32 tensors for portable callers.
func (s *ArchSession) CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil {
		return nil, core.NewError("native.CaptureKV: nil session")
	}
	if opts.BlockStartToken < 0 {
		return nil, core.NewError("native.CaptureKV: block start token must be >= 0")
	}
	if s.pos <= 0 {
		return nil, core.NewError("native.CaptureKV: empty cache")
	}
	if s.pos > s.maxLen {
		return nil, core.NewError("native.CaptureKV: position outside maxLen")
	}
	if len(s.cachedIDs) != s.pos {
		return nil, core.NewError("native.CaptureKV: cached ids do not match position")
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return nil, err
	}
	layers := make([]kv.LayerSnapshot, len(s.state.specs))
	for li, spec := range s.state.specs {
		layers[li] = kv.LayerSnapshot{
			Layer:      li,
			CacheIndex: spec.CacheIndex,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    s.stateCacheMaxSize(spec),
		}
	}
	for _, view := range views {
		start, tokenCount, err := nativeKVLayerCaptureWindow(view, s.pos)
		if err != nil {
			return nil, err
		}
		keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, s.pos)
		if err != nil {
			return nil, err
		}
		if len(keyRows) != tokenCount*view.rowBytes || len(valueRows) != tokenCount*view.rowBytes {
			return nil, core.NewError("native.CaptureKV: layer payload size mismatch")
		}
		keySlab := make([]byte, len(keyRows))
		valueSlab := make([]byte, len(valueRows))
		nativeKVTokenRowsToLayerSlab(keySlab, keyRows, tokenCount, view.kvHeads, view.headDim)
		nativeKVTokenRowsToLayerSlab(valueSlab, valueRows, tokenCount, view.kvHeads, view.headDim)
		shape := []int32{1, int32(view.kvHeads), int32(tokenCount), int32(view.headDim)}
		layer := kv.LayerSnapshot{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KeyDType:   nativeKVSnapshotDTypeBF16,
			KeyBytes:   keySlab,
			KeyShape:   append([]int32(nil), shape...),
			ValueDType: nativeKVSnapshotDTypeBF16,
			ValueBytes: valueSlab,
			ValueShape: append([]int32(nil), shape...),
		}
		if !opts.RawKVOnly {
			layer.Heads = nativeKVLayerSlabHeads(keySlab, valueSlab, tokenCount, view.kvHeads, view.headDim)
		}
		layers[view.layer] = layer
	}
	logits, logitShape, err := s.captureKVLogits()
	if err != nil {
		return nil, err
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Tokens:        append([]int32(nil), s.cachedIDs...),
		TokenOffset:   s.pos,
		NumLayers:     len(s.state.specs),
		NumHeads:      s.arch.MaxKVHeads(),
		SeqLen:        s.pos,
		HeadDim:       s.arch.MaxHeadDim(),
		NumQueryHeads: s.arch.Heads,
		LogitShape:    logitShape,
		Logits:        logits,
		Layers:        layers,
	}, nil
}

// RestoreKV restores a root KV snapshot into the resident native cache. It
// accepts native BF16 layer slabs directly and falls back to per-head float32
// tensors by converting them once into the native BF16 slab layout.
func (s *ArchSession) RestoreKV(snapshot *kv.Snapshot) error {
	if s == nil {
		return core.NewError("native.RestoreKV: nil session")
	}
	if snapshot == nil {
		return core.NewError("native.RestoreKV: nil snapshot")
	}
	if snapshot.Version > kv.SnapshotVersion {
		return core.NewError("native.RestoreKV: unsupported snapshot version")
	}
	position := kv.EffectiveTokenOffset(snapshot)
	if position <= 0 || position > s.maxLen {
		return core.NewError("native.RestoreKV: position outside maxLen")
	}
	if snapshot.NumLayers > 0 && snapshot.NumLayers != len(s.state.specs) {
		return core.NewError("native.RestoreKV: layer count mismatch")
	}
	targetViews, err := s.stateLayerViews()
	if err != nil {
		return err
	}
	if len(targetViews) > 0 && len(snapshot.Layers) == 0 {
		return core.NewError("native.RestoreKV: snapshot has no layers")
	}
	for _, view := range targetViews {
		layer, ok := nativeKVSnapshotLayer(snapshot, view.layer)
		if !ok {
			return core.NewError("native.RestoreKV: missing layer")
		}
		if layer.CacheIndex >= 0 && layer.CacheIndex != view.cacheIndex {
			return core.NewError("native.RestoreKV: cache-index mismatch")
		}
		if layer.CacheMode != "" && view.cacheMode != "" && layer.CacheMode != view.cacheMode {
			return core.NewError("native.RestoreKV: cache-mode mismatch")
		}
		if layer.MaxSize > 0 && layer.MaxSize != view.maxSize {
			return core.NewError("native.RestoreKV: cache max-size mismatch")
		}
		keySlab, valueSlab, tokenCount, err := nativeKVLayerSnapshotSlabs(layer, view)
		if err != nil {
			return err
		}
		wantTokens := position
		if view.maxSize > 0 && position > view.cacheRows {
			wantTokens = view.cacheRows
		}
		if tokenCount != wantTokens {
			return core.NewError("native.RestoreKV: layer window length mismatch")
		}
		keyRows := make([]byte, tokenCount*view.rowBytes)
		valueRows := make([]byte, tokenCount*view.rowBytes)
		nativeKVLayerSlabToTokenRows(keyRows, keySlab, tokenCount, view.kvHeads, view.headDim)
		nativeKVLayerSlabToTokenRows(valueRows, valueSlab, tokenCount, view.kvHeads, view.headDim)
		block := SessionStateLayerBlock{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   keyRows,
			ValueBytes: valueRows,
		}
		if err := restoreStateBlockLayer(view, position-tokenCount, tokenCount, position, block); err != nil {
			return err
		}
	}
	if err := s.restoreKVSnapshotMetadata(snapshot, position); err != nil {
		return err
	}
	return nil
}

func (s *ArchSession) captureKVLogits() ([]float32, []int32, error) {
	var logits []byte
	switch {
	case len(s.retainedLogits) == s.arch.Vocab*bf16Size:
		logits = s.retainedLogits
	case len(s.retainedHidden) == s.arch.Hidden*bf16Size:
		var err error
		logits, err = s.BoundaryLogits()
		if err != nil {
			return nil, nil, err
		}
	default:
		return nil, nil, nil
	}
	if len(logits) == 0 {
		return nil, nil, nil
	}
	if len(logits) != s.arch.Vocab*bf16Size {
		return nil, nil, core.NewError("native.CaptureKV: boundary logits size mismatch")
	}
	return bf16ToF32Slice(logits), []int32{1, int32(s.arch.Vocab)}, nil
}

func (s *ArchSession) restoreKVSnapshotMetadata(snapshot *kv.Snapshot, position int) error {
	cachedIDs := append([]int32(nil), snapshot.Tokens...)
	if len(snapshot.Generated) > 0 && len(cachedIDs)+len(snapshot.Generated) <= position {
		cachedIDs = append(cachedIDs, snapshot.Generated...)
	}
	if len(cachedIDs) > position {
		return core.NewError("native.RestoreKV: cached ids exceed position")
	}
	s.pos = position
	s.cachedIDs = append(s.cachedIDs[:0], cachedIDs...)
	s.clearCachedPromptHidden()
	s.resetRetainedHidden()
	if len(snapshot.Logits) == 0 {
		return nil
	}
	if len(snapshot.LogitShape) > 0 {
		total := 1
		for _, dim := range snapshot.LogitShape {
			if dim <= 0 {
				return core.NewError("native.RestoreKV: invalid logit shape")
			}
			total *= int(dim)
		}
		if total != len(snapshot.Logits) {
			return core.NewError("native.RestoreKV: logit shape mismatch")
		}
	}
	if len(snapshot.Logits) != s.arch.Vocab {
		return core.NewError("native.RestoreKV: logits size mismatch")
	}
	s.rememberRetainedLogits(f32ToBf16Slice(snapshot.Logits))
	return nil
}

func nativeKVSnapshotLayer(snapshot *kv.Snapshot, layerIndex int) (kv.LayerSnapshot, bool) {
	if snapshot == nil || layerIndex < 0 {
		return kv.LayerSnapshot{}, false
	}
	if layerIndex < len(snapshot.Layers) {
		layer := snapshot.Layers[layerIndex]
		if layer.Layer == layerIndex {
			return layer, true
		}
	}
	for _, layer := range snapshot.Layers {
		if layer.Layer == layerIndex {
			return layer, true
		}
	}
	return kv.LayerSnapshot{}, false
}

func nativeKVLayerCaptureWindow(view sessionStateLayerView, position int) (int, int, error) {
	if view.rowBytes <= 0 || view.cacheRows <= 0 {
		return 0, 0, core.NewError("native.CaptureKV: invalid layer view geometry")
	}
	if position <= 0 {
		return 0, 0, core.NewError("native.CaptureKV: empty cache")
	}
	tokenCount := position
	if view.maxSize > 0 && position > view.cacheRows {
		tokenCount = view.cacheRows
	}
	return position - tokenCount, tokenCount, nil
}

func nativeKVLayerSnapshotSlabs(layer kv.LayerSnapshot, view sessionStateLayerView) ([]byte, []byte, int, error) {
	if len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 {
		keySeq, err := nativeKVValidateLayerRaw(layer.KeyBytes, layer.KeyDType, layer.KeyShape, view)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "native layer key", err)
		}
		valueSeq, err := nativeKVValidateLayerRaw(layer.ValueBytes, layer.ValueDType, layer.ValueShape, view)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "native layer value", err)
		}
		if keySeq != valueSeq {
			return nil, nil, 0, core.NewError("native.RestoreKV: layer key/value window mismatch")
		}
		return layer.KeyBytes, layer.ValueBytes, keySeq, nil
	}
	return nativeKVHeadSnapshotSlabs(layer, view)
}

func nativeKVValidateLayerRaw(raw []byte, dtype string, shape []int32, view sessionStateLayerView) (int, error) {
	if len(raw) == 0 || len(shape) != 4 {
		return 0, core.NewError("missing native BF16 slab")
	}
	if !nativeKVIsBF16DType(dtype) {
		return 0, core.NewError("unsupported native dtype")
	}
	if shape[0] != 1 || int(shape[1]) != view.kvHeads || int(shape[3]) != view.headDim {
		return 0, core.NewError("native slab shape mismatch")
	}
	tokenCount := int(shape[2])
	if tokenCount <= 0 {
		return 0, core.NewError("native slab token count invalid")
	}
	if len(raw) != tokenCount*view.rowBytes {
		return 0, core.NewError("native slab byte length mismatch")
	}
	return tokenCount, nil
}

func nativeKVHeadSnapshotSlabs(layer kv.LayerSnapshot, view sessionStateLayerView) ([]byte, []byte, int, error) {
	if len(layer.Heads) != view.kvHeads {
		return nil, nil, 0, core.NewError("native.RestoreKV: head count mismatch")
	}
	tokenCount := 0
	for _, head := range layer.Heads {
		keySeq, err := nativeKVHeadSnapshotSeqLen(head.Key, head.KeyBytes, head.KeyDType, view.headDim)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head key", err)
		}
		valueSeq, err := nativeKVHeadSnapshotSeqLen(head.Value, head.ValueBytes, head.ValueDType, view.headDim)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head value", err)
		}
		if keySeq != valueSeq {
			return nil, nil, 0, core.NewError("native.RestoreKV: head key/value window mismatch")
		}
		if tokenCount == 0 {
			tokenCount = keySeq
			continue
		}
		if keySeq != tokenCount {
			return nil, nil, 0, core.NewError("native.RestoreKV: head window length mismatch")
		}
	}
	if tokenCount <= 0 {
		return nil, nil, 0, core.NewError("native.RestoreKV: missing head payload")
	}
	keySlab := make([]byte, view.kvHeads*tokenCount*view.headDim*bf16Size)
	valueSlab := make([]byte, len(keySlab))
	for headIndex, head := range layer.Heads {
		headOff := headIndex * tokenCount * view.headDim * bf16Size
		if err := nativeKVFillHeadBF16(keySlab[headOff:headOff+tokenCount*view.headDim*bf16Size], head.Key, head.KeyBytes, head.KeyDType, tokenCount, view.headDim); err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head key", err)
		}
		if err := nativeKVFillHeadBF16(valueSlab[headOff:headOff+tokenCount*view.headDim*bf16Size], head.Value, head.ValueBytes, head.ValueDType, tokenCount, view.headDim); err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head value", err)
		}
	}
	return keySlab, valueSlab, tokenCount, nil
}

func nativeKVHeadSnapshotSeqLen(values []float32, raw []byte, dtype string, headDim int) (int, error) {
	if headDim <= 0 {
		return 0, core.NewError("invalid head dim")
	}
	if len(raw) > 0 {
		if !nativeKVIsBF16DType(dtype) {
			return 0, core.NewError("unsupported head raw dtype")
		}
		rowBytes := headDim * bf16Size
		if len(raw)%rowBytes != 0 {
			return 0, core.NewError("head raw byte length mismatch")
		}
		return len(raw) / rowBytes, nil
	}
	if len(values) == 0 {
		return 0, core.NewError("missing head tensor")
	}
	if len(values)%headDim != 0 {
		return 0, core.NewError("head tensor length mismatch")
	}
	return len(values) / headDim, nil
}

func nativeKVFillHeadBF16(dst []byte, values []float32, raw []byte, dtype string, tokenCount, headDim int) error {
	want := tokenCount * headDim * bf16Size
	if len(dst) != want {
		return core.NewError("native.RestoreKV: destination size mismatch")
	}
	if len(raw) > 0 {
		if !nativeKVIsBF16DType(dtype) || len(raw) != want {
			return core.NewError("native.RestoreKV: raw head payload mismatch")
		}
		copy(dst, raw)
		return nil
	}
	if len(values) != tokenCount*headDim {
		return core.NewError("native.RestoreKV: float32 head payload mismatch")
	}
	for i, v := range values {
		h := f32ToBF16(v)
		dst[i*bf16Size], dst[i*bf16Size+1] = byte(h), byte(h>>8)
	}
	return nil
}

func nativeKVLayerSlabHeads(keySlab, valueSlab []byte, tokenCount, heads, headDim int) []kv.HeadSnapshot {
	if tokenCount <= 0 || heads <= 0 || headDim <= 0 {
		return nil
	}
	headBytes := tokenCount * headDim * bf16Size
	out := make([]kv.HeadSnapshot, heads)
	for head := 0; head < heads; head++ {
		off := head * headBytes
		out[head] = kv.HeadSnapshot{
			Key:   bf16ToF32Slice(keySlab[off : off+headBytes]),
			Value: bf16ToF32Slice(valueSlab[off : off+headBytes]),
		}
	}
	return out
}

func nativeKVTokenRowsToLayerSlab(dst, src []byte, tokenCount, heads, headDim int) {
	rowBytes := heads * headDim * bf16Size
	headBytes := headDim * bf16Size
	for head := 0; head < heads; head++ {
		for token := 0; token < tokenCount; token++ {
			srcOff := token*rowBytes + head*headBytes
			dstOff := (head*tokenCount + token) * headBytes
			copy(dst[dstOff:dstOff+headBytes], src[srcOff:srcOff+headBytes])
		}
	}
}

func nativeKVLayerSlabToTokenRows(dst, src []byte, tokenCount, heads, headDim int) {
	rowBytes := heads * headDim * bf16Size
	headBytes := headDim * bf16Size
	for token := 0; token < tokenCount; token++ {
		for head := 0; head < heads; head++ {
			srcOff := (head*tokenCount + token) * headBytes
			dstOff := token*rowBytes + head*headBytes
			copy(dst[dstOff:dstOff+headBytes], src[srcOff:srcOff+headBytes])
		}
	}
}

func nativeKVIsBF16DType(dtype string) bool {
	switch strings.ToLower(dtype) {
	case "bfloat16", "bf16":
		return true
	default:
		return false
	}
}
