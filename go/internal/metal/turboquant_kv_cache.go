// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

const defaultTurboQuantKVCachePageSize = defaultPagedKVPageSize

// TurboQuantKVCache is the reference compressed K/V cache for the explicit
// turboquant research mode. It keeps compressed page payloads as the owned
// state and restores MLX arrays only as a compatibility bridge for the existing
// attention path.
type TurboQuantKVCache struct {
	payloads []TurboQuantKVReferencePagePayload
	keys     *Array
	values   *Array

	offset   int
	length   int
	maxSize  int
	pageSize int
	step     int

	batch   int32
	heads   int32
	headDim int32

	cacheIndex  int
	layer       int
	layerType   string
	sharedOwner int

	lastErr error
}

// TurboQuantKVCachePayloadEstimate sums the compressed payload sections
// currently owned by a TurboQuant cache. PayloadBytes is the actual section
// data before alignment padding; PaddedPayloadBytes is the byte span retained
// by the page payload buffers.
type TurboQuantKVCachePayloadEstimate struct {
	Pages                     int     `json:"pages"`
	PageVectors               uint64  `json:"page_vectors"`
	PageElements              uint64  `json:"page_elements"`
	KeyCentroidBytes          uint64  `json:"key_centroid_bytes"`
	KeyQJLSignBytes           uint64  `json:"key_qjl_sign_bytes,omitempty"`
	KeyNormBytes              uint64  `json:"key_norm_bytes"`
	KeyResidualNormBytes      uint64  `json:"key_residual_norm_bytes,omitempty"`
	ValueCentroidBytes        uint64  `json:"value_centroid_bytes"`
	ValueNormBytes            uint64  `json:"value_norm_bytes"`
	OutlierMaskBytes          uint64  `json:"outlier_mask_bytes,omitempty"`
	PayloadBytes              uint64  `json:"payload_bytes"`
	PaddedPayloadBytes        uint64  `json:"padded_payload_bytes"`
	AlignmentPaddingBytes     uint64  `json:"alignment_padding_bytes,omitempty"`
	FP16BaselineBytes         uint64  `json:"fp16_baseline_bytes"`
	PayloadToFP16Ratio        float64 `json:"payload_to_fp16_ratio,omitempty"`
	PaddedPayloadToFP16Ratio  float64 `json:"padded_payload_to_fp16_ratio,omitempty"`
	PayloadSavingsRatio       float64 `json:"payload_savings_ratio,omitempty"`
	PaddedPayloadSavingsRatio float64 `json:"padded_payload_savings_ratio,omitempty"`
}

func NewTurboQuantKVCache(maxSize, pageSize int) *TurboQuantKVCache {
	if pageSize <= 0 {
		pageSize = defaultTurboQuantKVCachePageSize
	}
	return &TurboQuantKVCache{
		maxSize:     maxSize,
		pageSize:    pageSize,
		step:        pageSize,
		layerType:   "unknown",
		sharedOwner: 0,
	}
}

func (c *TurboQuantKVCache) SetLayerIdentity(cacheIndex, layer, sharedOwner int, layerType string) {
	if c == nil {
		return
	}
	c.cacheIndex = cacheIndex
	c.layer = layer
	c.sharedOwner = sharedOwner
	if layerType != "" {
		c.layerType = layerType
	}
}

func (c *TurboQuantKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	if c == nil {
		return k, v
	}
	c.lastErr = nil
	batch, heads, incomingLen, headDim, err := turboQuantKVArrayShape(k, v)
	if err != nil {
		c.lastErr = err
		return k, v
	}
	if seqLen > 0 && seqLen < incomingLen {
		incomingLen = seqLen
	}
	if c.length > 0 && (c.batch != batch || c.heads != heads || c.headDim != headDim) {
		c.lastErr = core.NewError("mlx: TurboQuant KV cache shape changed across updates")
		return k, v
	}

	incomingKeys := k.Floats()
	incomingValues := v.Floats()
	if incomingLen != int(k.Dim(2)) {
		incomingKeys = turboQuantKVExtractSeq(incomingKeys, int(batch), int(heads), int(k.Dim(2)), int(headDim), 0, incomingLen)
		incomingValues = turboQuantKVExtractSeq(incomingValues, int(batch), int(heads), int(v.Dim(2)), int(headDim), 0, incomingLen)
	}

	newOffset := c.offset + incomingLen
	if c.length == 0 || c.maxSize <= 0 || c.length+incomingLen <= c.maxSize {
		payloads, err := c.encodePayloads(incomingKeys, incomingValues, batch, heads, incomingLen, headDim, c.offset)
		if err != nil {
			c.lastErr = err
			return k, v
		}
		c.payloads = append(c.payloads, payloads...)
		c.offset = newOffset
		c.length += incomingLen
		c.batch = batch
		c.heads = heads
		c.headDim = headDim
		outK, outV, err := c.restoreCurrentArrays()
		if err != nil {
			c.lastErr = err
			return k, v
		}
		return outK, outV
	}

	keys, values := incomingKeys, incomingValues
	totalLen := incomingLen
	previousKeys, previousValues, err := c.decodeFloatData()
	if err != nil {
		c.lastErr = err
		return k, v
	}
	keys = turboQuantKVConcatSeq(previousKeys, c.length, incomingKeys, incomingLen, int(batch), int(heads), int(headDim))
	values = turboQuantKVConcatSeq(previousValues, c.length, incomingValues, incomingLen, int(batch), int(heads), int(headDim))
	totalLen = c.length + incomingLen

	visibleLen := totalLen
	if c.maxSize > 0 && visibleLen > c.maxSize {
		drop := visibleLen - c.maxSize
		keys = turboQuantKVExtractSeq(keys, int(batch), int(heads), totalLen, int(headDim), drop, c.maxSize)
		values = turboQuantKVExtractSeq(values, int(batch), int(heads), totalLen, int(headDim), drop, c.maxSize)
		visibleLen = c.maxSize
	}

	tokenOffset := newOffset - visibleLen
	payloads, err := c.encodePayloads(keys, values, batch, heads, visibleLen, headDim, tokenOffset)
	if err != nil {
		c.lastErr = err
		return k, v
	}

	c.payloads = payloads
	c.offset = newOffset
	c.length = visibleLen
	c.batch = batch
	c.heads = heads
	c.headDim = headDim
	outK, outV, err := c.restoreCurrentArrays()
	if err != nil {
		c.lastErr = err
		return k, v
	}
	return outK, outV
}

func (c *TurboQuantKVCache) Offset() int {
	if c == nil {
		return 0
	}
	return c.offset
}

func (c *TurboQuantKVCache) Len() int {
	if c == nil {
		return 0
	}
	return c.length
}

func (c *TurboQuantKVCache) State() []*Array {
	if c == nil || c.length <= 0 {
		return nil
	}
	if c.keys == nil || c.values == nil || !c.keys.Valid() || !c.values.Valid() {
		if _, _, err := c.restoreCurrentArrays(); err != nil {
			c.lastErr = err
			return nil
		}
	}
	return []*Array{c.keys, c.values}
}

func (c *TurboQuantKVCache) AppendState(dst []*Array) []*Array {
	for _, state := range c.State() {
		if state != nil && state.Valid() {
			dst = append(dst, state)
		}
	}
	return dst
}

func (c *TurboQuantKVCache) AppendDirtyState(dst []*Array) []*Array {
	return c.AppendState(dst)
}

func (c *TurboQuantKVCache) ReadState() ([]*Array, []*Array) {
	if c == nil || c.length <= 0 {
		return nil, nil
	}
	keys, values, err := c.decodePayloadArrays()
	if err != nil {
		c.lastErr = err
		return nil, nil
	}
	state := []*Array{keys, values}
	return state, state
}

func (c *TurboQuantKVCache) Reset() {
	if c == nil {
		return
	}
	Free(c.keys, c.values)
	c.keys = nil
	c.values = nil
	c.payloads = nil
	c.offset = 0
	c.length = 0
	c.lastErr = nil
}

func (c *TurboQuantKVCache) Detach() {
	if c == nil {
		return
	}
	Free(c.keys, c.values)
	c.keys = nil
	c.values = nil
}

func (c *TurboQuantKVCache) Err() error {
	if c == nil {
		return nil
	}
	return c.lastErr
}

// PayloadEstimate reports the compressed payload bytes currently retained by
// the cache, including side-channel metadata and alignment padding. It is a
// compressed-state accounting helper, not a live MLX active-memory sampler.
func (c *TurboQuantKVCache) PayloadEstimate() (TurboQuantKVCachePayloadEstimate, error) {
	if c == nil || len(c.payloads) == 0 {
		return TurboQuantKVCachePayloadEstimate{}, core.NewError("mlx: TurboQuant KV cache has no payloads")
	}
	estimate := TurboQuantKVCachePayloadEstimate{Pages: len(c.payloads)}
	for _, payload := range c.payloads {
		if err := payload.validateSections(); err != nil {
			return TurboQuantKVCachePayloadEstimate{}, err
		}
		pageEstimate, err := payload.Layout.EstimatePayloadBytes()
		if err != nil {
			return TurboQuantKVCachePayloadEstimate{}, err
		}
		payloadBytes := payload.UnpaddedByteCount()
		if payloadBytes != pageEstimate.TotalBytes {
			return TurboQuantKVCachePayloadEstimate{}, core.NewError(core.Sprintf("mlx: TurboQuant KV payload byte accounting mismatch: payload=%d estimate=%d", payloadBytes, pageEstimate.TotalBytes))
		}
		paddedBytes := uint64(len(payload.Data))
		if paddedBytes < payloadBytes {
			return TurboQuantKVCachePayloadEstimate{}, core.NewError("mlx: TurboQuant KV payload padding is invalid")
		}

		estimate.PageVectors += pageEstimate.PageVectors
		estimate.PageElements += pageEstimate.PageElements
		estimate.KeyCentroidBytes += pageEstimate.KeyCentroidBytes
		estimate.KeyQJLSignBytes += pageEstimate.KeyQJLSignBytes
		estimate.KeyNormBytes += pageEstimate.KeyNormBytes
		estimate.KeyResidualNormBytes += pageEstimate.KeyResidualNormBytes
		estimate.ValueCentroidBytes += pageEstimate.ValueCentroidBytes
		estimate.ValueNormBytes += pageEstimate.ValueNormBytes
		estimate.OutlierMaskBytes += pageEstimate.OutlierMaskBytes
		estimate.PayloadBytes += payloadBytes
		estimate.PaddedPayloadBytes += paddedBytes
		estimate.AlignmentPaddingBytes += paddedBytes - payloadBytes
		estimate.FP16BaselineBytes += pageEstimate.FP16BaselineBytes
	}
	if estimate.FP16BaselineBytes > 0 {
		baseline := float64(estimate.FP16BaselineBytes)
		estimate.PayloadToFP16Ratio = float64(estimate.PayloadBytes) / baseline
		estimate.PaddedPayloadToFP16Ratio = float64(estimate.PaddedPayloadBytes) / baseline
		estimate.PayloadSavingsRatio = 1 - estimate.PayloadToFP16Ratio
		estimate.PaddedPayloadSavingsRatio = 1 - estimate.PaddedPayloadToFP16Ratio
	}
	return estimate, nil
}

func turboQuantKVCachesPayloadEstimate(caches []Cache) *TurboQuantKVCachePayloadEstimate {
	var total TurboQuantKVCachePayloadEstimate
	seen := false
	for _, cache := range caches {
		turbo, ok := cache.(*TurboQuantKVCache)
		if !ok || turbo == nil || len(turbo.payloads) == 0 {
			continue
		}
		estimate, err := turbo.PayloadEstimate()
		if err != nil {
			return nil
		}
		seen = true
		total.Pages += estimate.Pages
		total.PageVectors += estimate.PageVectors
		total.PageElements += estimate.PageElements
		total.KeyCentroidBytes += estimate.KeyCentroidBytes
		total.KeyQJLSignBytes += estimate.KeyQJLSignBytes
		total.KeyNormBytes += estimate.KeyNormBytes
		total.KeyResidualNormBytes += estimate.KeyResidualNormBytes
		total.ValueCentroidBytes += estimate.ValueCentroidBytes
		total.ValueNormBytes += estimate.ValueNormBytes
		total.OutlierMaskBytes += estimate.OutlierMaskBytes
		total.PayloadBytes += estimate.PayloadBytes
		total.PaddedPayloadBytes += estimate.PaddedPayloadBytes
		total.AlignmentPaddingBytes += estimate.AlignmentPaddingBytes
		total.FP16BaselineBytes += estimate.FP16BaselineBytes
	}
	if !seen {
		return nil
	}
	if total.FP16BaselineBytes > 0 {
		baseline := float64(total.FP16BaselineBytes)
		total.PayloadToFP16Ratio = float64(total.PayloadBytes) / baseline
		total.PaddedPayloadToFP16Ratio = float64(total.PaddedPayloadBytes) / baseline
		total.PayloadSavingsRatio = 1 - total.PayloadToFP16Ratio
		total.PaddedPayloadSavingsRatio = 1 - total.PaddedPayloadToFP16Ratio
	}
	return &total
}

func (c *TurboQuantKVCache) encodePayloads(keys, values []float32, batch, heads int32, seqLen int, headDim int32, tokenOffset int) ([]TurboQuantKVReferencePagePayload, error) {
	if seqLen <= 0 {
		return nil, core.NewError("mlx: TurboQuant KV cache cannot encode empty state")
	}
	pageSize := c.pageSize
	if pageSize <= 0 {
		pageSize = defaultTurboQuantKVCachePageSize
	}
	payloads := make([]TurboQuantKVReferencePagePayload, 0, (seqLen+pageSize-1)/pageSize)
	for start := 0; start < seqLen; start += pageSize {
		take := min(pageSize, seqLen-start)
		layout := c.referencePageLayout(batch, heads, int32(take), headDim, tokenOffset+start, take)
		page, err := encodeTurboQuantKVReferencePageFromSeq(keys, values, int(batch), int(heads), seqLen, int(headDim), start, layout)
		if err != nil {
			return nil, err
		}
		payload, err := page.PackedPayload()
		if err != nil {
			return nil, err
		}
		payloads = append(payloads, payload)
	}
	return payloads, nil
}

func (c *TurboQuantKVCache) referencePageLayout(batch, heads, seqLen, headDim int32, tokenOffset, pageTokens int) TurboQuantKVPageLayout {
	outlierMask := turboQuantKVOutlierMask(headDim, headDim/2)
	return TurboQuantKVPageLayout{
		Version:     TurboQuantKVLayoutVersion,
		Codec:       TurboQuantKVCodecName,
		CacheIndex:  c.cacheIndex,
		Layer:       c.layer,
		LayerType:   c.layerType,
		SharedOwner: c.sharedOwner,
		Shape:       TurboQuantKVShape{Batch: batch, Heads: heads, SeqLen: seqLen, HeadDim: headDim},
		TokenOffset: tokenOffset,
		PageTokens:  pageTokens,
		PageSize:    c.pageSize,
		LocalWindow: c.maxSize,
		Key: TurboQuantKVCodec{
			Algorithm:          TurboQuantKVAlgorithmProd,
			NormalBits:         3,
			OutlierBits:        4,
			OutlierPolicy:      TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:        outlierMask,
			NormPolicy:         TurboQuantKVNormPolicyExplicitVectorBF16V1,
			ResidualNormPolicy: TurboQuantKVResidualNormPolicyExplicitVectorBF16V1,
			RotationSeed:       0x54514b0000000001,
			QJLSeed:            0x5451510000000001,
			CodebookID:         TurboQuantKVReferenceCodebookUniform,
		},
		Value: TurboQuantKVCodec{
			Algorithm:     TurboQuantKVAlgorithmMSE,
			NormalBits:    3,
			OutlierBits:   4,
			OutlierPolicy: TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:   outlierMask,
			NormPolicy:    TurboQuantKVNormPolicyExplicitVectorBF16V1,
			RotationSeed:  0x5451560000000001,
			CodebookID:    TurboQuantKVReferenceCodebookUniform,
		},
	}
}

func (c *TurboQuantKVCache) restoreCurrentArrays() (*Array, *Array, error) {
	keys, values, err := c.decodePayloadArrays()
	if err != nil {
		return nil, nil, err
	}
	Free(c.keys, c.values)
	c.keys = keys
	c.values = values
	return keys, values, nil
}

func (c *TurboQuantKVCache) decodeFloatData() ([]float32, []float32, error) {
	if c == nil {
		return nil, nil, core.NewError("mlx: TurboQuant KV cache has no payloads")
	}
	keys, values, _, _, _, _, err := turboQuantKVDecodePayloadFloatData(c.payloads)
	return keys, values, err
}

func (c *TurboQuantKVCache) decodePayloadArrays() (*Array, *Array, error) {
	if c == nil {
		return nil, nil, core.NewError("mlx: TurboQuant KV cache has no payloads")
	}
	keys, values, batch, heads, seqLen, headDim, err := turboQuantKVDecodePayloadFloatData(c.payloads)
	if err != nil {
		return nil, nil, err
	}
	shape := [4]int{batch, heads, seqLen, headDim}
	keyArray, keyErr := fromPinnedFloat32Values(keys, shape[:])
	valueArray, valueErr := fromPinnedFloat32Values(values, shape[:])
	if keyErr != nil || valueErr != nil {
		Free(keyArray, valueArray)
		if keyErr != nil {
			return nil, nil, keyErr
		}
		return nil, nil, valueErr
	}
	return keyArray, valueArray, nil
}

func turboQuantKVDecodePayloadFloatData(payloads []TurboQuantKVReferencePagePayload) ([]float32, []float32, int, int, int, int, error) {
	batch, heads, totalTokens, headDim, elements, err := turboQuantKVPayloadFloatDataShape(payloads)
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	keys := make([]float32, elements)
	values := make([]float32, elements)
	if _, _, _, _, err := turboQuantKVDecodePayloadFloatDataInto(payloads, keys, values); err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	return keys, values, batch, heads, totalTokens, headDim, nil
}

func turboQuantKVDecodePayloadFloatDataInto(payloads []TurboQuantKVReferencePagePayload, keys, values []float32) (int, int, int, int, error) {
	batch, heads, totalTokens, headDim, elements, err := turboQuantKVPayloadFloatDataShape(payloads)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	if len(keys) != elements || len(values) != elements {
		return 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV payload destination shape is invalid")
	}
	scratch := borrowTurboQuantKVReferenceDecodeScratch(headDim)
	defer releaseTurboQuantKVReferenceDecodeScratch(scratch)
	tokenStart := 0
	for _, payload := range payloads {
		if err := payload.decodeBaseFloatDataInto(keys, values, totalTokens, tokenStart, scratch.rotated, scratch.normalised); err != nil {
			return 0, 0, 0, 0, err
		}
		tokenStart += payload.Layout.PageTokens
	}
	return batch, heads, totalTokens, headDim, nil
}

func turboQuantKVPayloadFloatDataShape(payloads []TurboQuantKVReferencePagePayload) (int, int, int, int, int, error) {
	if len(payloads) == 0 {
		return 0, 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV cache has no payloads")
	}
	first := payloads[0].Layout
	if err := first.Validate(); err != nil {
		return 0, 0, 0, 0, 0, err
	}
	batch := int(first.Shape.Batch)
	heads := int(first.Shape.Heads)
	headDim := int(first.Shape.HeadDim)
	totalTokens := 0
	for _, payload := range payloads {
		layout := payload.Layout
		if err := layout.Validate(); err != nil {
			return 0, 0, 0, 0, 0, err
		}
		if layout.Shape.Batch != first.Shape.Batch ||
			layout.Shape.Heads != first.Shape.Heads ||
			layout.Shape.HeadDim != first.Shape.HeadDim {
			return 0, 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV payload shapes differ")
		}
		totalTokens += layout.PageTokens
	}
	if totalTokens <= 0 {
		return 0, 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV payload token length is invalid")
	}
	return batch, heads, totalTokens, headDim, batch * heads * totalTokens * headDim, nil
}

func snapshotTurboQuantCache(cache *TurboQuantKVCache, tokenLen int) (cacheSnapshot, bool, error) {
	if cache == nil || tokenLen <= 0 || tokenLen > cache.Len() || len(cache.payloads) == 0 {
		return cacheSnapshot{}, false, nil
	}
	payloads, err := turboQuantKVPayloadPrefix(cache.payloads, tokenLen)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	return cacheSnapshot{
		mode:          KVCacheModeTurboQuant,
		turboPayloads: payloads,
		offset:        cache.Offset(),
		length:        tokenLen,
		step:          cache.pageSize,
		maxSize:       cache.maxSize,
		rotating:      cache.maxSize > 0,
	}, true, nil
}

func inspectTurboQuantKVCacheRange(cache *TurboQuantKVCache, start, end int) (kvCacheSnapshot, bool) {
	if cache == nil || start < 0 || end <= start || end > cache.Len() {
		return kvCacheSnapshot{}, false
	}
	payloads, err := turboQuantKVPayloadPrefix(cache.payloads, end)
	if err != nil {
		cache.lastErr = err
		return kvCacheSnapshot{}, false
	}
	if start > 0 {
		keys, values, err := decodeTurboQuantKVSnapshotFloatArrays(payloads)
		if err != nil {
			cache.lastErr = err
			return kvCacheSnapshot{}, false
		}
		keySlice := Slice4(keys, 0, 0, int32(start), 0, int32(keys.Dim(0)), int32(keys.Dim(1)), int32(end), int32(keys.Dim(3)))
		valueSlice := Slice4(values, 0, 0, int32(start), 0, int32(values.Dim(0)), int32(values.Dim(1)), int32(end), int32(values.Dim(3)))
		layout := payloads[0].Layout
		page, encodeErr := EncodeTurboQuantKVReferencePage(keySlice.Floats(), valueSlice.Floats(), TurboQuantKVPageLayout{
			Version:     TurboQuantKVLayoutVersion,
			Codec:       TurboQuantKVCodecName,
			CacheIndex:  layout.CacheIndex,
			Layer:       layout.Layer,
			LayerType:   layout.LayerType,
			SharedOwner: layout.SharedOwner,
			Shape:       TurboQuantKVShape{Batch: int32(keys.Dim(0)), Heads: int32(keys.Dim(1)), SeqLen: int32(end - start), HeadDim: int32(keys.Dim(3))},
			TokenOffset: payloads[0].Layout.TokenOffset + start,
			PageTokens:  end - start,
			PageSize:    max(end-start, 1),
			LocalWindow: payloads[0].Layout.LocalWindow,
			Key:         layout.Key,
			Value:       layout.Value,
		})
		Free(keys, values, keySlice, valueSlice)
		if encodeErr != nil {
			cache.lastErr = encodeErr
			return kvCacheSnapshot{}, false
		}
		payload, err := page.PackedPayload()
		if err != nil {
			cache.lastErr = err
			return kvCacheSnapshot{}, false
		}
		payloads = []TurboQuantKVReferencePagePayload{payload}
	}
	headDim := int(cache.headDim)
	numHeads := int(cache.heads)
	if (headDim == 0 || numHeads == 0) && len(payloads) > 0 {
		headDim = int(payloads[0].Layout.Shape.HeadDim)
		numHeads = int(payloads[0].Layout.Shape.Heads)
	}
	return kvCacheSnapshot{
		NumHeads:           numHeads,
		HeadDim:            headDim,
		CacheMode:          KVCacheModeTurboQuant,
		TurboQuantPayloads: turboQuantKVClonePayloads(payloads),
	}, true
}

func appendRestoreTurboQuantCacheSnapshot(dst []*Array, snapshot cacheSnapshot, prefixLen, offset int) (Cache, []*Array, error) {
	if prefixLen <= 0 {
		return nil, nil, core.NewError("prompt cache: invalid TurboQuant prefix length")
	}
	payloads, err := turboQuantKVPayloadPrefix(snapshot.turboPayloads, prefixLen)
	if err != nil {
		return nil, nil, err
	}
	if offset <= 0 {
		offset = prefixLen
	}
	pageSize := snapshot.step
	if pageSize <= 0 {
		pageSize = defaultTurboQuantKVCachePageSize
	}
	cache := NewTurboQuantKVCache(snapshot.maxSize, pageSize)
	cache.payloads = payloads
	cache.offset = offset
	cache.length = prefixLen
	if len(payloads) > 0 {
		layout := payloads[0].Layout
		cache.batch = layout.Shape.Batch
		cache.heads = layout.Shape.Heads
		cache.headDim = layout.Shape.HeadDim
		cache.SetLayerIdentity(layout.CacheIndex, layout.Layer, layout.SharedOwner, layout.LayerType)
	}
	keys, values, err := cache.restoreCurrentArrays()
	if err != nil {
		return nil, nil, err
	}
	return cache, append(dst, keys, values), nil
}

func decodeTurboQuantKVSnapshotFloatArrays(payloads []TurboQuantKVReferencePagePayload) (*Array, *Array, error) {
	if len(payloads) == 0 {
		return nil, nil, errTurboQuantSnapshotLayout
	}
	cache := NewTurboQuantKVCache(0, 0)
	cache.payloads = turboQuantKVClonePayloads(payloads)
	return cache.decodePayloadArrays()
}

func turboQuantKVPayloadPrefix(payloads []TurboQuantKVReferencePagePayload, tokenLen int) ([]TurboQuantKVReferencePagePayload, error) {
	if tokenLen <= 0 || len(payloads) == 0 {
		return nil, core.NewError("mlx: TurboQuant KV payload prefix is empty")
	}
	out := make([]TurboQuantKVReferencePagePayload, 0, len(payloads))
	remaining := tokenLen
	for _, payload := range payloads {
		if remaining <= 0 {
			break
		}
		if err := payload.Layout.Validate(); err != nil {
			return nil, err
		}
		pageTokens := payload.Layout.PageTokens
		if pageTokens <= 0 {
			return nil, core.NewError("mlx: TurboQuant KV payload page length is invalid")
		}
		if pageTokens <= remaining {
			out = append(out, turboQuantKVClonePayload(payload))
			remaining -= pageTokens
			continue
		}
		prefix, err := turboQuantKVPayloadPagePrefix(payload, remaining)
		if err != nil {
			return nil, err
		}
		out = append(out, prefix)
		remaining = 0
	}
	if remaining > 0 {
		return nil, core.NewError("mlx: TurboQuant KV payload shorter than prefix")
	}
	return out, nil
}

func turboQuantKVPayloadPagePrefix(payload TurboQuantKVReferencePagePayload, tokenLen int) (TurboQuantKVReferencePagePayload, error) {
	keyArray, valueArray, err := payload.DecodeBaseArrays()
	if err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	defer Free(keyArray, valueArray)
	keyPrefix, err := viewPagePrefix(keyArray, tokenLen)
	if err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	valuePrefix, err := viewPagePrefix(valueArray, tokenLen)
	if err != nil {
		Free(keyPrefix)
		return TurboQuantKVReferencePagePayload{}, err
	}
	defer Free(keyPrefix, valuePrefix)
	layout := payload.Layout
	layout.Shape.SeqLen = int32(tokenLen)
	layout.PageTokens = tokenLen
	page, err := EncodeTurboQuantKVReferencePage(keyPrefix.Floats(), valuePrefix.Floats(), layout)
	if err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	return page.PackedPayload()
}

func turboQuantKVClonePayloads(payloads []TurboQuantKVReferencePagePayload) []TurboQuantKVReferencePagePayload {
	out := make([]TurboQuantKVReferencePagePayload, len(payloads))
	for idx := range payloads {
		out[idx] = turboQuantKVClonePayload(payloads[idx])
	}
	return out
}

func cloneTurboQuantKVPayloads(payloads []TurboQuantKVReferencePagePayload) []TurboQuantKVReferencePagePayload {
	return turboQuantKVClonePayloads(payloads)
}

func turboQuantKVPayloadTokenLen(payloads []TurboQuantKVReferencePagePayload) int {
	var total int
	for _, payload := range payloads {
		if err := payload.Layout.Validate(); err != nil {
			return 0
		}
		total += payload.Layout.PageTokens
	}
	return total
}

func turboQuantKVClonePayload(payload TurboQuantKVReferencePagePayload) TurboQuantKVReferencePagePayload {
	payload.Sections = append([]TurboQuantKVReferencePagePayloadSection(nil), payload.Sections...)
	payload.Data = append([]byte(nil), payload.Data...)
	return payload
}

func turboQuantKVArrayShape(k, v *Array) (int32, int32, int, int32, error) {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV cache received invalid arrays")
	}
	if k.NumDims() < 4 || v.NumDims() < 4 {
		return 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV cache requires rank-4 K/V arrays")
	}
	var kBuf, vBuf [maxTensorRank]int32
	kShape := k.ShapeInto(kBuf[:0])
	vShape := v.ShapeInto(vBuf[:0])
	if len(kShape) < 4 || len(vShape) < 4 ||
		kShape[0] != vShape[0] || kShape[1] != vShape[1] ||
		kShape[2] != vShape[2] || kShape[3] != vShape[3] {
		return 0, 0, 0, 0, core.NewError("mlx: TurboQuant KV cache K/V shapes differ")
	}
	return kShape[0], kShape[1], int(kShape[2]), kShape[3], nil
}

func turboQuantKVConcatSeq(left []float32, leftSeq int, right []float32, rightSeq int, batch, heads, headDim int) []float32 {
	if leftSeq <= 0 {
		return append([]float32(nil), right...)
	}
	if rightSeq <= 0 {
		return append([]float32(nil), left...)
	}
	totalSeq := leftSeq + rightSeq
	out := make([]float32, batch*heads*totalSeq*headDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			dstBase := ((b*heads + h) * totalSeq) * headDim
			leftBase := ((b*heads + h) * leftSeq) * headDim
			rightBase := ((b*heads + h) * rightSeq) * headDim
			copy(out[dstBase:dstBase+leftSeq*headDim], left[leftBase:leftBase+leftSeq*headDim])
			copy(out[dstBase+leftSeq*headDim:dstBase+totalSeq*headDim], right[rightBase:rightBase+rightSeq*headDim])
		}
	}
	return out
}

func turboQuantKVExtractSeq(data []float32, batch, heads, seqLen, headDim, start, take int) []float32 {
	if start == 0 && take == seqLen {
		return data
	}
	out := make([]float32, batch*heads*take*headDim)
	var dst int
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			src := ((b*heads+h)*seqLen + start) * headDim
			n := take * headDim
			copy(out[dst:dst+n], data[src:src+n])
			dst += n
		}
	}
	return out
}
