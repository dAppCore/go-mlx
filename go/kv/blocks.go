// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	stdio "io"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

const (
	// KVSnapshotStateBlockKind identifies one State chunk containing a KV block.
	KVSnapshotStateBlockKind = "go-mlx/kv-snapshot-block"
	// StateBlockBundleKind identifies a collection of State KV blocks.
	StateBlockBundleKind = "go-mlx/kv-snapshot-block-bundle"
	// StateBlockVersion is the block envelope schema version.
	StateBlockVersion = 1

	// KVSnapshotMemvidBlockKind identifies one old memvid-named chunk
	// containing a KV block.
	//
	// Deprecated: use KVSnapshotStateBlockKind.
	KVSnapshotMemvidBlockKind = KVSnapshotStateBlockKind
	// MemvidBlockBundleKind identifies a collection of old memvid-named KV
	// blocks.
	//
	// Deprecated: use StateBlockBundleKind.
	MemvidBlockBundleKind = StateBlockBundleKind
	// MemvidBlockVersion is the block envelope schema version.
	//
	// Deprecated: use StateBlockVersion.
	MemvidBlockVersion = StateBlockVersion

	kvSnapshotStatePayloadRaw        = "raw"
	kvSnapshotStatePayloadJSONBase64 = "json-base64"
)

// Block is one contiguous token range from a KV snapshot.
type Block struct {
	Index      int
	TokenStart int
	TokenCount int
	Hash       string
	Snapshot   *Snapshot
}

// StateTokenBlock is the token-only view of one durable State KV block.
type StateTokenBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Hash       string
	Tokens     []int32
}

// StateBlockOptions controls durable State-backed KV block storage.
type StateBlockOptions struct {
	BlockSize         int
	KVEncoding        Encoding
	URI               string
	Title             string
	Kind              string
	Track             string
	Tags              map[string]string
	Labels            []string
	ReusePrefix       *StateBlockBundle
	ReusePrefixTokens int
}

// MemvidBlockOptions controls old memvid-named KV block storage.
//
// Deprecated: use StateBlockOptions. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockOptions = StateBlockOptions

// StateBlockBundle is a portable manifest for durable State KV blocks.
type StateBlockBundle struct {
	Version      int             `json:"version"`
	Kind         string          `json:"kind"`
	SnapshotHash string          `json:"snapshot_hash,omitempty"`
	KVEncoding   Encoding        `json:"kv_encoding,omitempty"`
	Architecture string          `json:"architecture,omitempty"`
	TokenCount   int             `json:"token_count,omitempty"`
	TokenOffset  int             `json:"token_offset,omitempty"`
	BlockSize    int             `json:"block_size,omitempty"`
	NumLayers    int             `json:"num_layers,omitempty"`
	NumHeads     int             `json:"num_heads,omitempty"`
	SeqLen       int             `json:"seq_len,omitempty"`
	HeadDim      int             `json:"head_dim,omitempty"`
	ReusedBlocks int             `json:"reused_blocks,omitempty"`
	Blocks       []StateBlockRef `json:"blocks,omitempty"`
}

// MemvidBlockBundle is a portable manifest for old memvid-named KV blocks.
//
// Deprecated: use StateBlockBundle. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockBundle = StateBlockBundle

// StateBlockRef links one logical KV block to a durable State chunk.
type StateBlockRef struct {
	Index            int            `json:"index"`
	TokenStart       int            `json:"token_start"`
	TokenCount       int            `json:"token_count"`
	KVHash           string         `json:"kv_hash,omitempty"`
	PayloadEncoding  string         `json:"payload_encoding,omitempty"`
	PayloadByteCount int            `json:"payload_byte_count,omitempty"`
	State            state.ChunkRef `json:"state,omitempty"`
	// Deprecated: retained only so older bundles using json:"memvid" can wake.
	Memvid state.ChunkRef `json:"memvid,omitempty"`
}

// MemvidBlockRef links one logical KV block to an old memvid-named chunk.
//
// Deprecated: use StateBlockRef. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockRef = StateBlockRef

type kvSnapshotStateBlockEnvelope struct {
	Version          int    `json:"version"`
	Kind             string `json:"kind"`
	BlockIndex       int    `json:"block_index"`
	TokenStart       int    `json:"token_start"`
	TokenCount       int    `json:"token_count"`
	KVHash           string `json:"kv_hash"`
	KVEncoding       string `json:"kv_encoding,omitempty"`
	BinaryEncoding   string `json:"binary_encoding"`
	PayloadByteCount int    `json:"payload_byte_count,omitempty"`
	Data             string `json:"data"`
}

// SplitBlocks splits a KV snapshot into contiguous token-range blocks.
func (s *Snapshot) SplitBlocks(blockSize int) ([]Block, error) {
	blocks := []Block{}
	err := s.walkBlocks(blockSize, true, func(block Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	return blocks, nil
}

// RangeBlocks streams contiguous token-range blocks to yield without retaining
// every sliced block at once. Returning false from yield stops iteration.
func (s *Snapshot) RangeBlocks(blockSize int, yield func(Block) bool) error {
	if yield == nil {
		return core.NewError("mlx: KV snapshot block yield is nil")
	}
	return s.walkBlocks(blockSize, true, func(block Block) (bool, error) {
		return yield(block), nil
	})
}

func (s *Snapshot) walkBlocks(blockSize int, includeHash bool, yield func(Block) (bool, error)) error {
	if s == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	if blockSize <= 0 {
		return core.NewError("mlx: KV snapshot block size must be > 0")
	}
	seqLen := EffectiveSeqLen(s)
	if seqLen <= 0 || len(s.Tokens) != seqLen {
		return core.NewError("mlx: KV snapshot block split requires tokens matching sequence length")
	}
	if s.HeadDim <= 0 {
		return core.NewError("mlx: KV snapshot block split requires head dimension")
	}
	baseOffset := EffectiveTokenOffset(s) - seqLen
	if baseOffset < 0 {
		baseOffset = 0
	}
	boundaries, err := s.blockBoundaries(blockSize, seqLen)
	if err != nil {
		return err
	}
	for i := 0; i < len(boundaries)-1; i++ {
		start := boundaries[i]
		end := boundaries[i+1]
		blockSnapshot, err := s.SliceBlock(start, end, baseOffset, end == seqLen)
		if err != nil {
			return err
		}
		var hash string
		if includeHash {
			hash, err = HashSnapshot(blockSnapshot)
			if err != nil {
				return err
			}
		}
		ok, err := yield(Block{
			Index:      i,
			TokenStart: start,
			TokenCount: end - start,
			Hash:       hash,
			Snapshot:   blockSnapshot,
		})
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
	}
	return nil
}

func (s *Snapshot) blockBoundaries(blockSize, seqLen int) ([]int, error) {
	seen := map[int]bool{0: true, seqLen: true}
	for next := blockSize; next < seqLen; next += blockSize {
		seen[next] = true
	}
	for _, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "layer window", err)
		}
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
	return boundaries, nil
}

func (s *Snapshot) SliceBlock(start, end, baseOffset int, final bool) (*Snapshot, error) {
	if start < 0 || end <= start || end > len(s.Tokens) {
		return nil, core.NewError("mlx: invalid KV snapshot block range")
	}
	seqLen := EffectiveSeqLen(s)
	layers := make([]LayerSnapshot, len(s.Layers))
	for layerIndex, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "layer window", err)
		}
		windowStart := seqLen - windowLen
		overlapStart := max(start, windowStart)
		overlapEnd := min(end, seqLen)
		layers[layerIndex] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
		}
		if windowLen <= 0 || overlapStart >= overlapEnd {
			continue
		}
		localStart := overlapStart - windowStart
		localEnd := overlapEnd - windowStart
		keyLayerBytes, keyLayerShape, err := sliceKVSnapshotLayerRawTensor(layer.KeyBytes, layer.KeyDType, layer.KeyShape, localStart, localEnd)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer key tensor", err)
		}
		valueLayerBytes, valueLayerShape, err := sliceKVSnapshotLayerRawTensor(layer.ValueBytes, layer.ValueDType, layer.ValueShape, localStart, localEnd)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer value tensor", err)
		}
		layers[layerIndex].KeyDType = layer.KeyDType
		layers[layerIndex].KeyBytes = keyLayerBytes
		layers[layerIndex].KeyShape = keyLayerShape
		layers[layerIndex].ValueDType = layer.ValueDType
		layers[layerIndex].ValueBytes = valueLayerBytes
		layers[layerIndex].ValueShape = valueLayerShape
		layers[layerIndex].Heads = make([]HeadSnapshot, len(layer.Heads))
		for headIndex, head := range layer.Heads {
			key, err := sliceKVSnapshotTensor(head.Key, localStart, localEnd, s.HeadDim, windowLen)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice key tensor", err)
			}
			value, err := sliceKVSnapshotTensor(head.Value, localStart, localEnd, s.HeadDim, windowLen)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice value tensor", err)
			}
			keyBytes, err := sliceKVSnapshotRawTensor(head.KeyBytes, head.KeyDType, localStart, localEnd, windowLen, len(head.Key))
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice native key tensor", err)
			}
			valueBytes, err := sliceKVSnapshotRawTensor(head.ValueBytes, head.ValueDType, localStart, localEnd, windowLen, len(head.Value))
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice native value tensor", err)
			}
			layers[layerIndex].Heads[headIndex] = HeadSnapshot{
				Key:        key,
				KeyDType:   head.KeyDType,
				KeyBytes:   keyBytes,
				Value:      value,
				ValueDType: head.ValueDType,
				ValueBytes: valueBytes,
			}
		}
	}
	block := &Snapshot{
		Version:       effectiveVersion(s, KVSnapshotEncodingFloat32),
		Architecture:  s.Architecture,
		Tokens:        append([]int32(nil), s.Tokens[start:end]...),
		TokenOffset:   baseOffset + end,
		NumLayers:     s.NumLayers,
		NumHeads:      s.NumHeads,
		SeqLen:        end - start,
		HeadDim:       s.HeadDim,
		NumQueryHeads: s.NumQueryHeads,
		Layers:        layers,
	}
	if final {
		block.Generated = append([]int32(nil), s.Generated...)
		block.LogitShape = append([]int32(nil), s.LogitShape...)
		block.Logits = append([]float32(nil), s.Logits...)
	}
	return block, nil
}

func kvSnapshotLayerWindowLen(layer LayerSnapshot, seqLen, headDim int) (int, error) {
	windowLen := 0
	for _, length := range []int{
		kvSnapshotLayerRawWindowLen(layer.KeyBytes, layer.KeyDType, layer.KeyShape, seqLen),
		kvSnapshotLayerRawWindowLen(layer.ValueBytes, layer.ValueDType, layer.ValueShape, seqLen),
	} {
		if length < 0 {
			return 0, core.NewError("mlx: KV snapshot layer raw shape does not match sequence dimensions")
		}
		if length <= 0 {
			continue
		}
		if windowLen == 0 {
			windowLen = length
			continue
		}
		if windowLen != length {
			return 0, core.NewError("mlx: KV snapshot layer mixes cache window lengths")
		}
	}
	for _, head := range layer.Heads {
		for _, length := range []int{
			kvSnapshotTensorWindowLen(len(head.Key), seqLen, headDim),
			kvSnapshotTensorWindowLen(len(head.Value), seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.KeyBytes, head.KeyDType, seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.ValueBytes, head.ValueDType, seqLen, headDim),
		} {
			if length < 0 {
				return 0, core.NewError("mlx: KV snapshot tensor shape does not match sequence/head dimensions")
			}
			if length <= 0 {
				continue
			}
			if windowLen == 0 {
				windowLen = length
				continue
			}
			if windowLen != length {
				return 0, core.NewError("mlx: KV snapshot layer mixes cache window lengths")
			}
		}
	}
	return windowLen, nil
}

func kvSnapshotTensorWindowLen(valueCount, seqLen, headDim int) int {
	if valueCount <= 0 {
		return 0
	}
	if seqLen > 0 && valueCount%seqLen == 0 {
		return seqLen
	}
	if headDim > 0 && valueCount%headDim == 0 {
		return valueCount / headDim
	}
	return -1
}

func kvSnapshotRawTensorWindowLen(raw []byte, dtype string, seqLen, headDim int) int {
	if len(raw) == 0 {
		return 0
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 || len(raw)%bytesPerValue != 0 {
		return -1
	}
	return kvSnapshotTensorWindowLen(len(raw)/bytesPerValue, seqLen, headDim)
}

func kvSnapshotLayerRawWindowLen(raw []byte, dtype string, shape []int32, seqLen int) int {
	if len(raw) == 0 {
		return 0
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 || len(shape) != 4 {
		return -1
	}
	elements := 1
	for _, dim := range shape {
		if dim <= 0 {
			return -1
		}
		elements *= int(dim)
	}
	if len(raw) != elements*bytesPerValue {
		return -1
	}
	if seqLen > 0 && int(shape[2]) > seqLen {
		return -1
	}
	return int(shape[2])
}

func sliceKVSnapshotTensor(values []float32, start, end, headDim, seqLen int) ([]float32, error) {
	if len(values) == 0 {
		return nil, nil
	}
	if seqLen <= 0 {
		return nil, core.NewError("mlx: KV snapshot tensor shape does not match sequence/head dimensions")
	}
	if headDim <= 0 || len(values) != seqLen*headDim {
		if len(values)%seqLen != 0 {
			return nil, core.NewError("mlx: KV snapshot tensor shape does not match sequence/head dimensions")
		}
		headDim = len(values) / seqLen
	}
	begin := start * headDim
	finish := end * headDim
	if begin < 0 || finish > len(values) || begin >= finish {
		return nil, core.NewError("mlx: invalid KV snapshot tensor block range")
	}
	return append([]float32(nil), values[begin:finish]...), nil
}

func sliceKVSnapshotRawTensor(raw []byte, dtype string, start, end, seqLen, valueCount int) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 {
		return nil, core.NewError("mlx: unsupported KV snapshot raw tensor dtype")
	}
	if valueCount <= 0 {
		if len(raw)%bytesPerValue != 0 {
			return nil, core.NewError("mlx: KV snapshot raw tensor byte length is invalid")
		}
		valueCount = len(raw) / bytesPerValue
	}
	if seqLen <= 0 || valueCount%seqLen != 0 || len(raw) != valueCount*bytesPerValue {
		return nil, core.NewError("mlx: KV snapshot raw tensor shape does not match sequence length")
	}
	headDim := valueCount / seqLen
	begin := start * headDim * bytesPerValue
	finish := end * headDim * bytesPerValue
	if begin < 0 || finish > len(raw) || begin >= finish {
		return nil, core.NewError("mlx: invalid KV snapshot raw tensor block range")
	}
	return append([]byte(nil), raw[begin:finish]...), nil
}

func sliceKVSnapshotLayerRawTensor(raw []byte, dtype string, shape []int32, start, end int) ([]byte, []int32, error) {
	if len(raw) == 0 {
		return nil, nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 || len(shape) != 4 {
		return nil, nil, core.NewError("mlx: unsupported KV snapshot layer raw tensor")
	}
	B, H, L, D := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || start < 0 || end <= start || end > L {
		return nil, nil, core.NewError("mlx: invalid KV snapshot layer raw tensor range")
	}
	if len(raw) != B*H*L*D*bytesPerValue {
		return nil, nil, core.NewError("mlx: KV snapshot layer raw tensor byte length mismatch")
	}
	take := end - start
	out := make([]byte, B*H*take*D*bytesPerValue)
	dst := 0
	rowBytes := take * D * bytesPerValue
	for b := range B {
		for h := range H {
			src := (((b*H+h)*L + start) * D) * bytesPerValue
			copy(out[dst:dst+rowBytes], raw[src:src+rowBytes])
			dst += rowBytes
		}
	}
	outShape := append([]int32(nil), shape...)
	outShape[2] = int32(take)
	return out, outShape, nil
}

// AssembleBlocks reassembles contiguous blocks produced by SplitBlocks.
func AssembleBlocks(blocks []Block) (*Snapshot, error) {
	if len(blocks) == 0 {
		return nil, core.NewError("mlx: KV snapshot blocks are empty")
	}
	if err := validateKVSnapshotBlockOrder(blocks); err != nil {
		return nil, err
	}
	first := blocks[0].Snapshot
	if first == nil {
		return nil, core.NewError("mlx: KV snapshot block is nil")
	}
	assembled := &Snapshot{
		Version:       first.Version,
		Architecture:  first.Architecture,
		NumLayers:     first.NumLayers,
		NumHeads:      first.NumHeads,
		HeadDim:       first.HeadDim,
		NumQueryHeads: first.NumQueryHeads,
		Layers:        emptyKVSnapshotLayers(first.Layers),
	}
	for _, block := range blocks {
		if block.Snapshot == nil {
			return nil, core.NewError("mlx: KV snapshot block is nil")
		}
		if err := appendKVSnapshotBlock(assembled, block.Snapshot); err != nil {
			return nil, err
		}
	}
	last := blocks[len(blocks)-1].Snapshot
	assembled.Generated = append([]int32(nil), last.Generated...)
	assembled.TokenOffset = last.TokenOffset
	assembled.LogitShape = append([]int32(nil), last.LogitShape...)
	assembled.Logits = append([]float32(nil), last.Logits...)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

func validateKVSnapshotBlockOrder(blocks []Block) error {
	nextStart := 0
	for index, block := range blocks {
		if block.Index != index {
			return core.NewError("mlx: KV snapshot blocks are not ordered by index")
		}
		if block.TokenStart != nextStart || block.TokenCount <= 0 {
			return core.NewError("mlx: KV snapshot blocks are not contiguous")
		}
		if block.Snapshot == nil || len(block.Snapshot.Tokens) != block.TokenCount {
			return core.NewError("mlx: KV snapshot block token count mismatch")
		}
		nextStart += block.TokenCount
	}
	return nil
}

func emptyKVSnapshotLayers(layers []LayerSnapshot) []LayerSnapshot {
	out := make([]LayerSnapshot, len(layers))
	for i, layer := range layers {
		out[i] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			KeyDType:   layer.KeyDType,
			KeyShape:   append([]int32(nil), layer.KeyShape...),
			ValueDType: layer.ValueDType,
			ValueShape: append([]int32(nil), layer.ValueShape...),
		}
		if len(layer.Heads) > 0 {
			out[i].Heads = make([]HeadSnapshot, len(layer.Heads))
		}
	}
	return out
}

func appendKVSnapshotBlock(dst *Snapshot, block *Snapshot) error {
	if block.Architecture != "" && dst.Architecture != "" && block.Architecture != dst.Architecture {
		return core.NewError("mlx: KV snapshot block architecture mismatch")
	}
	if block.HeadDim != dst.HeadDim || block.NumHeads != dst.NumHeads || block.NumLayers != dst.NumLayers {
		return core.NewError("mlx: KV snapshot block shape mismatch")
	}
	if len(block.Layers) != len(dst.Layers) {
		return core.NewError("mlx: KV snapshot block layer count mismatch")
	}
	dst.Tokens = append(dst.Tokens, block.Tokens...)
	dst.SeqLen += block.SeqLen
	for layerIndex, layer := range block.Layers {
		if len(layer.KeyBytes) > 0 {
			dstLayer := &dst.Layers[layerIndex]
			if err := appendKVSnapshotLayerRawBlock(&dstLayer.KeyDType, &dstLayer.KeyBytes, &dstLayer.KeyShape, layer.KeyDType, layer.KeyBytes, layer.KeyShape); err != nil {
				return core.E("AssembleBlocks", "append native layer key tensor", err)
			}
		}
		if len(layer.ValueBytes) > 0 {
			dstLayer := &dst.Layers[layerIndex]
			if err := appendKVSnapshotLayerRawBlock(&dstLayer.ValueDType, &dstLayer.ValueBytes, &dstLayer.ValueShape, layer.ValueDType, layer.ValueBytes, layer.ValueShape); err != nil {
				return core.E("AssembleBlocks", "append native layer value tensor", err)
			}
		}
		if len(layer.Heads) == 0 {
			continue
		}
		if len(dst.Layers[layerIndex].Heads) == 0 {
			dst.Layers[layerIndex].Heads = make([]HeadSnapshot, len(layer.Heads))
		}
		if len(layer.Heads) != len(dst.Layers[layerIndex].Heads) {
			return core.NewError("mlx: KV snapshot block head count mismatch")
		}
		for headIndex, head := range layer.Heads {
			dstHead := &dst.Layers[layerIndex].Heads[headIndex]
			dstHead.Key = append(dstHead.Key, head.Key...)
			dstHead.Value = append(dstHead.Value, head.Value...)
			if err := appendKVSnapshotRawBlock(&dstHead.KeyDType, &dstHead.KeyBytes, head.KeyDType, head.KeyBytes); err != nil {
				return core.E("AssembleBlocks", "append native key tensor", err)
			}
			if err := appendKVSnapshotRawBlock(&dstHead.ValueDType, &dstHead.ValueBytes, head.ValueDType, head.ValueBytes); err != nil {
				return core.E("AssembleBlocks", "append native value tensor", err)
			}
		}
	}
	return nil
}

func appendKVSnapshotLayerRawBlock(dstDType *string, dstBytes *[]byte, dstShape *[]int32, dtype string, raw []byte, shape []int32) error {
	if len(raw) == 0 {
		return nil
	}
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 || len(shape) != 4 {
		return core.NewError("mlx: unsupported KV snapshot layer raw tensor")
	}
	blockShape := append([]int32(nil), shape...)
	B, H, L, D := int(blockShape[0]), int(blockShape[1]), int(blockShape[2]), int(blockShape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || len(raw) != B*H*L*D*bytesPerValue {
		return core.NewError("mlx: KV snapshot layer raw tensor shape mismatch")
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return core.NewError("mlx: KV snapshot layer raw tensor dtype mismatch")
	}
	if len(*dstBytes) == 0 {
		*dstBytes = append((*dstBytes)[:0], raw...)
		*dstShape = blockShape
		return nil
	}
	if len(*dstShape) != 4 || int((*dstShape)[0]) != B || int((*dstShape)[1]) != H || int((*dstShape)[3]) != D {
		return core.NewError("mlx: KV snapshot layer raw tensor shape mismatch")
	}
	oldShape := append([]int32(nil), (*dstShape)...)
	oldLen := int(oldShape[2])
	if oldLen <= 0 || len(*dstBytes) != B*H*oldLen*D*bytesPerValue {
		return core.NewError("mlx: KV snapshot layer raw tensor byte length mismatch")
	}
	totalLen := oldLen + L
	merged := make([]byte, B*H*totalLen*D*bytesPerValue)
	oldRowBytes := oldLen * D * bytesPerValue
	newRowBytes := L * D * bytesPerValue
	totalRowBytes := totalLen * D * bytesPerValue
	for b := range B {
		for h := range H {
			row := b*H + h
			dstStart := row * totalRowBytes
			oldStart := row * oldRowBytes
			newStart := row * newRowBytes
			copy(merged[dstStart:dstStart+oldRowBytes], (*dstBytes)[oldStart:oldStart+oldRowBytes])
			copy(merged[dstStart+oldRowBytes:dstStart+oldRowBytes+newRowBytes], raw[newStart:newStart+newRowBytes])
		}
	}
	*dstBytes = merged
	(*dstShape)[2] = int32(totalLen)
	return nil
}

func appendKVSnapshotRawBlock(dstDType *string, dstBytes *[]byte, dtype string, raw []byte) error {
	if len(raw) == 0 {
		return nil
	}
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 {
		return core.NewError("mlx: unsupported KV snapshot raw tensor dtype")
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return core.NewError("mlx: KV snapshot raw tensor dtype mismatch")
	}
	*dstBytes = append(*dstBytes, raw...)
	return nil
}

// SaveStateBlocks stores each KV block as a separate State chunk and returns a
// manifest.
func (s *Snapshot) SaveStateBlocks(ctx context.Context, store state.Writer, opts StateBlockOptions) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = defaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	bundle := &StateBlockBundle{
		Version:      StateBlockVersion,
		Kind:         StateBlockBundleKind,
		KVEncoding:   encoding,
		Architecture: s.Architecture,
		TokenCount:   len(s.Tokens),
		TokenOffset:  EffectiveTokenOffset(s),
		BlockSize:    blockSize,
		NumLayers:    s.NumLayers,
		NumHeads:     s.NumHeads,
		SeqLen:       EffectiveSeqLen(s),
		HeadDim:      s.HeadDim,
		Blocks:       []StateBlockRef{},
	}
	blockHashes := []string{}
	err = s.walkBlocks(blockSize, false, func(block Block) (bool, error) {
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		blockHashes = append(blockHashes, hash)
		bundle.Blocks = append(bundle.Blocks, StateBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			State:            ref,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle, blockHashes)
	return bundle, nil
}

// SaveMemvidBlocks stores each KV block as a separate memvid chunk and returns
// a manifest.
//
// Deprecated: use SaveStateBlocks.
func (s *Snapshot) SaveMemvidBlocks(ctx context.Context, store state.Writer, opts StateBlockOptions) (*StateBlockBundle, error) {
	return s.SaveStateBlocks(ctx, store, opts)
}

// SaveStateBlocksFromStream stores streamed KV blocks into a durable State
// bundle without retaining all sliced blocks in memory.
func SaveStateBlocksFromStream(ctx context.Context, store state.Writer, opts StateBlockOptions, stream func(func(Block) (bool, error)) error) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if stream == nil {
		return nil, core.NewError("mlx: State KV block stream is nil")
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = defaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		KVEncoding: encoding,
		BlockSize:  blockSize,
		Blocks:     []StateBlockRef{},
	}
	blockHashes := []string{}
	err = stream(func(block Block) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if block.Snapshot == nil {
			return false, core.NewError("mlx: streamed KV snapshot block is nil")
		}
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		applyKVSnapshotStateBundleBlock(bundle, block)
		blockHashes = append(blockHashes, hash)
		bundle.Blocks = append(bundle.Blocks, StateBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			State:            ref,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle, blockHashes)
	return bundle, nil
}

// SaveMemvidBlocksFromStream stores streamed KV blocks in a memvid-backed
// bundle without retaining all sliced blocks in memory.
//
// Deprecated: use SaveStateBlocksFromStream.
func SaveMemvidBlocksFromStream(ctx context.Context, store state.Writer, opts StateBlockOptions, stream func(func(Block) (bool, error)) error) (*StateBlockBundle, error) {
	return SaveStateBlocksFromStream(ctx, store, opts, stream)
}

func applyKVSnapshotStateBundleBlock(bundle *StateBlockBundle, block Block) {
	if bundle == nil || block.Snapshot == nil {
		return
	}
	snapshot := block.Snapshot
	if bundle.Architecture == "" {
		bundle.Architecture = snapshot.Architecture
	}
	if bundle.NumLayers == 0 {
		bundle.NumLayers = snapshot.NumLayers
	}
	if bundle.NumHeads == 0 {
		bundle.NumHeads = snapshot.NumHeads
	}
	if bundle.HeadDim == 0 {
		bundle.HeadDim = snapshot.HeadDim
	}
	if bundle.SeqLen < block.TokenStart+block.TokenCount {
		bundle.SeqLen = block.TokenStart + block.TokenCount
	}
	if bundle.TokenCount < block.TokenStart+block.TokenCount {
		bundle.TokenCount = block.TokenStart + block.TokenCount
	}
	if snapshot.TokenOffset > bundle.TokenOffset {
		bundle.TokenOffset = snapshot.TokenOffset
	}
}

func kvSnapshotStateBlockBundleHash(bundle *StateBlockBundle, blockHashes []string) string {
	if bundle == nil {
		return ""
	}
	builder := core.NewBuilder()
	// Pre-size to the exact final length so Builder never resizes mid-write.
	// Each block hash is 64 hex chars + 1 separator; the head fields run ~80
	// chars typical (architecture + 3 ints + encoding + 5 separators).
	size := len(bundle.Architecture) + len(string(bundle.KVEncoding)) + 5*1 + 30
	for _, hash := range blockHashes {
		size += 1 + len(hash)
	}
	builder.Grow(size)
	builder.WriteString(bundle.Architecture)
	builder.WriteString("|")
	builder.WriteString(string(bundle.KVEncoding))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(bundle.TokenCount))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(bundle.TokenOffset))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(bundle.BlockSize))
	for _, hash := range blockHashes {
		builder.WriteString("|")
		builder.WriteString(hash)
	}
	// SHA256HexString uses core.AsBytes under the hood — skips the
	// []byte copy of the Builder.String() roundtrip on every block-
	// bundle hash computation.
	return core.SHA256HexString(builder.String())
}

func saveOrReuseKVSnapshotStateBlock(ctx context.Context, store state.Writer, block Block, opts StateBlockOptions, encoding Encoding) (state.ChunkRef, string, string, int, bool, error) {
	if reused, hash, ok, err := reusableKVSnapshotStateBlockRef(block, opts, encoding); err != nil {
		return state.ChunkRef{}, "", "", 0, false, err
	} else if ok {
		return stateBlockChunkRef(reused), hash, reused.PayloadEncoding, reused.PayloadByteCount, true, nil
	}
	ref, hash, payloadEncoding, payloadByteCount, err := saveKVSnapshotStateBlock(ctx, store, block, opts, encoding)
	return ref, hash, payloadEncoding, payloadByteCount, false, err
}

func reusableKVSnapshotStateBlockRef(block Block, opts StateBlockOptions, encoding Encoding) (StateBlockRef, string, bool, error) {
	parent := opts.ReusePrefix
	if parent == nil || len(parent.Blocks) == 0 {
		return StateBlockRef{}, "", false, nil
	}
	if parent.KVEncoding != "" && parent.KVEncoding != encoding {
		return StateBlockRef{}, "", false, nil
	}
	reuseLimit := opts.ReusePrefixTokens
	if reuseLimit <= 0 {
		reuseLimit = parent.TokenCount
	}
	if block.TokenStart < 0 || block.TokenCount <= 0 || block.TokenStart+block.TokenCount > reuseLimit {
		return StateBlockRef{}, "", false, nil
	}
	hash, err := hashStateBlockPayload(block, encoding)
	if err != nil {
		return StateBlockRef{}, "", false, err
	}
	for _, ref := range parent.Blocks {
		if ref.TokenStart != block.TokenStart || ref.TokenCount != block.TokenCount {
			continue
		}
		if ref.KVHash != "" && ref.KVHash != hash {
			continue
		}
		reused := ref
		reused.Index = block.Index
		reused.TokenStart = block.TokenStart
		reused.TokenCount = block.TokenCount
		reused.KVHash = hash
		return reused, hash, true, nil
	}
	return StateBlockRef{}, hash, false, nil
}

func hashStateBlockPayload(block Block, encoding Encoding) (string, error) {
	if block.Snapshot == nil {
		return "", core.NewError("mlx: KV snapshot block is nil")
	}
	hash := sha256.New()
	if err := block.Snapshot.writeWithOptions(hash, SaveOptions{KVEncoding: encoding}); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func saveKVSnapshotStateBlock(ctx context.Context, store state.Writer, block Block, opts StateBlockOptions, encoding Encoding) (state.ChunkRef, string, string, int, error) {
	if streamStore, ok := store.(state.BinaryStreamWriter); ok {
		payloadSize, err := block.Snapshot.encodedSizeWithOptions(SaveOptions{KVEncoding: encoding})
		if err != nil {
			return state.ChunkRef{}, "", "", 0, err
		}
		hash := sha256.New()
		ref, err := streamStore.PutBytesStream(ctx, payloadSize, kvSnapshotStateBlockPutOptions(block, opts, "", string(encoding), kvSnapshotStatePayloadRaw), func(writer stdio.Writer) error {
			return block.Snapshot.writeWithOptions(stdio.MultiWriter(writer, hash), SaveOptions{KVEncoding: encoding})
		})
		if err != nil {
			return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "stream raw State block", err)
		}
		return ref, hex.EncodeToString(hash.Sum(nil)), kvSnapshotStatePayloadRaw, payloadSize, nil
	}
	data, err := block.Snapshot.bytesWithOptions(SaveOptions{KVEncoding: encoding})
	if err != nil {
		return state.ChunkRef{}, "", "", 0, err
	}
	hash := core.SHA256Hex(data)
	if binaryStore, ok := store.(state.BinaryWriter); ok {
		ref, err := binaryStore.PutBytes(ctx, data, kvSnapshotStateBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotStatePayloadRaw))
		if err != nil {
			return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "write raw State block", err)
		}
		return ref, hash, kvSnapshotStatePayloadRaw, len(data), nil
	}
	envelope := kvSnapshotStateBlockEnvelope{
		Version:          StateBlockVersion,
		Kind:             KVSnapshotStateBlockKind,
		BlockIndex:       block.Index,
		TokenStart:       block.TokenStart,
		TokenCount:       block.TokenCount,
		KVHash:           hash,
		KVEncoding:       string(encoding),
		BinaryEncoding:   "base64",
		PayloadByteCount: len(data),
		Data:             core.Base64Encode(data),
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(envelope), kvSnapshotStateBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotStatePayloadJSONBase64))
	if err != nil {
		return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "write State block", err)
	}
	return ref, hash, kvSnapshotStatePayloadJSONBase64, len(data), nil
}

// SaveStateBlockBundle stores the KV block manifest in the same
// State store as its referenced blocks.
func SaveStateBlockBundle(ctx context.Context, store state.Writer, bundle *StateBlockBundle, uri string) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return state.ChunkRef{}, core.NewError("mlx: state store is nil")
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, core.NewError("mlx: State KV block bundle URI is required")
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return state.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(bundle), state.PutOptions{
		URI:    uri,
		Title:  "go-mlx State block bundle",
		Kind:   StateBlockBundleKind,
		Track:  "session-kv-blocks",
		Labels: []string{"go-mlx", "kv-snapshot-block-bundle"},
	})
	if err != nil {
		return state.ChunkRef{}, core.E("Snapshot.SaveStateBlockBundle", "write State bundle", err)
	}
	return ref, nil
}

// SaveMemvidBlockBundle stores the KV block manifest in the same
// old memvid-named store as its referenced blocks.
//
// Deprecated: use SaveStateBlockBundle.
func SaveMemvidBlockBundle(ctx context.Context, store state.Writer, bundle *MemvidBlockBundle, uri string) (state.ChunkRef, error) {
	return SaveStateBlockBundle(ctx, store, bundle, uri)
}

func kvSnapshotStateBlockPutOptions(block Block, opts StateBlockOptions, hash, kvEncoding, payloadEncoding string) state.PutOptions {
	kind := opts.Kind
	if kind == "" {
		kind = KVSnapshotStateBlockKind
	}
	track := opts.Track
	if track == "" {
		track = "session-kv-blocks"
	}
	tags := cloneKVSnapshotStateTags(opts.Tags)
	if hash != "" {
		tags["kv_hash"] = hash
	}
	tags["kv_encoding"] = kvEncoding
	tags["payload_encoding"] = payloadEncoding
	tags["block_index"] = core.Itoa(block.Index)
	tags["token_start"] = core.Itoa(block.TokenStart)
	tags["token_count"] = core.Itoa(block.TokenCount)
	// Pre-size for the deterministic 2 appended labels — avoids the
	// geometric-grow path on every per-block State save.
	labels := make([]string, len(opts.Labels), len(opts.Labels)+2)
	copy(labels, opts.Labels)
	labels = append(labels, "go-mlx", "kv-snapshot-block")
	baseURI := firstNonEmpty(opts.URI, "mlx://kv-snapshot-blocks")
	// Direct string concatenation skips the fmt.Sprintf parse + format
	// state machinery on every per-block save (~SaveStateBlocks fires once
	// per checkpointed block during prefill).
	indexStr := core.Itoa(block.Index)
	return state.PutOptions{
		URI:    baseURI + "/block/" + indexStr,
		Title:  firstNonEmpty(opts.Title, "go-mlx KV block "+indexStr),
		Kind:   kind,
		Track:  track,
		Tags:   tags,
		Labels: labels,
	}
}

// LoadFromStateBlocks restores a full KV snapshot from a State block manifest.
func LoadFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle) (*Snapshot, error) {
	return LoadFromStateBlocksWithOptions(ctx, store, bundle, LoadOptions{})
}

// LoadFromMemvidBlocks restores a full KV snapshot from a memvid block manifest.
//
// Deprecated: use LoadFromStateBlocks.
func LoadFromMemvidBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle) (*Snapshot, error) {
	return LoadFromStateBlocks(ctx, store, bundle)
}

// LoadStateBlockBundle restores a KV block manifest by URI from the
// same State store as its referenced blocks.
func LoadStateBlockBundle(ctx context.Context, store state.Store, uri string) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if core.Trim(uri) == "" {
		return nil, core.NewError("mlx: State KV block bundle URI is required")
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadStateBlockBundle", "resolve State bundle", err)
	}
	var bundle StateBlockBundle
	if result := core.JSONUnmarshalString(chunk.Text, &bundle); !result.OK {
		return nil, core.E("LoadStateBlockBundle", "parse bundle", ResultError(result))
	}
	if err := ValidateStateBlockBundle(&bundle); err != nil {
		return nil, err
	}
	return &bundle, nil
}

// LoadMemvidBlockBundle restores a KV block manifest by URI from an old
// memvid-named store.
//
// Deprecated: use LoadStateBlockBundle.
func LoadMemvidBlockBundle(ctx context.Context, store state.Store, uri string) (*MemvidBlockBundle, error) {
	return LoadStateBlockBundle(ctx, store, uri)
}

// LoadFromStateBlocksWithOptions restores a full KV snapshot from a
// State block manifest with explicit decode options.
func LoadFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if bundle == nil {
		return nil, core.NewError("mlx: State KV block bundle is nil")
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return nil, core.NewError("mlx: unsupported State KV block bundle version")
	}
	if bundle.Kind != StateBlockBundleKind {
		return nil, core.NewError("mlx: invalid State KV block bundle kind")
	}
	blocks := make([]Block, 0, len(bundle.Blocks))
	for _, ref := range bundle.Blocks {
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	snapshot, err := AssembleBlocks(blocks)
	if err != nil {
		return nil, err
	}
	if bundle.TokenOffset > 0 && snapshot.TokenOffset != bundle.TokenOffset {
		return nil, core.NewError("mlx: State KV block token offset mismatch")
	}
	return snapshot, nil
}

// LoadFromMemvidBlocksWithOptions restores a full KV snapshot from a
// memvid block manifest with explicit decode options.
//
// Deprecated: use LoadFromStateBlocksWithOptions.
func LoadFromMemvidBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	return LoadFromStateBlocksWithOptions(ctx, store, bundle, opts)
}

// LoadPrefixFromStateBlocks restores only the State KV blocks needed
// to cover prefixTokens. The returned snapshot is suitable for prompt-cache
// warmup; non-final prefixes intentionally omit logits.
func LoadPrefixFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) (*Snapshot, error) {
	return LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, LoadOptions{})
}

// LoadPrefixFromMemvidBlocks restores only the memvid KV blocks needed
// to cover prefixTokens. The returned snapshot is suitable for prompt-cache
// warmup; non-final prefixes intentionally omit logits.
//
// Deprecated: use LoadPrefixFromStateBlocks.
func LoadPrefixFromMemvidBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) (*Snapshot, error) {
	return LoadPrefixFromStateBlocks(ctx, store, bundle, prefixTokens)
}

// LoadPrefixFromStateBlocksWithOptions restores only the State KV
// blocks needed to cover prefixTokens with explicit decode options.
func LoadPrefixFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 || prefixTokens == bundle.TokenCount {
		return LoadFromStateBlocksWithOptions(ctx, store, bundle, opts)
	}
	if prefixTokens > bundle.TokenCount {
		return nil, core.NewError("mlx: State KV prefix exceeds bundle token count")
	}
	refs := stateBlockRefsForPrefix(bundle, prefixTokens)
	if len(refs) == 0 {
		return nil, core.NewError("mlx: State KV prefix has no covering blocks")
	}
	blocks := make([]Block, 0, len(refs))
	for _, ref := range refs {
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	snapshot, err := AssembleBlocks(blocks)
	if err != nil {
		return nil, err
	}
	if len(snapshot.Tokens) == prefixTokens {
		if prefixTokens < bundle.TokenCount {
			ClearTerminalState(snapshot)
		}
		return snapshot, nil
	}
	if len(snapshot.Tokens) < prefixTokens {
		return nil, core.NewError("mlx: State KV prefix blocks do not cover requested tokens")
	}
	baseOffset := EffectiveTokenOffset(snapshot) - EffectiveSeqLen(snapshot)
	if baseOffset < 0 {
		baseOffset = 0
	}
	trimmed, err := snapshot.SliceBlock(0, prefixTokens, baseOffset, false)
	if err != nil {
		return nil, err
	}
	return trimmed, nil
}

// LoadPrefixFromMemvidBlocksWithOptions restores only the memvid KV
// blocks needed to cover prefixTokens with explicit decode options.
//
// Deprecated: use LoadPrefixFromStateBlocksWithOptions.
func LoadPrefixFromMemvidBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	return LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, opts)
}

// LoadPrefixTokensFromStateBlocks restores only token IDs from a State block
// manifest. It intentionally avoids K/V assembly, which is the correct wake
// path for folded State because the compact prompt will be prefetched again.
func LoadPrefixTokensFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) ([]int32, error) {
	return LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, LoadOptions{})
}

// LoadPrefixTokensFromStateBlocksWithOptions restores only token IDs from the
// blocks needed to cover prefixTokens with explicit decode options.
func LoadPrefixTokensFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) ([]int32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return nil, core.NewError("mlx: State token prefix exceeds bundle token count")
	}
	refs := stateBlockRefsForPrefix(bundle, prefixTokens)
	if len(refs) == 0 {
		return nil, core.NewError("mlx: State token prefix has no covering blocks")
	}
	tokens := make([]int32, 0, prefixTokens)
	nextStart := 0
	for expectedIndex, ref := range refs {
		if ref.Index != expectedIndex || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return nil, core.NewError("mlx: State token blocks are not contiguous")
		}
		block, err := LoadStateBlockTokensWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		if len(block.Tokens) != block.TokenCount {
			return nil, core.NewError("mlx: State token block token count mismatch")
		}
		if block.Index != ref.Index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return nil, core.NewError("mlx: State token block metadata mismatch")
		}
		tokens = append(tokens, block.Tokens...)
		nextStart += ref.TokenCount
		if len(tokens) >= prefixTokens {
			break
		}
	}
	if len(tokens) < prefixTokens {
		return nil, core.NewError("mlx: State token prefix blocks do not cover requested tokens")
	}
	return tokens[:prefixTokens], nil
}

func stateBlockRefsForPrefix(bundle *StateBlockBundle, prefixTokens int) []StateBlockRef {
	refs := make([]StateBlockRef, 0, len(bundle.Blocks))
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		refs = append(refs, ref)
		if ref.TokenStart+ref.TokenCount >= prefixTokens {
			break
		}
	}
	return refs
}

func ValidateStateBlockBundle(bundle *StateBlockBundle) error {
	if bundle == nil {
		return core.NewError("mlx: State KV block bundle is nil")
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return core.NewError("mlx: unsupported State KV block bundle version")
	}
	if bundle.Kind != StateBlockBundleKind {
		return core.NewError("mlx: invalid State KV block bundle kind")
	}
	if bundle.TokenCount <= 0 {
		return core.NewError("mlx: State KV block bundle token count is empty")
	}
	if len(bundle.Blocks) == 0 {
		return core.NewError("mlx: State KV block bundle has no blocks")
	}
	return nil
}

// ValidateMemvidBlockBundle checks an old memvid-named KV block bundle.
//
// Deprecated: use ValidateStateBlockBundle.
func ValidateMemvidBlockBundle(bundle *MemvidBlockBundle) error {
	return ValidateStateBlockBundle(bundle)
}

func ClearTerminalState(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	snapshot.Generated = nil
	snapshot.LogitShape = nil
	snapshot.Logits = nil
}

func loadKVSnapshotStateBlock(ctx context.Context, store state.Store, ref StateBlockRef) (Block, error) {
	return LoadStateBlockWithOptions(ctx, store, ref, LoadOptions{})
}

// LoadStateBlockWithOptions loads one durable State KV block with explicit
// decode options.
func LoadStateBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
		return loadRawKVSnapshotStateBlockWithOptions(ctx, store, ref, opts)
	}
	chunk, err := state.Resolve(ctx, store, stateBlockChunkRef(ref).ChunkID)
	if err != nil {
		return Block{}, core.E("LoadFromStateBlocks", "resolve State block", err)
	}
	var envelope kvSnapshotStateBlockEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return Block{}, core.E("LoadFromStateBlocks", "parse block envelope", ResultError(result))
	}
	data, err := decodeKVSnapshotStateBlockEnvelope(envelope, ref.KVHash)
	if err != nil {
		return Block{}, err
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return Block{}, err
	}
	return Block{
		Index:      envelope.BlockIndex,
		TokenStart: envelope.TokenStart,
		TokenCount: envelope.TokenCount,
		Hash:       envelope.KVHash,
		Snapshot:   snapshot,
	}, nil
}

// LoadMemvidBlockWithOptions loads one memvid KV block with explicit decode
// options.
//
// Deprecated: use LoadStateBlockWithOptions.
func LoadMemvidBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	return LoadStateBlockWithOptions(ctx, store, ref, opts)
}

// LoadStateBlockTokens loads only token IDs from one durable State KV block.
func LoadStateBlockTokens(ctx context.Context, store state.Store, ref StateBlockRef) (StateTokenBlock, error) {
	return LoadStateBlockTokensWithOptions(ctx, store, ref, LoadOptions{})
}

// LoadStateBlockTokensWithOptions loads only token IDs from one durable State
// KV block. Decode options are accepted for symmetry with full block loading;
// tensor payloads are skipped rather than decoded.
func LoadStateBlockTokensWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, _ LoadOptions) (StateTokenBlock, error) {
	if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
		data, err := loadRawStateBlockPayload(ctx, store, ref)
		if err != nil {
			return StateTokenBlock{}, err
		}
		tokens, err := parseKVSnapshotTokens(data)
		if err != nil {
			return StateTokenBlock{}, err
		}
		return StateTokenBlock{
			Index:      ref.Index,
			TokenStart: ref.TokenStart,
			TokenCount: ref.TokenCount,
			Hash:       ref.KVHash,
			Tokens:     tokens,
		}, nil
	}
	chunk, err := state.Resolve(ctx, store, stateBlockChunkRef(ref).ChunkID)
	if err != nil {
		return StateTokenBlock{}, core.E("LoadFromStateBlocks", "resolve State token block", err)
	}
	var envelope kvSnapshotStateBlockEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return StateTokenBlock{}, core.E("LoadFromStateBlocks", "parse token block envelope", ResultError(result))
	}
	data, err := decodeKVSnapshotStateBlockEnvelope(envelope, ref.KVHash)
	if err != nil {
		return StateTokenBlock{}, err
	}
	tokens, err := parseKVSnapshotTokens(data)
	if err != nil {
		return StateTokenBlock{}, err
	}
	return StateTokenBlock{
		Index:      envelope.BlockIndex,
		TokenStart: envelope.TokenStart,
		TokenCount: envelope.TokenCount,
		Hash:       envelope.KVHash,
		Tokens:     tokens,
	}, nil
}

func loadRawKVSnapshotStateBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	data, err := loadRawStateBlockPayload(ctx, store, ref)
	if err != nil {
		return Block{}, err
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return Block{}, err
	}
	return Block{
		Index:      ref.Index,
		TokenStart: ref.TokenStart,
		TokenCount: ref.TokenCount,
		Hash:       ref.KVHash,
		Snapshot:   snapshot,
	}, nil
}

func loadRawStateBlockPayload(ctx context.Context, store state.Store, ref StateBlockRef) ([]byte, error) {
	chunk, err := state.ResolveRefBytes(ctx, store, stateBlockChunkRef(ref))
	if err != nil {
		return nil, core.E("LoadFromStateBlocks", "resolve raw State block", err)
	}
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}
	if ref.PayloadByteCount > 0 && len(data) != ref.PayloadByteCount {
		return nil, core.NewError("mlx: State raw KV block payload length mismatch")
	}
	hash := core.SHA256Hex(data)
	if ref.KVHash != "" && hash != ref.KVHash {
		return nil, core.NewError("mlx: State raw KV block hash mismatch")
	}
	return data, nil
}

// StateBlockChunkRef returns the current State chunk ref for a block,
// falling back to the deprecated json:"memvid" ref for older bundles.
func StateBlockChunkRef(ref StateBlockRef) state.ChunkRef {
	if ref.State.ChunkID != 0 || ref.State.Segment != "" || ref.State.Codec != "" || ref.State.HasFrameOffset {
		return ref.State
	}
	return ref.Memvid
}

func stateBlockChunkRef(ref StateBlockRef) state.ChunkRef {
	return StateBlockChunkRef(ref)
}

func decodeKVSnapshotStateBlockEnvelope(envelope kvSnapshotStateBlockEnvelope, expectedHash string) ([]byte, error) {
	if envelope.Version <= 0 || envelope.Version > StateBlockVersion {
		return nil, core.NewError("mlx: unsupported State KV block version")
	}
	if envelope.Kind != KVSnapshotStateBlockKind {
		return nil, core.NewError("mlx: invalid State KV block kind")
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, core.NewError("mlx: unsupported State KV block binary encoding")
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadFromStateBlocks", "decode block payload", ResultError(decoded))
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, core.NewError("mlx: State KV block decoded to non-byte data")
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, core.NewError("mlx: State KV block payload length mismatch")
	}
	hash := core.SHA256Hex(data)
	if envelope.KVHash != "" && hash != envelope.KVHash {
		return nil, core.NewError("mlx: State KV block hash mismatch")
	}
	if expectedHash != "" && hash != expectedHash {
		return nil, core.NewError("mlx: State KV block ref hash mismatch")
	}
	return data, nil
}

func EffectiveSeqLen(snapshot *Snapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.SeqLen > 0 {
		return snapshot.SeqLen
	}
	return len(snapshot.Tokens)
}
