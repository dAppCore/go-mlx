// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	stdio "io"
	"strconv"

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

// kvSnapshotStateBlockDefaultLabels is the per-block label pair used
// when the caller passes empty StateBlockOptions.Labels — shared
// across blocks so the per-block PutOptions skips a slice allocation.
// State stores treat PutOptions.Labels as read-only input.
var kvSnapshotStateBlockDefaultLabels = []string{"go-mlx", "kv-snapshot-block"}

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Sharing instances also makes errors.Is comparable for
// callers distinguishing "store nil" from "block range invalid" without
// parsing message text.
var (
	errBlockRangeInvalid           = core.NewError("mlx: invalid KV snapshot block range")
	errLayerRawTensorRangeInvalid  = core.NewError("mlx: invalid KV snapshot layer raw tensor range")
	errRawTensorBlockRangeInvalid  = core.NewError("mlx: invalid KV snapshot raw tensor block range")
	errTensorBlockRangeInvalid     = core.NewError("mlx: invalid KV snapshot tensor block range")
	errBundleKindInvalid           = core.NewError("mlx: invalid State KV block bundle kind")
	errBlockKindInvalid            = core.NewError("mlx: invalid State KV block kind")
	errBlockArchMismatch           = core.NewError("mlx: KV snapshot block architecture mismatch")
	errBlockHeadCountMismatch      = core.NewError("mlx: KV snapshot block head count mismatch")
	errBlockNil                    = core.NewError("mlx: KV snapshot block is nil")
	errBlockLayerCountMismatch     = core.NewError("mlx: KV snapshot block layer count mismatch")
	errBlockMetadataMismatch       = core.NewError("mlx: KV snapshot block metadata mismatch")
	errBlockCompressedPayloadSplit = core.NewError("mlx: KV snapshot compressed payload block requires full range")
	errBlockShapeMismatch          = core.NewError("mlx: KV snapshot block shape mismatch")
	errBlockSizeTooSmall           = core.NewError("mlx: KV snapshot block size must be > 0")
	errBlockSplitNeedsHeadDim      = core.NewError("mlx: KV snapshot block split requires head dimension")
	errBlockSplitNeedsTokens       = core.NewError("mlx: KV snapshot block split requires tokens matching sequence length")
	errBlockTokenCountMismatch     = core.NewError("mlx: KV snapshot block token count mismatch")
	errBlockYieldNil               = core.NewError("mlx: KV snapshot block yield is nil")
	errBlocksEmpty                 = core.NewError("mlx: KV snapshot blocks are empty")
	errBlocksNotContiguous         = core.NewError("mlx: KV snapshot blocks are not contiguous")
	errBlocksOutOfOrder            = core.NewError("mlx: KV snapshot blocks are not ordered by index")
	errSnapshotNil                 = core.NewError("mlx: KV snapshot is nil")
	errLayerMixesWindowLens        = core.NewError("mlx: KV snapshot layer mixes cache window lengths")
	errLayerRawShapeMismatch       = core.NewError("mlx: KV snapshot layer raw shape does not match sequence dimensions")
	errLayerRawByteLenMismatch     = core.NewError("mlx: KV snapshot layer raw tensor byte length mismatch")
	errLayerRawDtypeMismatch       = core.NewError("mlx: KV snapshot layer raw tensor dtype mismatch")
	errLayerRawTensorShape         = core.NewError("mlx: KV snapshot layer raw tensor shape mismatch")
	errRawTensorByteLenInvalid     = core.NewError("mlx: KV snapshot raw tensor byte length is invalid")
	errRawTensorDtypeMismatch      = core.NewError("mlx: KV snapshot raw tensor dtype mismatch")
	errRawTensorShapeSeq           = core.NewError("mlx: KV snapshot raw tensor shape does not match sequence length")
	errTensorShapeSeqHead          = core.NewError("mlx: KV snapshot tensor shape does not match sequence/head dimensions")
	errBundleNoBlocks              = core.NewError("mlx: State KV block bundle has no blocks")
	errBundleNil                   = core.NewError("mlx: State KV block bundle is nil")
	errBundleTokenCountEmpty       = core.NewError("mlx: State KV block bundle token count is empty")
	errBundleURIRequired           = core.NewError("mlx: State KV block bundle URI is required")
	errBlockNonByteData            = core.NewError("mlx: State KV block decoded to non-byte data")
	errBlockHashMismatch           = core.NewError("mlx: State KV block hash mismatch")
	errBlockPayloadLenMismatch     = core.NewError("mlx: State KV block payload length mismatch")
	errBlockRefHashMismatch        = core.NewError("mlx: State KV block ref hash mismatch")
	errBlockStreamNil              = core.NewError("mlx: State KV block stream is nil")
	errBlockTokenOffsetMismatch    = core.NewError("mlx: State KV block token offset mismatch")
	errPrefixBlocksNoCover         = core.NewError("mlx: State KV prefix blocks do not cover requested tokens")
	errPrefixExceedsBundle         = core.NewError("mlx: State KV prefix exceeds bundle token count")
	errPrefixNoCoveringBlocks      = core.NewError("mlx: State KV prefix has no covering blocks")
	errRawBlockHashMismatch        = core.NewError("mlx: State raw KV block hash mismatch")
	errRawBlockPayloadLenMismatch  = core.NewError("mlx: State raw KV block payload length mismatch")
	errStateStoreNil               = core.NewError("mlx: state store is nil")
	errTokenBlockMetadata          = core.NewError("mlx: State token block metadata mismatch")
	errTokenBlockTokenCount        = core.NewError("mlx: State token block token count mismatch")
	errTokenBlocksNotContiguous    = core.NewError("mlx: State token blocks are not contiguous")
	errTokenPrefixNoCover          = core.NewError("mlx: State token prefix blocks do not cover requested tokens")
	errTokenPrefixExceeds          = core.NewError("mlx: State token prefix exceeds bundle token count")
	errTokenPrefixNoBlocks         = core.NewError("mlx: State token prefix has no covering blocks")
	errStreamedBlockNil            = core.NewError("mlx: streamed KV snapshot block is nil")
	errUnsupportedLayerRawTensor   = core.NewError("mlx: unsupported KV snapshot layer raw tensor")
	errUnsupportedRawTensorDtype   = core.NewError("mlx: unsupported KV snapshot raw tensor dtype")
	errUnsupportedBlockEncoding    = core.NewError("mlx: unsupported State KV block binary encoding")
	errUnsupportedBundleVersion    = core.NewError("mlx: unsupported State KV block bundle version")
	errUnsupportedBlockVersion     = core.NewError("mlx: unsupported State KV block version")
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
	// walkBlocks emits one block per blockSize-aligned range; mirror the
	// SaveStateBlocks estimate so growth-loop reallocs vanish for typical
	// snapshots. A layer-window adjustment may add one extra boundary —
	// the +1 absorbs it without overshoot.
	expectedBlocks := 1
	if blockSize > 0 && s != nil && len(s.Tokens) > 0 {
		expectedBlocks = (len(s.Tokens)+blockSize-1)/blockSize + 1
	}
	blocks := make([]Block, 0, expectedBlocks)
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
		return errBlockYieldNil
	}
	return s.walkBlocks(blockSize, true, func(block Block) (bool, error) {
		return yield(block), nil
	})
}

func (s *Snapshot) walkBlocks(blockSize int, includeHash bool, yield func(Block) (bool, error)) error {
	if s == nil {
		return errSnapshotNil
	}
	if blockSize <= 0 {
		return errBlockSizeTooSmall
	}
	seqLen := EffectiveSeqLen(s)
	if seqLen <= 0 || len(s.Tokens) != seqLen {
		return errBlockSplitNeedsTokens
	}
	if s.HeadDim <= 0 {
		return errBlockSplitNeedsHeadDim
	}
	baseOffset := EffectiveTokenOffset(s) - seqLen
	if baseOffset < 0 {
		baseOffset = 0
	}
	boundaries, err := s.blockBoundaries(blockSize, seqLen)
	if err != nil {
		return err
	}
	// includeHash signals an external observer of the block snapshots —
	// SplitBlocks / RangeBlocks return blocks to the caller, so each
	// snapshot needs cloned slices for independent ownership. The internal
	// SaveStateBlocks path passes includeHash=false; it encodes + hashes
	// each block within yield and discards the snapshot before the next
	// iteration, so non-cloning sub-views are safe.
	cloneSlices := includeHash
	for i := 0; i < len(boundaries)-1; i++ {
		start := boundaries[i]
		end := boundaries[i+1]
		blockSnapshot, err := s.sliceBlockInternal(start, end, baseOffset, end == seqLen, cloneSlices)
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
	if snapshotHasLayerCompressedPayloads(s) {
		return []int{0, seqLen}, nil
	}
	// Build directly into a sorted, dedup'd slice — boundary count is
	// O(seqLen/blockSize) + O(layers), typically <10. Mapping was the
	// 4th-largest alloc source on SaveStateBlocks.
	expected := 2 + (seqLen / blockSize) + len(s.Layers)
	boundaries := make([]int, 0, expected)
	// Deterministic boundaries are pre-sorted: 0, blockSize, 2*blockSize, ..., seqLen.
	boundaries = append(boundaries, 0)
	for next := blockSize; next < seqLen; next += blockSize {
		boundaries = append(boundaries, next)
	}
	boundaries = append(boundaries, seqLen)
	for _, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "layer window", err)
		}
		if windowLen <= 0 || windowLen >= seqLen {
			continue
		}
		boundaries = kvBoundaryInsert(boundaries, seqLen-windowLen)
	}
	return boundaries, nil
}

// kvBoundaryInsert keeps boundaries sorted + deduped while inserting v.
// boundaries is small (≤ seqLen/blockSize + few layer-window slots)
// so linear scan beats map ops or a binary search + memmove.
func kvBoundaryInsert(boundaries []int, v int) []int {
	for i, b := range boundaries {
		if b == v {
			return boundaries
		}
		if b > v {
			boundaries = append(boundaries, 0)
			copy(boundaries[i+1:], boundaries[i:])
			boundaries[i] = v
			return boundaries
		}
	}
	return append(boundaries, v)
}

func kvBlockPayloadSlices(payloads [][]byte, clone bool) [][]byte {
	if len(payloads) == 0 {
		return nil
	}
	out := make([][]byte, len(payloads))
	for i := range payloads {
		if clone {
			out[i] = core.SliceClone(payloads[i])
			continue
		}
		out[i] = payloads[i]
	}
	return out
}

func (s *Snapshot) SliceBlock(start, end, baseOffset int, final bool) (*Snapshot, error) {
	return s.sliceBlockInternal(start, end, baseOffset, final, true)
}

// sliceBlockInternal is the implementation of SliceBlock. When cloneSlices
// is false, per-head Key/Value/KeyBytes/ValueBytes return as sub-views of
// the parent snapshot — used only by walkBlocks(includeHash=false), the
// SaveStateBlocks path that immediately encodes and discards each block.
func (s *Snapshot) sliceBlockInternal(start, end, baseOffset int, final bool, cloneSlices bool) (*Snapshot, error) {
	if start < 0 || end <= start || end > len(s.Tokens) {
		return nil, errBlockRangeInvalid
	}
	seqLen := EffectiveSeqLen(s)
	layers := make([]LayerSnapshot, len(s.Layers))
	// Heads-slab: one backing slice across all layers collapses N per-layer
	// make([]HeadSnapshot,...) into a single allocation. Hot during
	// SaveStateBlocks — fires per checkpoint block × number of layers.
	// Layers with no overlap (windowLen <= 0) skip head slicing entirely;
	// the slab still under-uses the backing buffer in that case but never
	// over-allocates because we size against NumHeads.
	var headSlab []HeadSnapshot
	var slabCursor int
	if s.NumHeads > 0 && len(s.Layers) > 0 {
		headSlab = make([]HeadSnapshot, len(s.Layers)*s.NumHeads)
	}
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
			CacheMode:  layer.CacheMode,
		}
		if len(layer.TurboQuantPayloads) > 0 {
			if start != 0 || end != seqLen {
				return nil, errBlockCompressedPayloadSplit
			}
			layers[layerIndex].TurboQuantPayloads = kvBlockPayloadSlices(layer.TurboQuantPayloads, cloneSlices)
			continue
		}
		if windowLen <= 0 || overlapStart >= overlapEnd {
			continue
		}
		localStart := overlapStart - windowStart
		localEnd := overlapEnd - windowStart
		keyLayerBytes, keyLayerShape, err := sliceKVSnapshotLayerRawTensorOpt(layer.KeyBytes, layer.KeyDType, layer.KeyShape, localStart, localEnd, cloneSlices)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer key tensor", err)
		}
		valueLayerBytes, valueLayerShape, err := sliceKVSnapshotLayerRawTensorOpt(layer.ValueBytes, layer.ValueDType, layer.ValueShape, localStart, localEnd, cloneSlices)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer value tensor", err)
		}
		layers[layerIndex].KeyDType = layer.KeyDType
		layers[layerIndex].KeyBytes = keyLayerBytes
		layers[layerIndex].KeyShape = keyLayerShape
		layers[layerIndex].ValueDType = layer.ValueDType
		layers[layerIndex].ValueBytes = valueLayerBytes
		layers[layerIndex].ValueShape = valueLayerShape
		headCount := len(layer.Heads)
		if headSlab != nil && slabCursor+headCount <= len(headSlab) {
			layers[layerIndex].Heads = headSlab[slabCursor : slabCursor+headCount : slabCursor+headCount]
			slabCursor += headCount
		} else {
			layers[layerIndex].Heads = make([]HeadSnapshot, headCount)
		}
		for headIndex, head := range layer.Heads {
			key, err := sliceKVSnapshotTensorOpt(head.Key, localStart, localEnd, s.HeadDim, windowLen, cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice key tensor", err)
			}
			value, err := sliceKVSnapshotTensorOpt(head.Value, localStart, localEnd, s.HeadDim, windowLen, cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice value tensor", err)
			}
			keyBytes, err := sliceKVSnapshotRawTensorOpt(head.KeyBytes, head.KeyDType, localStart, localEnd, windowLen, len(head.Key), cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice native key tensor", err)
			}
			valueBytes, err := sliceKVSnapshotRawTensorOpt(head.ValueBytes, head.ValueDType, localStart, localEnd, windowLen, len(head.Value), cloneSlices)
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
	var tokens []int32
	if cloneSlices {
		tokens = core.SliceClone(s.Tokens[start:end])
	} else {
		tokens = s.Tokens[start:end]
	}
	block := &Snapshot{
		Version:       effectiveVersion(s, KVSnapshotEncodingFloat32),
		Architecture:  s.Architecture,
		Tokens:        tokens,
		TokenOffset:   baseOffset + end,
		NumLayers:     s.NumLayers,
		NumHeads:      s.NumHeads,
		SeqLen:        end - start,
		HeadDim:       s.HeadDim,
		NumQueryHeads: s.NumQueryHeads,
		Layers:        layers,
	}
	if final {
		if cloneSlices {
			block.Generated = core.SliceClone(s.Generated)
			block.LogitShape = core.SliceClone(s.LogitShape)
			block.Logits = core.SliceClone(s.Logits)
		} else {
			block.Generated = s.Generated
			block.LogitShape = s.LogitShape
			block.Logits = s.Logits
		}
	}
	return block, nil
}

func kvSnapshotLayerWindowLen(layer LayerSnapshot, seqLen, headDim int) (int, error) {
	// Inline the per-length collect+iterate to skip a [2]int + [4]int
	// slice literal alloc per layer + per head (SaveStateBlocks fires
	// once per checkpointed block, with O(layers × heads) alloc count).
	windowLen := 0
	for _, length := range [2]int{
		kvSnapshotLayerRawWindowLen(layer.KeyBytes, layer.KeyDType, layer.KeyShape, seqLen),
		kvSnapshotLayerRawWindowLen(layer.ValueBytes, layer.ValueDType, layer.ValueShape, seqLen),
	} {
		if length < 0 {
			return 0, errLayerRawShapeMismatch
		}
		if length <= 0 {
			continue
		}
		if windowLen == 0 {
			windowLen = length
			continue
		}
		if windowLen != length {
			return 0, errLayerMixesWindowLens
		}
	}
	for _, head := range layer.Heads {
		for _, length := range [4]int{
			kvSnapshotTensorWindowLen(len(head.Key), seqLen, headDim),
			kvSnapshotTensorWindowLen(len(head.Value), seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.KeyBytes, head.KeyDType, seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.ValueBytes, head.ValueDType, seqLen, headDim),
		} {
			if length < 0 {
				return 0, errTensorShapeSeqHead
			}
			if length <= 0 {
				continue
			}
			if windowLen == 0 {
				windowLen = length
				continue
			}
			if windowLen != length {
				return 0, errLayerMixesWindowLens
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
	return sliceKVSnapshotTensorOpt(values, start, end, headDim, seqLen, true)
}

// sliceKVSnapshotTensorOpt slices a head Key/Value tensor. clone=false
// returns a sub-view of values (zero-alloc) — only the internal
// SaveStateBlocks walkBlocks path uses this, because the block snapshot
// is encoded + discarded within the yield call.
func sliceKVSnapshotTensorOpt(values []float32, start, end, headDim, seqLen int, clone bool) ([]float32, error) {
	if len(values) == 0 {
		return nil, nil
	}
	if seqLen <= 0 {
		return nil, errTensorShapeSeqHead
	}
	if headDim <= 0 || len(values) != seqLen*headDim {
		if len(values)%seqLen != 0 {
			return nil, errTensorShapeSeqHead
		}
		headDim = len(values) / seqLen
	}
	begin := start * headDim
	finish := end * headDim
	if begin < 0 || finish > len(values) || begin >= finish {
		return nil, errTensorBlockRangeInvalid
	}
	if clone {
		return core.SliceClone(values[begin:finish]), nil
	}
	return values[begin:finish:finish], nil
}

func sliceKVSnapshotRawTensor(raw []byte, dtype string, start, end, seqLen, valueCount int) ([]byte, error) {
	return sliceKVSnapshotRawTensorOpt(raw, dtype, start, end, seqLen, valueCount, true)
}

// sliceKVSnapshotRawTensorOpt slices a head's raw-byte tensor. clone=false
// returns a sub-view — see sliceKVSnapshotTensorOpt for the safe-use rule.
func sliceKVSnapshotRawTensorOpt(raw []byte, dtype string, start, end, seqLen, valueCount int, clone bool) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 {
		return nil, errUnsupportedRawTensorDtype
	}
	if valueCount <= 0 {
		if len(raw)%bytesPerValue != 0 {
			return nil, errRawTensorByteLenInvalid
		}
		valueCount = len(raw) / bytesPerValue
	}
	if seqLen <= 0 || valueCount%seqLen != 0 || len(raw) != valueCount*bytesPerValue {
		return nil, errRawTensorShapeSeq
	}
	headDim := valueCount / seqLen
	begin := start * headDim * bytesPerValue
	finish := end * headDim * bytesPerValue
	if begin < 0 || finish > len(raw) || begin >= finish {
		return nil, errRawTensorBlockRangeInvalid
	}
	if clone {
		return core.SliceClone(raw[begin:finish]), nil
	}
	return raw[begin:finish:finish], nil
}

func sliceKVSnapshotLayerRawTensor(raw []byte, dtype string, shape []int32, start, end int) ([]byte, []int32, error) {
	return sliceKVSnapshotLayerRawTensorOpt(raw, dtype, shape, start, end, true)
}

// sliceKVSnapshotLayerRawTensorOpt slices a native layer slab. clone=false can
// return a borrowed sub-view only when the requested sequence range is
// physically contiguous in the [B,H,L,D] row-major storage; for Gemma-style
// single K/V head slabs this keeps SaveStateBlocks from copying every block
// before the State writer immediately serialises it.
func sliceKVSnapshotLayerRawTensorOpt(raw []byte, dtype string, shape []int32, start, end int, clone bool) ([]byte, []int32, error) {
	if len(raw) == 0 {
		return nil, nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 || len(shape) != 4 {
		return nil, nil, errUnsupportedLayerRawTensor
	}
	B, H, L, D := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || start < 0 || end <= start || end > L {
		return nil, nil, errLayerRawTensorRangeInvalid
	}
	if len(raw) != B*H*L*D*bytesPerValue {
		return nil, nil, errLayerRawByteLenMismatch
	}
	take := end - start
	rowBytes := take * D * bytesPerValue
	if !clone && B*H == 1 {
		begin := start * D * bytesPerValue
		finish := begin + rowBytes
		outShape := core.SliceClone(shape)
		outShape[2] = int32(take)
		return raw[begin:finish:finish], outShape, nil
	}
	out := make([]byte, B*H*take*D*bytesPerValue)
	dst := 0
	for b := range B {
		for h := range H {
			src := (((b*H+h)*L + start) * D) * bytesPerValue
			copy(out[dst:dst+rowBytes], raw[src:src+rowBytes])
			dst += rowBytes
		}
	}
	outShape := core.SliceClone(shape)
	outShape[2] = int32(take)
	return out, outShape, nil
}

// AssembleBlocks reassembles contiguous blocks produced by SplitBlocks.
func AssembleBlocks(blocks []Block) (*Snapshot, error) {
	if len(blocks) == 0 {
		return nil, errBlocksEmpty
	}
	totalTokens, err := validateKVSnapshotBlockOrder(blocks)
	if err != nil {
		return nil, err
	}
	first := blocks[0].Snapshot
	if first == nil {
		return nil, errBlockNil
	}
	assembled := &Snapshot{
		Version:       first.Version,
		Architecture:  first.Architecture,
		NumLayers:     first.NumLayers,
		NumHeads:      first.NumHeads,
		HeadDim:       first.HeadDim,
		NumQueryHeads: first.NumQueryHeads,
		Layers:        emptyKVSnapshotLayers(first.Layers),
		// Pre-size Tokens against the validated total — append-block
		// accumulates a known count, so geometric grow is pure waste.
		Tokens: make([]int32, 0, totalTokens),
	}
	// Pre-size the per-head KeyBytes/ValueBytes buffers against the summed
	// raw payload across all blocks. appendKVSnapshotRawBlock otherwise
	// rides through Go's geometric grow on every block — once on first
	// arrival, plus one or two grows by block 3. The pre-sum pass walks
	// blocks × layers × heads but does no allocs.
	preSizeAssembledRawBytes(assembled, blocks)
	for _, block := range blocks {
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if err := appendKVSnapshotBlock(assembled, block.Snapshot); err != nil {
			return nil, err
		}
	}
	last := blocks[len(blocks)-1].Snapshot
	assembled.Generated = core.SliceClone(last.Generated)
	assembled.TokenOffset = last.TokenOffset
	assembled.LogitShape = core.SliceClone(last.LogitShape)
	assembled.Logits = core.SliceClone(last.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

// preSizeAssembledRawBytes pre-allocates per-head raw byte buffers in the
// assembled snapshot against the total payload across all blocks. Saves
// the appendKVSnapshotRawBlock geometric-grow path during AssembleBlocks.
func preSizeAssembledRawBytes(assembled *Snapshot, blocks []Block) {
	if assembled == nil || len(assembled.Layers) == 0 || len(blocks) == 0 {
		return
	}
	for layerIndex := range assembled.Layers {
		var layerKeyTotal, layerValueTotal int
		for _, block := range blocks {
			if block.Snapshot == nil || layerIndex >= len(block.Snapshot.Layers) {
				continue
			}
			srcLayer := block.Snapshot.Layers[layerIndex]
			layerKeyTotal += len(srcLayer.KeyBytes)
			layerValueTotal += len(srcLayer.ValueBytes)
		}
		dstLayer := &assembled.Layers[layerIndex]
		if layerKeyTotal > 0 {
			dstLayer.KeyBytes = make([]byte, 0, layerKeyTotal)
		}
		if layerValueTotal > 0 {
			dstLayer.ValueBytes = make([]byte, 0, layerValueTotal)
		}
		for headIndex := range assembled.Layers[layerIndex].Heads {
			var keyTotal, valueTotal int
			for _, block := range blocks {
				if block.Snapshot == nil || layerIndex >= len(block.Snapshot.Layers) {
					continue
				}
				srcLayer := block.Snapshot.Layers[layerIndex]
				if headIndex >= len(srcLayer.Heads) {
					continue
				}
				srcHead := srcLayer.Heads[headIndex]
				keyTotal += len(srcHead.KeyBytes)
				valueTotal += len(srcHead.ValueBytes)
			}
			dstHead := &assembled.Layers[layerIndex].Heads[headIndex]
			if keyTotal > 0 {
				dstHead.KeyBytes = make([]byte, 0, keyTotal)
			}
			if valueTotal > 0 {
				dstHead.ValueBytes = make([]byte, 0, valueTotal)
			}
		}
	}
}

func validateKVSnapshotBlockOrder(blocks []Block) (int, error) {
	nextStart := 0
	for index, block := range blocks {
		if block.Index != index {
			return 0, errBlocksOutOfOrder
		}
		if block.TokenStart != nextStart || block.TokenCount <= 0 {
			return 0, errBlocksNotContiguous
		}
		if block.Snapshot == nil || len(block.Snapshot.Tokens) != block.TokenCount {
			return 0, errBlockTokenCountMismatch
		}
		nextStart += block.TokenCount
	}
	return nextStart, nil
}

func emptyKVSnapshotLayers(layers []LayerSnapshot) []LayerSnapshot {
	out := make([]LayerSnapshot, len(layers))
	// Heads-slab: one backing slice across all layers — typical assembled
	// snapshots carry uniform NumHeads per layer (the first block sets
	// shape so we use it as the slab size). Layers with a divergent head
	// count fall back to per-layer make.
	var slabHeadsPerLayer int
	for _, layer := range layers {
		if len(layer.Heads) > slabHeadsPerLayer {
			slabHeadsPerLayer = len(layer.Heads)
		}
	}
	var headSlab []HeadSnapshot
	var slabCursor int
	if slabHeadsPerLayer > 0 {
		headSlab = make([]HeadSnapshot, len(layers)*slabHeadsPerLayer)
	}
	for i, layer := range layers {
		out[i] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			CacheMode:  layer.CacheMode,
			KeyDType:   layer.KeyDType,
			KeyShape:   core.SliceClone(layer.KeyShape),
			ValueDType: layer.ValueDType,
			ValueShape: core.SliceClone(layer.ValueShape),
		}
		headCount := len(layer.Heads)
		if headCount > 0 {
			if headSlab != nil && slabCursor+headCount <= len(headSlab) {
				out[i].Heads = headSlab[slabCursor : slabCursor+headCount : slabCursor+headCount]
				slabCursor += headCount
			} else {
				out[i].Heads = make([]HeadSnapshot, headCount)
			}
		}
	}
	return out
}

func appendKVSnapshotBlock(dst *Snapshot, block *Snapshot) error {
	if block.Architecture != "" && dst.Architecture != "" && block.Architecture != dst.Architecture {
		return errBlockArchMismatch
	}
	if block.HeadDim != dst.HeadDim || block.NumHeads != dst.NumHeads || block.NumLayers != dst.NumLayers {
		return errBlockShapeMismatch
	}
	if len(block.Layers) != len(dst.Layers) {
		return errBlockLayerCountMismatch
	}
	dst.Tokens = append(dst.Tokens, block.Tokens...)
	dst.SeqLen += block.SeqLen
	for layerIndex, layer := range block.Layers {
		dstLayer := &dst.Layers[layerIndex]
		if layer.CacheMode != "" {
			if dstLayer.CacheMode != "" && dstLayer.CacheMode != layer.CacheMode {
				return errBlockMetadataMismatch
			}
			dstLayer.CacheMode = layer.CacheMode
		}
		if len(layer.TurboQuantPayloads) > 0 {
			dstLayer.TurboQuantPayloads = append(dstLayer.TurboQuantPayloads, cloneKVByteSlices(layer.TurboQuantPayloads)...)
		}
		if len(layer.KeyBytes) > 0 {
			if err := appendKVSnapshotLayerRawBlock(&dstLayer.KeyDType, &dstLayer.KeyBytes, &dstLayer.KeyShape, layer.KeyDType, layer.KeyBytes, layer.KeyShape); err != nil {
				return core.E("AssembleBlocks", "append native layer key tensor", err)
			}
		}
		if len(layer.ValueBytes) > 0 {
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
			return errBlockHeadCountMismatch
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
		return errUnsupportedLayerRawTensor
	}
	B, H, L, D := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || len(raw) != B*H*L*D*bytesPerValue {
		return errLayerRawTensorShape
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return errLayerRawDtypeMismatch
	}
	if len(*dstBytes) == 0 {
		// First-arrival path is the only owner of the new shape — clone
		// happens here, not unconditionally on every call. Subsequent
		// calls rewrite dstShape[2] in-place after validating B/H/D.
		*dstBytes = append((*dstBytes)[:0], raw...)
		*dstShape = core.SliceClone(shape)
		return nil
	}
	if len(*dstShape) != 4 || int((*dstShape)[0]) != B || int((*dstShape)[1]) != H || int((*dstShape)[3]) != D {
		return errLayerRawTensorShape
	}
	// oldShape was previously cloned + read for oldLen — direct read
	// from dstShape eliminates the clone alloc; we only need shape[2]
	// (the sequence-length dim) and shape is rewritten in-place below.
	oldLen := int((*dstShape)[2])
	if oldLen <= 0 || len(*dstBytes) != B*H*oldLen*D*bytesPerValue {
		return errLayerRawByteLenMismatch
	}
	totalLen := oldLen + L
	if B*H == 1 {
		*dstBytes = append(*dstBytes, raw...)
		(*dstShape)[2] = int32(totalLen)
		return nil
	}
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
		return errUnsupportedRawTensorDtype
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return errRawTensorDtypeMismatch
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
		return nil, errSnapshotNil
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = defaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	// Pre-size block-tracking slices against the expected block count —
	// SaveStateBlocks walks blockSize-aligned ranges, so the count is
	// known within a layer-window adjustment of (seqLen + blockSize - 1) /
	// blockSize. Saves the geometric-grow append cycle per block.
	expectedBlocks := 1
	if blockSize > 0 && len(s.Tokens) > 0 {
		expectedBlocks = (len(s.Tokens) + blockSize - 1) / blockSize
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
		Blocks:       make([]StateBlockRef, 0, expectedBlocks),
	}
	err = s.walkBlocks(blockSize, false, func(block Block) (bool, error) {
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
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
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle)
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
		return nil, errStateStoreNil
	}
	if stream == nil {
		return nil, errBlockStreamNil
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
	err = stream(func(block Block) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if block.Snapshot == nil {
			return false, errStreamedBlockNil
		}
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		applyKVSnapshotStateBundleBlock(bundle, block)
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
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle)
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

func kvSnapshotStateBlockBundleHash(bundle *StateBlockBundle) string {
	if bundle == nil {
		return ""
	}
	builder := core.NewBuilder()
	// Pre-size to the exact final length so Builder never resizes mid-write.
	// Each block hash is 64 hex chars + 1 separator; the head fields run ~80
	// chars typical (architecture + 3 ints + encoding + 5 separators).
	size := len(bundle.Architecture) + len(string(bundle.KVEncoding)) + 5*1 + 30
	for _, ref := range bundle.Blocks {
		size += 1 + len(ref.KVHash)
	}
	builder.Grow(size)
	builder.WriteString(bundle.Architecture)
	builder.WriteString("|")
	builder.WriteString(string(bundle.KVEncoding))
	builder.WriteString("|")
	// strconv.AppendInt writes directly into the builder's growing
	// internal buffer; skips the three intermediate strings core.Itoa
	// would mint per call.
	var scratch [20]byte
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.TokenCount), 10))
	builder.WriteString("|")
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.TokenOffset), 10))
	builder.WriteString("|")
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.BlockSize), 10))
	for _, ref := range bundle.Blocks {
		builder.WriteString("|")
		builder.WriteString(ref.KVHash)
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
		return "", errBlockNil
	}
	hash := sha256.New()
	if err := block.Snapshot.writeWithOptions(hash, SaveOptions{KVEncoding: encoding}); err != nil {
		return "", err
	}
	var sum [sha256.Size]byte
	return hex.EncodeToString(hash.Sum(sum[:0])), nil
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
		var sum [sha256.Size]byte
		return ref, hex.EncodeToString(hash.Sum(sum[:0])), kvSnapshotStatePayloadRaw, payloadSize, nil
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
		return state.ChunkRef{}, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, errBundleURIRequired
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
	// Compute the index string once and reuse — block.Index is used in
	// tags, URI, and the default Title. The previous code minted three
	// separate copies via core.Itoa.
	indexStr := core.Itoa(block.Index)
	tags["block_index"] = indexStr
	tags["token_start"] = core.Itoa(block.TokenStart)
	tags["token_count"] = core.Itoa(block.TokenCount)
	// Skip the per-block labels make when the caller supplied no extra
	// labels — the default two-element pair is identical across blocks,
	// share a single package-global slice. State stores treat Labels as
	// read-only input; mutating the returned PutOptions is contract-
	// violating already.
	var labels []string
	if len(opts.Labels) == 0 {
		labels = kvSnapshotStateBlockDefaultLabels
	} else {
		// Pre-size for the deterministic 2 appended labels — avoids the
		// geometric-grow path on every per-block State save.
		labels = make([]string, len(opts.Labels), len(opts.Labels)+2)
		copy(labels, opts.Labels)
		labels = append(labels, "go-mlx", "kv-snapshot-block")
	}
	baseURI := firstNonEmpty(opts.URI, "mlx://kv-snapshot-blocks")
	// Direct string concatenation skips the fmt.Sprintf parse + format
	// state machinery on every per-block save (~SaveStateBlocks fires once
	// per checkpointed block during prefill). Avoid materialising the
	// default title when opts.Title is non-empty — the previous code
	// concatenated "go-mlx KV block " + indexStr unconditionally.
	title := opts.Title
	if title == "" {
		title = "go-mlx KV block " + indexStr
	}
	return state.PutOptions{
		URI:    baseURI + "/block/" + indexStr,
		Title:  title,
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
		return nil, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return nil, errBundleURIRequired
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
		return nil, errStateStoreNil
	}
	if bundle == nil {
		return nil, errBundleNil
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return nil, errUnsupportedBundleVersion
	}
	if bundle.Kind != StateBlockBundleKind {
		return nil, errBundleKindInvalid
	}
	if len(bundle.Blocks) == 0 {
		return nil, errBlocksEmpty
	}
	// Stream-assemble: load each block, fold into the assembled snapshot,
	// then release the per-block snapshot pointer. Avoids holding every
	// per-block []float32 / []byte alive until AssembleBlocks runs.
	snapshot, err := loadAndAssembleStateBlocks(ctx, store, bundle, opts)
	if err != nil {
		return nil, err
	}
	if bundle.TokenOffset > 0 && snapshot.TokenOffset != bundle.TokenOffset {
		return nil, errBlockTokenOffsetMismatch
	}
	return snapshot, nil
}

// loadAndAssembleStateBlocks streams blocks from a State bundle into a
// single assembled snapshot without retaining the per-block Snapshot
// pointers between iterations. The first block defines the assembled
// shape (Architecture, Layer count, head dimensions, raw tensor dtypes
// + shapes) — subsequent blocks fold into the same skeleton.
func loadAndAssembleStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	// Validate ordering up front against bundle.Blocks rather than after
	// loading every snapshot. The full block snapshots aren't required
	// for ordering checks.
	totalTokens := 0
	nextStart := 0
	for index, ref := range bundle.Blocks {
		if ref.Index != index {
			return nil, errBlocksOutOfOrder
		}
		if ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return nil, errBlocksNotContiguous
		}
		nextStart += ref.TokenCount
		totalTokens += ref.TokenCount
	}
	var assembled *Snapshot
	var lastBlock *Snapshot
	for index, ref := range bundle.Blocks {
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if block.Index != index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return nil, errBlockMetadataMismatch
		}
		if len(block.Snapshot.Tokens) != ref.TokenCount {
			return nil, errBlockTokenCountMismatch
		}
		if assembled == nil {
			first := block.Snapshot
			assembled = &Snapshot{
				Version:       first.Version,
				Architecture:  first.Architecture,
				NumLayers:     first.NumLayers,
				NumHeads:      first.NumHeads,
				HeadDim:       first.HeadDim,
				NumQueryHeads: first.NumQueryHeads,
				Layers:        emptyKVSnapshotLayers(first.Layers),
				Tokens:        make([]int32, 0, totalTokens),
			}
			// Pre-size assembled per-head byte buffers from bundle metadata
			// rather than walking the full block list — the bundle's
			// PayloadByteCount sums the raw block payload sizes, which
			// approximates the head byte counts when payload encoding is
			// raw. Falls back to no pre-size when bytes counts aren't
			// available; appendKVSnapshotRawBlock then handles growth.
			preSizeAssembledRawBytesFromFirst(assembled, first, len(bundle.Blocks))
		}
		if err := appendKVSnapshotBlock(assembled, block.Snapshot); err != nil {
			return nil, err
		}
		lastBlock = block.Snapshot
	}
	if assembled == nil || lastBlock == nil {
		return nil, errBlocksEmpty
	}
	assembled.Generated = core.SliceClone(lastBlock.Generated)
	assembled.TokenOffset = lastBlock.TokenOffset
	assembled.LogitShape = core.SliceClone(lastBlock.LogitShape)
	assembled.Logits = core.SliceClone(lastBlock.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

func loadAndAssembleStateBlockPrefix(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	blockCount, err := stateBlockPrefixCoverage(bundle, prefixTokens)
	if err != nil {
		return nil, err
	}
	var assembled *Snapshot
	var lastBlock *Snapshot
	for index := range blockCount {
		ref := bundle.Blocks[index]
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if block.Index != ref.Index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return nil, errBlockMetadataMismatch
		}
		if len(block.Snapshot.Tokens) != ref.TokenCount {
			return nil, errBlockTokenCountMismatch
		}
		blockSnapshot := block.Snapshot
		if ref.TokenStart+ref.TokenCount > prefixTokens {
			trimEnd := prefixTokens - ref.TokenStart
			if trimEnd <= 0 {
				break
			}
			baseOffset := EffectiveTokenOffset(blockSnapshot) - EffectiveSeqLen(blockSnapshot)
			if baseOffset < 0 {
				baseOffset = ref.TokenStart
			}
			blockSnapshot, err = blockSnapshot.SliceBlock(0, trimEnd, baseOffset, false)
			if err != nil {
				return nil, err
			}
		}
		if assembled == nil {
			first := blockSnapshot
			assembled = &Snapshot{
				Version:       first.Version,
				Architecture:  first.Architecture,
				NumLayers:     first.NumLayers,
				NumHeads:      first.NumHeads,
				HeadDim:       first.HeadDim,
				NumQueryHeads: first.NumQueryHeads,
				Layers:        emptyKVSnapshotLayers(first.Layers),
				Tokens:        make([]int32, 0, prefixTokens),
			}
			preSizeAssembledRawBytesFromFirst(assembled, first, blockCount)
		}
		if err := appendKVSnapshotBlock(assembled, blockSnapshot); err != nil {
			return nil, err
		}
		lastBlock = blockSnapshot
	}
	if assembled == nil || lastBlock == nil {
		return nil, errPrefixNoCoveringBlocks
	}
	assembled.Generated = core.SliceClone(lastBlock.Generated)
	assembled.TokenOffset = lastBlock.TokenOffset
	assembled.LogitShape = core.SliceClone(lastBlock.LogitShape)
	assembled.Logits = core.SliceClone(lastBlock.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

func stateBlockPrefixCoverage(bundle *StateBlockBundle, prefixTokens int) (int, error) {
	if bundle == nil || len(bundle.Blocks) == 0 {
		return 0, errPrefixNoCoveringBlocks
	}
	nextStart := 0
	totalTokens := 0
	blockCount := 0
	for index, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != index {
			return 0, errBlocksOutOfOrder
		}
		if ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errBlocksNotContiguous
		}
		nextStart += ref.TokenCount
		totalTokens += ref.TokenCount
		blockCount++
		if totalTokens >= prefixTokens {
			break
		}
	}
	if blockCount == 0 {
		return 0, errPrefixNoCoveringBlocks
	}
	if totalTokens < prefixTokens {
		return 0, errPrefixBlocksNoCover
	}
	return blockCount, nil
}

// preSizeAssembledRawBytesFromFirst pre-allocates per-head KeyBytes /
// ValueBytes buffers in assembled by extrapolating from the first
// block's byte count × the block count — cheaper than the full-blocks
// pre-pass when blocks are uniformly sized.
func preSizeAssembledRawBytesFromFirst(assembled *Snapshot, first *Snapshot, blockCount int) {
	if assembled == nil || first == nil || blockCount <= 0 {
		return
	}
	for layerIndex := range assembled.Layers {
		if layerIndex >= len(first.Layers) {
			continue
		}
		firstLayer := first.Layers[layerIndex]
		dstLayer := &assembled.Layers[layerIndex]
		if keyCap := len(firstLayer.KeyBytes) * blockCount; keyCap > 0 {
			dstLayer.KeyBytes = make([]byte, 0, keyCap)
		}
		if valueCap := len(firstLayer.ValueBytes) * blockCount; valueCap > 0 {
			dstLayer.ValueBytes = make([]byte, 0, valueCap)
		}
		for headIndex := range assembled.Layers[layerIndex].Heads {
			if headIndex >= len(firstLayer.Heads) {
				continue
			}
			firstHead := firstLayer.Heads[headIndex]
			dstHead := &dstLayer.Heads[headIndex]
			if keyCap := len(firstHead.KeyBytes) * blockCount; keyCap > 0 {
				dstHead.KeyBytes = make([]byte, 0, keyCap)
			}
			if valueCap := len(firstHead.ValueBytes) * blockCount; valueCap > 0 {
				dstHead.ValueBytes = make([]byte, 0, valueCap)
			}
		}
	}
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
		return nil, errStateStoreNil
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 || prefixTokens == bundle.TokenCount {
		return LoadFromStateBlocksWithOptions(ctx, store, bundle, opts)
	}
	if prefixTokens > bundle.TokenCount {
		return nil, errPrefixExceedsBundle
	}
	snapshot, err := loadAndAssembleStateBlockPrefix(ctx, store, bundle, prefixTokens, opts)
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
		return nil, errPrefixBlocksNoCover
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
		return nil, errStateStoreNil
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return nil, errTokenPrefixExceeds
	}
	// Inline iteration over bundle.Blocks skips the intermediate
	// stateBlockRefsForPrefix slice allocation — we already break when the
	// running token count covers prefixTokens, the same condition
	// stateBlockRefsForPrefix uses to truncate.
	if len(bundle.Blocks) == 0 {
		return nil, errTokenPrefixNoBlocks
	}
	tokens := make([]int32, 0, prefixTokens)
	nextStart := 0
	expectedIndex := 0
	covered := false
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != expectedIndex || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return nil, errTokenBlocksNotContiguous
		}
		// Fast path: when the block is raw-payload-stored (the predominant
		// case after the SaveStateBlocks switch to BinaryWriter), parse
		// tokens directly into the result slice. Avoids the per-block
		// []int32 allocation that LoadStateBlockTokensWithOptions would
		// otherwise pay through parseKVSnapshotTokens.
		var blockTokenCount int
		var err error
		if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
			data, derr := loadRawStateBlockPayload(ctx, store, ref)
			if derr != nil {
				return nil, derr
			}
			before := len(tokens)
			tokens, err = parseKVSnapshotTokensInto(tokens, data)
			if err != nil {
				return nil, err
			}
			blockTokenCount = len(tokens) - before
		} else {
			block, lerr := LoadStateBlockTokensWithOptions(ctx, store, ref, opts)
			if lerr != nil {
				return nil, lerr
			}
			if block.Index != ref.Index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
				return nil, errTokenBlockMetadata
			}
			tokens = append(tokens, block.Tokens...)
			blockTokenCount = len(block.Tokens)
		}
		if blockTokenCount != ref.TokenCount {
			return nil, errTokenBlockTokenCount
		}
		nextStart += ref.TokenCount
		expectedIndex++
		covered = true
		if len(tokens) >= prefixTokens {
			break
		}
	}
	if !covered {
		return nil, errTokenPrefixNoBlocks
	}
	if len(tokens) < prefixTokens {
		return nil, errTokenPrefixNoCover
	}
	return tokens[:prefixTokens], nil
}

func ValidateStateBlockBundle(bundle *StateBlockBundle) error {
	if bundle == nil {
		return errBundleNil
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return errUnsupportedBundleVersion
	}
	if bundle.Kind != StateBlockBundleKind {
		return errBundleKindInvalid
	}
	if bundle.TokenCount <= 0 {
		return errBundleTokenCountEmpty
	}
	if len(bundle.Blocks) == 0 {
		return errBundleNoBlocks
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
	chunk, err := state.BorrowRefBytes(ctx, store, stateBlockChunkRef(ref))
	if err != nil {
		return nil, core.E("LoadFromStateBlocks", "resolve raw State block", err)
	}
	data := chunk.Data
	if ref.PayloadByteCount > 0 && len(data) != ref.PayloadByteCount {
		return nil, errRawBlockPayloadLenMismatch
	}
	hash := core.SHA256Hex(data)
	if ref.KVHash != "" && hash != ref.KVHash {
		return nil, errRawBlockHashMismatch
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
		return nil, errUnsupportedBlockVersion
	}
	if envelope.Kind != KVSnapshotStateBlockKind {
		return nil, errBlockKindInvalid
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, errUnsupportedBlockEncoding
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadFromStateBlocks", "decode block payload", ResultError(decoded))
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, errBlockNonByteData
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, errBlockPayloadLenMismatch
	}
	hash := core.SHA256Hex(data)
	if envelope.KVHash != "" && hash != envelope.KVHash {
		return nil, errBlockHashMismatch
	}
	if expectedHash != "" && hash != expectedHash {
		return nil, errBlockRefHashMismatch
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
