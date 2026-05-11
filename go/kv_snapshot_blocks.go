// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	stdio "io"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
)

const (
	// KVSnapshotMemvidBlockKind identifies one memvid chunk containing a KV block.
	KVSnapshotMemvidBlockKind = "go-mlx/kv-snapshot-block"
	// KVSnapshotMemvidBlockBundleKind identifies a collection of memvid KV blocks.
	KVSnapshotMemvidBlockBundleKind = "go-mlx/kv-snapshot-block-bundle"
	// KVSnapshotMemvidBlockVersion is the block envelope schema version.
	KVSnapshotMemvidBlockVersion = 1

	kvSnapshotMemvidPayloadRaw        = "raw"
	kvSnapshotMemvidPayloadJSONBase64 = "json-base64"
)

// KVSnapshotBlock is one contiguous token range from a KV snapshot.
type KVSnapshotBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Hash       string
	Snapshot   *KVSnapshot
}

// KVSnapshotMemvidBlockOptions controls memvid-backed KV block storage.
type KVSnapshotMemvidBlockOptions struct {
	BlockSize         int
	KVEncoding        KVSnapshotEncoding
	URI               string
	Title             string
	Kind              string
	Track             string
	Tags              map[string]string
	Labels            []string
	ReusePrefix       *KVSnapshotMemvidBlockBundle
	ReusePrefixTokens int
}

// KVSnapshotMemvidBlockBundle is a portable manifest for memvid KV blocks.
type KVSnapshotMemvidBlockBundle struct {
	Version      int                        `json:"version"`
	Kind         string                     `json:"kind"`
	SnapshotHash string                     `json:"snapshot_hash,omitempty"`
	KVEncoding   KVSnapshotEncoding         `json:"kv_encoding,omitempty"`
	Architecture string                     `json:"architecture,omitempty"`
	TokenCount   int                        `json:"token_count,omitempty"`
	TokenOffset  int                        `json:"token_offset,omitempty"`
	BlockSize    int                        `json:"block_size,omitempty"`
	NumLayers    int                        `json:"num_layers,omitempty"`
	NumHeads     int                        `json:"num_heads,omitempty"`
	SeqLen       int                        `json:"seq_len,omitempty"`
	HeadDim      int                        `json:"head_dim,omitempty"`
	ReusedBlocks int                        `json:"reused_blocks,omitempty"`
	Blocks       []KVSnapshotMemvidBlockRef `json:"blocks,omitempty"`
}

// KVSnapshotMemvidBlockRef links one logical KV block to a memvid chunk.
type KVSnapshotMemvidBlockRef struct {
	Index            int             `json:"index"`
	TokenStart       int             `json:"token_start"`
	TokenCount       int             `json:"token_count"`
	KVHash           string          `json:"kv_hash,omitempty"`
	PayloadEncoding  string          `json:"payload_encoding,omitempty"`
	PayloadByteCount int             `json:"payload_byte_count,omitempty"`
	Memvid           memvid.ChunkRef `json:"memvid"`
}

type kvSnapshotMemvidBlockEnvelope struct {
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
func (s *KVSnapshot) SplitBlocks(blockSize int) ([]KVSnapshotBlock, error) {
	blocks := []KVSnapshotBlock{}
	err := s.walkBlocks(blockSize, true, func(block KVSnapshotBlock) (bool, error) {
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
func (s *KVSnapshot) RangeBlocks(blockSize int, yield func(KVSnapshotBlock) bool) error {
	if yield == nil {
		return core.NewError("mlx: KV snapshot block yield is nil")
	}
	return s.walkBlocks(blockSize, true, func(block KVSnapshotBlock) (bool, error) {
		return yield(block), nil
	})
}

func (s *KVSnapshot) walkBlocks(blockSize int, includeHash bool, yield func(KVSnapshotBlock) (bool, error)) error {
	if s == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	if blockSize <= 0 {
		return core.NewError("mlx: KV snapshot block size must be > 0")
	}
	seqLen := effectiveKVSnapshotSeqLen(s)
	if seqLen <= 0 || len(s.Tokens) != seqLen {
		return core.NewError("mlx: KV snapshot block split requires tokens matching sequence length")
	}
	if s.HeadDim <= 0 {
		return core.NewError("mlx: KV snapshot block split requires head dimension")
	}
	baseOffset := effectiveKVSnapshotTokenOffset(s) - seqLen
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
		blockSnapshot, err := s.sliceBlock(start, end, baseOffset, end == seqLen)
		if err != nil {
			return err
		}
		var hash string
		if includeHash {
			hash, err = hashKVSnapshot(blockSnapshot)
			if err != nil {
				return err
			}
		}
		ok, err := yield(KVSnapshotBlock{
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

func (s *KVSnapshot) blockBoundaries(blockSize, seqLen int) ([]int, error) {
	seen := map[int]bool{0: true, seqLen: true}
	for next := blockSize; next < seqLen; next += blockSize {
		seen[next] = true
	}
	for _, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("KVSnapshot.SplitBlocks", "layer window", err)
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

func (s *KVSnapshot) sliceBlock(start, end, baseOffset int, final bool) (*KVSnapshot, error) {
	if start < 0 || end <= start || end > len(s.Tokens) {
		return nil, core.NewError("mlx: invalid KV snapshot block range")
	}
	seqLen := effectiveKVSnapshotSeqLen(s)
	layers := make([]KVLayerSnapshot, len(s.Layers))
	for layerIndex, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("KVSnapshot.SplitBlocks", "layer window", err)
		}
		windowStart := seqLen - windowLen
		overlapStart := max(start, windowStart)
		overlapEnd := min(end, seqLen)
		layers[layerIndex] = KVLayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
		}
		if windowLen <= 0 || overlapStart >= overlapEnd {
			continue
		}
		localStart := overlapStart - windowStart
		localEnd := overlapEnd - windowStart
		layers[layerIndex].Heads = make([]KVHeadSnapshot, len(layer.Heads))
		for headIndex, head := range layer.Heads {
			key, err := sliceKVSnapshotTensor(head.Key, localStart, localEnd, s.HeadDim, windowLen)
			if err != nil {
				return nil, core.E("KVSnapshot.SplitBlocks", "slice key tensor", err)
			}
			value, err := sliceKVSnapshotTensor(head.Value, localStart, localEnd, s.HeadDim, windowLen)
			if err != nil {
				return nil, core.E("KVSnapshot.SplitBlocks", "slice value tensor", err)
			}
			keyBytes, err := sliceKVSnapshotRawTensor(head.KeyBytes, head.KeyDType, localStart, localEnd, windowLen, len(head.Key))
			if err != nil {
				return nil, core.E("KVSnapshot.SplitBlocks", "slice native key tensor", err)
			}
			valueBytes, err := sliceKVSnapshotRawTensor(head.ValueBytes, head.ValueDType, localStart, localEnd, windowLen, len(head.Value))
			if err != nil {
				return nil, core.E("KVSnapshot.SplitBlocks", "slice native value tensor", err)
			}
			layers[layerIndex].Heads[headIndex] = KVHeadSnapshot{
				Key:        key,
				KeyDType:   head.KeyDType,
				KeyBytes:   keyBytes,
				Value:      value,
				ValueDType: head.ValueDType,
				ValueBytes: valueBytes,
			}
		}
	}
	block := &KVSnapshot{
		Version:       effectiveKVSnapshotVersion(s, KVSnapshotEncodingFloat32),
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

func kvSnapshotLayerWindowLen(layer KVLayerSnapshot, seqLen, headDim int) (int, error) {
	windowLen := 0
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

// AssembleKVSnapshotBlocks reassembles contiguous blocks produced by SplitBlocks.
func AssembleKVSnapshotBlocks(blocks []KVSnapshotBlock) (*KVSnapshot, error) {
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
	assembled := &KVSnapshot{
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

func validateKVSnapshotBlockOrder(blocks []KVSnapshotBlock) error {
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

func emptyKVSnapshotLayers(layers []KVLayerSnapshot) []KVLayerSnapshot {
	out := make([]KVLayerSnapshot, len(layers))
	for i, layer := range layers {
		out[i] = KVLayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
		}
		if len(layer.Heads) > 0 {
			out[i].Heads = make([]KVHeadSnapshot, len(layer.Heads))
		}
	}
	return out
}

func appendKVSnapshotBlock(dst *KVSnapshot, block *KVSnapshot) error {
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
		if len(layer.Heads) == 0 {
			continue
		}
		if len(dst.Layers[layerIndex].Heads) == 0 {
			dst.Layers[layerIndex].Heads = make([]KVHeadSnapshot, len(layer.Heads))
		}
		if len(layer.Heads) != len(dst.Layers[layerIndex].Heads) {
			return core.NewError("mlx: KV snapshot block head count mismatch")
		}
		for headIndex, head := range layer.Heads {
			dstHead := &dst.Layers[layerIndex].Heads[headIndex]
			dstHead.Key = append(dstHead.Key, head.Key...)
			dstHead.Value = append(dstHead.Value, head.Value...)
			if err := appendKVSnapshotRawBlock(&dstHead.KeyDType, &dstHead.KeyBytes, head.KeyDType, head.KeyBytes); err != nil {
				return core.E("AssembleKVSnapshotBlocks", "append native key tensor", err)
			}
			if err := appendKVSnapshotRawBlock(&dstHead.ValueDType, &dstHead.ValueBytes, head.ValueDType, head.ValueBytes); err != nil {
				return core.E("AssembleKVSnapshotBlocks", "append native value tensor", err)
			}
		}
	}
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

// SaveMemvidBlocks stores each KV block as a separate memvid chunk and returns a manifest.
func (s *KVSnapshot) SaveMemvidBlocks(ctx context.Context, store memvid.Writer, opts KVSnapshotMemvidBlockOptions) (*KVSnapshotMemvidBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = DefaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	bundle := &KVSnapshotMemvidBlockBundle{
		Version:      KVSnapshotMemvidBlockVersion,
		Kind:         KVSnapshotMemvidBlockBundleKind,
		KVEncoding:   encoding,
		Architecture: s.Architecture,
		TokenCount:   len(s.Tokens),
		TokenOffset:  effectiveKVSnapshotTokenOffset(s),
		BlockSize:    blockSize,
		NumLayers:    s.NumLayers,
		NumHeads:     s.NumHeads,
		SeqLen:       effectiveKVSnapshotSeqLen(s),
		HeadDim:      s.HeadDim,
		Blocks:       []KVSnapshotMemvidBlockRef{},
	}
	blockHashes := []string{}
	err = s.walkBlocks(blockSize, false, func(block KVSnapshotBlock) (bool, error) {
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotMemvidBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		blockHashes = append(blockHashes, hash)
		bundle.Blocks = append(bundle.Blocks, KVSnapshotMemvidBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotMemvidBlockBundleHash(bundle, blockHashes)
	return bundle, nil
}

func SaveMemvidBlocksFromStream(ctx context.Context, store memvid.Writer, opts KVSnapshotMemvidBlockOptions, stream func(func(KVSnapshotBlock) (bool, error)) error) (*KVSnapshotMemvidBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	if stream == nil {
		return nil, core.NewError("mlx: memvid KV block stream is nil")
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = DefaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	bundle := &KVSnapshotMemvidBlockBundle{
		Version:    KVSnapshotMemvidBlockVersion,
		Kind:       KVSnapshotMemvidBlockBundleKind,
		KVEncoding: encoding,
		BlockSize:  blockSize,
		Blocks:     []KVSnapshotMemvidBlockRef{},
	}
	blockHashes := []string{}
	err = stream(func(block KVSnapshotBlock) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if block.Snapshot == nil {
			return false, core.NewError("mlx: streamed KV snapshot block is nil")
		}
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotMemvidBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		applyKVSnapshotMemvidBundleBlock(bundle, block)
		blockHashes = append(blockHashes, hash)
		bundle.Blocks = append(bundle.Blocks, KVSnapshotMemvidBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	if err := validateKVSnapshotMemvidBlockBundle(bundle); err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotMemvidBlockBundleHash(bundle, blockHashes)
	return bundle, nil
}

func applyKVSnapshotMemvidBundleBlock(bundle *KVSnapshotMemvidBlockBundle, block KVSnapshotBlock) {
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

func kvSnapshotMemvidBlockBundleHash(bundle *KVSnapshotMemvidBlockBundle, blockHashes []string) string {
	if bundle == nil {
		return ""
	}
	builder := core.NewBuilder()
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
	return core.SHA256Hex([]byte(builder.String()))
}

func saveOrReuseKVSnapshotMemvidBlock(ctx context.Context, store memvid.Writer, block KVSnapshotBlock, opts KVSnapshotMemvidBlockOptions, encoding KVSnapshotEncoding) (memvid.ChunkRef, string, string, int, bool, error) {
	if reused, hash, ok, err := reusableKVSnapshotMemvidBlockRef(block, opts, encoding); err != nil {
		return memvid.ChunkRef{}, "", "", 0, false, err
	} else if ok {
		return reused.Memvid, hash, reused.PayloadEncoding, reused.PayloadByteCount, true, nil
	}
	ref, hash, payloadEncoding, payloadByteCount, err := saveKVSnapshotMemvidBlock(ctx, store, block, opts, encoding)
	return ref, hash, payloadEncoding, payloadByteCount, false, err
}

func reusableKVSnapshotMemvidBlockRef(block KVSnapshotBlock, opts KVSnapshotMemvidBlockOptions, encoding KVSnapshotEncoding) (KVSnapshotMemvidBlockRef, string, bool, error) {
	parent := opts.ReusePrefix
	if parent == nil || len(parent.Blocks) == 0 {
		return KVSnapshotMemvidBlockRef{}, "", false, nil
	}
	if parent.KVEncoding != "" && parent.KVEncoding != encoding {
		return KVSnapshotMemvidBlockRef{}, "", false, nil
	}
	reuseLimit := opts.ReusePrefixTokens
	if reuseLimit <= 0 {
		reuseLimit = parent.TokenCount
	}
	if block.TokenStart < 0 || block.TokenCount <= 0 || block.TokenStart+block.TokenCount > reuseLimit {
		return KVSnapshotMemvidBlockRef{}, "", false, nil
	}
	hash, err := hashKVSnapshotMemvidBlockPayload(block, encoding)
	if err != nil {
		return KVSnapshotMemvidBlockRef{}, "", false, err
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
	return KVSnapshotMemvidBlockRef{}, hash, false, nil
}

func hashKVSnapshotMemvidBlockPayload(block KVSnapshotBlock, encoding KVSnapshotEncoding) (string, error) {
	if block.Snapshot == nil {
		return "", core.NewError("mlx: KV snapshot block is nil")
	}
	hash := sha256.New()
	if err := block.Snapshot.writeWithOptions(hash, KVSnapshotSaveOptions{KVEncoding: encoding}); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func saveKVSnapshotMemvidBlock(ctx context.Context, store memvid.Writer, block KVSnapshotBlock, opts KVSnapshotMemvidBlockOptions, encoding KVSnapshotEncoding) (memvid.ChunkRef, string, string, int, error) {
	if streamStore, ok := store.(memvid.BinaryStreamWriter); ok {
		payloadSize, err := block.Snapshot.encodedSizeWithOptions(KVSnapshotSaveOptions{KVEncoding: encoding})
		if err != nil {
			return memvid.ChunkRef{}, "", "", 0, err
		}
		hash := sha256.New()
		ref, err := streamStore.PutBytesStream(ctx, payloadSize, kvSnapshotMemvidBlockPutOptions(block, opts, "", string(encoding), kvSnapshotMemvidPayloadRaw), func(writer stdio.Writer) error {
			return block.Snapshot.writeWithOptions(stdio.MultiWriter(writer, hash), KVSnapshotSaveOptions{KVEncoding: encoding})
		})
		if err != nil {
			return memvid.ChunkRef{}, "", "", 0, core.E("KVSnapshot.SaveMemvidBlocks", "stream raw memvid block", err)
		}
		return ref, hex.EncodeToString(hash.Sum(nil)), kvSnapshotMemvidPayloadRaw, payloadSize, nil
	}
	data, err := block.Snapshot.bytesWithOptions(KVSnapshotSaveOptions{KVEncoding: encoding})
	if err != nil {
		return memvid.ChunkRef{}, "", "", 0, err
	}
	hash := core.SHA256Hex(data)
	if binaryStore, ok := store.(memvid.BinaryWriter); ok {
		ref, err := binaryStore.PutBytes(ctx, data, kvSnapshotMemvidBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotMemvidPayloadRaw))
		if err != nil {
			return memvid.ChunkRef{}, "", "", 0, core.E("KVSnapshot.SaveMemvidBlocks", "write raw memvid block", err)
		}
		return ref, hash, kvSnapshotMemvidPayloadRaw, len(data), nil
	}
	envelope := kvSnapshotMemvidBlockEnvelope{
		Version:          KVSnapshotMemvidBlockVersion,
		Kind:             KVSnapshotMemvidBlockKind,
		BlockIndex:       block.Index,
		TokenStart:       block.TokenStart,
		TokenCount:       block.TokenCount,
		KVHash:           hash,
		KVEncoding:       string(encoding),
		BinaryEncoding:   "base64",
		PayloadByteCount: len(data),
		Data:             core.Base64Encode(data),
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(envelope), kvSnapshotMemvidBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotMemvidPayloadJSONBase64))
	if err != nil {
		return memvid.ChunkRef{}, "", "", 0, core.E("KVSnapshot.SaveMemvidBlocks", "write memvid block", err)
	}
	return ref, hash, kvSnapshotMemvidPayloadJSONBase64, len(data), nil
}

// SaveKVSnapshotMemvidBlockBundle stores the KV block manifest in the same
// memvid store as its referenced blocks.
func SaveKVSnapshotMemvidBlockBundle(ctx context.Context, store memvid.Writer, bundle *KVSnapshotMemvidBlockBundle, uri string) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid store is nil")
	}
	if core.Trim(uri) == "" {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid KV block bundle URI is required")
	}
	if err := validateKVSnapshotMemvidBlockBundle(bundle); err != nil {
		return memvid.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(bundle), memvid.PutOptions{
		URI:    uri,
		Title:  "go-mlx KV block bundle",
		Kind:   KVSnapshotMemvidBlockBundleKind,
		Track:  "session-kv-blocks",
		Labels: []string{"go-mlx", "kv-snapshot-block-bundle"},
	})
	if err != nil {
		return memvid.ChunkRef{}, core.E("KVSnapshot.SaveMemvidBlockBundle", "write memvid bundle", err)
	}
	return ref, nil
}

func kvSnapshotMemvidBlockPutOptions(block KVSnapshotBlock, opts KVSnapshotMemvidBlockOptions, hash, kvEncoding, payloadEncoding string) memvid.PutOptions {
	kind := opts.Kind
	if kind == "" {
		kind = KVSnapshotMemvidBlockKind
	}
	track := opts.Track
	if track == "" {
		track = "session-kv-blocks"
	}
	tags := cloneKVSnapshotMemvidTags(opts.Tags)
	if hash != "" {
		tags["kv_hash"] = hash
	}
	tags["kv_encoding"] = kvEncoding
	tags["payload_encoding"] = payloadEncoding
	tags["block_index"] = core.Itoa(block.Index)
	tags["token_start"] = core.Itoa(block.TokenStart)
	tags["token_count"] = core.Itoa(block.TokenCount)
	labels := append([]string(nil), opts.Labels...)
	labels = append(labels, "go-mlx", "kv-snapshot-block")
	baseURI := firstNonEmptyString(opts.URI, "mlx://kv-snapshot-blocks")
	return memvid.PutOptions{
		URI:    core.Sprintf("%s/block/%d", baseURI, block.Index),
		Title:  firstNonEmptyString(opts.Title, core.Sprintf("go-mlx KV block %d", block.Index)),
		Kind:   kind,
		Track:  track,
		Tags:   tags,
		Labels: labels,
	}
}

// LoadKVSnapshotFromMemvidBlocks restores a full KV snapshot from a memvid block manifest.
func LoadKVSnapshotFromMemvidBlocks(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle) (*KVSnapshot, error) {
	return LoadKVSnapshotFromMemvidBlocksWithOptions(ctx, store, bundle, KVSnapshotLoadOptions{})
}

// LoadKVSnapshotMemvidBlockBundle restores a KV block manifest by URI from the
// same memvid store as its referenced blocks.
func LoadKVSnapshotMemvidBlockBundle(ctx context.Context, store memvid.Store, uri string) (*KVSnapshotMemvidBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	if core.Trim(uri) == "" {
		return nil, core.NewError("mlx: memvid KV block bundle URI is required")
	}
	chunk, err := memvid.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadKVSnapshotMemvidBlockBundle", "resolve memvid bundle", err)
	}
	var bundle KVSnapshotMemvidBlockBundle
	if result := core.JSONUnmarshalString(chunk.Text, &bundle); !result.OK {
		return nil, core.E("LoadKVSnapshotMemvidBlockBundle", "parse bundle", kvSnapshotResultError(result))
	}
	if err := validateKVSnapshotMemvidBlockBundle(&bundle); err != nil {
		return nil, err
	}
	return &bundle, nil
}

// LoadKVSnapshotFromMemvidBlocksWithOptions restores a full KV snapshot from a
// memvid block manifest with explicit decode options.
func LoadKVSnapshotFromMemvidBlocksWithOptions(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle, opts KVSnapshotLoadOptions) (*KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	if bundle == nil {
		return nil, core.NewError("mlx: memvid KV block bundle is nil")
	}
	if bundle.Version <= 0 || bundle.Version > KVSnapshotMemvidBlockVersion {
		return nil, core.NewError("mlx: unsupported memvid KV block bundle version")
	}
	if bundle.Kind != KVSnapshotMemvidBlockBundleKind {
		return nil, core.NewError("mlx: invalid memvid KV block bundle kind")
	}
	blocks := make([]KVSnapshotBlock, 0, len(bundle.Blocks))
	for _, ref := range bundle.Blocks {
		block, err := loadKVSnapshotMemvidBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	snapshot, err := AssembleKVSnapshotBlocks(blocks)
	if err != nil {
		return nil, err
	}
	if bundle.TokenOffset > 0 && snapshot.TokenOffset != bundle.TokenOffset {
		return nil, core.NewError("mlx: memvid KV block token offset mismatch")
	}
	return snapshot, nil
}

// LoadKVSnapshotPrefixFromMemvidBlocks restores only the memvid KV blocks needed
// to cover prefixTokens. The returned snapshot is suitable for prompt-cache
// warmup; non-final prefixes intentionally omit logits.
func LoadKVSnapshotPrefixFromMemvidBlocks(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle, prefixTokens int) (*KVSnapshot, error) {
	return LoadKVSnapshotPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, KVSnapshotLoadOptions{})
}

// LoadKVSnapshotPrefixFromMemvidBlocksWithOptions restores only the memvid KV
// blocks needed to cover prefixTokens with explicit decode options.
func LoadKVSnapshotPrefixFromMemvidBlocksWithOptions(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle, prefixTokens int, opts KVSnapshotLoadOptions) (*KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	if err := validateKVSnapshotMemvidBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 || prefixTokens == bundle.TokenCount {
		return LoadKVSnapshotFromMemvidBlocksWithOptions(ctx, store, bundle, opts)
	}
	if prefixTokens > bundle.TokenCount {
		return nil, core.NewError("mlx: memvid KV prefix exceeds bundle token count")
	}
	refs := make([]KVSnapshotMemvidBlockRef, 0, len(bundle.Blocks))
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		refs = append(refs, ref)
		if ref.TokenStart+ref.TokenCount >= prefixTokens {
			break
		}
	}
	if len(refs) == 0 {
		return nil, core.NewError("mlx: memvid KV prefix has no covering blocks")
	}
	blocks := make([]KVSnapshotBlock, 0, len(refs))
	for _, ref := range refs {
		block, err := loadKVSnapshotMemvidBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	snapshot, err := AssembleKVSnapshotBlocks(blocks)
	if err != nil {
		return nil, err
	}
	if len(snapshot.Tokens) == prefixTokens {
		if prefixTokens < bundle.TokenCount {
			clearKVSnapshotTerminalState(snapshot)
		}
		return snapshot, nil
	}
	if len(snapshot.Tokens) < prefixTokens {
		return nil, core.NewError("mlx: memvid KV prefix blocks do not cover requested tokens")
	}
	baseOffset := effectiveKVSnapshotTokenOffset(snapshot) - effectiveKVSnapshotSeqLen(snapshot)
	if baseOffset < 0 {
		baseOffset = 0
	}
	trimmed, err := snapshot.sliceBlock(0, prefixTokens, baseOffset, false)
	if err != nil {
		return nil, err
	}
	return trimmed, nil
}

func validateKVSnapshotMemvidBlockBundle(bundle *KVSnapshotMemvidBlockBundle) error {
	if bundle == nil {
		return core.NewError("mlx: memvid KV block bundle is nil")
	}
	if bundle.Version <= 0 || bundle.Version > KVSnapshotMemvidBlockVersion {
		return core.NewError("mlx: unsupported memvid KV block bundle version")
	}
	if bundle.Kind != KVSnapshotMemvidBlockBundleKind {
		return core.NewError("mlx: invalid memvid KV block bundle kind")
	}
	if bundle.TokenCount <= 0 {
		return core.NewError("mlx: memvid KV block bundle token count is empty")
	}
	if len(bundle.Blocks) == 0 {
		return core.NewError("mlx: memvid KV block bundle has no blocks")
	}
	return nil
}

func clearKVSnapshotTerminalState(snapshot *KVSnapshot) {
	if snapshot == nil {
		return
	}
	snapshot.Generated = nil
	snapshot.LogitShape = nil
	snapshot.Logits = nil
}

func loadKVSnapshotMemvidBlock(ctx context.Context, store memvid.Store, ref KVSnapshotMemvidBlockRef) (KVSnapshotBlock, error) {
	return loadKVSnapshotMemvidBlockWithOptions(ctx, store, ref, KVSnapshotLoadOptions{})
}

func loadKVSnapshotMemvidBlockWithOptions(ctx context.Context, store memvid.Store, ref KVSnapshotMemvidBlockRef, opts KVSnapshotLoadOptions) (KVSnapshotBlock, error) {
	if ref.PayloadEncoding == kvSnapshotMemvidPayloadRaw {
		return loadRawKVSnapshotMemvidBlockWithOptions(ctx, store, ref, opts)
	}
	chunk, err := memvid.Resolve(ctx, store, ref.Memvid.ChunkID)
	if err != nil {
		return KVSnapshotBlock{}, core.E("LoadKVSnapshotFromMemvidBlocks", "resolve memvid block", err)
	}
	var envelope kvSnapshotMemvidBlockEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return KVSnapshotBlock{}, core.E("LoadKVSnapshotFromMemvidBlocks", "parse block envelope", kvSnapshotResultError(result))
	}
	data, err := decodeKVSnapshotMemvidBlockEnvelope(envelope, ref.KVHash)
	if err != nil {
		return KVSnapshotBlock{}, err
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return KVSnapshotBlock{}, err
	}
	return KVSnapshotBlock{
		Index:      envelope.BlockIndex,
		TokenStart: envelope.TokenStart,
		TokenCount: envelope.TokenCount,
		Hash:       envelope.KVHash,
		Snapshot:   snapshot,
	}, nil
}

func loadRawKVSnapshotMemvidBlockWithOptions(ctx context.Context, store memvid.Store, ref KVSnapshotMemvidBlockRef, opts KVSnapshotLoadOptions) (KVSnapshotBlock, error) {
	chunk, err := memvid.ResolveRefBytes(ctx, store, ref.Memvid)
	if err != nil {
		return KVSnapshotBlock{}, core.E("LoadKVSnapshotFromMemvidBlocks", "resolve raw memvid block", err)
	}
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}
	if ref.PayloadByteCount > 0 && len(data) != ref.PayloadByteCount {
		return KVSnapshotBlock{}, core.NewError("mlx: memvid raw KV block payload length mismatch")
	}
	hash := core.SHA256Hex(data)
	if ref.KVHash != "" && hash != ref.KVHash {
		return KVSnapshotBlock{}, core.NewError("mlx: memvid raw KV block hash mismatch")
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return KVSnapshotBlock{}, err
	}
	return KVSnapshotBlock{
		Index:      ref.Index,
		TokenStart: ref.TokenStart,
		TokenCount: ref.TokenCount,
		Hash:       ref.KVHash,
		Snapshot:   snapshot,
	}, nil
}

func decodeKVSnapshotMemvidBlockEnvelope(envelope kvSnapshotMemvidBlockEnvelope, expectedHash string) ([]byte, error) {
	if envelope.Version <= 0 || envelope.Version > KVSnapshotMemvidBlockVersion {
		return nil, core.NewError("mlx: unsupported memvid KV block version")
	}
	if envelope.Kind != KVSnapshotMemvidBlockKind {
		return nil, core.NewError("mlx: invalid memvid KV block kind")
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, core.NewError("mlx: unsupported memvid KV block binary encoding")
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadKVSnapshotFromMemvidBlocks", "decode block payload", kvSnapshotResultError(decoded))
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, core.NewError("mlx: memvid KV block decoded to non-byte data")
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, core.NewError("mlx: memvid KV block payload length mismatch")
	}
	hash := core.SHA256Hex(data)
	if envelope.KVHash != "" && hash != envelope.KVHash {
		return nil, core.NewError("mlx: memvid KV block hash mismatch")
	}
	if expectedHash != "" && hash != expectedHash {
		return nil, core.NewError("mlx: memvid KV block ref hash mismatch")
	}
	return data, nil
}

func effectiveKVSnapshotSeqLen(snapshot *KVSnapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.SeqLen > 0 {
		return snapshot.SeqLen
	}
	return len(snapshot.Tokens)
}
