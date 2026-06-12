// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	"dappco.re/go/mlx/pkg/metal"
)

// buildGemma4SlidingMask materialises a DENSE [B, L, L] f32 mask — 134MB at
// L=4096 (#76). Production prefill avoids it by chunking
// (FixedSlidingPrefillChunkLimit keeps L bounded); long UNCHUNKED prefills
// from lib callers pay the quadratic cost. If that ever lands on a hot
// path, replace with a banded/compressed mask rather than raising limits.
func buildGemma4SlidingMask(batchSize, seqLen, window int32) *metal.Array {
	negInf := float32(math.Inf(-1))
	data := make([]float32, int(batchSize)*int(seqLen)*int(seqLen))
	for b := range batchSize {
		base := int(b) * int(seqLen) * int(seqLen)
		for i := range seqLen {
			for j := range seqLen {
				if j <= i && i-j < window {
					data[base+int(i)*int(seqLen)+int(j)] = 0
				} else {
					data[base+int(i)*int(seqLen)+int(j)] = negInf
				}
			}
		}
	}
	return metal.FromValues(data, int(batchSize), 1, int(seqLen), int(seqLen))
}

func buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window int32) *metal.Array {
	negInf := float32(math.Inf(-1))
	data := make([]float32, int(batchSize)*int(queryLen)*int(keyLen))
	for b := range batchSize {
		base := int(b) * int(queryLen) * int(keyLen)
		for i := range queryLen {
			queryPos := offset + i
			for j := range keyLen {
				keyPos := keyStart + j
				allowed := keyPos <= queryPos
				if window > 0 && allowed {
					allowed = queryPos-keyPos < window
				}
				if allowed {
					data[base+int(i)*int(keyLen)+int(j)] = 0
				} else {
					data[base+int(i)*int(keyLen)+int(j)] = negInf
				}
			}
		}
	}
	return metal.FromValues(data, int(batchSize), 1, int(queryLen), int(keyLen))
}

type gemma4CachedAttentionMaskKey struct {
	batchSize int32
	queryLen  int32
	keyLen    int32
	offset    int32
	keyStart  int32
	window    int32
}

type gemma4RuntimeMaskCache struct {
	masks map[gemma4CachedAttentionMaskKey]*metal.Array
	owned []*metal.Array
}

func newGemma4RuntimeMaskCache() *gemma4RuntimeMaskCache {
	return &gemma4RuntimeMaskCache{}
}

func (c *gemma4RuntimeMaskCache) CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window int32) *metal.Array {
	if c == nil {
		return buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window)
	}
	key := gemma4CachedAttentionMaskKey{
		batchSize: batchSize,
		queryLen:  queryLen,
		keyLen:    keyLen,
		offset:    offset,
		keyStart:  keyStart,
		window:    window,
	}
	if c.masks == nil {
		c.masks = make(map[gemma4CachedAttentionMaskKey]*metal.Array)
	}
	if mask := c.masks[key]; mask != nil && mask.Valid() {
		return mask
	}
	mask := buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window)
	if mask == nil || !mask.Valid() {
		metal.Free(mask)
		return nil
	}
	c.masks[key] = mask
	c.owned = append(c.owned, mask)
	return mask
}

func (c *gemma4RuntimeMaskCache) Free() {
	if c == nil {
		return
	}
	metal.Free(c.owned...)
	c.owned = nil
	c.masks = nil
}

func gemma4CanUseOffsetCausalAttention(queryLen, keyLen, window int32) bool {
	if queryLen <= 1 || keyLen <= 0 {
		return false
	}
	if window <= 0 {
		return true
	}
	return queryLen <= window && keyLen <= window+queryLen-1
}

func gemma4SlidingCausalContextLen(queryLen, keyLen, window int32) int {
	if queryLen <= 1 || keyLen <= 0 || window <= 0 || queryLen > window {
		return int(keyLen)
	}
	needed := window + queryLen - 1
	if needed >= keyLen {
		return int(keyLen)
	}
	return int(needed)
}

func fixedSingleTokenCausalMaskFromHost(batchSize int32, capacity, offset int) *metal.Array {
	if batchSize <= 0 || capacity <= 0 {
		return nil
	}
	data := make([]float32, int(batchSize)*capacity)
	for b := range int(batchSize) {
		base := b * capacity
		for i := range capacity {
			if i > offset {
				data[base+i] = -1e9
			}
		}
	}
	return metal.FromValues(data, int(batchSize), 1, 1, capacity)
}

type fixedGemma4AttentionMaskSet struct {
	batchSize int32
	seqLen    int32
	disabled  bool
	masks     map[fixedGemma4AttentionMaskKey]*metal.Array
	owned     []*metal.Array
}

type fixedGemma4AttentionMaskKey struct {
	capacity int
	offset   int
}

func newFixedGemma4AttentionMaskSet(batchSize, seqLen int32, mask *metal.Array) *fixedGemma4AttentionMaskSet {
	return &fixedGemma4AttentionMaskSet{
		batchSize: batchSize,
		seqLen:    seqLen,
		disabled:  !metal.FixedSharedMaskEnabled() || mask != nil || seqLen != 1,
	}
}

func (s *fixedGemma4AttentionMaskSet) ForLayer(cache metal.Cache, prev sharedKV) *metal.Array {
	if s == nil || s.disabled {
		return nil
	}
	capacity, offset, ok := fixedGemma4AttentionMaskCapacityOffset(cache, prev, s.seqLen)
	if !ok {
		return nil
	}
	key := fixedGemma4AttentionMaskKey{capacity: capacity, offset: offset}
	if s.masks == nil {
		s.masks = make(map[fixedGemma4AttentionMaskKey]*metal.Array)
	}
	if mask := s.masks[key]; mask != nil && mask.Valid() {
		return mask
	}
	mask := fixedSingleTokenCausalMaskFromHost(s.batchSize, capacity, offset)
	if mask == nil || !mask.Valid() {
		metal.Free(mask)
		return nil
	}
	s.masks[key] = mask
	s.owned = append(s.owned, mask)
	return mask
}

func (s *fixedGemma4AttentionMaskSet) Free() {
	if s == nil {
		return
	}
	metal.Free(s.owned...)
	s.owned = nil
	s.masks = nil
}

func fixedGemma4AttentionMaskCapacityOffset(cache metal.Cache, prev sharedKV, seqLen int32) (int, int, bool) {
	if seqLen != 1 {
		return 0, 0, false
	}
	if fixed, ok := cache.(*metal.FixedKVCache); ok && fixed != nil && fixed.MaxSize() > 0 {
		offset := fixed.Offset()
		if offset >= 0 && offset+int(seqLen) <= fixed.MaxSize() {
			return fixed.MaxSize(), offset, true
		}
		return 0, 0, false
	}
	if prev.Fixed && prev.Keys != nil && prev.Keys.Valid() && prev.Keys.NumDims() == 4 {
		capacity := int(prev.Keys.Dim(2))
		offset := prev.Offset
		if capacity > 0 && offset >= 0 && offset+int(seqLen) <= capacity {
			return capacity, offset, true
		}
	}
	return 0, 0, false
}

func gemma4CombineMasks(base, extra *metal.Array) *metal.Array {
	if base == nil {
		return extra
	}
	if extra == nil {
		return base
	}
	combined := metal.Minimum(base, extra)
	return combined
}

// Forward runs the Gemma 4 text model forward pass.
