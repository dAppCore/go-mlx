// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

const (
	defaultPagedKVPageSize       = 512
	hyperLongPagedKVPageSize     = 1024
	hyperLongPagedKVSizeBoundary = 65536
)

var enablePagedKVPrealloc = core.Env("GO_MLX_ENABLE_PAGED_KV_PREALLOC") == "1"

func pagedKVPreallocEnabled() bool {
	return enablePagedKVPrealloc || pagedKVPreallocRuntimeEnabled()
}

// Cache manages key-value pairs for transformer attention layers.
//
//	cache := metal.NewKVCache()              // unbounded — grows with context
//	cache := metal.NewRotatingKVCache(4096)  // bounded — slides at maxSize tokens
//
//	k, v = cache.Update(k, v, seqLen)       // append new tokens; returns full K/V slice
//	cache.Detach()                           // break graph after Eval to free Metal memory
type Cache interface {
	// Update adds new key/value tensors and returns the full cached K/V.
	Update(k, v *Array, seqLen int) (*Array, *Array)
	// Offset returns the total number of tokens processed.
	Offset() int
	// Len returns the number of cached tokens (may differ from Offset for rotating caches).
	Len() int
	// State returns the cached K/V arrays, or nil if empty.
	State() []*Array
	// Reset clears the cache for a new generation session.
	Reset()
	// Detach replaces internal K/V arrays with copies that have no graph parents.
	// Call after Eval to allow Metal memory from prior graph operations to be freed.
	Detach()
}

// KVCacheMode names the native storage strategy used for K/V tensors.
type KVCacheMode string

const (
	KVCacheModeDefault KVCacheMode = ""
	KVCacheModeFP16    KVCacheMode = "fp16"
	KVCacheModeQ8      KVCacheMode = "q8"
	KVCacheModeKQ8VQ4  KVCacheMode = "k-q8-v-q4"
	KVCacheModePaged   KVCacheMode = "paged"
	KVCacheModeFixed   KVCacheMode = "fixed"
)

type readableCache interface {
	ReadState() (state []*Array, owned []*Array)
}

func cacheReadState(cache Cache) (state []*Array, owned []*Array) {
	if cache == nil {
		return nil, nil
	}
	if readable, ok := cache.(readableCache); ok {
		return readable.ReadState()
	}
	if rotating, ok := cache.(*RotatingKVCache); ok {
		state = rotating.orderedState()
		return state, state
	}
	return cache.State(), nil
}

// KVCache implements an unbounded cache that grows as needed.
// Pre-allocates in chunks of `step` tokens to reduce allocations.
type KVCache struct {
	keys, values *Array
	offset       int
	step         int
}

// NewKVCache creates a new unbounded KV cache with 256-token chunks.
func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

func (c *KVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	prev := c.offset
	shape := k.Shape()
	if len(shape) < 4 {
		// K/V must be [B, H, L, D] — if not, pass through unchanged
		if c.keys == nil {
			c.keys, c.values = k, v
		}
		c.offset += seqLen
		return c.keys, c.values
	}
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	// Grow buffer if needed.
	if c.keys == nil || (prev+seqLen) > int(c.keys.Shape()[2]) {
		nSteps := (c.step + seqLen - 1) / c.step
		newK := Zeros([]int32{B, H, int32(nSteps * c.step), Dk}, k.Dtype())
		newV := Zeros([]int32{B, H, int32(nSteps * c.step), Dv}, v.Dtype())

		if c.keys != nil {
			oldK, oldV := c.keys, c.values
			if prev%c.step != 0 {
				oldK = Slice(oldK, []int32{0, 0, 0, 0}, []int32{B, H, int32(prev), Dk})
				oldV = Slice(oldV, []int32{0, 0, 0, 0}, []int32{B, H, int32(prev), Dv})
				Free(c.keys, c.values)
			}
			c.keys = Concatenate([]*Array{oldK, newK}, 2)
			c.values = Concatenate([]*Array{oldV, newV}, 2)
			Free(oldK, oldV, newK, newV)
		} else {
			c.keys, c.values = newK, newV
		}
	}

	c.offset += seqLen
	oldK, oldV := c.keys, c.values
	c.keys = SliceUpdateInplace(c.keys, k, []int32{0, 0, int32(prev), 0}, []int32{B, H, int32(c.offset), Dk})
	c.values = SliceUpdateInplace(c.values, v, []int32{0, 0, int32(prev), 0}, []int32{B, H, int32(c.offset), Dv})
	Free(oldK, oldV)

	return Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.offset), Dk}),
		Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.offset), Dv})
}

func (c *KVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

func (c *KVCache) Offset() int { return c.offset }
func (c *KVCache) Len() int    { return c.offset }

func (c *KVCache) Reset() {
	Free(c.keys, c.values)
	c.keys = nil
	c.values = nil
	c.offset = 0
}

func (c *KVCache) Detach() {
	if c.keys == nil {
		return
	}
	Detach(c.keys, c.values)
}

// RotatingKVCache implements a bounded sliding window cache.
type RotatingKVCache struct {
	keys, values *Array
	offset       int
	maxSize      int
	step         int
	idx          int
}

// NewRotatingKVCache creates a cache bounded to maxSize tokens.
func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, step: 256}
}

func (c *RotatingKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	if seqLen > 1 {
		return c.updateConcat(k, v, seqLen)
	}
	return c.updateInPlace(k, v)
}

func (c *RotatingKVCache) updateInPlace(k, v *Array) (*Array, *Array) {
	shape := k.Shape()
	if len(shape) < 4 {
		if c.keys == nil {
			c.keys, c.values = k, v
		}
		c.offset++
		return c.keys, c.values
	}
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	if c.keys == nil || (c.idx >= int(c.keys.Shape()[2]) && int(c.keys.Shape()[2]) < c.maxSize) {
		var cap int
		if c.keys != nil {
			cap = int(c.keys.Shape()[2])
		}
		newSize := min(c.step, c.maxSize-cap)
		newK := Zeros([]int32{B, H, int32(newSize), Dk}, k.Dtype())
		newV := Zeros([]int32{B, H, int32(newSize), Dv}, v.Dtype())
		if c.keys != nil {
			oldK, oldV := c.keys, c.values
			c.keys = Concatenate([]*Array{oldK, newK}, 2)
			c.values = Concatenate([]*Array{oldV, newV}, 2)
			Free(oldK, oldV, newK, newV)
		} else {
			c.keys, c.values = newK, newV
		}
	}

	if c.idx >= c.maxSize {
		c.idx = 0
	}

	oldK, oldV := c.keys, c.values
	c.keys = SliceUpdateInplace(c.keys, k, []int32{0, 0, int32(c.idx), 0}, []int32{B, H, int32(c.idx + 1), Dk})
	c.values = SliceUpdateInplace(c.values, v, []int32{0, 0, int32(c.idx), 0}, []int32{B, H, int32(c.idx + 1), Dv})
	Free(oldK, oldV)

	c.offset++
	c.idx++

	validLen := int32(min(c.offset, c.maxSize))
	start := 0
	if c.offset > c.maxSize {
		start = c.idx
		if start >= c.maxSize {
			start = 0
		}
	}
	return rotatingCacheWindow(c.keys, start, validLen), rotatingCacheWindow(c.values, start, validLen)
}

func (c *RotatingKVCache) updateConcat(k, v *Array, seqLen int) (*Array, *Array) {
	shape := k.Shape()
	if len(shape) < 4 {
		// K/V must be [B, H, L, D] — if not, pass through unchanged
		if c.keys == nil {
			c.keys, c.values = k, v
		}
		c.offset += seqLen
		return c.keys, c.values
	}
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	var fullK, fullV *Array
	if c.keys == nil {
		fullK, fullV = k.Clone(), v.Clone()
	} else {
		oldK, oldV := c.keys, c.values
		fullK = Concatenate([]*Array{oldK, k}, 2)
		fullV = Concatenate([]*Array{oldV, v}, 2)
		Free(oldK, oldV)
	}
	c.offset += seqLen

	cap := int(fullK.Shape()[2])
	if trim := cap - c.maxSize; trim > 0 {
		// Preserve the full multi-token prompt for the current attention pass,
		// while storing only the bounded sliding window for future decode steps.
		c.keys = Slice(fullK, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dk})
		c.values = Slice(fullV, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dv})
		c.idx = int(c.keys.Shape()[2])
		return Slice(fullK, []int32{0, 0, 0, 0}, []int32{B, H, int32(cap), Dk}),
			Slice(fullV, []int32{0, 0, 0, 0}, []int32{B, H, int32(cap), Dv})
	}

	c.keys, c.values = fullK, fullV
	c.idx = int(c.keys.Shape()[2])
	// Return Slice views so callers can Free them without destroying the cache.
	// (updateInPlace and KVCache.Update already return Slice views.)
	return Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.idx), Dk}),
		Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.idx), Dv})
}

func rotatingCacheWindow(buffer *Array, start int, validLen int32) *Array {
	if buffer == nil || !buffer.Valid() {
		return nil
	}
	shape := buffer.Shape()
	if validLen <= 0 {
		starts := make([]int32, len(shape))
		ends := make([]int32, len(shape))
		return Slice(buffer, starts, ends)
	}
	if len(shape) < 4 {
		return buffer.Clone()
	}
	if start <= 0 || int32(start) >= validLen {
		return Slice(buffer, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], validLen, shape[3]})
	}

	tail := Slice(buffer, []int32{0, 0, int32(start), 0}, []int32{shape[0], shape[1], validLen, shape[3]})
	head := Slice(buffer, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(start), shape[3]})
	ordered := Concatenate([]*Array{tail, head}, 2)
	Free(tail, head)
	return ordered
}

func (c *RotatingKVCache) orderedState() []*Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	start := 0
	if c.offset > c.maxSize {
		start = c.idx
		if start >= c.maxSize {
			start = 0
		}
	}
	validLen := int32(c.Len())
	return []*Array{
		rotatingCacheWindow(c.keys, start, validLen),
		rotatingCacheWindow(c.values, start, validLen),
	}
}

func (c *RotatingKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

func (c *RotatingKVCache) Offset() int { return c.offset }
func (c *RotatingKVCache) Len() int {
	length := min(c.offset, c.maxSize)
	if c.keys == nil || !c.keys.Valid() {
		return length
	}
	shape := c.keys.Shape()
	if len(shape) >= 3 && int(shape[2]) < length {
		return int(shape[2])
	}
	return length
}

func (c *RotatingKVCache) Reset() {
	Free(c.keys, c.values)
	c.keys = nil
	c.values = nil
	c.offset = 0
	c.idx = 0
}

func (c *RotatingKVCache) Detach() {
	if c.keys == nil {
		return
	}
	Detach(c.keys, c.values)
}

// FixedKVCache keeps K/V storage at one stable capacity for single-token
// decode. It is an experimental cache used by compiled Gemma 4 decode probes;
// normal callers should prefer the public paged or rotating cache modes.
type FixedKVCache struct {
	keys, values              *Array
	slidingIndices, lastIndex *Array
	storageDType              DType
	hasStorageDType           bool
	offset                    int
	length                    int
	maxSize                   int
}

// FixedKVState is a caller-owned view of a fixed-capacity K/V cache.
type FixedKVState struct {
	Keys   *Array
	Values *Array
	Owned  []*Array
	Length int
}

// Free releases cloned fixed-cache handles.
func (s FixedKVState) Free() {
	Free(s.Owned...)
}

// NewFixedKVCache creates a fixed-capacity KV cache.
func NewFixedKVCache(maxSize int) *FixedKVCache {
	return &FixedKVCache{maxSize: maxSize}
}

func NewFixedKVCacheWithDType(maxSize int, dtype DType) *FixedKVCache {
	cache := NewFixedKVCache(maxSize)
	cache.storageDType = dtype
	cache.hasStorageDType = true
	return cache
}

func (c *FixedKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return nil, nil
	}
	k, v, owned := c.storageKV(k, v)
	defer Free(owned...)
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || c.maxSize <= 0 {
		if c.keys == nil {
			c.keys, c.values = k.Clone(), v.Clone()
		}
		c.offset += seqLen
		c.length = min(c.offset, c.maxSize)
		return c.keys.Clone(), c.values.Clone()
	}
	totalLen := int(kShape[2])
	if seqLen <= 0 || seqLen > totalLen {
		seqLen = totalLen
	}
	c.ensureShape(kShape[0], kShape[1], kShape[3], vShape[3], k.Dtype(), v.Dtype())
	if c.offset+seqLen > c.maxSize {
		return c.updateOverflow(k, v, seqLen)
	}
	writeK, writeV := k, v
	writeLen := seqLen
	if writeLen > c.maxSize {
		start := writeLen - c.maxSize
		writeK = Slice(k, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(writeLen), kShape[3]})
		writeV = Slice(v, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(writeLen), vShape[3]})
		defer Free(writeK, writeV)
		writeLen = c.maxSize
	}

	start := c.offset

	oldK, oldV := c.keys, c.values
	c.keys = SliceUpdateInplace(c.keys, writeK, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(start + writeLen), kShape[3]})
	c.values = SliceUpdateInplace(c.values, writeV, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(start + writeLen), vShape[3]})
	Free(oldK, oldV)

	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	return c.validState()
}

func (c *FixedKVCache) updateOverflow(k, v *Array, seqLen int) (*Array, *Array) {
	prevK, prevV := c.validState()
	var fullK, fullV *Array
	if prevK == nil || prevV == nil {
		fullK, fullV = k.Clone(), v.Clone()
	} else {
		fullK = Concatenate([]*Array{prevK, k}, 2)
		fullV = Concatenate([]*Array{prevV, v}, 2)
		Free(prevK, prevV)
	}
	tailK, tailV := cacheTail(fullK, fullV, c.maxSize)
	c.replaceFromTail(tailK, tailV)
	if tailK != fullK {
		Free(tailK, tailV)
	}
	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	if seqLen > 1 {
		return c.overflowAttentionContext(fullK, fullV)
	}
	tailStateK, tailStateV := c.validState()
	if tailStateK != nil && tailStateV != nil {
		return tailStateK, tailStateV
	}
	return cacheTail(fullK, fullV, c.maxSize)
}

func (c *FixedKVCache) overflowAttentionContext(fullK, fullV *Array) (*Array, *Array) {
	kShape := fullK.Shape()
	vShape := fullV.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || c.maxSize <= 0 {
		return fullK, fullV
	}
	totalLen := int(kShape[2])
	if totalLen <= c.maxSize {
		return fullK, fullV
	}
	prefixLen := totalLen - c.maxSize
	prefixK := Slice(fullK, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(prefixLen), kShape[3]})
	prefixV := Slice(fullV, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(prefixLen), vShape[3]})
	tailK, tailV := c.validState()
	if tailK == nil || tailV == nil {
		Free(prefixK, prefixV, tailK, tailV)
		return fullK, fullV
	}
	outK := Concatenate([]*Array{prefixK, tailK}, 2)
	outV := Concatenate([]*Array{prefixV, tailV}, 2)
	Free(prefixK, prefixV, tailK, tailV, fullK, fullV)
	return outK, outV
}

func (c *FixedKVCache) ensureShape(batch, heads, keyDim, valueDim int32, keyType, valueType DType) {
	if c.keys != nil && c.values != nil {
		kShape := c.keys.Shape()
		vShape := c.values.Shape()
		if len(kShape) >= 4 && len(vShape) >= 4 &&
			kShape[0] == batch && kShape[1] == heads && kShape[2] == int32(c.maxSize) && kShape[3] == keyDim &&
			vShape[0] == batch && vShape[1] == heads && vShape[2] == int32(c.maxSize) && vShape[3] == valueDim {
			return
		}
	}
	Free(c.keys, c.values, c.slidingIndices, c.lastIndex)
	c.keys = Zeros([]int32{batch, heads, int32(c.maxSize), keyDim}, keyType)
	c.values = Zeros([]int32{batch, heads, int32(c.maxSize), valueDim}, valueType)
	c.slidingIndices = nil
	c.lastIndex = nil
	c.offset = 0
	c.length = 0
}

func (c *FixedKVCache) slidingUpdateInputs() (*Array, *Array) {
	if c.maxSize <= 0 {
		return nil, nil
	}
	if c.slidingIndices != nil && c.slidingIndices.Valid() && c.lastIndex != nil && c.lastIndex.Valid() {
		return c.slidingIndices, c.lastIndex
	}
	Free(c.slidingIndices, c.lastIndex)
	indices := make([]int32, c.maxSize)
	for i := 0; i < c.maxSize; i++ {
		next := i + 1
		if next >= c.maxSize {
			next = c.maxSize - 1
		}
		indices[i] = int32(next)
	}
	c.slidingIndices = FromValues(indices, c.maxSize)
	c.lastIndex = FromValue(c.maxSize - 1)
	return c.slidingIndices, c.lastIndex
}

func (c *FixedKVCache) replaceFromTail(k, v *Array) {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return
	}
	k, v, owned := c.storageKV(k, v)
	defer Free(owned...)
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		return
	}
	Free(c.keys, c.values)
	c.keys = Zeros([]int32{kShape[0], kShape[1], int32(c.maxSize), kShape[3]}, k.Dtype())
	c.values = Zeros([]int32{vShape[0], vShape[1], int32(c.maxSize), vShape[3]}, v.Dtype())
	tailLen := min(int(kShape[2]), c.maxSize)
	oldK, oldV := c.keys, c.values
	c.keys = SliceUpdateInplace(c.keys, k, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(tailLen), kShape[3]})
	c.values = SliceUpdateInplace(c.values, v, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(tailLen), vShape[3]})
	Free(oldK, oldV)
}

func (c *FixedKVCache) validState() (*Array, *Array) {
	if c.keys == nil || c.values == nil {
		return nil, nil
	}
	kShape := c.keys.Shape()
	vShape := c.values.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || c.length <= 0 {
		return nil, nil
	}
	return Slice(c.keys, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(c.length), kShape[3]}),
		Slice(c.values, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(c.length), vShape[3]})
}

// FixedState returns cloned full-capacity K/V handles for compiled decode.
func (c *FixedKVCache) FixedState() FixedKVState {
	state := FixedKVState{Length: c.length}
	if c.keys == nil || c.values == nil {
		return state
	}
	state.Keys = c.keys.Clone()
	state.Values = c.values.Clone()
	state.Owned = []*Array{state.Keys, state.Values}
	return state
}

// BorrowedFixedState returns cache-owned full-capacity K/V handles for hot
// native decode paths. Callers must not free the returned state.
func (c *FixedKVCache) BorrowedFixedState() FixedKVState {
	state := FixedKVState{Length: c.length}
	if c.keys == nil || c.values == nil {
		return state
	}
	state.Keys = c.keys
	state.Values = c.values
	return state
}

func (c *FixedKVCache) ReplaceFixedFromNative(k, v *Array, seqLen int) FixedKVState {
	Free(c.keys, c.values)
	c.keys = k
	c.values = v
	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	return c.FixedState()
}

func (c *FixedKVCache) ReplaceFixedFromNativeBorrowed(k, v *Array, seqLen int) FixedKVState {
	Free(c.keys, c.values)
	c.keys = k
	c.values = v
	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	return c.BorrowedFixedState()
}

func (c *FixedKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

func (c *FixedKVCache) ReadState() ([]*Array, []*Array) {
	k, v := c.validState()
	if k == nil || v == nil {
		Free(k, v)
		return nil, nil
	}
	state := []*Array{k, v}
	return state, state
}

func (c *FixedKVCache) Offset() int { return c.offset }
func (c *FixedKVCache) Len() int    { return c.length }

func (c *FixedKVCache) Reset() {
	Free(c.keys, c.values, c.slidingIndices, c.lastIndex)
	c.keys = nil
	c.values = nil
	c.slidingIndices = nil
	c.lastIndex = nil
	c.offset = 0
	c.length = 0
}

func (c *FixedKVCache) Detach() {
	if c.keys == nil {
		return
	}
	Detach(c.keys, c.values)
}

func (c *FixedKVCache) storageKV(k, v *Array) (*Array, *Array, []*Array) {
	if c == nil || !c.hasStorageDType {
		return k, v, nil
	}
	return cacheStorageKV(k, v, c.storageDType)
}

// QuantizedKVCache stores cache tensors in int8 lanes and dequantizes them
// only for the attention call. keyBits/valueBits control the logical quantizer
// range; q4 values currently use int8 storage until packed q4 kernels land.
type QuantizedKVCache struct {
	keys, values       *Array
	keyScale           *Array
	valueScale         *Array
	keyDtype           DType
	valueDtype         DType
	keyShape           []int32
	valueShape         []int32
	offset             int
	maxSize            int
	step               int
	keyBits, valueBits int
}

// NewQuantizedKVCache creates a cache using symmetric q8/q4 K/V storage.
func NewQuantizedKVCache(maxSize, keyBits, valueBits int) *QuantizedKVCache {
	if keyBits <= 0 {
		keyBits = 8
	}
	if valueBits <= 0 {
		valueBits = keyBits
	}
	return &QuantizedKVCache{maxSize: maxSize, step: 256, keyBits: keyBits, valueBits: valueBits}
}

func (c *QuantizedKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	shape := k.Shape()
	if len(shape) < 4 {
		fullK := k.Clone()
		fullV := v.Clone()
		c.storeQuantized(fullK, fullV)
		c.offset += seqLen
		return fullK, fullV
	}

	prevK, prevV := c.dequantizedState()
	var fullK, fullV *Array
	if prevK == nil {
		fullK = k.Clone()
		fullV = v.Clone()
	} else {
		fullK = Concatenate([]*Array{prevK, k}, 2)
		fullV = Concatenate([]*Array{prevV, v}, 2)
		Free(prevK, prevV)
	}
	c.offset += seqLen

	storeK, storeV := fullK, fullV
	if c.maxSize > 0 {
		storeK, storeV = cacheTail(fullK, fullV, c.maxSize)
	}
	c.storeQuantized(storeK, storeV)
	if storeK != fullK {
		Free(storeK, storeV)
	}
	return fullK, fullV
}

func (c *QuantizedKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values, c.keyScale, c.valueScale}
}

func (c *QuantizedKVCache) ReadState() ([]*Array, []*Array) {
	k, v := c.dequantizedState()
	if k == nil || v == nil {
		Free(k, v)
		return nil, nil
	}
	state := []*Array{k, v}
	return state, state
}

func (c *QuantizedKVCache) Offset() int { return c.offset }

func (c *QuantizedKVCache) Len() int {
	if c.keys == nil {
		return 0
	}
	if c.maxSize > 0 {
		return min(c.offset, c.maxSize)
	}
	shape := c.keys.Shape()
	if len(shape) >= 3 {
		return int(shape[2])
	}
	return c.offset
}

func (c *QuantizedKVCache) Reset() {
	Free(c.keys, c.values, c.keyScale, c.valueScale)
	c.keys = nil
	c.values = nil
	c.keyScale = nil
	c.valueScale = nil
	c.offset = 0
}

func (c *QuantizedKVCache) Detach() {
	// Quantized cache tensors are state for future decode steps. Some MLX
	// quantize/dequantize graphs are not captured directly by logits eval, so
	// detaching here can make the next decode step unevaluable.
}

func (c *QuantizedKVCache) storeQuantized(k, v *Array) {
	oldK, oldV, oldKS, oldVS := c.keys, c.values, c.keyScale, c.valueScale
	c.keyDtype = k.Dtype()
	c.valueDtype = v.Dtype()
	c.keys, c.keyScale, c.keyShape = quantizeCacheArray(k, c.keyBits)
	c.values, c.valueScale, c.valueShape = quantizeCacheArray(v, c.valueBits)
	Free(oldK, oldV, oldKS, oldVS)
}

func (c *QuantizedKVCache) dequantizedState() (*Array, *Array) {
	if c.keys == nil || c.values == nil {
		return nil, nil
	}
	return dequantizeCacheArray(c.keys, c.keyScale, c.keyDtype, c.keyShape, c.keyBits),
		dequantizeCacheArray(c.values, c.valueScale, c.valueDtype, c.valueShape, c.valueBits)
}

// PagedKVCache stores K/V tensors in block arrays to avoid repeatedly growing
// one large allocation. Attention receives a concatenated view for each step.
type PagedKVCache struct {
	kPages, vPages                     []*Array
	pageLens                           []int
	materializedKeys, materializedVals *Array
	pageShape                          pagedKVPageShape
	borrowedKeysScratch                []*Array
	borrowedValuesScratch              []*Array
	borrowedOwnedScratch               []*Array
	materializedLength                 int
	storageDType                       DType
	hasStorageDType                    bool
	offset                             int
	length                             int
	maxSize                            int
	pageSize                           int
}

type pagedKVPageShape struct {
	set    bool
	kBatch int32
	kHeads int32
	kDim   int32
	vBatch int32
	vHeads int32
	vDim   int32
}

// PagedKVState is a view of a paged K/V cache. Keys and Values may borrow
// cache-owned arrays; Owned lists transient visible slices that callers must
// release with Free.
type PagedKVState struct {
	Keys   []*Array
	Values []*Array
	Owned  []*Array
	Length int
}

// Free releases transient visible slices returned with the page state.
func (s PagedKVState) Free() {
	Free(s.Owned...)
}

func repeatPagedState(state PagedKVState, factor int32) (keys, values, owned []*Array) {
	if factor <= 1 {
		return state.Keys, state.Values, nil
	}
	keys = make([]*Array, len(state.Keys))
	values = make([]*Array, len(state.Values))
	owned = make([]*Array, 0, len(state.Keys)+len(state.Values))
	for i, page := range state.Keys {
		keys[i] = RepeatKV(page, factor)
		owned = append(owned, keys[i])
	}
	for i, page := range state.Values {
		values[i] = RepeatKV(page, factor)
		owned = append(owned, values[i])
	}
	return keys, values, owned
}

func pagedStateNeedsMaterializedRepeat(state PagedKVState, factor int32) bool {
	if factor <= 1 || len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return false
	}
	for i, key := range state.Keys {
		value := state.Values[i]
		if key == nil || value == nil || !key.Valid() || !value.Valid() || key.NumDims() < 4 || value.NumDims() < 4 {
			return true
		}
		if key.Dim(1) != 1 || value.Dim(1) != 1 {
			return true
		}
	}
	return false
}

// NewPagedKVCache creates a page/block-oriented cache.
func NewPagedKVCache(maxSize, pageSize int) *PagedKVCache {
	pageSize = resolvePagedKVPageSize(maxSize, pageSize)
	return &PagedKVCache{maxSize: maxSize, pageSize: pageSize}
}

func NewPagedKVCacheWithDType(maxSize, pageSize int, dtype DType) *PagedKVCache {
	cache := NewPagedKVCache(maxSize, pageSize)
	cache.storageDType = dtype
	cache.hasStorageDType = true
	return cache
}

func resolvePagedKVPageSize(maxSize, requested int) int {
	pageSize := requested
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
		if maxSize > hyperLongPagedKVSizeBoundary {
			pageSize = hyperLongPagedKVPageSize
		}
	}
	if parsed := core.ParseInt(core.Trim(RuntimeGateValue("GO_MLX_PAGED_KV_PAGE_SIZE")), 10, 64); parsed.OK {
		if value := int(parsed.Value.(int64)); value > 0 {
			pageSize = value
		}
	}
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	if maxSize > 0 && pageSize > maxSize {
		pageSize = maxSize
	}
	return pageSize
}

func (c *PagedKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	added := c.appendPages(k, v, seqLen)
	c.offset += added
	c.length += added

	fullK, fullV := c.concatenatedState()
	if c.maxSize > 0 && c.length > c.maxSize {
		c.trimToMaxSize()
	}
	return fullK, fullV
}

// UpdatePages adds new K/V tensors and returns cloned page handles without
// concatenating the full cache. Use this for decode-time paged attention.
func (c *PagedKVCache) UpdatePages(k, v *Array, seqLen int) PagedKVState {
	added := c.appendPages(k, v, seqLen)
	c.offset += added
	c.length += added
	c.trimToMaxSize()
	return c.PageState()
}

// UpdateBorrowedPages adds new K/V tensors and returns page handles that borrow
// full physical pages from the cache. Partial preallocated pages are still
// returned as owned visible slices. Use this only for immediate decode attention
// before the cache mutates again.
func (c *PagedKVCache) UpdateBorrowedPages(k, v *Array, seqLen int) PagedKVState {
	added := c.appendPages(k, v, seqLen)
	c.offset += added
	c.length += added
	c.trimToMaxSize()
	return c.BorrowedPageState()
}

func (c *PagedKVCache) UpdateBorrowedPagesMaterialized(k, v *Array, seqLen int) (PagedKVState, *Array, *Array) {
	added := c.appendPages(k, v, seqLen)
	c.offset += added
	c.length += added
	c.trimToMaxSize()
	state := c.BorrowedPageState()
	if added <= 0 || c.maxSize <= 0 {
		return state, nil, nil
	}
	if c.materializedLength == c.length-added && c.appendMaterialized(k, v, added) {
		keys, values := c.materializedVisibleState()
		return state, keys, values
	}
	c.resetMaterialized()
	if c.initMaterializedFromPages(state) {
		keys, values := c.materializedVisibleState()
		return state, keys, values
	}
	return state, nil, nil
}

func (c *PagedKVCache) ReplaceSinglePageFromNative(k, v *Array, seqLen int) PagedKVState {
	Free(c.kPages...)
	Free(c.vPages...)
	c.resetMaterialized()
	c.kPages = []*Array{k}
	c.vPages = []*Array{v}
	c.pageLens = []int{seqLen}
	c.recordPageShape(k.Shape(), v.Shape())
	c.offset += seqLen
	c.length += seqLen
	return c.PageState()
}

// PageState returns cloned page handles for callers that need an independently
// freeable view of the current page list.
func (c *PagedKVCache) PageState() PagedKVState {
	state := PagedKVState{Length: c.length}
	if len(c.kPages) == 0 || len(c.vPages) == 0 {
		return state
	}
	state.Keys = make([]*Array, len(c.kPages))
	state.Values = make([]*Array, len(c.vPages))
	state.Owned = make([]*Array, 0, len(c.kPages)+len(c.vPages))
	for i, page := range c.kPages {
		state.Keys[i] = c.visiblePage(page, i)
		state.Owned = append(state.Owned, state.Keys[i])
	}
	for i, page := range c.vPages {
		state.Values[i] = c.visiblePage(page, i)
		state.Owned = append(state.Owned, state.Values[i])
	}
	return state
}

// BorrowedPageState returns page handles for attention kernels that consume
// block tables or page lists directly. Full pages are borrowed from the cache to
// avoid per-token clone graph churn; only partial preallocated views are owned.
func (c *PagedKVCache) BorrowedPageState() PagedKVState {
	state := PagedKVState{Length: c.length}
	if len(c.kPages) == 0 || len(c.vPages) == 0 {
		return state
	}
	state.Keys = c.borrowedKeys(len(c.kPages))
	state.Values = c.borrowedValues(len(c.vPages))
	state.Owned = nil
	for i, page := range c.kPages {
		visible, owned := c.borrowVisiblePage(page, i)
		state.Keys[i] = visible
		if owned {
			if state.Owned == nil {
				state.Owned = c.borrowedOwned(0, len(c.kPages)+len(c.vPages))
			}
			state.Owned = append(state.Owned, visible)
		}
	}
	for i, page := range c.vPages {
		visible, owned := c.borrowVisiblePage(page, i)
		state.Values[i] = visible
		if owned {
			if state.Owned == nil {
				state.Owned = c.borrowedOwned(0, len(c.kPages)+len(c.vPages))
			}
			state.Owned = append(state.Owned, visible)
		}
	}
	return state
}

func (c *PagedKVCache) State() []*Array {
	if len(c.kPages) == 0 {
		return nil
	}
	out := make([]*Array, 0, len(c.kPages)+len(c.vPages))
	out = append(out, c.kPages...)
	out = append(out, c.vPages...)
	return out
}

func (c *PagedKVCache) ReadState() ([]*Array, []*Array) {
	k, v := c.concatenatedState()
	if k == nil || v == nil {
		Free(k, v)
		return nil, nil
	}
	state := []*Array{k, v}
	return state, state
}

func (c *PagedKVCache) Offset() int { return c.offset }
func (c *PagedKVCache) Len() int    { return c.length }

func (c *PagedKVCache) Reset() {
	Free(c.kPages...)
	Free(c.vPages...)
	c.resetMaterialized()
	c.kPages = nil
	c.vPages = nil
	c.pageLens = nil
	c.pageShape = pagedKVPageShape{}
	c.borrowedKeysScratch = nil
	c.borrowedValuesScratch = nil
	c.borrowedOwnedScratch = nil
	c.offset = 0
	c.length = 0
}

func (c *PagedKVCache) Detach() {
	// Paged attention reuses page views directly across decode steps. Some MLX
	// page views are not captured by the final logits eval; detaching them can
	// turn the next decode step into an unevaluable graph. Snapshot paths use
	// contiguous caches until native page-state snapshots land.
	if c.materializedKeys != nil || c.materializedVals != nil {
		Detach(c.materializedKeys, c.materializedVals)
	}
}

func (c *PagedKVCache) concatenatedState() (*Array, *Array) {
	kPages, vPages, owned := c.visiblePages()
	defer Free(owned...)
	return concatenatePagedState(kPages, vPages)
}

func (c *PagedKVCache) appendPages(k, v *Array, seqLen int) int {
	k, v, owned := c.storageKV(k, v)
	defer Free(owned...)
	if pagedKVPreallocEnabled() {
		return c.appendPagesPrealloc(k, v, seqLen)
	}
	return c.appendPagesConcat(k, v, seqLen)
}

func (c *PagedKVCache) storageKV(k, v *Array) (*Array, *Array, []*Array) {
	if c == nil || !c.hasStorageDType {
		return k, v, nil
	}
	return cacheStorageKV(k, v, c.storageDType)
}

func cacheStorageKV(k, v *Array, dtype DType) (*Array, *Array, []*Array) {
	if DTypeByteSize(dtype) <= 0 {
		return k, v, nil
	}
	owned := make([]*Array, 0, 2)
	if k != nil && k.Valid() && k.Dtype() != dtype {
		k = AsType(k, dtype)
		owned = append(owned, k)
	}
	if v != nil && v.Valid() && v.Dtype() != dtype {
		v = AsType(v, dtype)
		owned = append(owned, v)
	}
	return k, v, owned
}

func (c *PagedKVCache) appendPagesConcat(k, v *Array, seqLen int) int {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return 0
	}
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		c.kPages = append(c.kPages, k.Clone())
		c.vPages = append(c.vPages, v.Clone())
		c.pageLens = append(c.pageLens, seqLen)
		return seqLen
	}
	totalLen := int(kShape[2])
	if seqLen <= 0 || seqLen > totalLen {
		seqLen = totalLen
	}
	for start := 0; start < seqLen; {
		remaining := seqLen - start
		if c.canAppendToLastPage(kShape, vShape) {
			last := len(c.kPages) - 1
			room := c.pageSize - c.pageLen(last)
			if room > 0 {
				take := min(room, remaining)
				c.appendToLastPage(k, v, kShape, vShape, start, take)
				start += take
				continue
			}
		}
		take := min(c.pageSize, remaining)
		pageK, ownedK := cachePageView(k, kShape, start, take, totalLen)
		pageV, ownedV := cachePageView(v, vShape, start, take, int(vShape[2]))
		if !ownedK {
			pageK = pageK.Clone()
		}
		if !ownedV {
			pageV = pageV.Clone()
		}
		c.kPages = append(c.kPages, pageK)
		c.vPages = append(c.vPages, pageV)
		c.pageLens = append(c.pageLens, take)
		c.recordPageShape(kShape, vShape)
		start += take
	}
	return seqLen
}

func (c *PagedKVCache) appendPagesPrealloc(k, v *Array, seqLen int) int {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return 0
	}
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		return c.appendPagesConcat(k, v, seqLen)
	}
	totalLen := int(kShape[2])
	if seqLen <= 0 || seqLen > totalLen {
		seqLen = totalLen
	}
	for start := 0; start < seqLen; {
		remaining := seqLen - start
		if c.canAppendToLastPage(kShape, vShape) {
			last := len(c.kPages) - 1
			room := c.pageSize - c.pageLen(last)
			if room > 0 {
				take := min(room, remaining)
				c.appendToLastPagePrealloc(k, v, kShape, vShape, start, take)
				start += take
				continue
			}
		}
		take := min(c.pageSize, remaining)
		c.appendNewPagePrealloc(k, v, kShape, vShape, start, take)
		start += take
	}
	return seqLen
}

func (c *PagedKVCache) canAppendToLastPage(kShape, vShape []int32) bool {
	if len(c.kPages) == 0 || len(c.vPages) == 0 {
		return false
	}
	lastK := c.kPages[len(c.kPages)-1]
	lastV := c.vPages[len(c.vPages)-1]
	if c.pageLen(len(c.kPages)-1) >= c.pageSize {
		return false
	}
	if c.pageShape.set {
		return c.pageShape.matches(kShape, vShape)
	}
	lastKShape := lastK.Shape()
	lastVShape := lastV.Shape()
	ok := len(lastKShape) >= 4 &&
		len(lastVShape) >= 4 &&
		lastKShape[0] == kShape[0] &&
		lastKShape[1] == kShape[1] &&
		lastKShape[3] == kShape[3] &&
		lastVShape[0] == vShape[0] &&
		lastVShape[1] == vShape[1] &&
		lastVShape[3] == vShape[3]
	if ok {
		c.recordPageShape(kShape, vShape)
	}
	return ok
}

func (c *PagedKVCache) appendToLastPage(k, v *Array, kShape, vShape []int32, start, take int) {
	pieceK, ownedK := cachePageView(k, kShape, start, take, int(kShape[2]))
	pieceV, ownedV := cachePageView(v, vShape, start, take, int(vShape[2]))
	last := len(c.kPages) - 1
	oldK, oldV := c.kPages[last], c.vPages[last]
	c.kPages[last] = Concatenate([]*Array{oldK, pieceK}, 2)
	c.vPages[last] = Concatenate([]*Array{oldV, pieceV}, 2)
	c.pageLens[last] += take
	c.recordPageShape(kShape, vShape)
	Free(oldK, oldV)
	if ownedK {
		Free(pieceK)
	}
	if ownedV {
		Free(pieceV)
	}
}

func (c *PagedKVCache) appendToLastPagePrealloc(k, v *Array, kShape, vShape []int32, start, take int) {
	pieceK, ownedK := cachePageView(k, kShape, start, take, int(kShape[2]))
	pieceV, ownedV := cachePageView(v, vShape, start, take, int(vShape[2]))
	last := len(c.kPages) - 1
	writeStart := c.pageLen(last)
	oldK, oldV := c.kPages[last], c.vPages[last]
	c.kPages[last] = SliceUpdateInplace(oldK, pieceK, []int32{0, 0, int32(writeStart), 0}, []int32{kShape[0], kShape[1], int32(writeStart + take), kShape[3]})
	c.vPages[last] = SliceUpdateInplace(oldV, pieceV, []int32{0, 0, int32(writeStart), 0}, []int32{vShape[0], vShape[1], int32(writeStart + take), vShape[3]})
	c.pageLens[last] = writeStart + take
	c.recordPageShape(kShape, vShape)
	Free(oldK, oldV)
	if ownedK {
		Free(pieceK)
	}
	if ownedV {
		Free(pieceV)
	}
}

func (c *PagedKVCache) appendNewPagePrealloc(k, v *Array, kShape, vShape []int32, start, take int) {
	pieceK, ownedK := cachePageView(k, kShape, start, take, int(kShape[2]))
	pieceV, ownedV := cachePageView(v, vShape, start, take, int(vShape[2]))
	pageK := Zeros([]int32{kShape[0], kShape[1], int32(c.pageSize), kShape[3]}, k.Dtype())
	pageV := Zeros([]int32{vShape[0], vShape[1], int32(c.pageSize), vShape[3]}, v.Dtype())
	updatedK := SliceUpdateInplace(pageK, pieceK, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(take), kShape[3]})
	updatedV := SliceUpdateInplace(pageV, pieceV, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(take), vShape[3]})
	c.kPages = append(c.kPages, updatedK)
	c.vPages = append(c.vPages, updatedV)
	c.pageLens = append(c.pageLens, take)
	c.recordPageShape(kShape, vShape)
	Free(pageK, pageV)
	if ownedK {
		Free(pieceK)
	}
	if ownedV {
		Free(pieceV)
	}
}

func cachePageView(a *Array, shape []int32, start, take, totalLen int) (*Array, bool) {
	if start == 0 && take == totalLen {
		return a, false
	}
	return Slice(a, []int32{0, 0, int32(start), 0}, []int32{shape[0], shape[1], int32(start + take), shape[3]}), true
}

func (c *PagedKVCache) trimToMaxSize() {
	if c.maxSize <= 0 || c.length <= c.maxSize {
		return
	}
	c.resetMaterialized()
	excess := c.length - c.maxSize
	for excess > 0 && len(c.kPages) > 0 && len(c.vPages) > 0 {
		pageLen := c.pageLen(0)
		if pageLen <= 0 {
			Free(c.kPages[0], c.vPages[0])
			c.kPages = c.kPages[1:]
			c.vPages = c.vPages[1:]
			c.pageLens = c.pageLens[1:]
			continue
		}
		if pageLen <= excess {
			Free(c.kPages[0], c.vPages[0])
			c.kPages = c.kPages[1:]
			c.vPages = c.vPages[1:]
			c.pageLens = c.pageLens[1:]
			c.length -= pageLen
			excess -= pageLen
			continue
		}
		c.trimFirstPage(excess)
		c.length -= excess
		excess = 0
	}
	if c.length > c.maxSize {
		c.length = c.maxSize
	}
}

func (c *PagedKVCache) trimFirstPage(tokens int) {
	if tokens <= 0 || len(c.kPages) == 0 || len(c.vPages) == 0 {
		return
	}
	kShape := c.kPages[0].Shape()
	vShape := c.vPages[0].Shape()
	pageLen := c.pageLen(0)
	if len(kShape) < 4 || len(vShape) < 4 || tokens >= pageLen {
		return
	}
	oldK, oldV := c.kPages[0], c.vPages[0]
	newLen := pageLen - tokens
	tailK := Slice(oldK, []int32{0, 0, int32(tokens), 0}, []int32{kShape[0], kShape[1], int32(pageLen), kShape[3]})
	tailV := Slice(oldV, []int32{0, 0, int32(tokens), 0}, []int32{vShape[0], vShape[1], int32(pageLen), vShape[3]})
	if pagedKVPreallocEnabled() {
		pageK := Zeros([]int32{kShape[0], kShape[1], int32(c.pageSize), kShape[3]}, oldK.Dtype())
		pageV := Zeros([]int32{vShape[0], vShape[1], int32(c.pageSize), vShape[3]}, oldV.Dtype())
		c.kPages[0] = SliceUpdateInplace(pageK, tailK, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(newLen), kShape[3]})
		c.vPages[0] = SliceUpdateInplace(pageV, tailV, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(newLen), vShape[3]})
		Free(pageK, pageV)
	} else {
		c.kPages[0] = tailK
		c.vPages[0] = tailV
		tailK, tailV = nil, nil
	}
	c.pageLens[0] = newLen
	Free(oldK, oldV, tailK, tailV)
}

func (c *PagedKVCache) recordPageShape(kShape, vShape []int32) {
	if len(kShape) < 4 || len(vShape) < 4 {
		return
	}
	c.pageShape = pagedKVPageShape{
		set:    true,
		kBatch: kShape[0],
		kHeads: kShape[1],
		kDim:   kShape[3],
		vBatch: vShape[0],
		vHeads: vShape[1],
		vDim:   vShape[3],
	}
}

func (s pagedKVPageShape) matches(kShape, vShape []int32) bool {
	return len(kShape) >= 4 &&
		len(vShape) >= 4 &&
		s.kBatch == kShape[0] &&
		s.kHeads == kShape[1] &&
		s.kDim == kShape[3] &&
		s.vBatch == vShape[0] &&
		s.vHeads == vShape[1] &&
		s.vDim == vShape[3]
}

func (c *PagedKVCache) pageLen(i int) int {
	if i >= 0 && i < len(c.pageLens) && c.pageLens[i] > 0 {
		return c.pageLens[i]
	}
	if i >= 0 && i < len(c.kPages) {
		return pagedArrayLen(c.kPages[i])
	}
	return 0
}

func pagedPageLensForPages(pages []*Array, totalLen int) []int {
	if len(pages) == 0 {
		return nil
	}
	lens := make([]int, len(pages))
	remaining := totalLen
	for i, page := range pages {
		length := pagedArrayLen(page)
		if remaining > 0 && length > remaining {
			length = remaining
		}
		if length < 0 {
			length = 0
		}
		lens[i] = length
		remaining -= length
	}
	return lens
}

func (c *PagedKVCache) visiblePage(page *Array, i int) *Array {
	if page == nil || !page.Valid() {
		return nil
	}
	shape := page.Shape()
	length := c.pageLen(i)
	if len(shape) < 4 || length <= 0 || length >= int(shape[2]) {
		return page.Clone()
	}
	return Slice(page, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(length), shape[3]})
}

func (c *PagedKVCache) borrowVisiblePage(page *Array, i int) (*Array, bool) {
	if page == nil || !page.Valid() {
		return nil, false
	}
	length := c.pageLen(i)
	if c.pageSize > 0 && length >= c.pageSize {
		return page, false
	}
	shape := page.Shape()
	if len(shape) < 4 || length <= 0 || length >= int(shape[2]) {
		return page, false
	}
	return Slice(page, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(length), shape[3]}), true
}

func (c *PagedKVCache) borrowedKeys(n int) []*Array {
	if cap(c.borrowedKeysScratch) < n {
		c.borrowedKeysScratch = make([]*Array, n)
	}
	keys := c.borrowedKeysScratch[:n]
	clear(keys)
	return keys
}

func (c *PagedKVCache) borrowedValues(n int) []*Array {
	if cap(c.borrowedValuesScratch) < n {
		c.borrowedValuesScratch = make([]*Array, n)
	}
	values := c.borrowedValuesScratch[:n]
	clear(values)
	return values
}

func (c *PagedKVCache) borrowedOwned(length, capacity int) []*Array {
	if cap(c.borrowedOwnedScratch) < capacity {
		c.borrowedOwnedScratch = make([]*Array, length, capacity)
	}
	owned := c.borrowedOwnedScratch[:length]
	clear(c.borrowedOwnedScratch[:cap(c.borrowedOwnedScratch)])
	return owned
}

func (c *PagedKVCache) visiblePages() (kPages, vPages, owned []*Array) {
	if len(c.kPages) == 0 || len(c.vPages) == 0 || len(c.kPages) != len(c.vPages) {
		return nil, nil, nil
	}
	kPages = make([]*Array, len(c.kPages))
	vPages = make([]*Array, len(c.vPages))
	owned = make([]*Array, 0, len(c.kPages)+len(c.vPages))
	for i := range c.kPages {
		kPages[i] = c.visiblePage(c.kPages[i], i)
		vPages[i] = c.visiblePage(c.vPages[i], i)
		owned = append(owned, kPages[i], vPages[i])
	}
	return kPages, vPages, owned
}

func pagedArrayLen(page *Array) int {
	if page == nil || !page.Valid() {
		return 0
	}
	shape := page.Shape()
	if len(shape) < 3 {
		return 0
	}
	return int(shape[2])
}

func concatenatePagedState(kPages, vPages []*Array) (*Array, *Array) {
	if len(kPages) == 0 || len(vPages) == 0 || len(kPages) != len(vPages) {
		return nil, nil
	}
	if len(kPages) == 1 {
		return kPages[0].Clone(), vPages[0].Clone()
	}
	return Concatenate(kPages, 2), Concatenate(vPages, 2)
}

func (c *PagedKVCache) resetMaterialized() {
	Free(c.materializedKeys, c.materializedVals)
	c.materializedKeys = nil
	c.materializedVals = nil
	c.materializedLength = 0
}

func (c *PagedKVCache) appendMaterialized(k, v *Array, seqLen int) bool {
	if c.materializedKeys == nil || c.materializedVals == nil || seqLen <= 0 || c.maxSize <= 0 {
		return false
	}
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || c.materializedLength+seqLen > c.maxSize {
		return false
	}
	if !c.materializedShapesMatch(kShape, vShape) {
		return false
	}
	writeK, writeV := k, v
	totalLen := int(kShape[2])
	if totalLen <= 0 {
		return false
	}
	if seqLen > totalLen {
		seqLen = totalLen
	}
	if totalLen != seqLen {
		start := totalLen - seqLen
		writeK = Slice(k, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(totalLen), kShape[3]})
		writeV = Slice(v, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(totalLen), vShape[3]})
		defer Free(writeK, writeV)
	}
	start := c.materializedLength
	oldK, oldV := c.materializedKeys, c.materializedVals
	c.materializedKeys = SliceUpdateInplace(c.materializedKeys, writeK, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(start + seqLen), kShape[3]})
	c.materializedVals = SliceUpdateInplace(c.materializedVals, writeV, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(start + seqLen), vShape[3]})
	Free(oldK, oldV)
	c.materializedLength += seqLen
	return c.materializedLength == c.length
}

func (c *PagedKVCache) initMaterializedFromPages(state PagedKVState) bool {
	if c.maxSize <= 0 || state.Length <= 0 || len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return false
	}
	fullK, fullV := concatenatePagedState(state.Keys, state.Values)
	if fullK == nil || fullV == nil || !fullK.Valid() || !fullV.Valid() {
		Free(fullK, fullV)
		return false
	}
	kShape := fullK.Shape()
	vShape := fullV.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || state.Length > c.maxSize {
		Free(fullK, fullV)
		return false
	}
	c.materializedKeys = Zeros([]int32{kShape[0], kShape[1], int32(c.maxSize), kShape[3]}, fullK.Dtype())
	c.materializedVals = Zeros([]int32{vShape[0], vShape[1], int32(c.maxSize), vShape[3]}, fullV.Dtype())
	oldK, oldV := c.materializedKeys, c.materializedVals
	c.materializedKeys = SliceUpdateInplace(c.materializedKeys, fullK, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(state.Length), kShape[3]})
	c.materializedVals = SliceUpdateInplace(c.materializedVals, fullV, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(state.Length), vShape[3]})
	Free(oldK, oldV, fullK, fullV)
	c.materializedLength = state.Length
	return true
}

func (c *PagedKVCache) materializedVisibleState() (*Array, *Array) {
	if c.materializedKeys == nil || c.materializedVals == nil || c.materializedLength <= 0 {
		return nil, nil
	}
	kShape := c.materializedKeys.Shape()
	vShape := c.materializedVals.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		return nil, nil
	}
	return Slice(c.materializedKeys, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(c.materializedLength), kShape[3]}),
		Slice(c.materializedVals, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(c.materializedLength), vShape[3]})
}

func (c *PagedKVCache) materializedShapesMatch(kShape, vShape []int32) bool {
	if c.materializedKeys == nil || c.materializedVals == nil {
		return false
	}
	mkShape := c.materializedKeys.Shape()
	mvShape := c.materializedVals.Shape()
	return len(mkShape) >= 4 && len(mvShape) >= 4 &&
		mkShape[0] == kShape[0] &&
		mkShape[1] == kShape[1] &&
		mkShape[2] == int32(c.maxSize) &&
		mkShape[3] == kShape[3] &&
		mvShape[0] == vShape[0] &&
		mvShape[1] == vShape[1] &&
		mvShape[2] == int32(c.maxSize) &&
		mvShape[3] == vShape[3]
}

func cacheTail(k, v *Array, maxSize int) (*Array, *Array) {
	if maxSize <= 0 || k == nil || v == nil {
		return k, v
	}
	kShape := k.Shape()
	vShape := v.Shape()
	if len(kShape) < 4 || len(vShape) < 4 || int(kShape[2]) <= maxSize {
		return k, v
	}
	start := int(kShape[2]) - maxSize
	return Slice(k, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], kShape[2], kShape[3]}),
		Slice(v, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], vShape[2], vShape[3]})
}

func quantizeCacheArray(a *Array, bits int) (*Array, *Array, []int32) {
	shape := append([]int32(nil), a.Shape()...)
	levels := 1
	for range max(0, bits-1) {
		levels *= 2
	}
	maxValue := float32(levels - 1)
	if maxValue <= 0 {
		maxValue = 127
	}
	abs := Abs(a)
	maxAbs := maxAll(abs)
	eps := FromValue(float32(1e-6))
	clampedAbs := Maximum(maxAbs, eps)
	denom := FromValue(maxValue)
	scale := Divide(clampedAbs, denom)
	normalized := Divide(a, scale)
	rounded := Round(normalized)
	minValue := FromValue(-maxValue)
	maxBound := FromValue(maxValue)
	clipped := Clip(rounded, minValue, maxBound)
	q := AsType(clipped, DTypeInt8)
	Free(abs, maxAbs, eps, clampedAbs, denom, normalized, rounded, minValue, maxBound, clipped)
	if bits == 4 {
		packed := packQ4(q)
		Free(q)
		return packed, scale, shape
	}
	return q, scale, shape
}

func dequantizeCacheArray(q, scale *Array, dtype DType, shape []int32, bits int) *Array {
	source := q
	var unpacked *Array
	if bits == 4 {
		unpacked = unpackQ4(q, shape)
		source = unpacked
	}
	f := AsType(source, DTypeFloat32)
	deq := Mul(f, scale)
	Free(f, unpacked)
	if dtype == DTypeFloat32 || dtype == 0 {
		return deq
	}
	out := AsType(deq, dtype)
	Free(deq)
	return out
}

// packQ4 packs an int8 array's low-4-bit nibbles into a uint8 array half the
// length. The implementation reshapes the flat input to [pairs, 2] so the even
// and odd halves can be sliced as views — no Gather index arrays, no host-side
// int32 index allocations.
func packQ4(q *Array) *Array {
	shape := q.Shape()
	n := cacheElementCount(shape)
	flat := Reshape(q, int32(n))
	offset := AsType(FromValue(8), DTypeInt8)
	shifted := Add(flat, offset)
	shiftedU := AsType(shifted, DTypeUint8)
	Free(flat, offset, shifted)

	padded := shiftedU
	nP := n
	if n%2 != 0 {
		zero := Zeros([]int32{1}, DTypeUint8)
		padded = Concatenate([]*Array{shiftedU, zero}, 0)
		Free(shiftedU, zero)
		nP = n + 1
	}

	pairs := nP / 2
	paired := Reshape(padded, int32(pairs), int32(2))
	Free(padded)
	low := SliceAxis(paired, 1, 0, 1)
	high := SliceAxis(paired, 1, 1, 2)
	Free(paired)
	shift := AsType(FromValue(4), DTypeUint8)
	highShifted := LeftShift(high, shift)
	packed2D := BitwiseOr(low, highShifted)
	packed := Reshape(packed2D, int32(pairs))
	Free(low, high, shift, highShifted, packed2D)
	return packed
}

// unpackQ4 expands a uint8 array of packed Q4 nibbles back into a signed int8
// array of the original shape. The implementation reshapes pair-wise after
// extracting the low/high nibbles, replacing the previous PutAlongAxis +
// gather indices with structural ops only.
func unpackQ4(packed *Array, shape []int32) *Array {
	n := cacheElementCount(shape)
	if n == 0 {
		return Reshape(packed, shape...)
	}
	mask := AsType(FromValue(15), DTypeUint8)
	low := BitwiseAnd(packed, mask)
	shift := AsType(FromValue(4), DTypeUint8)
	high := RightShift(packed, shift)
	Free(mask, shift)

	pairs := int(low.Shape()[0])
	lowE := ExpandDims(low, 1)
	highE := ExpandDims(high, 1)
	Free(low, high)
	stacked := Concatenate([]*Array{lowE, highE}, 1)
	Free(lowE, highE)

	flatLen := pairs * 2
	flat := Reshape(stacked, int32(flatLen))
	Free(stacked)

	outU := flat
	if flatLen > n {
		outU = Slice(flat, []int32{0}, []int32{int32(n)})
		Free(flat)
	}

	outInt := AsType(outU, DTypeInt8)
	offset := AsType(FromValue(8), DTypeInt8)
	signed := Subtract(outInt, offset)
	reshaped := Reshape(signed, shape...)
	Free(outU, outInt, offset, signed)
	return reshaped
}

func cacheElementCount(shape []int32) int {
	if len(shape) == 0 {
		return 1
	}
	total := 1
	for _, dim := range shape {
		total *= int(dim)
	}
	return total
}

func maxAll(a *Array) *Array {
	current := a
	owned := false
	for len(current.Shape()) > 0 {
		next := MaxAxis(current, 0, false)
		if owned {
			Free(current)
		}
		current = next
		owned = true
	}
	if !owned {
		return current.Clone()
	}
	return current
}
