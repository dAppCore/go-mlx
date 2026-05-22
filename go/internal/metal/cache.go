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

// stateAppender is an optional interface implemented by caches that can append
// their state arrays into a caller-provided slice — bypasses the per-call
// `[]*Array{...}` literal allocation that `State()` produces. Used by hot
// prefill paths (prompt_cache.prefillCacheStateArrays) where Gemma 4's 26-cache
// fan-out previously paid 27 allocs per dispatch (one per State() call plus the
// outer slice). Caches that don't implement this gracefully fall back to State().
type stateAppender interface {
	AppendState(dst []*Array) []*Array
}

// appendCacheState appends a cache's live state arrays into dst. Prefers
// AppendState (alloc-free) when implemented; falls back to State() copy.
func appendCacheState(dst []*Array, c Cache) []*Array {
	if c == nil {
		return dst
	}
	if a, ok := c.(stateAppender); ok {
		return a.AppendState(dst)
	}
	for _, state := range c.State() {
		if state != nil && state.Valid() {
			dst = append(dst, state)
		}
	}
	return dst
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
				oldK = Slice4(oldK, 0, 0, 0, 0, B, H, int32(prev), Dk)
				oldV = Slice4(oldV, 0, 0, 0, 0, B, H, int32(prev), Dv)
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
	c.keys = SliceUpdateInplace4(c.keys, k, 0, 0, int32(prev), 0, B, H, int32(c.offset), Dk)
	c.values = SliceUpdateInplace4(c.values, v, 0, 0, int32(prev), 0, B, H, int32(c.offset), Dv)
	Free(oldK, oldV)

	return Slice4(c.keys, 0, 0, 0, 0, B, H, int32(c.offset), Dk),
		Slice4(c.values, 0, 0, 0, 0, B, H, int32(c.offset), Dv)
}

func (c *KVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *KVCache) AppendState(dst []*Array) []*Array {
	if c.keys == nil {
		return dst
	}
	if c.keys != nil && c.keys.Valid() {
		dst = append(dst, c.keys)
	}
	if c.values != nil && c.values.Valid() {
		dst = append(dst, c.values)
	}
	return dst
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
//
// Storage is held in temporal order in a single buffer of shape
// `[B, H, idx, D]` where `idx` is the count of valid tokens (capped at
// maxSize). Below cap the buffer grows in `c.step` (=256) slots at a time
// via [Concatenate]; each single-token Update writes the new token at slot
// `idx` via [SliceUpdateInplace] and bumps `idx`. Past cap the buffer stays
// pinned at maxSize: each append drops the oldest slot via a metadata-only
// [Slice] and concatenates the freshly written token at the tail.
//
// The legacy ring layout (write at `idx mod maxSize` and rebuild a
// temporally-ordered view via Slice+Slice+Concat on every return) triggered
// IDEAS.md §1 dynamic KV concatenation. The pre-existing in-place
// [SliceUpdateInplace] write IS being hit on the past-cap path; the cost
// surfaced by W7-E's bench data comes from `rotatingCacheWindow` allocating
// a fresh O(maxSize) ordered buffer per Update on top of the in-place write.
// Holding the buffer in temporal order folds the return path into a direct
// reference (`return c.keys, c.values`) and replaces the two write-side
// graph nodes per token (SliceUpdate + ordered-view Concat) with one
// (Concat that performs the drop+append in a single graph op), halving the
// per-token Metal data movement past cap without inflating the per-Update
// buffer size that the long-chain bench is sensitive to.
type RotatingKVCache struct {
	// keys, values hold the temporally-ordered window. Below cap the L
	// dimension equals the legacy growth state (idx slots, pre-allocated up
	// to c.step ahead); at/past cap it equals exactly maxSize.
	keys, values *Array
	offset       int
	maxSize      int
	step         int
	// idx is the temporal length of valid content in keys/values
	// (0..maxSize). Once idx reaches maxSize it stays there, and each
	// single-token Update past cap performs a drop+append via Slice+Concat.
	idx int
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

	// Past-cap fast path: temporally drop-and-append.
	//
	// The previous ring layout did SliceUpdateInplace at idx (write step) then
	// Slice+Slice+Concat in [rotatingCacheWindow] (ordered-view step) — two
	// graph nodes whose outputs are both shape [B,H,maxSize,D] and both
	// trigger a fresh O(maxSize) Metal buffer at Eval. The drop+append below
	// achieves the same temporally-ordered window via a single Concat — one
	// fresh buffer per K/V per token instead of two.
	if c.keys != nil && c.idx >= c.maxSize {
		oldK, oldV := c.keys, c.values
		prefixK := Slice4(oldK, 0, 0, 1, 0, B, H, int32(c.maxSize), Dk)
		prefixV := Slice4(oldV, 0, 0, 1, 0, B, H, int32(c.maxSize), Dv)
		c.keys = Concatenate([]*Array{prefixK, k}, 2)
		c.values = Concatenate([]*Array{prefixV, v}, 2)
		Free(oldK, oldV, prefixK, prefixV)
		c.offset++
		// idx stays at maxSize — buffer is now full and temporally ordered.
		// Return Slice views so caller Free() does not invalidate c.keys.
		return Slice4(c.keys, 0, 0, 0, 0, B, H, int32(c.maxSize), Dk),
			Slice4(c.values, 0, 0, 0, 0, B, H, int32(c.maxSize), Dv)
	}

	// Below cap: grow + write at temporal tail (same as legacy growth path).
	if c.keys == nil || (c.idx >= int(c.keys.Shape()[2]) && int(c.keys.Shape()[2]) < c.maxSize) {
		cur := 0
		if c.keys != nil {
			cur = int(c.keys.Shape()[2])
		}
		newSize := min(c.step, c.maxSize-cur)
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

	// Write at the temporal tail. Below cap this is a single in-place
	// SliceUpdate (the IDEAS.md "good shape" pre-allocated buffer with
	// offset indexing).
	oldK, oldV := c.keys, c.values
	c.keys = SliceUpdateInplace4(c.keys, k, 0, 0, int32(c.idx), 0, B, H, int32(c.idx+1), Dk)
	c.values = SliceUpdateInplace4(c.values, v, 0, 0, int32(c.idx), 0, B, H, int32(c.idx+1), Dv)
	Free(oldK, oldV)

	c.offset++
	c.idx++

	// Below cap the storage may extend past idx (pre-allocated headroom);
	// return a view bounded to the valid window.
	window := min(c.offset, c.maxSize)
	return Slice4(c.keys, 0, 0, 0, 0, B, H, int32(window), Dk),
		Slice4(c.values, 0, 0, 0, 0, B, H, int32(window), Dv)
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

	// Compose the current temporally-ordered prefix (slots [0, idx)) with the
	// incoming multi-token segment.
	var prevK, prevV *Array
	if c.keys != nil && c.keys.Valid() && c.idx > 0 {
		prevK = Slice4(c.keys, 0, 0, 0, 0, B, H, int32(c.idx), Dk)
		prevV = Slice4(c.values, 0, 0, 0, 0, B, H, int32(c.idx), Dv)
	}

	var fullK, fullV *Array
	if prevK == nil {
		fullK, fullV = k.Clone(), v.Clone()
	} else {
		fullK = Concatenate([]*Array{prevK, k}, 2)
		fullV = Concatenate([]*Array{prevV, v}, 2)
		Free(prevK, prevV)
	}
	if c.keys != nil {
		Free(c.keys, c.values)
		c.keys, c.values = nil, nil
	}
	c.offset += seqLen

	full := int(fullK.Shape()[2])
	if trim := full - c.maxSize; trim > 0 {
		// Preserve the full multi-token prompt for the current attention pass,
		// while storing only the bounded sliding window for future decode steps.
		c.keys = Slice4(fullK, 0, 0, int32(trim), 0, B, H, int32(full), Dk)
		c.values = Slice4(fullV, 0, 0, int32(trim), 0, B, H, int32(full), Dv)
		c.idx = int(c.keys.Shape()[2])
		return Slice4(fullK, 0, 0, 0, 0, B, H, int32(full), Dk),
			Slice4(fullV, 0, 0, 0, 0, B, H, int32(full), Dv)
	}

	c.keys, c.values = fullK, fullV
	c.idx = full
	// Return Slice views so callers can Free them without destroying the cache.
	return Slice4(c.keys, 0, 0, 0, 0, B, H, int32(c.idx), Dk),
		Slice4(c.values, 0, 0, 0, 0, B, H, int32(c.idx), Dv)
}

func (c *RotatingKVCache) orderedState() []*Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	shape := c.keys.Shape()
	if len(shape) < 4 {
		return []*Array{c.keys.Clone(), c.values.Clone()}
	}
	// Storage is always temporally ordered (the past-cap drop+append keeps
	// it that way), so the ordered view is just a leading Slice — no
	// Slice+Slice+Concat reorder.
	window := c.Len()
	if window <= 0 || window > int(shape[2]) {
		window = int(shape[2])
	}
	if window <= 0 {
		starts := []int32{0, 0, 0, 0}
		ends := []int32{shape[0], shape[1], 0, shape[3]}
		return []*Array{Slice(c.keys, starts, ends), Slice(c.values, starts, ends)}
	}
	dv := c.values.Shape()[3]
	return []*Array{
		Slice4(c.keys, 0, 0, 0, 0, shape[0], shape[1], int32(window), shape[3]),
		Slice4(c.values, 0, 0, 0, 0, shape[0], shape[1], int32(window), dv),
	}
}

func (c *RotatingKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	// Buffer storage is always temporally ordered and shape[2] is either the
	// growth-step length (below cap) or exactly maxSize (at/past cap), so the
	// raw arrays are the canonical reference. Returning them directly keeps
	// the legacy contract that Reset/Free invalidates State() callers' handles.
	return []*Array{c.keys, c.values}
}

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *RotatingKVCache) AppendState(dst []*Array) []*Array {
	if c.keys == nil {
		return dst
	}
	if c.keys != nil && c.keys.Valid() {
		dst = append(dst, c.keys)
	}
	if c.values != nil && c.values.Valid() {
		dst = append(dst, c.values)
	}
	return dst
}

func (c *RotatingKVCache) Offset() int { return c.offset }
func (c *RotatingKVCache) Len() int {
	length := min(c.offset, c.maxSize)
	if c.keys == nil || !c.keys.Valid() {
		return length
	}
	// c.idx is the temporal count of valid tokens (bounded by maxSize). If
	// the storage was restored from a smaller snapshot, fall back to its L
	// dimension.
	if c.idx < length {
		length = c.idx
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
//
// Once ensureShape has materialised c.keys / c.values, the per-axis dims
// (batch, heads, keyDim, valueDim) are stable for the rest of the cache's
// lifetime — Reset() is the only path that invalidates them. The cached
// shape lets the steady-state single-token Update path avoid calling
// Array.Shape(), which allocates a fresh []int32 on every call.
//
// FixedKVCache resolves the MLX dispatch stream once per Update via the
// local fixedKVCacheUpdateStream variable, then threads it through the
// 4–6 MLX ops the Update produces.  This collapses the DefaultStream() →
// currentDefaultDevice() defer-record allocation from per-op down to
// per-Update.  The cache does NOT persist the stream across Updates,
// because callers may install a temporary default stream via
// withGenerationStream between calls.
type FixedKVCache struct {
	keys, values              *Array
	slidingIndices, lastIndex *Array
	storageDType              DType
	hasStorageDType           bool
	offset                    int
	length                    int
	maxSize                   int

	// shapeCached is true once batch/heads/keyDim/valueDim hold the
	// dims of the currently-materialised c.keys / c.values buffers.
	shapeCached bool
	batch       int32
	heads       int32
	keyDim      int32
	valueDim    int32
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
	// Resolve the dispatch stream once up-front and thread it through
	// every MLX op in this Update — AsType conversions on the FP16
	// path, the two slice-update writes, and the two slice reads in
	// validState.  Cuts ~5 DefaultStream() → currentDefaultDevice()
	// defer-record allocations per token on the FP16 single-token
	// decode loop.
	stream := DefaultStream()
	k, v, ownK, ownV := c.storageKVPair(k, v, stream)
	defer freeOwnedPair(ownK, ownV)
	// Use Dim accessors (single cgo call, no slice alloc) instead of
	// Shape() — the steady-state single-token decode loop hits this path
	// hundreds of times per generation, and every fresh []int32 escapes
	// to the heap.
	if k.NumDims() < 4 || v.NumDims() < 4 || c.maxSize <= 0 {
		if c.keys == nil {
			c.keys, c.values = k.Clone(), v.Clone()
		}
		c.offset += seqLen
		c.length = min(c.offset, c.maxSize)
		return c.keys.Clone(), c.values.Clone()
	}
	kBatch := int32(k.Dim(0))
	kHeads := int32(k.Dim(1))
	totalLen := k.Dim(2)
	kKeyDim := int32(k.Dim(3))
	vValueDim := int32(v.Dim(3))
	if seqLen <= 0 || seqLen > totalLen {
		seqLen = totalLen
	}
	c.ensureShape(kBatch, kHeads, kKeyDim, vValueDim, k.Dtype(), v.Dtype())
	if c.offset+seqLen > c.maxSize {
		return c.updateOverflow(k, v, seqLen)
	}
	writeK, writeV := k, v
	writeLen := seqLen
	if writeLen > c.maxSize {
		start := writeLen - c.maxSize
		writeK = Slice4(k, 0, 0, int32(start), 0, kBatch, kHeads, int32(writeLen), kKeyDim)
		writeV = Slice4(v, 0, 0, int32(start), 0, kBatch, kHeads, int32(writeLen), vValueDim)
		defer Free(writeK, writeV)
		writeLen = c.maxSize
	}

	start := c.offset

	oldK, oldV := c.keys, c.values
	// Use the FixedKVCache-specific 4D slice-update helper — stack-allocated
	// cgo int arrays save three [4]C.int heap allocations per call versus
	// the generic SliceUpdateInplace.  Two calls per Update × hundreds of
	// tokens per decode loop.  Stream was resolved at the top of Update.
	c.keys = fixedKVCacheSliceUpdate4D(c.keys, writeK, kBatch, kHeads, int32(start), int32(start+writeLen), kKeyDim, stream)
	c.values = fixedKVCacheSliceUpdate4D(c.values, writeV, kBatch, kHeads, int32(start), int32(start+writeLen), vValueDim, stream)
	Free(oldK, oldV)

	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	return c.validStateWithStream(stream)
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
	prefixK := Slice4(fullK, 0, 0, 0, 0, kShape[0], kShape[1], int32(prefixLen), kShape[3])
	prefixV := Slice4(fullV, 0, 0, 0, 0, vShape[0], vShape[1], int32(prefixLen), vShape[3])
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
	// Steady-state fast path: trust the cached dims rather than allocating
	// fresh []int32 via Array.Shape() on every Update.
	if c.shapeCached && c.keys != nil && c.values != nil &&
		c.batch == batch && c.heads == heads &&
		c.keyDim == keyDim && c.valueDim == valueDim {
		return
	}
	if c.keys != nil && c.values != nil {
		// First call after a shape change — fall back to the Dim accessor
		// (cgo call, no slice alloc) to validate the existing buffers.
		if c.keys.NumDims() >= 4 && c.values.NumDims() >= 4 &&
			int32(c.keys.Dim(0)) == batch && int32(c.keys.Dim(1)) == heads &&
			int32(c.keys.Dim(2)) == int32(c.maxSize) && int32(c.keys.Dim(3)) == keyDim &&
			int32(c.values.Dim(0)) == batch && int32(c.values.Dim(1)) == heads &&
			int32(c.values.Dim(2)) == int32(c.maxSize) && int32(c.values.Dim(3)) == valueDim {
			c.batch, c.heads, c.keyDim, c.valueDim = batch, heads, keyDim, valueDim
			c.shapeCached = true
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
	c.batch, c.heads, c.keyDim, c.valueDim = batch, heads, keyDim, valueDim
	c.shapeCached = true
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
	stream := DefaultStream()
	k, v, ownK, ownV := c.storageKVPair(k, v, stream)
	defer freeOwnedPair(ownK, ownV)
	if k.NumDims() < 4 || v.NumDims() < 4 {
		return
	}
	kBatch := int32(k.Dim(0))
	kHeads := int32(k.Dim(1))
	kSeq := k.Dim(2)
	kKeyDim := int32(k.Dim(3))
	vValueDim := int32(v.Dim(3))
	Free(c.keys, c.values)
	c.keys = Zeros([]int32{kBatch, kHeads, int32(c.maxSize), kKeyDim}, k.Dtype())
	c.values = Zeros([]int32{kBatch, kHeads, int32(c.maxSize), vValueDim}, v.Dtype())
	tailLen := min(kSeq, c.maxSize)
	oldK, oldV := c.keys, c.values
	c.keys = fixedKVCacheSliceUpdate4D(c.keys, k, kBatch, kHeads, 0, int32(tailLen), kKeyDim, stream)
	c.values = fixedKVCacheSliceUpdate4D(c.values, v, kBatch, kHeads, 0, int32(tailLen), vValueDim, stream)
	Free(oldK, oldV)
	c.batch, c.heads, c.keyDim, c.valueDim = kBatch, kHeads, kKeyDim, vValueDim
	c.shapeCached = true
}

func (c *FixedKVCache) validState() (*Array, *Array) {
	return c.validStateWithStream(DefaultStream())
}

// validStateWithStream is the alloc-conscious variant used by Update's
// hot path, which has already resolved the stream once for its slice-
// update ops.  External callers go through validState which re-resolves.
func (c *FixedKVCache) validStateWithStream(stream *Stream) (*Array, *Array) {
	if c.keys == nil || c.values == nil || c.length <= 0 {
		return nil, nil
	}
	// Cached dims are stable for the lifetime of c.keys / c.values — use
	// the pooled-cgo-int fixedKVCacheSlice4D helper to skip both the
	// Shape() []int32 allocs and Slice's three [4]C.int heap allocs.
	if c.shapeCached {
		return fixedKVCacheSlice4D(c.keys, c.batch, c.heads, 0, int32(c.length), c.keyDim, stream),
			fixedKVCacheSlice4D(c.values, c.batch, c.heads, 0, int32(c.length), c.valueDim, stream)
	}
	// Fallback for paths that bypass ensureShape (legacy / pre-cache state).
	if c.keys.NumDims() < 4 || c.values.NumDims() < 4 {
		return nil, nil
	}
	return Slice4(c.keys, 0, 0, 0, 0, int32(c.keys.Dim(0)), int32(c.keys.Dim(1)), int32(c.length), int32(c.keys.Dim(3))),
		Slice4(c.values, 0, 0, 0, 0, int32(c.values.Dim(0)), int32(c.values.Dim(1)), int32(c.length), int32(c.values.Dim(3)))
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
	// Caller-supplied buffers — shape cache is no longer valid until
	// validState's fallback or the next ensureShape re-establishes it.
	c.shapeCached = false
	return c.FixedState()
}

func (c *FixedKVCache) ReplaceFixedFromNativeBorrowed(k, v *Array, seqLen int) FixedKVState {
	Free(c.keys, c.values)
	c.keys = k
	c.values = v
	c.offset += seqLen
	c.length = min(c.offset, c.maxSize)
	c.shapeCached = false
	return c.BorrowedFixedState()
}

func (c *FixedKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *FixedKVCache) AppendState(dst []*Array) []*Array {
	if c.keys == nil {
		return dst
	}
	if c.keys != nil && c.keys.Valid() {
		dst = append(dst, c.keys)
	}
	if c.values != nil && c.values.Valid() {
		dst = append(dst, c.values)
	}
	return dst
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
	c.shapeCached = false
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

// storageKVPair is the slice-free variant of storageKV.  Returns the dtype-
// converted k', v' alongside the *Array handles to free (or nil if no
// conversion was required).  Avoids the []*Array backing-array allocation
// that cacheStorageKV does — important on the per-token decode loop where
// every Update converts F32→F16 for the cache buffer.
//
// stream is the pre-resolved MLX stream; passing it through to the
// FP16-conversion AsType ops avoids two more DefaultStream() lookups
// per Update on the FP16 storage path.
//
//	convK, convV, ownK, ownV := c.storageKVPair(k, v, stream)
//	defer freeOwnedPair(ownK, ownV)
func (c *FixedKVCache) storageKVPair(k, v *Array, stream *Stream) (convK, convV, ownK, ownV *Array) {
	if c == nil || !c.hasStorageDType {
		return k, v, nil, nil
	}
	if DTypeByteSize(c.storageDType) <= 0 {
		return k, v, nil, nil
	}
	convK, convV = k, v
	if k != nil && k.Valid() && k.Dtype() != c.storageDType {
		convK = fixedKVCacheAsType(k, c.storageDType, stream)
		ownK = convK
	}
	if v != nil && v.Valid() && v.Dtype() != c.storageDType {
		convV = fixedKVCacheAsType(v, c.storageDType, stream)
		ownV = convV
	}
	return convK, convV, ownK, ownV
}

// freeOwnedPair releases the two slots from storageKVPair without an
// intermediate []*Array.  A single call into the variadic Free with two
// fixed args lets the compiler use a stack-allocated backing array.
//
//	defer freeOwnedPair(ownK, ownV)
func freeOwnedPair(ownK, ownV *Array) {
	if ownK == nil && ownV == nil {
		return
	}
	Free(ownK, ownV)
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
	// Scratch buffers for visiblePages — reused across Update calls so the
	// per-token concatenatedState() path doesn't allocate three []*Array
	// slices each time.  The slices are consumed within concatenatedState
	// (kPages/vPages feed Concatenate, owned is Free'd) so they're safe to
	// reuse on the next call.
	visibleKScratch     []*Array
	visibleVScratch     []*Array
	visibleOwnedScratch []*Array
	// Scratch buffers for K/V shape readouts — Dim() into these from inside
	// appendPagesPrealloc/Concat instead of calling Shape() which allocates a
	// new []int32 every time.  The slices are passed down to helpers within
	// the same call frame (canAppendToLastPage, append* helpers, cachePageView)
	// and never retained beyond the Update.
	kShapeScratch []int32
	vShapeScratch []int32
	materializedLength  int
	storageDType        DType
	hasStorageDType     bool
	offset              int
	length              int
	maxSize             int
	pageSize            int
	// preallocStorage is true when pages have storage = c.pageSize (prealloc
	// path); false when storage equals the actual fill length (concat path).
	// Set lazily on first page append; cleared on Reset.  Used by visiblePage
	// to skip page.Shape() allocations — the cached pageShape + this flag
	// fully describe the slice/clone branch without a per-call cgo Shape().
	preallocStorage bool
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

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *PagedKVCache) AppendState(dst []*Array) []*Array {
	if len(c.kPages) == 0 {
		return dst
	}
	for _, page := range c.kPages {
		if page != nil && page.Valid() {
			dst = append(dst, page)
		}
	}
	for _, page := range c.vPages {
		if page != nil && page.Valid() {
			dst = append(dst, page)
		}
	}
	return dst
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
	c.visibleKScratch = nil
	c.visibleVScratch = nil
	c.visibleOwnedScratch = nil
	c.kShapeScratch = nil
	c.vShapeScratch = nil
	c.preallocStorage = false
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
	if len(kPages) == 1 && len(vPages) == 1 {
		// Single-page fast path: the visible-page slice/clone is already a
		// fresh Array suitable for return — skip the redundant Clone inside
		// concatenatePagedState by handing ownership directly to the caller
		// and dropping the two pages from the owned-free list.
		fullK, fullV := kPages[0], vPages[0]
		owned = pagedOwnedExcept(owned, fullK, fullV)
		Free(owned...)
		return fullK, fullV
	}
	defer Free(owned...)
	return concatenatePagedState(kPages, vPages)
}

// pagedOwnedExcept returns owned with the entries equal to k or v removed.
// Used by concatenatedState's single-page fast path to skip the Clone+Free
// dance — kPages[0] and vPages[0] flow out to the caller, so they must not
// be Free'd in the owned-list cleanup.
func pagedOwnedExcept(owned []*Array, k, v *Array) []*Array {
	if len(owned) == 0 {
		return owned
	}
	out := owned[:0]
	for _, a := range owned {
		if a == k || a == v {
			continue
		}
		out = append(out, a)
	}
	return out
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
	kShape, vShape, ok := c.populateShapeScratch(k, v)
	if !ok {
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
	// Use scratch slices populated via Dim() instead of k.Shape()/v.Shape() —
	// each Shape() call allocates a fresh []int32 on every token-Update, while
	// Dim is a single cgo read.  The scratch is only read within this call
	// frame; helpers receive []int32 views and don't retain them.
	kShape, vShape, ok := c.populateShapeScratch(k, v)
	if !ok {
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

// populateShapeScratch fills the cache's K/V shape scratch slices from the
// arrays' Dim() values and returns views over them.  Saves two Shape() heap
// allocations per appendPages*  call.  The returned slices are only valid
// until the next populateShapeScratch / Reset.
func (c *PagedKVCache) populateShapeScratch(k, v *Array) (kShape, vShape []int32, ok bool) {
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		return nil, nil, false
	}
	if k.NumDims() < 4 || v.NumDims() < 4 {
		return nil, nil, false
	}
	if cap(c.kShapeScratch) < 4 {
		c.kShapeScratch = make([]int32, 4)
	} else {
		c.kShapeScratch = c.kShapeScratch[:4]
	}
	if cap(c.vShapeScratch) < 4 {
		c.vShapeScratch = make([]int32, 4)
	} else {
		c.vShapeScratch = c.vShapeScratch[:4]
	}
	c.kShapeScratch[0] = int32(k.Dim(0))
	c.kShapeScratch[1] = int32(k.Dim(1))
	c.kShapeScratch[2] = int32(k.Dim(2))
	c.kShapeScratch[3] = int32(k.Dim(3))
	c.vShapeScratch[0] = int32(v.Dim(0))
	c.vShapeScratch[1] = int32(v.Dim(1))
	c.vShapeScratch[2] = int32(v.Dim(2))
	c.vShapeScratch[3] = int32(v.Dim(3))
	return c.kShapeScratch, c.vShapeScratch, true
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
	// pagedSliceUpdate4D consolidates the three [4]C.int cgo allocations of
	// SliceUpdateInplace into a single 12-element make — 2 makes per token
	// vs 6 for the generic helper.
	stream := DefaultStream()
	c.kPages[last] = pagedSliceUpdate4D(oldK, pieceK, kShape[0], kShape[1], int32(writeStart), int32(writeStart+take), kShape[3], stream)
	c.vPages[last] = pagedSliceUpdate4D(oldV, pieceV, vShape[0], vShape[1], int32(writeStart), int32(writeStart+take), vShape[3], stream)
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
	stream := DefaultStream()
	updatedK := pagedSliceUpdate4D(pageK, pieceK, kShape[0], kShape[1], 0, int32(take), kShape[3], stream)
	updatedV := pagedSliceUpdate4D(pageV, pieceV, vShape[0], vShape[1], 0, int32(take), vShape[3], stream)
	c.kPages = append(c.kPages, updatedK)
	c.vPages = append(c.vPages, updatedV)
	c.pageLens = append(c.pageLens, take)
	c.recordPageShape(kShape, vShape)
	c.preallocStorage = true
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
	return Slice4(a, 0, 0, int32(start), 0, shape[0], shape[1], int32(start+take), shape[3]), true
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
	tailK := Slice4(oldK, 0, 0, int32(tokens), 0, kShape[0], kShape[1], int32(pageLen), kShape[3])
	tailV := Slice4(oldV, 0, 0, int32(tokens), 0, vShape[0], vShape[1], int32(pageLen), vShape[3])
	if pagedKVPreallocEnabled() {
		pageK := Zeros([]int32{kShape[0], kShape[1], int32(c.pageSize), kShape[3]}, oldK.Dtype())
		pageV := Zeros([]int32{vShape[0], vShape[1], int32(c.pageSize), vShape[3]}, oldV.Dtype())
		c.kPages[0] = SliceUpdateInplace4(pageK, tailK, 0, 0, 0, 0, kShape[0], kShape[1], int32(newLen), kShape[3])
		c.vPages[0] = SliceUpdateInplace4(pageV, tailV, 0, 0, 0, 0, vShape[0], vShape[1], int32(newLen), vShape[3])
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
	length := c.pageLen(i)
	// Fast path: when the cached pageShape is set we know batch/heads/dim for
	// the K and V sides, and the storage seq-length is c.pageSize for prealloc
	// pages or pageLens[i] for concat pages.  This lets us skip the per-call
	// page.Shape() allocation and decide Slice vs Clone using cached info, and
	// route through pagedSlice4D which consolidates the three cgo-int slice
	// allocations of metal.Slice into a single make.
	if c.pageShape.set && length > 0 {
		if isK, ok := c.identifyPage(page, i); ok {
			storage := length
			if c.preallocStorage {
				storage = c.pageSize
			}
			if length >= storage {
				return page.Clone()
			}
			stream := DefaultStream()
			if isK {
				return pagedSlice4D(page, c.pageShape.kBatch, c.pageShape.kHeads, int32(length), c.pageShape.kDim, stream)
			}
			return pagedSlice4D(page, c.pageShape.vBatch, c.pageShape.vHeads, int32(length), c.pageShape.vDim, stream)
		}
	}
	shape := page.Shape()
	if len(shape) < 4 || length <= 0 || length >= int(shape[2]) {
		return page.Clone()
	}
	return Slice4(page, 0, 0, 0, 0, shape[0], shape[1], int32(length), shape[3])
}

func (c *PagedKVCache) borrowVisiblePage(page *Array, i int) (*Array, bool) {
	if page == nil || !page.Valid() {
		return nil, false
	}
	length := c.pageLen(i)
	if c.pageSize > 0 && length >= c.pageSize {
		return page, false
	}
	// Fast path: avoid page.Shape() when the cached pageShape is set.  Storage
	// is c.pageSize for prealloc pages; for concat pages the page is fully
	// filled (length == pageLens[i] == shape[2]) so borrow returns the page
	// directly without slicing.  Routes the slice via pagedSlice4D (one cgo-int
	// make instead of three) when a partial view is needed.
	if c.pageShape.set && length > 0 {
		if isK, ok := c.identifyPage(page, i); ok {
			storage := length
			if c.preallocStorage {
				storage = c.pageSize
			}
			if length >= storage {
				return page, false
			}
			stream := DefaultStream()
			if isK {
				return pagedSlice4D(page, c.pageShape.kBatch, c.pageShape.kHeads, int32(length), c.pageShape.kDim, stream), true
			}
			return pagedSlice4D(page, c.pageShape.vBatch, c.pageShape.vHeads, int32(length), c.pageShape.vDim, stream), true
		}
	}
	shape := page.Shape()
	if len(shape) < 4 || length <= 0 || length >= int(shape[2]) {
		return page, false
	}
	return Slice4(page, 0, 0, 0, 0, shape[0], shape[1], int32(length), shape[3]), true
}

// identifyPage returns (isK, ok) — isK is true when the page is the i-th K
// page, false when it is the i-th V page.  ok is false when the page doesn't
// match either, which can happen when the caller has cloned pages out of the
// cache.  Falls through to the legacy page.Shape() path in that case.
func (c *PagedKVCache) identifyPage(page *Array, i int) (bool, bool) {
	if i >= 0 && i < len(c.kPages) && c.kPages[i] == page {
		return true, true
	}
	if i >= 0 && i < len(c.vPages) && c.vPages[i] == page {
		return false, true
	}
	return false, false
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
	n := len(c.kPages)
	if n == 0 || len(c.vPages) == 0 || n != len(c.vPages) {
		return nil, nil, nil
	}
	// Reuse scratch buffers across Update calls — concatenatedState consumes
	// these slices within the same call (kPages/vPages flow into Concatenate,
	// owned is Free'd via defer), so reuse is safe.  Saves 3 allocs per Update.
	if cap(c.visibleKScratch) < n {
		c.visibleKScratch = make([]*Array, n)
	} else {
		c.visibleKScratch = c.visibleKScratch[:n]
	}
	if cap(c.visibleVScratch) < n {
		c.visibleVScratch = make([]*Array, n)
	} else {
		c.visibleVScratch = c.visibleVScratch[:n]
	}
	if cap(c.visibleOwnedScratch) < 2*n {
		c.visibleOwnedScratch = make([]*Array, 0, 2*n)
	} else {
		c.visibleOwnedScratch = c.visibleOwnedScratch[:0]
	}
	kPages = c.visibleKScratch
	vPages = c.visibleVScratch
	owned = c.visibleOwnedScratch
	for i := range c.kPages {
		kPages[i] = c.visiblePage(c.kPages[i], i)
		vPages[i] = c.visiblePage(c.vPages[i], i)
		owned = append(owned, kPages[i], vPages[i])
	}
	c.visibleOwnedScratch = owned
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
		writeK = Slice4(k, 0, 0, int32(start), 0, kShape[0], kShape[1], int32(totalLen), kShape[3])
		writeV = Slice4(v, 0, 0, int32(start), 0, vShape[0], vShape[1], int32(totalLen), vShape[3])
		defer Free(writeK, writeV)
	}
	start := c.materializedLength
	oldK, oldV := c.materializedKeys, c.materializedVals
	c.materializedKeys = SliceUpdateInplace4(c.materializedKeys, writeK, 0, 0, int32(start), 0, kShape[0], kShape[1], int32(start+seqLen), kShape[3])
	c.materializedVals = SliceUpdateInplace4(c.materializedVals, writeV, 0, 0, int32(start), 0, vShape[0], vShape[1], int32(start+seqLen), vShape[3])
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
	c.materializedKeys = SliceUpdateInplace4(c.materializedKeys, fullK, 0, 0, 0, 0, kShape[0], kShape[1], int32(state.Length), kShape[3])
	c.materializedVals = SliceUpdateInplace4(c.materializedVals, fullV, 0, 0, 0, 0, vShape[0], vShape[1], int32(state.Length), vShape[3])
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
	return Slice4(c.materializedKeys, 0, 0, 0, 0, kShape[0], kShape[1], int32(c.materializedLength), kShape[3]),
		Slice4(c.materializedVals, 0, 0, 0, 0, vShape[0], vShape[1], int32(c.materializedLength), vShape[3])
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
	return Slice4(k, 0, 0, int32(start), 0, kShape[0], kShape[1], kShape[2], kShape[3]),
		Slice4(v, 0, 0, int32(start), 0, vShape[0], vShape[1], vShape[2], vShape[3])
}

