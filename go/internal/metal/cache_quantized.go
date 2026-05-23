// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// QuantizedKVCache stores cache tensors in int8 lanes and dequantizes them
// only for the attention call. keyBits/valueBits control the logical quantizer
// range; q4 values currently use int8 storage until packed q4 kernels land.
//
// floatK / floatV cache the last dequantised K/V state so the next Update can
// skip the full unpack/upcast/multiply round-trip. They are populated lazily
// after Update and freed on Reset; snapshot/restore and ReadState() continue
// to operate on the quantised state, so save/load paths are unchanged.
//
// keyMaxBound / keyMinValue / valueMaxBound / valueMinValue / quantizeEps
// hoist the per-call FromValue scalars (constant for the cache's lifetime)
// onto the struct so quantizeCacheArray reuses one MLX scalar handle across
// all Updates rather than allocating + freeing four scalars per call.
//
// packOffsetI8 / packShiftU8 hoist the bit-pack constants used by packQ4
// (int8 8, uint8 4) so the Q4 storage path doesn't re-allocate them on
// every Update either.
type QuantizedKVCache struct {
	keys, values       *Array
	keyScale           *Array
	valueScale         *Array
	floatK, floatV     *Array
	keyMaxBound        *Array
	keyMinValue        *Array
	valueMaxBound      *Array
	valueMinValue      *Array
	quantizeEps        *Array
	packOffsetI8       *Array
	packShiftU8        *Array
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
	// NumDims() is a single cgo read whereas Shape() allocates a fresh
	// []int32 — and we only need to gate the rank-4 path below.
	if k.NumDims() < 4 {
		fullK := k.Clone()
		fullV := v.Clone()
		c.storeQuantized(fullK, fullV)
		c.cacheFloat(fullK, fullV)
		c.offset += seqLen
		return fullK, fullV
	}

	prevK, prevV := c.takeFloat()
	if prevK == nil {
		prevK, prevV = c.dequantizedState()
	}
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
	c.cacheFloat(storeK, storeV)
	if storeK != fullK {
		Free(storeK, storeV)
	}
	return fullK, fullV
}

// takeFloat returns the cached float K/V if present and clears the cache slots,
// transferring ownership to the caller. Returns (nil, nil) on miss.
func (c *QuantizedKVCache) takeFloat() (*Array, *Array) {
	k, v := c.floatK, c.floatV
	c.floatK = nil
	c.floatV = nil
	return k, v
}

// cacheFloat stores clones of k/v as the float-form cache for the next Update.
// Any previously-cached float arrays are released.
func (c *QuantizedKVCache) cacheFloat(k, v *Array) {
	old1, old2 := c.floatK, c.floatV
	if k != nil {
		c.floatK = k.Clone()
	} else {
		c.floatK = nil
	}
	if v != nil {
		c.floatV = v.Clone()
	} else {
		c.floatV = nil
	}
	Free(old1, old2)
}

func (c *QuantizedKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values, c.keyScale, c.valueScale}
}

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *QuantizedKVCache) AppendState(dst []*Array) []*Array {
	if c.keys == nil {
		return dst
	}
	if c.keys != nil && c.keys.Valid() {
		dst = append(dst, c.keys)
	}
	if c.values != nil && c.values.Valid() {
		dst = append(dst, c.values)
	}
	if c.keyScale != nil && c.keyScale.Valid() {
		dst = append(dst, c.keyScale)
	}
	if c.valueScale != nil && c.valueScale.Valid() {
		dst = append(dst, c.valueScale)
	}
	return dst
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
	Free(c.keys, c.values, c.keyScale, c.valueScale, c.floatK, c.floatV,
		c.keyMaxBound, c.keyMinValue, c.valueMaxBound, c.valueMinValue, c.quantizeEps,
		c.packOffsetI8, c.packShiftU8)
	c.keys = nil
	c.values = nil
	c.keyScale = nil
	c.valueScale = nil
	c.floatK = nil
	c.floatV = nil
	c.keyMaxBound = nil
	c.keyMinValue = nil
	c.valueMaxBound = nil
	c.valueMinValue = nil
	c.quantizeEps = nil
	c.packOffsetI8 = nil
	c.packShiftU8 = nil
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
	keyMax, keyMin, eps := c.ensureKeyScalars()
	packOff, packSh := c.ensurePackScalars(c.keyBits, c.valueBits)
	// Reuse the cache's shape backing across Updates — quantizeCacheArrayCached
	// will ShapeInto the passed buffer when its cap matches the source's
	// NumDims, skipping the per-call `[]int32` heap alloc that the previous
	// `append([]int32(nil), a.Shape()...)` pattern paid on every token.
	c.keys, c.keyScale, c.keyShape = quantizeCacheArrayCached(k, c.keyBits, keyMax, keyMin, eps, packOff, packSh, c.keyShape)
	valueMax, valueMin, _ := c.ensureValueScalars()
	c.values, c.valueScale, c.valueShape = quantizeCacheArrayCached(v, c.valueBits, valueMax, valueMin, eps, packOff, packSh, c.valueShape)
	Free(oldK, oldV, oldKS, oldVS)
}

// ensureKeyScalars lazily allocates the per-K quantise scalars (maxBound,
// minValue, eps) and returns shared handles. Scalars are derived from
// keyBits and are constant for the cache lifetime, so a single set is
// reused across every Update — cutting four MLX-scalar allocations per
// call.
func (c *QuantizedKVCache) ensureKeyScalars() (*Array, *Array, *Array) {
	if c.keyMaxBound == nil {
		maxValue := quantizeMaxValue(c.keyBits)
		c.keyMaxBound = FromValue(maxValue)
		c.keyMinValue = FromValue(-maxValue)
	}
	if c.quantizeEps == nil {
		c.quantizeEps = FromValue(float32(1e-6))
	}
	return c.keyMaxBound, c.keyMinValue, c.quantizeEps
}

// ensureValueScalars is the sibling helper for V quantisation. When
// keyBits == valueBits the cache could share one set, but the asymmetric
// K@q8/V@q4 mode (KVCacheModeKQ8VQ4) keeps the two scalar pairs
// independent so the quantiser graph keeps a fixed shape per branch.
func (c *QuantizedKVCache) ensureValueScalars() (*Array, *Array, *Array) {
	if c.valueMaxBound == nil {
		maxValue := quantizeMaxValue(c.valueBits)
		c.valueMaxBound = FromValue(maxValue)
		c.valueMinValue = FromValue(-maxValue)
	}
	if c.quantizeEps == nil {
		c.quantizeEps = FromValue(float32(1e-6))
	}
	return c.valueMaxBound, c.valueMinValue, c.quantizeEps
}

// ensurePackScalars lazily allocates the bit-pack constants used by packQ4
// (int8 8 sign-shift offset, uint8 4 shift count) when either K or V is
// stored at Q4. Returns (nil, nil) when neither branch needs them so the
// pure-Q8 path doesn't pay any setup cost.
func (c *QuantizedKVCache) ensurePackScalars(keyBits, valueBits int) (*Array, *Array) {
	if keyBits != 4 && valueBits != 4 {
		return nil, nil
	}
	if c.packOffsetI8 == nil {
		offTmp := FromValue(8)
		c.packOffsetI8 = AsType(offTmp, DTypeInt8)
		shTmp := FromValue(4)
		c.packShiftU8 = AsType(shTmp, DTypeUint8)
		Free(offTmp, shTmp)
	}
	return c.packOffsetI8, c.packShiftU8
}

func (c *QuantizedKVCache) dequantizedState() (*Array, *Array) {
	if c.keys == nil || c.values == nil {
		return nil, nil
	}
	return dequantizeCacheArray(c.keys, c.keyScale, c.keyDtype, c.keyShape, c.keyBits),
		dequantizeCacheArray(c.values, c.valueScale, c.valueDtype, c.valueShape, c.valueBits)
}

func quantizeCacheArray(a *Array, bits int) (*Array, *Array, []int32) {
	maxValue := quantizeMaxValue(bits)
	eps := FromValue(float32(1e-6))
	maxBound := FromValue(maxValue)
	minValue := FromValue(-maxValue)
	defer Free(eps, maxBound, minValue)
	return quantizeCacheArrayCached(a, bits, maxBound, minValue, eps, nil, nil, nil)
}

// quantizeCacheArrayCached is quantizeCacheArray with the bits-derived
// scalars supplied by the caller — letting the QuantizedKVCache reuse one
// scalar set across every Update rather than allocating fresh MLX scalars
// in the hot path. The caller owns eps/maxBound/minValue lifetime; pass
// nil for packOffsetI8/packShiftU8 to fall back to allocating them inside
// packQ4 (used by the non-cached entry point above).
//
// shapeBuf, when non-nil with sufficient cap, receives the source's shape
// via ShapeInto — letting the QuantizedKVCache reuse its keyShape /
// valueShape backing array across every Update and skip the per-call
// `[]int32` heap alloc that the previous `append([]int32(nil), ...)`
// pattern paid. Pass nil to fall back to allocating a fresh slice (used
// by snapshot paths in prompt_cache.go that need an independent copy).
func quantizeCacheArrayCached(a *Array, bits int, maxBound, minValue, eps, packOffsetI8, packShiftU8 *Array, shapeBuf []int32) (*Array, *Array, []int32) {
	ndim := a.NumDims()
	var shape []int32
	if cap(shapeBuf) >= ndim {
		shape = a.ShapeInto(shapeBuf[:0])
	} else {
		shape = append([]int32(nil), a.Shape()...)
	}
	abs := Abs(a)
	maxAbs := maxAll(abs)
	clampedAbs := Maximum(maxAbs, eps)
	scale := Divide(clampedAbs, maxBound)
	normalized := Divide(a, scale)
	rounded := Round(normalized)
	clipped := Clip(rounded, minValue, maxBound)
	q := AsType(clipped, DTypeInt8)
	Free(abs, maxAbs, clampedAbs, normalized, rounded, clipped)
	if bits == 4 {
		packed := packQ4Cached(q, packOffsetI8, packShiftU8)
		Free(q)
		return packed, scale, shape
	}
	return q, scale, shape
}

// quantizeMaxValue returns the symmetric-quantiser upper bound for `bits`
// (2^(bits-1) - 1). Falls back to 127 (q8) when bits == 0 — keeps prior
// behaviour for cache slots that were initialised without a bit width.
func quantizeMaxValue(bits int) float32 {
	levels := 1
	for range max(0, bits-1) {
		levels *= 2
	}
	maxValue := float32(levels - 1)
	if maxValue <= 0 {
		maxValue = 127
	}
	return maxValue
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
	return packQ4Cached(q, nil, nil)
}

// packQ4Cached is packQ4 with the bit-pack constants (int8 8 offset, uint8 4
// shift) supplied by the caller — letting the QuantizedKVCache reuse one
// pair across every Q4 Update rather than allocating fresh MLX scalars per
// call. Pass nil for both to fall back to per-call allocation.
//
// Element count is read via Size() (single cgo call into mlx_array_size)
// rather than Shape() + walk — Shape() allocates a fresh []int32 per call
// which would otherwise show up as one heap alloc per Q4 Update.
//
// Reshape1 / Reshape2 / Slice2 replace the variadic Reshape and SliceAxis
// calls (W11-AC): the rank-1/2 scalar-pass primitives skip the variadic
// []int32 escape on `Reshape(q, int32(n))` + `Reshape(padded, int32(pairs),
// int32(2))` + `Reshape(packed2D, int32(pairs))`, and replace the
// SliceAxis(paired,...) pair (which materialised `make([]int32, ndim)`
// twice per call) with register-passed scalar slices.
func packQ4Cached(q, offsetI8, shiftU8 *Array) *Array {
	n := q.Size()
	flat := Reshape1(q, int32(n))
	ownOffset := offsetI8 == nil
	offset := offsetI8
	if ownOffset {
		offset = AsType(FromValue(8), DTypeInt8)
	}
	shifted := Add(flat, offset)
	shiftedU := AsType(shifted, DTypeUint8)
	Free(flat, shifted)
	if ownOffset {
		Free(offset)
	}

	padded := shiftedU
	nP := n
	if n%2 != 0 {
		zero := Zeros([]int32{1}, DTypeUint8)
		padded = Concatenate([]*Array{shiftedU, zero}, 0)
		Free(shiftedU, zero)
		nP = n + 1
	}

	pairs := nP / 2
	paired := Reshape2(padded, int32(pairs), 2)
	Free(padded)
	low := Slice2(paired, 0, 0, int32(pairs), 1)
	high := Slice2(paired, 0, 1, int32(pairs), 2)
	Free(paired)
	ownShift := shiftU8 == nil
	shift := shiftU8
	if ownShift {
		shift = AsType(FromValue(4), DTypeUint8)
	}
	highShifted := LeftShift(high, shift)
	packed2D := BitwiseOr(low, highShifted)
	packed := Reshape1(packed2D, int32(pairs))
	Free(low, high, highShifted, packed2D)
	if ownShift {
		Free(shift)
	}
	return packed
}

// unpackQ4 expands a uint8 array of packed Q4 nibbles back into a signed int8
// array of the original shape. The implementation reshapes pair-wise after
// extracting the low/high nibbles, replacing the previous PutAlongAxis +
// gather indices with structural ops only.
//
// `pairs` is read via low.Dim(0) (single cgo call) rather than low.Shape()[0]
// (which allocates a fresh []int32 just to read one dim) — saves one heap
// alloc per dequantise on the rare Q4 dequant path.
//
// Reshape1 / Slice1 replace the rank-1 variadic Reshape / Slice calls
// (W11-AC): `Reshape(stacked, int32(flatLen))` paid one variadic-slice
// escape per dequant, and `Slice(flat, []int32{0}, []int32{int32(n)})`
// paid two more on the (rare) odd-length tail-trim. The final
// `Reshape(signed, shape...)` keeps the variadic form because the shape
// comes from the caller as a slice of arbitrary rank.
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

	pairs := low.Dim(0)
	lowE := ExpandDims(low, 1)
	highE := ExpandDims(high, 1)
	Free(low, high)
	stacked := Concatenate([]*Array{lowE, highE}, 1)
	Free(lowE, highE)

	flatLen := pairs * 2
	flat := Reshape1(stacked, int32(flatLen))
	Free(stacked)

	outU := flat
	if flatLen > n {
		outU = Slice1(flat, 0, int32(n))
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

// maxAll returns a scalar Array equal to the max-abs of all elements of a.
// The implementation flattens to 1-D (zero-copy reshape) then reduces in a
// single MaxAxis call, replacing the prior N-axis iterative reduction which
// materialised one intermediate per dimension.
//
// Element count is read via Size() + NumDims() (single cgo calls each)
// rather than Shape() + cacheElementCount walk — Shape() would allocate a
// fresh []int32 every call which is per-quantize, every Update.
func maxAll(a *Array) *Array {
	if a.NumDims() == 0 {
		return a.Clone()
	}
	n := a.Size()
	if n == 0 {
		return a.Clone()
	}
	flat := Reshape(a, int32(n))
	reduced := MaxAxis(flat, 0, false)
	Free(flat)
	return reduced
}
