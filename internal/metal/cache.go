//go:build darwin && arm64 && !nomlx

package metal

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
	return Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, validLen, Dk}),
		Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, validLen, Dv})
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

	if c.keys == nil {
		c.keys, c.values = k.Clone(), v.Clone()
	} else {
		oldK, oldV := c.keys, c.values
		c.keys = Concatenate([]*Array{oldK, k}, 2)
		c.values = Concatenate([]*Array{oldV, v}, 2)
		Free(oldK, oldV)
	}
	c.offset += seqLen

	cap := int(c.keys.Shape()[2])
	if trim := cap - c.maxSize; trim > 0 {
		oldK, oldV := c.keys, c.values
		c.keys = Slice(c.keys, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dk})
		c.values = Slice(c.values, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dv})
		Free(oldK, oldV)
	}

	c.idx = int(c.keys.Shape()[2])
	// Return Slice views so callers can Free them without destroying the cache.
	// (updateInPlace and KVCache.Update already return Slice views.)
	return Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.idx), Dk}),
		Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.idx), Dv})
}

func (c *RotatingKVCache) State() []*Array {
	if c.keys == nil {
		return nil
	}
	return []*Array{c.keys, c.values}
}

func (c *RotatingKVCache) Offset() int { return c.offset }
func (c *RotatingKVCache) Len() int    { return min(c.offset, c.maxSize) }

func (c *RotatingKVCache) Reset() {
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
