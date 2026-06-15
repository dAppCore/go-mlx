// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleNewKVCache() {
	cache := NewKVCache()

	core.Println(cache.Offset(), cache.Len(), cache.State() == nil, cache.Step())
	// Output: 0 0 true 256
}

func ExampleNewFixedKVCacheAtOffset() {
	// Restore a fixed-capacity cache to a previously checkpointed position
	// without allocating any storage — only the offset/length counters are set.
	cache := NewFixedKVCacheAtOffset(512, 37, 33)

	core.Println(cache.MaxSize(), cache.Offset(), cache.Len(), cache.State() == nil)
	// Output: 512 37 33 true
}

func ExampleCachesTruncateTo() {
	// A batch of storage-less caches can always truncate in place (each is below
	// the target), so the batch reports overall success.
	caches := []Cache{NewKVCache(), NewFixedKVCache(256)}

	core.Println(CachesTruncateTo(caches, 8))
	// Output: true
}

func ExampleKVCache_Update() {
	cache := NewKVCache()
	k, v := cacheExampleKV(1, 2, 3)
	outK, outV := cache.Update(k, v, 3)
	defer cache.Reset()
	defer Free(k, v, outK, outV)
	Materialize(outK, outV)

	core.Println(cache.Offset(), cache.Len(), outK.Shape(), outK.Floats())
	// Output: 3 3 [1 1 3 1] [1 2 3]
}

func ExampleKVCache_State() {
	cache := NewKVCache()
	k, v := cacheExampleKV(4, 5)
	outK, outV := cache.Update(k, v, 2)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	state := cache.State()
	core.Println(len(state), state[0].Shape(), state[1].Shape())
	// Output: 2 [1 1 256 1] [1 1 256 1]
}

func ExampleKVCache_Offset() {
	cache := NewKVCache()
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	core.Println(cache.Offset())
	// Output: 2
}

func ExampleKVCache_Len() {
	cache := NewKVCache()
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	core.Println(cache.Len())
	// Output: 2
}

func ExampleKVCache_Reset() {
	cache := NewKVCache()
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	Free(k, v, outK, outV)
	cache.Reset()

	core.Println(cache.Offset(), cache.Len(), cache.State() == nil)
	// Output: 0 0 true
}

func ExampleKVCache_Detach() {
	cache := NewKVCache()
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	defer cache.Reset()
	defer Free(k, v, outK, outV)
	Materialize(outK, outV)
	cache.Detach()

	core.Println(cache.Offset(), len(cache.State()), cache.State()[0].Valid())
	// Output: 2 2 true
}

func ExampleNewRotatingKVCache() {
	cache := NewRotatingKVCache(4)

	core.Println(cache.MaxSize(), cache.Offset(), cache.Len(), cache.State() == nil)
	// Output: 4 0 0 true
}

func ExampleRotatingKVCache_Update() {
	cache := NewRotatingKVCache(4)
	defer cache.Reset()

	var outK, outV *Array
	for i := 1; i <= 5; i++ {
		k, v := cacheExampleKV(float32(i))
		nextK, nextV := cache.Update(k, v, 1)
		Materialize(nextK, nextV)
		if outK != nil {
			Free(outK, outV)
		}
		Free(k, v)
		outK, outV = nextK, nextV
	}
	defer Free(outK, outV)

	core.Println(cache.Offset(), cache.Len(), outK.Shape(), outK.Floats())
	// Output: 5 4 [1 1 4 1] [2 3 4 5]
}

func ExampleRotatingKVCache_State() {
	cache := NewRotatingKVCache(4)
	k, v := cacheExampleKV(1, 2, 3, 4, 5)
	outK, outV := cache.Update(k, v, 5)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	state := cache.State()
	core.Println(outK.Shape(), state[0].Shape())
	// Output: [1 1 5 1] [1 1 4 1]
}

func ExampleRotatingKVCache_Offset() {
	cache := NewRotatingKVCache(4)
	k, v := cacheExampleKV(1, 2, 3, 4, 5)
	outK, outV := cache.Update(k, v, 5)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	core.Println(cache.Offset())
	// Output: 5
}

func ExampleRotatingKVCache_Len() {
	cache := NewRotatingKVCache(4)
	k, v := cacheExampleKV(1, 2, 3, 4, 5)
	outK, outV := cache.Update(k, v, 5)
	defer cache.Reset()
	defer Free(k, v, outK, outV)

	core.Println(cache.Len())
	// Output: 4
}

func ExampleRotatingKVCache_Reset() {
	cache := NewRotatingKVCache(4)
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	Free(k, v, outK, outV)
	cache.Reset()

	core.Println(cache.Offset(), cache.Len(), cache.State() == nil)
	// Output: 0 0 true
}

func ExampleRotatingKVCache_Detach() {
	cache := NewRotatingKVCache(4)
	k, v := cacheExampleKV(1, 2)
	outK, outV := cache.Update(k, v, 2)
	defer cache.Reset()
	defer Free(k, v, outK, outV)
	Materialize(outK, outV)
	cache.Detach()

	core.Println(cache.Offset(), cache.Len(), cache.State()[0].Valid())
	// Output: 2 2 true
}

func cacheExampleKV(values ...float32) (*Array, *Array) {
	k := FromValues(values, 1, 1, len(values), 1)
	v := FromValues(values, 1, 1, len(values), 1)
	return k, v
}
