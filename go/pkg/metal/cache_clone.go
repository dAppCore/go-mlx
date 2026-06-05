// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// cache_clone.go provides deep K/V cache cloning for runtime authors (e.g. a
// speculative-decode verifier that must explore draft tokens against a copy of
// the live cache without polluting it). Reconstructing each concrete cache type
// requires its private field layout, so the operation lives in package metal
// (RFC.model-sdk runtime-author surface) rather than leaking those fields.
package metal

import core "dappco.re/go"

var (
	errCloneCacheNil        = core.NewError("metal: cannot clone a nil cache")
	errCloneCacheStateEmpty = core.NewError("metal: cache state is empty")
)

// CloneCachePrefixes deep-copies a slice of caches, preserving each cache's
// visible token prefix. The clones are independent of the originals.
//
//	verify, err := metal.CloneCachePrefixes(liveCaches)
func CloneCachePrefixes(caches []Cache) ([]Cache, error) {
	cloned := make([]Cache, len(caches))
	for i, cache := range caches {
		next, err := CloneCachePrefix(cache)
		if err != nil {
			FreeCaches(cloned)
			return nil, core.E("metal.CloneCachePrefixes", core.Sprintf("clone cache %d", i), err)
		}
		cloned[i] = next
	}
	return cloned, nil
}

// CloneCachePrefix deep-copies a single cache, preserving its visible token
// prefix (Len). Empty caches clone to a fresh cache of the same type and
// geometry. The returned cache is independent of the original.
//
//	verifyCache, err := metal.CloneCachePrefix(liveCache)
func CloneCachePrefix(cache Cache) (Cache, error) {
	if cache == nil {
		return nil, errCloneCacheNil
	}
	if cache.Len() <= 0 {
		switch c := cache.(type) {
		case *RotatingKVCache:
			return NewRotatingKVCache(c.maxSize), nil
		case *FixedKVCache:
			return NewFixedKVCache(c.maxSize), nil
		case *PagedKVCache:
			if c.hasStorageDType {
				return NewPagedKVCacheWithDTypeAndPrealloc(c.maxSize, c.pageSize, c.storageDType, c.preallocPages), nil
			}
			return NewPagedKVCacheWithPrealloc(c.maxSize, c.pageSize, c.preallocPages), nil
		case *QuantizedKVCache:
			return NewQuantizedKVCache(c.maxSize, c.keyBits, c.valueBits), nil
		default:
			return NewKVCache(), nil
		}
	}
	switch c := cache.(type) {
	case *KVCache:
		state, owned := CacheReadState(c)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errCloneCacheStateEmpty
		}
		keys, values, err := cloneCacheStatePrefix(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &KVCache{keys: keys, values: values, offset: c.offset, step: c.step}, nil
	case *RotatingKVCache:
		state, owned := CacheReadState(c)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errCloneCacheStateEmpty
		}
		keys, values, err := cloneCacheStatePrefix(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &RotatingKVCache{keys: keys, values: values, offset: c.offset, maxSize: c.maxSize, step: c.step, idx: c.Len()}, nil
	case *FixedKVCache:
		state := c.FixedState()
		if state.Keys == nil || state.Values == nil {
			state.Free()
			return NewFixedKVCache(c.maxSize), nil
		}
		return &FixedKVCache{keys: state.Keys, values: state.Values, offset: c.offset, length: c.length, maxSize: c.maxSize}, nil
	case *PagedKVCache:
		pages := c.PageState()
		defer pages.Free()
		kPages, vPages, err := CopyPagedCachePrefix(pages.Keys, pages.Values, c.Len())
		if err != nil {
			return nil, err
		}
		return &PagedKVCache{
			kPages:          kPages,
			vPages:          vPages,
			pageLens:        PagedPageLensForPages(kPages, c.length),
			offset:          c.offset,
			length:          c.length,
			maxSize:         c.maxSize,
			pageSize:        c.pageSize,
			storageDType:    c.storageDType,
			hasStorageDType: c.hasStorageDType,
			preallocPages:   c.preallocPages,
		}, nil
	case *QuantizedKVCache:
		return &QuantizedKVCache{
			keys:       Copy(c.keys),
			values:     Copy(c.values),
			keyScale:   Copy(c.keyScale),
			valueScale: Copy(c.valueScale),
			keyDtype:   c.keyDtype,
			valueDtype: c.valueDtype,
			keyShape:   append([]int32(nil), c.keyShape...),
			valueShape: append([]int32(nil), c.valueShape...),
			offset:     c.offset,
			maxSize:    c.maxSize,
			step:       c.step,
			keyBits:    c.keyBits,
			valueBits:  c.valueBits,
		}, nil
	default:
		state, owned := CacheReadState(cache)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errCloneCacheStateEmpty
		}
		keys, values, err := cloneCacheStatePrefix(state[0], state[1], cache.Len())
		if err != nil {
			return nil, err
		}
		return &KVCache{keys: keys, values: values, offset: cache.Offset(), step: 256}, nil
	}
}

func cloneCacheStatePrefix(keys, values *Array, tokenLen int) (*Array, *Array, error) {
	keyCopy, err := CopyCachePrefix(keys, tokenLen)
	if err != nil {
		return nil, nil, err
	}
	valueCopy, err := CopyCachePrefix(values, tokenLen)
	if err != nil {
		Free(keyCopy)
		return nil, nil, err
	}
	return keyCopy, valueCopy, nil
}
