// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"crypto/sha256"
	"sync"
)

const defaultNativeVisionFeatureCacheEntries = 8

type nativeVisionFeatureKey = [sha256.Size]byte

func nativeVisionFeatureCacheKey(data []byte) nativeVisionFeatureKey {
	return sha256.Sum256(data)
}

type nativeVisionFeatureEntry struct {
	features   []byte
	softTokens int
}

type nativeVisionFeatureCache struct {
	mu       sync.Mutex
	capacity int
	entries  map[nativeVisionFeatureKey]nativeVisionFeatureEntry
	order    []nativeVisionFeatureKey
}

func newNativeVisionFeatureCache(capacity int) *nativeVisionFeatureCache {
	if capacity <= 0 {
		capacity = defaultNativeVisionFeatureCacheEntries
	}
	return &nativeVisionFeatureCache{
		capacity: capacity,
		entries:  make(map[nativeVisionFeatureKey]nativeVisionFeatureEntry, capacity),
	}
}

func (c *nativeVisionFeatureCache) get(key nativeVisionFeatureKey) ([]byte, int, bool) {
	if c == nil {
		return nil, 0, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok {
		return nil, 0, false
	}
	c.touchLocked(key)
	return append([]byte(nil), entry.features...), entry.softTokens, true
}

func (c *nativeVisionFeatureCache) put(key nativeVisionFeatureKey, features []byte, softTokens int) {
	if c == nil || len(features) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := nativeVisionFeatureEntry{features: append([]byte(nil), features...), softTokens: softTokens}
	if _, ok := c.entries[key]; ok {
		c.entries[key] = entry
		c.touchLocked(key)
		return
	}
	if len(c.order) >= c.capacity {
		tail := c.order[len(c.order)-1]
		c.order = c.order[:len(c.order)-1]
		delete(c.entries, tail)
	}
	c.entries[key] = entry
	c.order = append([]nativeVisionFeatureKey{key}, c.order...)
}

func (c *nativeVisionFeatureCache) touchLocked(key nativeVisionFeatureKey) {
	for i, k := range c.order {
		if k != key {
			continue
		}
		if i == 0 {
			return
		}
		copy(c.order[1:i+1], c.order[:i])
		c.order[0] = key
		return
	}
}
