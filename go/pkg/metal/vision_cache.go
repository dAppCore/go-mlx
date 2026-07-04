// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// The vision-feature LRU (#99): the vision tower runs ONCE per unique image.
// Chat clients replay the whole conversation every turn — the same image
// arrives again and again — and the vision lane skips the prompt cache by
// design, so without this every turn re-pays the SigLIP forward. Features
// are cached post-tower post-projector (the expensive half; pixel decode is
// noise), keyed by content hash.
//
// Safety model: MLX arrays are immutable and Clone() takes an independent
// refcounted handle on the same Metal buffer. The cache owns ONE handle per
// entry; every hit hands the caller a fresh Clone. Eviction frees only the
// cache's handle — a request still holding its clone keeps the buffer alive.
// Zero-copy hits, no use-after-free window.

package metal

import (
	"crypto/sha256"
	"sync"
)

// defaultVisionFeatureCacheEntries bounds the cache. Features are
// softTokens × hidden floats (single-digit MB per image) — eight entries
// covers a multi-image conversation without a meaningful RAM footprint.
const defaultVisionFeatureCacheEntries = 8

type visionFeatureKey = [sha256.Size]byte

func visionFeatureCacheKey(data []byte) visionFeatureKey {
	return sha256.Sum256(data)
}

type visionFeatureEntry struct {
	features   *Array // cache-owned handle
	softTokens int
}

type visionFeatureCache struct {
	mu       sync.Mutex
	capacity int
	entries  map[visionFeatureKey]*visionFeatureEntry
	// order is most-recent-first; tiny capacity makes a slice the right
	// structure.
	order  []visionFeatureKey
	hits   int
	misses int
}

func newVisionFeatureCache(capacity int) *visionFeatureCache {
	if capacity <= 0 {
		capacity = defaultVisionFeatureCacheEntries
	}
	return &visionFeatureCache{
		capacity: capacity,
		entries:  make(map[visionFeatureKey]*visionFeatureEntry, capacity),
	}
}

// get returns a caller-owned Clone of the cached features and the image's
// soft-token count.
func (c *visionFeatureCache) get(key visionFeatureKey) (*Array, int, bool) {
	if c == nil {
		return nil, 0, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, 0, false
	}
	c.hits++
	c.touchLocked(key)
	return entry.features.Clone(), entry.softTokens, true
}

// put stores features under key, TAKING OWNERSHIP of the handle (callers
// keep their own working handle and pass a Clone). Replacing or evicting
// frees only the cache-owned handle.
func (c *visionFeatureCache) put(key visionFeatureKey, features *Array, softTokens int) {
	if c == nil || features == nil || !features.Valid() {
		Free(features)
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if existing, ok := c.entries[key]; ok {
		Free(existing.features)
		existing.features = features
		existing.softTokens = softTokens
		c.touchLocked(key)
		return
	}
	if len(c.order) >= c.capacity {
		tail := c.order[len(c.order)-1]
		c.order = c.order[:len(c.order)-1]
		if evicted, ok := c.entries[tail]; ok {
			Free(evicted.features)
			delete(c.entries, tail)
		}
	}
	c.entries[key] = &visionFeatureEntry{features: features, softTokens: softTokens}
	c.order = append([]visionFeatureKey{key}, c.order...)
}

func (c *visionFeatureCache) touchLocked(key visionFeatureKey) {
	for i, k := range c.order {
		if k == key {
			if i == 0 {
				return
			}
			copy(c.order[1:i+1], c.order[:i])
			c.order[0] = key
			return
		}
	}
}

// stats reports lifetime hit/miss counters.
func (c *visionFeatureCache) stats() (hits, misses int) {
	if c == nil {
		return 0, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

// freeAll releases every cache-owned handle. Model.Close calls it.
func (c *visionFeatureCache) freeAll() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, entry := range c.entries {
		Free(entry.features)
	}
	c.entries = make(map[visionFeatureKey]*visionFeatureEntry, c.capacity)
	c.order = c.order[:0]
}

// visionFeatureCacheLazy returns the model's cache, creating it on first use.
func (m *Model) visionFeatureCacheLazy() *visionFeatureCache {
	m.visionCacheMu.Lock()
	defer m.visionCacheMu.Unlock()
	if m.visionCache == nil {
		m.visionCache = newVisionFeatureCache(0)
	}
	return m.visionCache
}

func (m *Model) closeVisionFeatureCache() {
	m.visionCacheMu.Lock()
	defer m.visionCacheMu.Unlock()
	m.visionCache.freeAll()
	m.visionCache = nil
}
