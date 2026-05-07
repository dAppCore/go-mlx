// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"time"

	"dappco.re/go"
)

type promptCacheEntry struct {
	tokens          []int32
	cacheableTokens int
	adapterHash     string
	caches          []cacheSnapshot
	logits          *Array
}

type cacheSnapshot struct {
	keys     *Array
	values   *Array
	offset   int
	length   int
	step     int
	maxSize  int
	rotating bool
}

func longestTokenPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (m *Model) acquirePromptCache() func() {
	if m == nil || !m.promptCacheEnabled {
		return func() {}
	}
	m.promptCacheMu.Lock()
	return m.promptCacheMu.Unlock
}

func (m *Model) promptCacheMinimum() int {
	if m == nil || m.promptCacheMinTokens <= 0 {
		return DefaultPromptCacheMinTokens
	}
	return m.promptCacheMinTokens
}

func (m *Model) promptCacheMatch(tokens []int32) (*promptCacheEntry, int) {
	if m == nil || !m.promptCacheEnabled || m.promptCache == nil {
		return nil, 0
	}
	entry := m.promptCache
	if entry.adapterHash != m.adapterCacheKey() {
		return nil, 0
	}
	prefixLen := longestTokenPrefix(tokens, entry.tokens)
	if prefixLen < m.promptCacheMinimum() || prefixLen > entry.cacheableTokens {
		return nil, 0
	}
	if prefixLen == len(tokens) && prefixLen != len(entry.tokens) {
		return nil, 0
	}
	return entry, prefixLen
}

func (m *Model) clearPromptCache() {
	if m == nil || m.promptCache == nil {
		return
	}
	m.promptCache.free()
	m.promptCache = nil
}

func (entry *promptCacheEntry) free() {
	if entry == nil {
		return
	}
	for _, snapshot := range entry.caches {
		Free(snapshot.keys, snapshot.values)
	}
	Free(entry.logits)
	entry.tokens = nil
	entry.caches = nil
	entry.logits = nil
}

type promptPreparation struct {
	caches          []Cache
	logits          *Array
	duration        time.Duration
	cacheHit        bool
	cacheHitTokens  int
	cacheMissTokens int
	restoreDuration time.Duration
}

func (m *Model) preparePrompt(ctx context.Context, tokens []int32) (promptPreparation, error) {
	start := time.Now()
	if entry, prefixLen := m.promptCacheMatch(tokens); entry != nil {
		restoreStart := time.Now()
		caches, logits, err := m.prefillFromPromptCache(ctx, entry, tokens, prefixLen)
		restoreDuration := time.Since(restoreStart)
		return promptPreparation{
			caches:          caches,
			logits:          logits,
			duration:        time.Since(start),
			cacheHit:        err == nil,
			cacheHitTokens:  prefixLen,
			cacheMissTokens: max(0, len(tokens)-prefixLen),
			restoreDuration: restoreDuration,
		}, err
	}

	caches := m.newCaches()
	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err != nil {
		freeCaches(caches)
		return promptPreparation{}, err
	}
	if err := m.storePromptCache(tokens, caches, logits); err != nil {
		Free(logits)
		freeCaches(caches)
		return promptPreparation{}, err
	}
	return promptPreparation{
		caches:          caches,
		logits:          logits,
		duration:        time.Since(start),
		cacheMissTokens: len(tokens),
	}, nil
}

func (m *Model) prefillTokenBlock(ctx context.Context, tokens []int32, caches []Cache) (*Array, error) {
	if len(tokens) == 0 {
		return nil, core.NewError("Model.Generate: empty prompt after tokenisation")
	}
	chunkSize := m.prefillChunkSize
	if chunkSize > 0 && len(tokens) > chunkSize {
		var logits *Array
		for start := 0; start < len(tokens); start += chunkSize {
			end := start + chunkSize
			if end > len(tokens) {
				end = len(tokens)
			}
			nextLogits, err := m.prefillTokenBlockOnce(ctx, tokens[start:end], caches)
			if err != nil {
				Free(logits)
				return nil, err
			}
			Free(logits)
			logits = nextLogits
		}
		return logits, nil
	}
	return m.prefillTokenBlockOnce(ctx, tokens, caches)
}

func (m *Model) prefillTokenBlockOnce(ctx context.Context, tokens []int32, caches []Cache) (*Array, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	vInput := FromValues(tokens, len(tokens))
	input := Reshape(vInput, 1, int32(len(tokens)))
	logits := m.model.Forward(input, caches)
	Free(vInput, input)

	if err := Eval(logits); err != nil {
		Free(logits)
		return nil, core.E("Model.Generate", "prefill", err)
	}
	detachEvalState(logits, caches)
	return logits, nil
}

func (m *Model) prefillFromPromptCache(ctx context.Context, entry *promptCacheEntry, tokens []int32, prefixLen int) ([]Cache, *Array, error) {
	caches, err := restorePromptCaches(entry.caches, prefixLen)
	if err != nil {
		return nil, nil, err
	}

	if prefixLen == len(tokens) && prefixLen == len(entry.tokens) {
		logits := Copy(entry.logits)
		if err := Eval(logits); err != nil {
			Free(logits)
			freeCaches(caches)
			return nil, nil, core.E("Model.Generate", "restore prompt logits", err)
		}
		Detach(logits)
		return caches, logits, nil
	}

	var logits *Array
	for _, id := range tokens[prefixLen:] {
		select {
		case <-ctx.Done():
			Free(logits)
			freeCaches(caches)
			return nil, nil, ctx.Err()
		default:
		}

		vInput := FromValues([]int32{id}, 1)
		input := Reshape(vInput, 1, 1)
		oldLogits := logits
		logits = m.model.Forward(input, caches)
		Free(vInput, input, oldLogits)
		if err := Eval(logits); err != nil {
			Free(logits)
			freeCaches(caches)
			return nil, nil, core.E("Model.Generate", "prompt cache suffix", err)
		}
		detachEvalState(logits, caches)
	}
	if logits == nil {
		freeCaches(caches)
		return nil, nil, core.NewError("Model.Generate: prompt cache hit had no suffix logits")
	}
	return caches, logits, nil
}

func (m *Model) storePromptCache(tokens []int32, caches []Cache, logits *Array) error {
	if m == nil || !m.promptCacheEnabled || len(tokens) < m.promptCacheMinimum() {
		return nil
	}
	entry, err := newPromptCacheEntry(tokens, caches, logits)
	if err != nil {
		return err
	}
	if entry == nil {
		return nil
	}
	entry.adapterHash = m.adapterCacheKey()
	m.clearPromptCache()
	m.promptCache = entry
	return nil
}

func (m *Model) adapterCacheKey() string {
	if m == nil {
		return ""
	}
	if m.adapterInfo.Hash != "" {
		return m.adapterInfo.Hash
	}
	if m.adapter != nil {
		return adapterInfoFromLoRA("", m.adapter).Hash
	}
	return ""
}

func newPromptCacheEntry(tokens []int32, caches []Cache, logits *Array) (*promptCacheEntry, error) {
	entry := &promptCacheEntry{
		tokens:          append([]int32(nil), tokens...),
		cacheableTokens: len(tokens),
		caches:          make([]cacheSnapshot, len(caches)),
	}
	var evalArrays []*Array
	for i, cache := range caches {
		snapshot, ok, err := snapshotCache(cache, len(tokens))
		if err != nil {
			entry.free()
			return nil, err
		}
		if !ok {
			entry.free()
			return nil, nil
		}
		entry.caches[i] = snapshot
		entry.cacheableTokens = min(entry.cacheableTokens, snapshot.offset)
		evalArrays = append(evalArrays, snapshot.keys, snapshot.values)
	}

	entry.logits = Copy(logits)
	evalArrays = append(evalArrays, entry.logits)
	if err := Eval(evalArrays...); err != nil {
		entry.free()
		return nil, core.E("prompt cache", "snapshot", err)
	}
	Detach(evalArrays...)
	return entry, nil
}

func snapshotCache(cache Cache, tokenLen int) (cacheSnapshot, bool, error) {
	if cache == nil || cache.State() == nil {
		return cacheSnapshot{}, false, nil
	}
	if cache.Offset() != cache.Len() || cache.Len() < tokenLen {
		return cacheSnapshot{}, false, nil
	}
	state := cache.State()
	if len(state) < 2 || !state[0].Valid() || !state[1].Valid() {
		return cacheSnapshot{}, false, nil
	}

	keys, err := copyCachePrefix(state[0], tokenLen)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	values, err := copyCachePrefix(state[1], tokenLen)
	if err != nil {
		Free(keys)
		return cacheSnapshot{}, false, err
	}

	snapshot := cacheSnapshot{
		keys:   keys,
		values: values,
		offset: tokenLen,
		length: tokenLen,
	}
	switch c := cache.(type) {
	case *RotatingKVCache:
		snapshot.rotating = true
		snapshot.maxSize = c.maxSize
		snapshot.step = c.step
	case *KVCache:
		snapshot.step = c.step
	default:
		Free(keys, values)
		return cacheSnapshot{}, false, nil
	}
	return snapshot, true, nil
}

func copyCachePrefix(array *Array, tokenLen int) (*Array, error) {
	if array == nil || !array.Valid() {
		return nil, core.NewError("prompt cache: invalid cache array")
	}
	shape := array.Shape()
	if len(shape) < 4 {
		return Copy(array), nil
	}
	if int(shape[2]) < tokenLen {
		return nil, core.NewError("prompt cache: cache shorter than prefix")
	}
	prefix := array
	if int(shape[2]) != tokenLen {
		prefix = Slice(array, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(tokenLen), shape[3]})
		defer Free(prefix)
	}
	return Copy(prefix), nil
}

func restorePromptCaches(snapshots []cacheSnapshot, prefixLen int) ([]Cache, error) {
	caches := make([]Cache, len(snapshots))
	var evalArrays []*Array
	for i, snapshot := range snapshots {
		keys, err := copyCachePrefix(snapshot.keys, prefixLen)
		if err != nil {
			freeCaches(caches)
			return nil, err
		}
		values, err := copyCachePrefix(snapshot.values, prefixLen)
		if err != nil {
			Free(keys)
			freeCaches(caches)
			return nil, err
		}
		evalArrays = append(evalArrays, keys, values)
		if snapshot.rotating {
			caches[i] = &RotatingKVCache{
				keys:    keys,
				values:  values,
				offset:  prefixLen,
				maxSize: snapshot.maxSize,
				step:    snapshot.step,
				idx:     prefixLen,
			}
			continue
		}
		caches[i] = &KVCache{
			keys:   keys,
			values: values,
			offset: prefixLen,
			step:   snapshot.step,
		}
	}
	if err := Eval(evalArrays...); err != nil {
		freeCaches(caches)
		return nil, core.E("prompt cache", "restore", err)
	}
	Detach(evalArrays...)
	return caches, nil
}
