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
	hidden          *Array
}

type cacheSnapshot struct {
	mode            KVCacheMode
	keys            *Array
	values          *Array
	keyScale        *Array
	valueScale      *Array
	keyDtype        DType
	valueDtype      DType
	keyShape        []int32
	valueShape      []int32
	keyBits         int
	valueBits       int
	kPages          []*Array
	vPages          []*Array
	offset          int
	length          int
	step            int
	maxSize         int
	rotating        bool
	storageDType    DType
	hasStorageDType bool
}

// appendArrays appends the snapshot's owned arrays onto out without
// allocating a new slice when out has enough capacity. Used by the
// restore hot path to build a single pre-sized eval slice across N
// snapshots.
func (snapshot cacheSnapshot) appendArrays(out []*Array) []*Array {
	if snapshot.keys != nil {
		out = append(out, snapshot.keys)
	}
	if snapshot.values != nil {
		out = append(out, snapshot.values)
	}
	if snapshot.keyScale != nil {
		out = append(out, snapshot.keyScale)
	}
	if snapshot.valueScale != nil {
		out = append(out, snapshot.valueScale)
	}
	out = append(out, snapshot.kPages...)
	out = append(out, snapshot.vPages...)
	return out
}

// snapshotArrayCount returns the maximum number of arrays the snapshot
// will yield via appendArrays — used to pre-size the eval slice on
// hot-restore paths without speculative growth.
func (snapshot cacheSnapshot) arrayCount() int {
	n := 0
	if snapshot.keys != nil {
		n++
	}
	if snapshot.values != nil {
		n++
	}
	if snapshot.keyScale != nil {
		n++
	}
	if snapshot.valueScale != nil {
		n++
	}
	return n + len(snapshot.kPages) + len(snapshot.vPages)
}

func freeCacheSnapshot(snapshot cacheSnapshot) {
	Free(snapshot.keys, snapshot.values, snapshot.keyScale, snapshot.valueScale)
	Free(snapshot.kPages...)
	Free(snapshot.vPages...)
}

// evalPromptCacheArrays runs Eval on arrays. On failure it re-evals each
// array individually to pinpoint the bad one, using labelAt(i) to render
// the per-item context. labelAt is only invoked on the failure path, so
// the happy path pays zero label-string alloc cost — important on
// Gemma 4 hot-restore where ~100 arrays are eval'd per call.
func evalPromptCacheArrays(scope string, arrays []*Array, labelAt func(i int) string) error {
	if err := Eval(arrays...); err != nil {
		for i, array := range arrays {
			if array == nil || !array.Valid() {
				continue
			}
			if itemErr := Eval(array); itemErr != nil {
				return core.E("prompt cache", scope+" "+labelAt(i), itemErr)
			}
		}
		return core.E("prompt cache", scope, err)
	}
	return nil
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
	if prefixLen == len(tokens) && prefixLen == len(entry.tokens) && (entry.logits == nil || !entry.logits.Valid()) {
		if prefixLen <= 1 {
			return nil, 0
		}
		return entry, prefixLen - 1
	}
	return entry, prefixLen
}

func (m *Model) promptCacheMatchWithHidden(tokens []int32) (*promptCacheEntry, int) {
	entry, prefixLen := m.promptCacheMatch(tokens)
	if entry == nil {
		return nil, 0
	}
	if prefixLen == len(tokens) && (entry.hidden == nil || !entry.hidden.Valid()) {
		if prefixLen <= 1 {
			return nil, 0
		}
		return entry, prefixLen - 1
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

// ClearPromptCache drops the model-owned prompt cache without touching loaded
// weights or adapter state.
func (m *Model) ClearPromptCache() {
	if m == nil {
		return
	}
	release := m.acquirePromptCache()
	defer release()
	m.clearPromptCache()
}

func (entry *promptCacheEntry) free() {
	if entry == nil {
		return
	}
	for _, snapshot := range entry.caches {
		freeCacheSnapshot(snapshot)
	}
	Free(entry.logits)
	Free(entry.hidden)
	entry.tokens = nil
	entry.caches = nil
	entry.logits = nil
	entry.hidden = nil
}

type promptPreparation struct {
	caches          []Cache
	logits          *Array
	hidden          *Array
	duration        time.Duration
	cacheHit        bool
	cacheHitTokens  int
	cacheMissTokens int
	restoreDuration time.Duration
}

const defaultLastTokenPrefillMinTokens = 512

func (m *Model) preparePrompt(ctx context.Context, tokens []int32, cfg GenerateConfig) (promptPreparation, error) {
	start := time.Now()
	requestFixedSize := m.generationFixedGemma4CacheSize(len(tokens), cfg.MaxTokens)
	if entry, prefixLen := m.promptCacheMatch(tokens); entry != nil {
		restoreStart := time.Now()
		caches, logits, err := m.prefillFromPromptCache(ctx, entry, tokens, prefixLen, requestFixedSize)
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

	caches := m.newCachesWithRequestFixedSize(requestFixedSize)
	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err != nil {
		freeCaches(caches)
		return promptPreparation{}, err
	}
	if m.runtimeCachesSnapshotSafe() {
		if err := m.storePromptCache(tokens, caches, logits); err != nil {
			Free(logits)
			freeCaches(caches)
			return promptPreparation{}, err
		}
	}
	return promptPreparation{
		caches:          caches,
		logits:          logits,
		duration:        time.Since(start),
		cacheMissTokens: len(tokens),
	}, nil
}

func (m *Model) runtimeCachesSnapshotSafe() bool {
	switch KVCacheMode(m.cacheMode) {
	case KVCacheModeKQ8VQ4:
		return false
	default:
		return true
	}
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
			if end < len(tokens) && len(caches) > 0 && RuntimeGateEnabled("GO_MLX_ENABLE_CACHE_ONLY_CHUNK_PREFILL") {
				if err := m.prefillTokenBlockCacheOnly(ctx, tokens[start:end], caches); err != nil {
					Free(logits)
					return nil, core.E("Model.Generate", core.Sprintf("prefill chunk %d:%d", start, end), err)
				}
				maybeClearGenerationCache()
				continue
			}
			nextLogits, err := m.prefillTokenBlockOnce(ctx, tokens[start:end], caches)
			if err != nil {
				Free(logits)
				return nil, core.E("Model.Generate", core.Sprintf("prefill chunk %d:%d", start, end), err)
			}
			Free(logits)
			logits = nextLogits
			maybeClearGenerationCache()
		}
		return logits, nil
	}
	logits, err := m.prefillTokenBlockOnce(ctx, tokens, caches)
	if err == nil {
		maybeClearGenerationCache()
	}
	return logits, err
}

func (m *Model) prefillTokenBlockCacheOnly(ctx context.Context, tokens []int32, caches []Cache) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	if len(tokens) == 0 {
		return core.NewError("Model.Generate: empty prefill cache-only block")
	}
	vInput := FromValues(tokens, len(tokens))
	input := Reshape(vInput, 1, int32(len(tokens)))
	logits := m.model.Forward(input, caches)
	Free(vInput, input)
	if logits == nil || !logits.Valid() {
		Free(logits)
		return core.NewError("Model.Generate: cache-only prefill returned nil logits")
	}
	cacheState := prefillCacheStateArrays(caches)
	if len(cacheState) == 0 {
		Free(logits)
		return core.NewError("Model.Generate: cache-only prefill produced no cache state")
	}
	if err := Eval(cacheState...); err != nil {
		Free(logits)
		return core.E("Model.Generate", "cache-only prefill", err)
	}
	Free(logits)
	detachCaches(caches)
	return nil
}

func prefillCacheStateArrays(caches []Cache) []*Array {
	// Pre-size to len(caches)*2 — the common KV case (keys + values per cache).
	// Quantized/paged caches contribute additional state arrays but Go's append
	// only realloc-grows when capacity is exceeded; over-capacity is cheap and
	// the hint matters most on Gemma 4 26-cache fan-outs where the unsized
	// nil-slice growth chain (0→1→2→4→8→16→32→64) dominated allocs.
	//
	// AppendState bypasses the per-cache `[]*Array{k,v}` slice literal that
	// State() returns — on a 26-cache Gemma 4 fan-out that was 27 allocs
	// (one per State()) plus the outer slice; now it's just the outer slice.
	arrays := make([]*Array, 0, len(caches)*2)
	for _, cache := range caches {
		arrays = appendCacheState(arrays, cache)
	}
	return arrays
}

func (m *Model) prefillTokenBlockOnce(ctx context.Context, tokens []int32, caches []Cache) (*Array, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	vInput := FromValues(tokens, len(tokens))
	input := Reshape(vInput, 1, int32(len(tokens)))
	logits, usedLastTokenPath := m.forwardLastTokenLogits(input, nil, caches)
	if logits == nil || !logits.Valid() {
		_ = lastError()
		Free(logits)
		usedLastTokenPath = false
		logits = m.model.Forward(input, caches)
	}
	Free(vInput)
	if logits == nil {
		Free(input)
		return nil, core.NewError("Model.Generate: model forward returned nil logits")
	}
	lastLogits, err := materializeLastTokenLogits(logits)
	if err != nil && usedLastTokenPath {
		fallbackLogits := m.model.Forward(input, caches)
		lastLogits, err = materializeLastTokenLogits(fallbackLogits)
	}
	Free(input)
	if err != nil {
		return nil, core.E("Model.Generate", "prefill", err)
	}
	if err := evalCachesBeforeDetach(caches); err != nil {
		Free(lastLogits)
		return nil, core.E("Model.Generate", "prefill cache state", err)
	}
	detachCaches(caches)
	return lastLogits, nil
}

func evalCachesBeforeDetach(caches []Cache) error {
	state := cacheStateArraysForDetach(caches)
	if len(state) == 0 {
		return nil
	}
	return Eval(state...)
}

func cacheStateArraysForDetach(caches []Cache) []*Array {
	arrays := make([]*Array, 0, len(caches)*2)
	for _, cache := range caches {
		if cache == nil {
			continue
		}
		if _, paged := cache.(*PagedKVCache); paged {
			continue
		}
		arrays = appendCacheState(arrays, cache)
	}
	return arrays
}

func (m *Model) forwardLastTokenLogits(tokens *Array, mask *Array, caches []Cache) (*Array, bool) {
	if m != nil && m.useLastTokenLogitsPrefill(tokens, mask) {
		if lastModel, ok := m.model.(LastTokenLogitsModel); ok {
			return lastModel.ForwardLastTokenLogits(tokens, mask, caches), true
		}
	}
	if mask != nil {
		return m.model.ForwardMasked(tokens, mask, caches), false
	}
	return m.model.Forward(tokens, caches), false
}

func (m *Model) useLastTokenLogitsPrefill(tokens *Array, mask *Array) bool {
	if m == nil {
		return false
	}
	switch core.Lower(core.Trim(core.Env("GO_MLX_ENABLE_LAST_LOGITS_PREFILL"))) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	}
	if mask != nil {
		return false
	}
	if _, ok := m.model.(LastTokenLogitsModel); !ok {
		return false
	}
	seqLen := prefillSequenceLength(tokens)
	minTokens := lastTokenPrefillMinTokens()
	return minTokens > 0 && seqLen >= minTokens
}

func prefillSequenceLength(tokens *Array) int {
	if tokens == nil || !tokens.Valid() {
		return 0
	}
	shape := tokens.Shape()
	switch {
	case len(shape) >= 2:
		return int(shape[1])
	case len(shape) == 1:
		return int(shape[0])
	default:
		return 0
	}
}

func lastTokenPrefillMinTokens() int {
	value := core.Trim(core.Env("GO_MLX_LAST_LOGITS_PREFILL_MIN_TOKENS"))
	if value == "" {
		return defaultLastTokenPrefillMinTokens
	}
	parsed := core.ParseInt(value, 10, 64)
	if !parsed.OK {
		return defaultLastTokenPrefillMinTokens
	}
	return int(parsed.Value.(int64))
}

func (m *Model) prefillFromPromptCache(ctx context.Context, entry *promptCacheEntry, tokens []int32, prefixLen, requestFixedSize int) ([]Cache, *Array, error) {
	caches, err := restorePromptCachesWithRequestFixedSize(entry.caches, prefixLen, requestFixedSize)
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

		vInput := fromSingleInt32(id)
		input := Reshape(vInput, 1, 1)
		oldLogits := logits
		nextLogits := m.model.Forward(input, caches)
		Free(vInput, input, oldLogits)
		logits, err = materializeLastTokenLogits(nextLogits)
		if err != nil {
			freeCaches(caches)
			return nil, nil, core.E("Model.Generate", "prompt cache suffix", err)
		}
		detachCaches(caches)
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

// RestorePromptCacheFromKV installs a captured KV prefix directly into the
// model-owned prompt cache. Prefix snapshots do not need logits; exact prompt
// hits replay only the final token to recover logits.
func (m *Model) RestorePromptCacheFromKV(ctx context.Context, snapshot *KVSnapshot) error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	if !m.promptCacheEnabled {
		return core.NewError("mlx: prompt cache is disabled")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return err
	}
	defer release()
	releasePromptCache := m.acquirePromptCache()
	defer releasePromptCache()

	var restoreErr error
	if deviceErr := m.withDevice(func() {
		entry, err := m.newPromptCacheEntryFromKVSnapshot(snapshot)
		if err == nil {
			m.clearPromptCache()
			m.promptCache = entry
		}
		restoreErr = err
	}); deviceErr != nil {
		return deviceErr
	}
	return restoreErr
}

// RestorePromptCacheFromKVBlocks installs a captured KV prefix from streamed
// contiguous blocks. Paged cache blocks are appended as page arrays, avoiding a
// full-prefix contiguous Metal allocation during restore.
func (m *Model) RestorePromptCacheFromKVBlocks(ctx context.Context, source KVSnapshotBlockSource) error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	if !m.promptCacheEnabled {
		return core.NewError("mlx: prompt cache is disabled")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return err
	}
	defer release()
	releasePromptCache := m.acquirePromptCache()
	defer releasePromptCache()

	var restoreErr error
	if deviceErr := m.withDevice(func() {
		entry, err := m.newPromptCacheEntryFromKVBlocks(ctx, source)
		if err == nil {
			m.clearPromptCache()
			m.promptCache = entry
		}
		restoreErr = err
	}); deviceErr != nil {
		return deviceErr
	}
	return restoreErr
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

func (m *Model) newPromptCacheEntryFromKVSnapshot(snapshot *KVSnapshot) (*promptCacheEntry, error) {
	if err := m.validatePromptCacheKVSnapshot(snapshot); err != nil {
		return nil, err
	}
	templates := m.newCaches()
	defer freeCaches(templates)
	if len(templates) == 0 {
		return nil, core.NewError("mlx: model has no KV caches")
	}
	entry := &promptCacheEntry{
		tokens:          append([]int32(nil), snapshot.Tokens...),
		cacheableTokens: len(snapshot.Tokens),
		adapterHash:     m.adapterCacheKey(),
		caches:          make([]cacheSnapshot, len(templates)),
	}
	populated := make([]bool, len(templates))
	for _, layer := range snapshot.Layers {
		if !kvLayerSnapshotHasState(layer) || layer.CacheIndex < 0 {
			continue
		}
		if layer.CacheIndex >= len(templates) {
			entry.free()
			return nil, core.NewError("mlx: KV snapshot cache index exceeds model cache count")
		}
		if populated[layer.CacheIndex] {
			continue
		}
		cacheSnapshot, err := cacheSnapshotFromKVLayer(snapshot, layer, templates[layer.CacheIndex])
		if err != nil {
			entry.free()
			return nil, err
		}
		entry.caches[layer.CacheIndex] = cacheSnapshot
		populated[layer.CacheIndex] = true
	}
	for i, ok := range populated {
		if !ok {
			entry.free()
			return nil, core.E("Model.RestorePromptCacheFromKV", core.Sprintf("missing cache %d", i), nil)
		}
	}
	totalArrays := 0
	for _, snapshot := range entry.caches {
		totalArrays += snapshot.arrayCount()
	}
	evalArrays := make([]*Array, 0, totalArrays)
	for _, snapshot := range entry.caches {
		evalArrays = snapshot.appendArrays(evalArrays)
	}
	if len(snapshot.Logits) > 0 || len(snapshot.LogitShape) > 0 {
		logits, err := restoreSnapshotLogits(snapshot)
		if err != nil {
			entry.free()
			return nil, err
		}
		entry.logits = logits
	}
	if err := Eval(evalArrays...); err != nil {
		entry.free()
		return nil, core.E("prompt cache", "restore KV snapshot", err)
	}
	Detach(evalArrays...)
	return entry, nil
}

func (m *Model) newPromptCacheEntryFromKVBlocks(ctx context.Context, source KVSnapshotBlockSource) (*promptCacheEntry, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prefixTokens := source.PrefixTokens
	if prefixTokens <= 0 {
		prefixTokens = source.TokenCount
	}
	if prefixTokens <= 0 {
		return nil, core.NewError("mlx: KV block source has no prefix tokens")
	}
	if source.TokenCount > 0 && prefixTokens > source.TokenCount {
		return nil, core.NewError("mlx: KV block prefix exceeds token count")
	}
	if source.BlockCount <= 0 {
		return nil, core.NewError("mlx: KV block source has no blocks")
	}
	if source.Load == nil {
		return nil, core.NewError("mlx: KV block source has no loader")
	}

	templates := m.newCaches()
	defer freeCaches(templates)
	if len(templates) == 0 {
		return nil, core.NewError("mlx: model has no KV caches")
	}
	entry := &promptCacheEntry{
		tokens:          make([]int32, 0, prefixTokens),
		cacheableTokens: prefixTokens,
		adapterHash:     m.adapterCacheKey(),
		caches:          make([]cacheSnapshot, len(templates)),
	}
	populated := make([]bool, len(templates))
	// Hoist populatedInBlock outside the block loop and zero per iteration.
	// Previously this was a per-block make([]bool, len(templates)); on a
	// 26-cache model with N blocks that's N+1 small slice allocs per
	// restore.
	populatedInBlock := make([]bool, len(templates))
	nextStart := 0
	var logitSnapshot *KVSnapshot

	for index := 0; index < source.BlockCount && nextStart < prefixTokens; index++ {
		select {
		case <-ctx.Done():
			entry.free()
			return nil, ctx.Err()
		default:
		}

		block, err := source.Load(ctx, index)
		if err != nil {
			entry.free()
			return nil, err
		}
		if block.Index != index {
			entry.free()
			return nil, core.NewError("mlx: KV block source returned unexpected block index")
		}
		if block.TokenStart != nextStart || block.TokenCount <= 0 {
			entry.free()
			return nil, core.NewError("mlx: KV block source returned non-contiguous blocks")
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			entry.free()
			return nil, core.NewError("mlx: KV block source returned tokens beyond prefix")
		}
		if block.Snapshot == nil || len(block.Snapshot.Tokens) != block.TokenCount {
			entry.free()
			return nil, core.NewError("mlx: KV block snapshot token count mismatch")
		}
		if err := m.validatePromptCacheKVSnapshot(block.Snapshot); err != nil {
			entry.free()
			return nil, err
		}

		clear(populatedInBlock)
		entry.tokens = append(entry.tokens, block.Snapshot.Tokens...)
		for _, layer := range block.Snapshot.Layers {
			if !kvLayerSnapshotHasState(layer) || layer.CacheIndex < 0 {
				continue
			}
			if layer.CacheIndex >= len(templates) {
				entry.free()
				return nil, core.NewError("mlx: KV snapshot cache index exceeds model cache count")
			}
			if populatedInBlock[layer.CacheIndex] {
				continue
			}
			populatedInBlock[layer.CacheIndex] = true
			part, err := cacheSnapshotFromKVLayer(block.Snapshot, layer, templates[layer.CacheIndex])
			if err != nil {
				entry.free()
				return nil, err
			}
			if !populated[layer.CacheIndex] {
				entry.caches[layer.CacheIndex] = part
				populated[layer.CacheIndex] = true
				continue
			}
			if err := appendCacheSnapshotBlock(&entry.caches[layer.CacheIndex], part); err != nil {
				freeCacheSnapshot(part)
				entry.free()
				return nil, err
			}
		}
		if len(block.Snapshot.Logits) > 0 || len(block.Snapshot.LogitShape) > 0 {
			logitSnapshot = block.Snapshot
		}
		nextStart += block.TokenCount
	}

	if nextStart != prefixTokens || len(entry.tokens) != prefixTokens {
		entry.free()
		return nil, core.NewError("mlx: KV block source does not cover requested prefix")
	}
	for i, ok := range populated {
		if !ok {
			entry.free()
			return nil, core.E("Model.RestorePromptCacheFromKVBlocks", core.Sprintf("missing cache %d", i), nil)
		}
	}
	if logitSnapshot != nil {
		logits, err := restoreSnapshotLogits(logitSnapshot)
		if err != nil {
			entry.free()
			return nil, err
		}
		entry.logits = logits
	}

	// Sum exact array count to size in one allocation. Hot path —
	// Gemma 4 26-cache block-source restore yields ~100 arrays; the
	// nil-slice realloc chain was load-bearing alloc cost.
	totalArrays := 0
	for _, snapshot := range entry.caches {
		totalArrays += snapshot.arrayCount()
	}
	if entry.logits != nil {
		totalArrays++
	}
	evalArrays := make([]*Array, 0, totalArrays)
	snapshotOffsets := make([]int, 0, len(entry.caches))
	for _, snapshot := range entry.caches {
		snapshotOffsets = append(snapshotOffsets, len(evalArrays))
		evalArrays = snapshot.appendArrays(evalArrays)
	}
	logitsIdx := -1
	if entry.logits != nil {
		logitsIdx = len(evalArrays)
		evalArrays = append(evalArrays, entry.logits)
	}
	if err := evalPromptCacheArrays("restore KV blocks", evalArrays, func(i int) string {
		if i == logitsIdx {
			return "logits"
		}
		ci := 0
		for ci+1 < len(snapshotOffsets) && snapshotOffsets[ci+1] <= i {
			ci++
		}
		return core.Sprintf("cache[%d].state[%d]", ci, i-snapshotOffsets[ci])
	}); err != nil {
		entry.free()
		return nil, err
	}
	Detach(evalArrays...)
	return entry, nil
}

func appendCacheSnapshotBlock(dst *cacheSnapshot, block cacheSnapshot) error {
	if dst == nil {
		return core.NewError("prompt cache: missing destination cache snapshot")
	}
	if dst.mode != block.mode {
		return core.NewError("prompt cache: cache block mode mismatch")
	}
	dstLen := snapshotCacheLength(*dst)
	blockLen := snapshotCacheLength(block)
	if dstLen <= 0 || blockLen <= 0 {
		return core.NewError("prompt cache: invalid cache block length")
	}
	if dst.mode == KVCacheModePaged {
		if len(block.kPages) == 0 || len(block.kPages) != len(block.vPages) {
			return core.NewError("prompt cache: invalid paged cache block")
		}
		if err := mergeCacheSnapshotStorageDType(dst, block); err != nil {
			return err
		}
		pageSize := dst.step
		if pageSize <= 0 {
			pageSize = block.step
		}
		if pageSize <= 0 {
			pageSize = defaultPagedKVPageSize
		}
		for i := range block.kPages {
			transferred, err := appendPagedCacheSnapshotPage(dst, block.kPages[i], block.vPages[i], pageSize)
			if err != nil {
				return err
			}
			if !transferred {
				Free(block.kPages[i], block.vPages[i])
			}
		}
		dst.length = dstLen + blockLen
		dst.offset = block.offset
		if dst.offset <= 0 {
			dst.offset = dst.length
		}
		if dst.step <= 0 {
			dst.step = block.step
		}
		if dst.maxSize <= 0 {
			dst.maxSize = block.maxSize
		}
		dst.rotating = dst.rotating || block.rotating
		return nil
	}

	leftK, leftV, err := cacheSnapshotFloatArrays(*dst)
	if err != nil {
		return err
	}
	rightK, rightV, err := cacheSnapshotFloatArrays(block)
	if err != nil {
		Free(leftK, leftV)
		return err
	}
	if err := validateCacheSnapshotConcat(leftK, rightK); err != nil {
		Free(leftK, leftV, rightK, rightV)
		return err
	}
	if err := validateCacheSnapshotConcat(leftV, rightV); err != nil {
		Free(leftK, leftV, rightK, rightV)
		return err
	}

	mergedK := Concatenate([]*Array{leftK, rightK}, 2)
	mergedV := Concatenate([]*Array{leftV, rightV}, 2)
	Free(leftK, leftV, rightK, rightV)
	mode := dst.mode
	keyDtype := dst.keyDtype
	valueDtype := dst.valueDtype
	keyBits := dst.keyBits
	valueBits := dst.valueBits
	step := dst.step
	maxSize := dst.maxSize
	rotating := dst.rotating || block.rotating
	offset := block.offset
	freeCacheSnapshot(*dst)

	*dst = cacheSnapshot{
		mode:     mode,
		offset:   offset,
		length:   dstLen + blockLen,
		step:     step,
		maxSize:  maxSize,
		rotating: rotating,
	}
	if dst.offset <= 0 {
		dst.offset = dst.length
	}
	if mode == KVCacheModeQ8 || mode == KVCacheModeKQ8VQ4 {
		if keyBits <= 0 {
			keyBits = 8
		}
		if valueBits <= 0 {
			valueBits = keyBits
		}
		dst.keyDtype = keyDtype
		dst.valueDtype = valueDtype
		dst.keyBits = keyBits
		dst.valueBits = valueBits
		dst.keys, dst.keyScale, dst.keyShape = quantizeCacheArray(mergedK, keyBits)
		dst.values, dst.valueScale, dst.valueShape = quantizeCacheArray(mergedV, valueBits)
		Free(mergedK, mergedV)
		return nil
	}
	dst.keys = mergedK
	dst.values = mergedV
	return nil
}

func mergeCacheSnapshotStorageDType(dst *cacheSnapshot, block cacheSnapshot) error {
	if dst == nil || !block.hasStorageDType {
		return nil
	}
	if dst.hasStorageDType && dst.storageDType != block.storageDType {
		return core.NewError("prompt cache: paged cache block storage dtype mismatch")
	}
	dst.storageDType = block.storageDType
	dst.hasStorageDType = true
	return nil
}

func appendPagedCacheSnapshotPage(dst *cacheSnapshot, keyPage, valuePage *Array, pageSize int) (bool, error) {
	if dst == nil || keyPage == nil || valuePage == nil || !keyPage.Valid() || !valuePage.Valid() {
		return false, core.NewError("prompt cache: invalid paged cache page")
	}
	if len(dst.kPages) != len(dst.vPages) {
		return false, core.NewError("prompt cache: invalid destination paged cache")
	}
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	pageLen := pagedArrayLen(keyPage)
	if pageLen <= 0 || pagedArrayLen(valuePage) != pageLen {
		return false, core.NewError("prompt cache: invalid paged cache page length")
	}
	if len(dst.kPages) > 0 {
		last := len(dst.kPages) - 1
		if err := validateCacheSnapshotConcat(dst.kPages[last], keyPage); err != nil {
			return false, err
		}
		if err := validateCacheSnapshotConcat(dst.vPages[last], valuePage); err != nil {
			return false, err
		}
	}
	if zeroCopyPagedRestoreRuntimeEnabled() {
		dst.kPages = append(dst.kPages, keyPage)
		dst.vPages = append(dst.vPages, valuePage)
		return true, nil
	}

	start := 0
	transferred := false
	for start < pageLen {
		last := len(dst.kPages) - 1
		if last >= 0 {
			room := pageSize - pagedArrayLen(dst.kPages[last])
			if room > 0 {
				take := min(room, pageLen-start)
				appendPagedCacheSnapshotPiece(dst, last, keyPage, valuePage, start, take)
				start += take
				continue
			}
		}
		take := min(pageSize, pageLen-start)
		if start == 0 && take == pageLen {
			dst.kPages = append(dst.kPages, keyPage)
			dst.vPages = append(dst.vPages, valuePage)
			transferred = true
			start += take
			continue
		}
		kPiece, vPiece := slicePagedCacheSnapshotPiece(keyPage, valuePage, start, take)
		dst.kPages = append(dst.kPages, Copy(kPiece))
		dst.vPages = append(dst.vPages, Copy(vPiece))
		Free(kPiece, vPiece)
		start += take
	}
	return transferred, nil
}

func appendPagedCacheSnapshotPiece(dst *cacheSnapshot, last int, keyPage, valuePage *Array, start, take int) {
	kPiece, vPiece := slicePagedCacheSnapshotPiece(keyPage, valuePage, start, take)
	oldK, oldV := dst.kPages[last], dst.vPages[last]
	dst.kPages[last] = Concatenate([]*Array{oldK, kPiece}, 2)
	dst.vPages[last] = Concatenate([]*Array{oldV, vPiece}, 2)
	Free(oldK, oldV, kPiece, vPiece)
}

func slicePagedCacheSnapshotPiece(keyPage, valuePage *Array, start, take int) (*Array, *Array) {
	kShape := keyPage.Shape()
	vShape := valuePage.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		return keyPage.Clone(), valuePage.Clone()
	}
	return Slice(keyPage, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(start + take), kShape[3]}),
		Slice(valuePage, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(start + take), vShape[3]})
}

func cacheSnapshotFloatArrays(snapshot cacheSnapshot) (*Array, *Array, error) {
	switch snapshot.mode {
	case KVCacheModePaged:
		keys, values := concatenatePagedState(snapshot.kPages, snapshot.vPages)
		if keys == nil || values == nil {
			Free(keys, values)
			return nil, nil, core.NewError("prompt cache: invalid paged cache snapshot")
		}
		return keys, values, nil
	case KVCacheModeQ8, KVCacheModeKQ8VQ4:
		if snapshot.keys == nil || snapshot.values == nil || snapshot.keyScale == nil || snapshot.valueScale == nil {
			return nil, nil, core.NewError("prompt cache: invalid quantized cache snapshot")
		}
		keyBits := snapshot.keyBits
		if keyBits <= 0 {
			keyBits = 8
		}
		valueBits := snapshot.valueBits
		if valueBits <= 0 {
			valueBits = keyBits
		}
		return dequantizeCacheArray(snapshot.keys, snapshot.keyScale, snapshot.keyDtype, snapshot.keyShape, keyBits),
			dequantizeCacheArray(snapshot.values, snapshot.valueScale, snapshot.valueDtype, snapshot.valueShape, valueBits), nil
	default:
		if snapshot.keys == nil || snapshot.values == nil {
			return nil, nil, core.NewError("prompt cache: invalid cache snapshot")
		}
		return Copy(snapshot.keys), Copy(snapshot.values), nil
	}
}

func validateCacheSnapshotConcat(left, right *Array) error {
	if left == nil || right == nil || !left.Valid() || !right.Valid() {
		return core.NewError("prompt cache: invalid cache concat arrays")
	}
	leftShape := left.Shape()
	rightShape := right.Shape()
	if len(leftShape) != len(rightShape) {
		return core.NewError("prompt cache: cache block rank mismatch")
	}
	if len(leftShape) < 3 {
		return nil
	}
	for i := range leftShape {
		if i == 2 {
			continue
		}
		if leftShape[i] != rightShape[i] {
			return core.NewError("prompt cache: cache block shape mismatch")
		}
	}
	return nil
}

func (m *Model) validatePromptCacheKVSnapshot(snapshot *KVSnapshot) error {
	if snapshot == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	if snapshot.Version <= 0 || snapshot.Version > KVSnapshotVersion {
		return core.NewError("mlx: unsupported KV snapshot version")
	}
	info := m.Info()
	if snapshot.Architecture != "" && info.Architecture != "" && snapshot.Architecture != info.Architecture {
		return core.NewError("mlx: KV snapshot architecture does not match model")
	}
	if len(snapshot.Tokens) == 0 {
		return core.NewError("mlx: KV snapshot has no tokens")
	}
	seqLen := snapshot.SeqLen
	if seqLen <= 0 {
		seqLen = len(snapshot.Tokens)
	}
	if seqLen <= 0 || len(snapshot.Tokens) != seqLen || snapshot.HeadDim <= 0 {
		return core.NewError("mlx: KV snapshot has invalid tensor dimensions")
	}
	if len(snapshot.Layers) == 0 {
		return core.NewError("mlx: KV snapshot has no layers")
	}
	return nil
}

func newPromptCacheEntry(tokens []int32, caches []Cache, logits *Array) (*promptCacheEntry, error) {
	return newPromptCacheEntryWithHidden(tokens, caches, logits, nil)
}

func newPromptCacheEntryWithHidden(tokens []int32, caches []Cache, logits, hidden *Array) (*promptCacheEntry, error) {
	entry := &promptCacheEntry{
		tokens:          append([]int32(nil), tokens...),
		cacheableTokens: len(tokens),
		caches:          make([]cacheSnapshot, len(caches)),
	}
	// Per-snapshot array offsets — used only on the failure path of
	// evalPromptCacheArrays to map array-index back to (cache, state).
	// Pre-size based on snapshotCache yielding up to ~4 arrays per cache
	// plus the 2 trailing logits/hidden entries; the closure stays cheap
	// on the happy path because labelAt is never invoked.
	evalArrays := make([]*Array, 0, len(caches)*4+2)
	snapshotOffsets := make([]int, 0, len(caches))
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
		snapshotOffsets = append(snapshotOffsets, len(evalArrays))
		evalArrays = snapshot.appendArrays(evalArrays)
	}

	entry.logits = Copy(logits)
	logitsIdx := len(evalArrays)
	evalArrays = append(evalArrays, entry.logits)
	hiddenIdx := -1
	if hidden != nil && hidden.Valid() {
		entry.hidden = Copy(hidden)
		hiddenIdx = len(evalArrays)
		evalArrays = append(evalArrays, entry.hidden)
	}
	if err := evalPromptCacheArrays("snapshot", evalArrays, func(i int) string {
		if i == logitsIdx {
			return "logits"
		}
		if i == hiddenIdx {
			return "hidden"
		}
		// Find the cache index by largest offset <= i.
		ci := 0
		for ci+1 < len(snapshotOffsets) && snapshotOffsets[ci+1] <= i {
			ci++
		}
		return core.Sprintf("cache[%d].state[%d]", ci, i-snapshotOffsets[ci])
	}); err != nil {
		entry.free()
		return nil, err
	}
	Detach(evalArrays...)
	return entry, nil
}

func snapshotCache(cache Cache, tokenLen int) (cacheSnapshot, bool, error) {
	if cache == nil || cache.State() == nil {
		return cacheSnapshot{}, false, nil
	}
	if fixed, ok := cache.(*FixedKVCache); ok {
		return snapshotFixedCache(fixed, tokenLen)
	}
	if paged, ok := cache.(*PagedKVCache); ok {
		restoreLen := min(paged.Len(), tokenLen)
		if restoreLen <= 0 {
			return cacheSnapshot{}, false, nil
		}
		return snapshotPagedCache(paged, restoreLen, paged.Offset())
	}
	if cache.Offset() != cache.Len() || cache.Len() < tokenLen {
		return cacheSnapshot{}, false, nil
	}
	switch c := cache.(type) {
	case *QuantizedKVCache:
		if c.keyBits != 8 || c.valueBits != 8 {
			return cacheSnapshot{}, false, nil
		}
		return snapshotQuantizedCache(c, tokenLen, tokenLen)
	case *PagedKVCache:
		return snapshotPagedCache(c, tokenLen, tokenLen)
	}
	state, ownedState := cacheReadState(cache)
	defer Free(ownedState...)
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
	case *FixedKVCache:
		snapshot.mode = KVCacheModeFixed
		snapshot.maxSize = c.maxSize
	default:
		Free(keys, values)
		return cacheSnapshot{}, false, nil
	}
	return snapshot, true, nil
}

func snapshotFixedCache(cache *FixedKVCache, tokenLen int) (cacheSnapshot, bool, error) {
	if cache == nil || tokenLen <= 0 || cache.Offset() < tokenLen || cache.Len() <= 0 {
		return cacheSnapshot{}, false, nil
	}
	state, ownedState := cacheReadState(cache)
	defer Free(ownedState...)
	if len(state) < 2 || !state[0].Valid() || !state[1].Valid() {
		return cacheSnapshot{}, false, nil
	}
	restoreLen := min(cache.Len(), tokenLen)
	keys, err := copyCachePrefix(state[0], restoreLen)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	values, err := copyCachePrefix(state[1], restoreLen)
	if err != nil {
		Free(keys)
		return cacheSnapshot{}, false, err
	}
	return cacheSnapshot{
		mode:            KVCacheModeFixed,
		keys:            keys,
		values:          values,
		offset:          tokenLen,
		length:          restoreLen,
		maxSize:         cache.maxSize,
		storageDType:    cache.storageDType,
		hasStorageDType: cache.hasStorageDType,
	}, true, nil
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

func snapshotQuantizedCache(cache *QuantizedKVCache, tokenLen, offset int) (cacheSnapshot, bool, error) {
	if cache == nil || cache.keys == nil || cache.values == nil || cache.keyScale == nil || cache.valueScale == nil {
		return cacheSnapshot{}, false, nil
	}
	if tokenLen <= 0 || tokenLen > cache.Len() {
		return cacheSnapshot{}, false, nil
	}
	mode := KVCacheModeQ8
	if cache.keyBits != 8 || cache.valueBits != 8 {
		mode = KVCacheModeKQ8VQ4
	}
	keys, keyShape, err := copyQuantizedCachePrefix(cache.keys, cache.keyShape, tokenLen, cache.keyBits)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	values, valueShape, err := copyQuantizedCachePrefix(cache.values, cache.valueShape, tokenLen, cache.valueBits)
	if err != nil {
		Free(keys)
		return cacheSnapshot{}, false, err
	}
	keyScale := Copy(cache.keyScale)
	valueScale := Copy(cache.valueScale)
	if offset <= 0 {
		offset = tokenLen
	}
	snapshot := cacheSnapshot{
		mode:       mode,
		keys:       keys,
		values:     values,
		keyScale:   keyScale,
		valueScale: valueScale,
		keyDtype:   cache.keyDtype,
		valueDtype: cache.valueDtype,
		keyShape:   keyShape,
		valueShape: valueShape,
		keyBits:    cache.keyBits,
		valueBits:  cache.valueBits,
		offset:     offset,
		length:     tokenLen,
		step:       cache.step,
		maxSize:    cache.maxSize,
		rotating:   cache.maxSize > 0,
	}
	return snapshot, true, nil
}

func copyQuantizedCachePrefix(array *Array, logicalShape []int32, tokenLen, bits int) (*Array, []int32, error) {
	if array == nil || !array.Valid() {
		return nil, nil, core.NewError("prompt cache: invalid quantized cache array")
	}
	shape := append([]int32(nil), logicalShape...)
	if len(shape) == 0 {
		shape = append([]int32(nil), array.Shape()...)
	}
	if bits == 4 {
		if len(shape) >= 3 && int(shape[2]) != tokenLen {
			return nil, nil, core.NewError("prompt cache: q4 prefix slicing is not supported")
		}
		return Copy(array), shape, nil
	}
	copied, err := copyCachePrefix(array, tokenLen)
	if err != nil {
		return nil, nil, err
	}
	if len(shape) >= 3 {
		shape[2] = int32(tokenLen)
	}
	return copied, shape, nil
}

func snapshotPagedCache(cache *PagedKVCache, tokenLen, offset int) (cacheSnapshot, bool, error) {
	if cache == nil || len(cache.kPages) == 0 || len(cache.vPages) == 0 {
		return cacheSnapshot{}, false, nil
	}
	if tokenLen <= 0 || tokenLen > cache.Len() {
		return cacheSnapshot{}, false, nil
	}
	visibleKPages, visibleVPages, ownedVisible := cache.visiblePages()
	defer Free(ownedVisible...)
	kPages, vPages, err := copyPagedCachePrefix(visibleKPages, visibleVPages, tokenLen)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	if offset <= 0 {
		offset = tokenLen
	}
	pageSize := cache.pageSize
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	return cacheSnapshot{
		mode:            KVCacheModePaged,
		kPages:          kPages,
		vPages:          vPages,
		offset:          offset,
		length:          tokenLen,
		step:            pageSize,
		maxSize:         cache.maxSize,
		rotating:        cache.maxSize > 0,
		storageDType:    cache.storageDType,
		hasStorageDType: cache.hasStorageDType,
	}, true, nil
}

func pageCacheArrays(keys, values *Array, pageSize int) ([]*Array, []*Array, bool, error) {
	if keys == nil || values == nil || !keys.Valid() || !values.Valid() {
		return nil, nil, false, core.NewError("prompt cache: invalid page source arrays")
	}
	kShape := keys.Shape()
	vShape := values.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		return []*Array{Copy(keys)}, []*Array{Copy(values)}, false, nil
	}
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	seqLen := int(kShape[2])
	if seqLen != int(vShape[2]) {
		return nil, nil, false, core.NewError("prompt cache: key/value page source length mismatch")
	}
	if seqLen <= pageSize {
		return []*Array{keys}, []*Array{values}, true, nil
	}
	kPages := make([]*Array, 0, (seqLen+pageSize-1)/pageSize)
	vPages := make([]*Array, 0, (seqLen+pageSize-1)/pageSize)
	for start := 0; start < seqLen; start += pageSize {
		end := min(seqLen, start+pageSize)
		kPage := Slice(keys, []int32{0, 0, int32(start), 0}, []int32{kShape[0], kShape[1], int32(end), kShape[3]})
		vPage := Slice(values, []int32{0, 0, int32(start), 0}, []int32{vShape[0], vShape[1], int32(end), vShape[3]})
		kPages = append(kPages, kPage)
		vPages = append(vPages, vPage)
	}
	return kPages, vPages, false, nil
}

func viewPagedCachePrefix(kPages, vPages []*Array, tokenLen int) ([]*Array, []*Array, error) {
	if len(kPages) == 0 || len(kPages) != len(vPages) {
		return nil, nil, core.NewError("prompt cache: invalid paged cache state")
	}
	remaining := tokenLen
	outK := make([]*Array, 0, len(kPages))
	outV := make([]*Array, 0, len(vPages))
	for i := range kPages {
		if remaining <= 0 {
			break
		}
		kPage := kPages[i]
		vPage := vPages[i]
		if kPage == nil || vPage == nil || !kPage.Valid() || !vPage.Valid() {
			Free(outK...)
			Free(outV...)
			return nil, nil, core.NewError("prompt cache: invalid paged cache page")
		}
		pageLen := pagedArrayLen(kPage)
		if pageLen <= 0 {
			Free(outK...)
			Free(outV...)
			return nil, nil, core.NewError("prompt cache: invalid paged cache page length")
		}
		take := min(pageLen, remaining)
		kView, err := viewPagePrefix(kPage, take)
		if err != nil {
			Free(outK...)
			Free(outV...)
			return nil, nil, err
		}
		vView, err := viewPagePrefix(vPage, take)
		if err != nil {
			Free(kView)
			Free(outK...)
			Free(outV...)
			return nil, nil, err
		}
		outK = append(outK, kView)
		outV = append(outV, vView)
		remaining -= take
	}
	if remaining > 0 {
		Free(outK...)
		Free(outV...)
		return nil, nil, core.NewError("prompt cache: paged cache shorter than prefix")
	}
	return outK, outV, nil
}

func viewPagePrefix(page *Array, tokenLen int) (*Array, error) {
	shape := page.Shape()
	if len(shape) < 4 {
		return page.Clone(), nil
	}
	if tokenLen > int(shape[2]) {
		return nil, core.NewError("prompt cache: page shorter than prefix")
	}
	if tokenLen == int(shape[2]) {
		return page.Clone(), nil
	}
	return Slice(page, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(tokenLen), shape[3]}), nil
}

func copyPagedCachePrefix(kPages, vPages []*Array, tokenLen int) ([]*Array, []*Array, error) {
	if len(kPages) == 0 || len(kPages) != len(vPages) {
		return nil, nil, core.NewError("prompt cache: invalid paged cache state")
	}
	remaining := tokenLen
	outK := make([]*Array, 0, len(kPages))
	outV := make([]*Array, 0, len(vPages))
	for i := range kPages {
		if remaining <= 0 {
			break
		}
		kPage := kPages[i]
		vPage := vPages[i]
		if kPage == nil || vPage == nil || !kPage.Valid() || !vPage.Valid() {
			Free(outK...)
			Free(outV...)
			return nil, nil, core.NewError("prompt cache: invalid paged cache page")
		}
		pageLen := pagedArrayLen(kPage)
		if pageLen <= 0 {
			Free(outK...)
			Free(outV...)
			return nil, nil, core.NewError("prompt cache: invalid paged cache page length")
		}
		take := min(pageLen, remaining)
		kCopy, err := copyPagePrefix(kPage, take)
		if err != nil {
			Free(outK...)
			Free(outV...)
			return nil, nil, err
		}
		vCopy, err := copyPagePrefix(vPage, take)
		if err != nil {
			Free(kCopy)
			Free(outK...)
			Free(outV...)
			return nil, nil, err
		}
		outK = append(outK, kCopy)
		outV = append(outV, vCopy)
		remaining -= take
	}
	if remaining > 0 {
		Free(outK...)
		Free(outV...)
		return nil, nil, core.NewError("prompt cache: paged cache shorter than prefix")
	}
	return outK, outV, nil
}

func copyPagePrefix(page *Array, tokenLen int) (*Array, error) {
	shape := page.Shape()
	if len(shape) < 4 {
		return Copy(page), nil
	}
	if tokenLen > int(shape[2]) {
		return nil, core.NewError("prompt cache: page shorter than prefix")
	}
	prefix := page
	if tokenLen != int(shape[2]) {
		prefix = Slice(page, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(tokenLen), shape[3]})
		defer Free(prefix)
	}
	return Copy(prefix), nil
}

func restorePromptCaches(snapshots []cacheSnapshot, prefixLen int) ([]Cache, error) {
	return restorePromptCachesWithRequestFixedSize(snapshots, prefixLen, 0)
}

func restorePromptCachesWithRequestFixedSize(snapshots []cacheSnapshot, prefixLen, requestFixedSize int) ([]Cache, error) {
	caches := make([]Cache, len(snapshots))
	// Pre-size to len(snapshots)*2 — common KV case (keys + values per
	// snapshot). Quantized snapshots contribute up to 4 (keys, values,
	// keyScale, valueScale); paged snapshots vary. The hint defeats the
	// nil-slice realloc chain on Gemma 4 26-snapshot hot-restores —
	// load-bearing for Virgil's hot-load substrate.
	evalArrays := make([]*Array, 0, len(snapshots)*2)
	for i, snapshot := range snapshots {
		restoreLen := snapshotCacheLength(snapshot)
		if restoreLen > prefixLen {
			restoreLen = prefixLen
		}
		if restoreLen <= 0 {
			continue
		}
		if requestFixedSize > 0 || snapshot.mode == KVCacheModeFixed {
			cache, arrays, err := restoreFixedCacheSnapshot(snapshot, restoreLen, prefixLen, requestFixedSize)
			if err != nil {
				freeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = append(evalArrays, arrays...)
			continue
		}
		if snapshot.mode == KVCacheModeQ8 || snapshot.mode == KVCacheModeKQ8VQ4 {
			cache, arrays, err := restoreQuantizedCacheSnapshot(snapshot, restoreLen, prefixLen)
			if err != nil {
				freeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = append(evalArrays, arrays...)
			continue
		}
		if snapshot.mode == KVCacheModePaged {
			cache, arrays, err := restorePagedCacheSnapshot(snapshot, restoreLen, prefixLen)
			if err != nil {
				freeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = append(evalArrays, arrays...)
			continue
		}
		keys, err := copyCachePrefix(snapshot.keys, restoreLen)
		if err != nil {
			freeCaches(caches)
			return nil, err
		}
		values, err := copyCachePrefix(snapshot.values, restoreLen)
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
				idx:     restoreLen,
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

func restoreFixedCacheSnapshot(snapshot cacheSnapshot, prefixLen, offset, requestFixedSize int) (Cache, []*Array, error) {
	if prefixLen <= 0 {
		return nil, nil, core.NewError("prompt cache: invalid fixed prefix length")
	}
	maxSize := requestFixedSize
	if maxSize <= 0 {
		maxSize = snapshot.maxSize
	}
	if fixedGemma4SlidingCacheBoundEnabled() && snapshot.maxSize > 0 {
		maxSize = min(maxSize, snapshot.maxSize)
	}
	if maxSize <= 0 {
		maxSize = prefixLen
	}
	if maxSize < prefixLen {
		return nil, nil, core.NewError("prompt cache: fixed cache capacity is smaller than prefix")
	}

	keys, values, err := cacheSnapshotFloatArrays(snapshot)
	if err != nil {
		return nil, nil, err
	}
	defer Free(keys, values)

	keyPrefix, err := copyCachePrefix(keys, prefixLen)
	if err != nil {
		return nil, nil, err
	}
	valuePrefix, err := copyCachePrefix(values, prefixLen)
	if err != nil {
		Free(keyPrefix)
		return nil, nil, err
	}

	kShape := keyPrefix.Shape()
	vShape := valuePrefix.Shape()
	if len(kShape) < 4 || len(vShape) < 4 {
		Free(keyPrefix, valuePrefix)
		return nil, nil, core.NewError("prompt cache: fixed cache restore requires rank-4 tensors")
	}
	if prefixLen > int(kShape[2]) || prefixLen > int(vShape[2]) {
		Free(keyPrefix, valuePrefix)
		return nil, nil, core.NewError("prompt cache: fixed cache prefix is shorter than requested")
	}
	if offset <= 0 {
		offset = prefixLen
	}

	storageDType, hasStorageDType := restoreCacheStorageDType(snapshot)
	if hasStorageDType {
		keyPrefix = castOwnedCacheArray(keyPrefix, storageDType)
		valuePrefix = castOwnedCacheArray(valuePrefix, storageDType)
	}
	defer Free(keyPrefix, valuePrefix)

	cache := NewFixedKVCache(maxSize)
	if hasStorageDType {
		cache = NewFixedKVCacheWithDType(maxSize, storageDType)
	}
	cache.keys = Zeros([]int32{kShape[0], kShape[1], int32(maxSize), kShape[3]}, keyPrefix.Dtype())
	cache.values = Zeros([]int32{vShape[0], vShape[1], int32(maxSize), vShape[3]}, valuePrefix.Dtype())
	oldK, oldV := cache.keys, cache.values
	cache.keys = SliceUpdateInplace(cache.keys, keyPrefix, []int32{0, 0, 0, 0}, []int32{kShape[0], kShape[1], int32(prefixLen), kShape[3]})
	cache.values = SliceUpdateInplace(cache.values, valuePrefix, []int32{0, 0, 0, 0}, []int32{vShape[0], vShape[1], int32(prefixLen), vShape[3]})
	Free(oldK, oldV)
	cache.offset = offset
	cache.length = prefixLen
	return cache, []*Array{cache.keys, cache.values}, nil
}

func restoreQuantizedCacheSnapshot(snapshot cacheSnapshot, prefixLen, offset int) (Cache, []*Array, error) {
	if prefixLen <= 0 {
		return nil, nil, core.NewError("prompt cache: invalid quantized prefix length")
	}
	keys, keyShape, err := copyQuantizedCachePrefix(snapshot.keys, snapshot.keyShape, prefixLen, snapshot.keyBits)
	if err != nil {
		return nil, nil, err
	}
	values, valueShape, err := copyQuantizedCachePrefix(snapshot.values, snapshot.valueShape, prefixLen, snapshot.valueBits)
	if err != nil {
		Free(keys)
		return nil, nil, err
	}
	keyScale := Copy(snapshot.keyScale)
	valueScale := Copy(snapshot.valueScale)
	if offset <= 0 {
		offset = prefixLen
	}
	step := snapshot.step
	if step <= 0 {
		step = defaultPagedKVPageSize
	}
	keyBits := snapshot.keyBits
	if keyBits <= 0 {
		keyBits = 8
	}
	valueBits := snapshot.valueBits
	if valueBits <= 0 {
		valueBits = keyBits
	}
	cache := &QuantizedKVCache{
		keys:       keys,
		values:     values,
		keyScale:   keyScale,
		valueScale: valueScale,
		keyDtype:   snapshot.keyDtype,
		valueDtype: snapshot.valueDtype,
		keyShape:   keyShape,
		valueShape: valueShape,
		offset:     offset,
		maxSize:    snapshot.maxSize,
		step:       step,
		keyBits:    keyBits,
		valueBits:  valueBits,
	}
	return cache, []*Array{keys, values, keyScale, valueScale}, nil
}

func restorePagedCacheSnapshot(snapshot cacheSnapshot, prefixLen, offset int) (Cache, []*Array, error) {
	if prefixLen <= 0 {
		return nil, nil, core.NewError("prompt cache: invalid paged prefix length")
	}
	kPages, vPages, err := viewPagedCachePrefix(snapshot.kPages, snapshot.vPages, prefixLen)
	if err != nil {
		return nil, nil, err
	}
	if offset <= 0 {
		offset = prefixLen
	}
	pageSize := snapshot.step
	if pageSize <= 0 {
		pageSize = defaultPagedKVPageSize
	}
	storageDType, hasStorageDType := restoreCacheStorageDType(snapshot)
	if hasStorageDType {
		castOwnedCachePages(kPages, vPages, storageDType)
	}
	cache := &PagedKVCache{
		kPages:          kPages,
		vPages:          vPages,
		pageLens:        pagedPageLensForPages(kPages, prefixLen),
		offset:          offset,
		length:          prefixLen,
		maxSize:         snapshot.maxSize,
		pageSize:        pageSize,
		storageDType:    storageDType,
		hasStorageDType: hasStorageDType,
	}
	arrays := make([]*Array, 0, len(kPages)+len(vPages))
	arrays = append(arrays, kPages...)
	arrays = append(arrays, vPages...)
	return cache, arrays, nil
}

func restoreCacheStorageDType(snapshot cacheSnapshot) (DType, bool) {
	if dtype, ok := kvCacheStorageDType(); ok {
		return dtype, true
	}
	if snapshot.hasStorageDType {
		return snapshot.storageDType, true
	}
	return DTypeFloat32, false
}

func castOwnedCacheArray(array *Array, dtype DType) *Array {
	if array == nil || !array.Valid() || DTypeByteSize(dtype) <= 0 || array.Dtype() == dtype {
		return array
	}
	cast := AsType(array, dtype)
	Free(array)
	return cast
}

func castOwnedCachePages(kPages, vPages []*Array, dtype DType) {
	for i := range kPages {
		kPages[i] = castOwnedCacheArray(kPages[i], dtype)
	}
	for i := range vPages {
		vPages[i] = castOwnedCacheArray(vPages[i], dtype)
	}
}
