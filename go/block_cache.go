// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
)

const (
	// DefaultCacheBlockSize is the token chunk size used for portable block
	// prefix identities when callers do not choose a size.
	DefaultCacheBlockSize = 128

	// BlockCacheDiskPathEnv enables disk-backed block metadata for loaded
	// inference adapters without adding provider/runtime dependencies.
	BlockCacheDiskPathEnv = "GO_MLX_BLOCK_CACHE_PATH"

	blockCacheMode        = "block-prefix"
	blockCacheDiskVersion = 1
)

// BlockCacheConfig configures the block-prefix cache metadata layer.
type BlockCacheConfig struct {
	BlockSize     int
	ModelHash     string
	AdapterHash   string
	TokenizerHash string
	Tokenize      func(prompt string) ([]int32, error)
	WarmPrompt    func(ctx context.Context, prompt string) error
	ClearRuntime  func()
	DiskPath      string
	MemvidStore   memvid.Writer
}

// BlockCacheService exposes stable block-prefix refs through
// inference.CacheService. It records block identities in memory, optionally
// persists them on disk, and delegates actual KV warming to the native prompt
// cache when a prompt warmer is configured.
type BlockCacheService struct {
	mu          sync.Mutex
	cfg         BlockCacheConfig
	blocks      map[string]inference.CacheBlockRef
	hits        uint64
	misses      uint64
	cleared     uint64
	evictions   uint64
	diskCorrupt uint64
	diskLoaded  bool
}

type blockCacheDiskRecord struct {
	Version   int                     `json:"version"`
	Ref       inference.CacheBlockRef `json:"ref"`
	Tokens    []int32                 `json:"tokens,omitempty"`
	MemvidRef *memvid.ChunkRef        `json:"memvid_ref,omitempty"`
}

type blockCacheMemvidPayload struct {
	Version       int                     `json:"version"`
	BlockID       string                  `json:"block_id"`
	Ref           inference.CacheBlockRef `json:"ref"`
	Tokens        []int32                 `json:"tokens,omitempty"`
	Encoding      string                  `json:"encoding,omitempty"`
	CacheMode     string                  `json:"cache_mode,omitempty"`
	PayloadFormat string                  `json:"payload_format,omitempty"`
}

// NewBlockCacheService returns a cache metadata service with stable prefix refs.
func NewBlockCacheService(cfg BlockCacheConfig) *BlockCacheService {
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = DefaultCacheBlockSize
	}
	return &BlockCacheService{
		cfg:    cfg,
		blocks: map[string]inference.CacheBlockRef{},
	}
}

// DefaultBlockCacheDiskPath returns the process-level opt-in path for
// persistent block-prefix metadata.
func DefaultBlockCacheDiskPath() string {
	return core.Trim(core.Env(BlockCacheDiskPathEnv))
}

// CacheStats reports in-memory block metadata and cumulative warm hit/miss
// counters.
func (service *BlockCacheService) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if err := cacheContextErr(ctx); err != nil {
		return inference.CacheStats{}, err
	}
	if service == nil {
		return inference.CacheStats{}, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheStats{}, err
	}
	return service.statsLocked(), nil
}

// CacheEntries returns stable cache block refs, optionally filtered by labels.
func (service *BlockCacheService) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	if err := cacheContextErr(ctx); err != nil {
		return nil, err
	}
	if service == nil {
		return nil, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return nil, err
	}
	entries := make([]inference.CacheBlockRef, 0, len(service.blocks))
	for _, ref := range service.blocks {
		if len(labels) > 0 && !blockRefMatchesLabels(ref, labels) {
			continue
		}
		entries = append(entries, cloneCacheBlockRef(ref))
	}
	sortCacheBlockRefs(entries)
	return entries, nil
}

// WarmCache creates stable block refs for the request and optionally warms the
// native prompt cache when a prompt and warmer are present.
func (service *BlockCacheService) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if err := cacheContextErr(ctx); err != nil {
		return inference.CacheWarmResult{}, err
	}
	if service == nil {
		return inference.CacheWarmResult{}, core.NewError("mlx: block cache service is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	tokens, err := service.requestTokens(req)
	if err != nil {
		return inference.CacheWarmResult{}, err
	}
	if len(tokens) == 0 {
		return inference.CacheWarmResult{}, core.NewError("mlx: cache warm requires prompt or tokens")
	}
	if service.cfg.WarmPrompt != nil && core.Trim(req.Prompt) != "" {
		if err := service.cfg.WarmPrompt(ctx, req.Prompt); err != nil {
			return inference.CacheWarmResult{}, err
		}
	}

	labels := service.compatibilityLabels(req)
	refs := service.blockRefs(req, tokens, labels)
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheWarmResult{}, err
	}
	for i, ref := range refs {
		if _, ok := service.blocks[ref.ID]; ok {
			service.hits++
			continue
		}
		service.misses++
		storedRef, err := service.writeDiskBlockLocked(ctx, ref, tokens[:ref.TokenStart+ref.TokenCount])
		if err != nil {
			return inference.CacheWarmResult{}, err
		}
		refs[i] = storedRef
		service.blocks[ref.ID] = storedRef
	}
	return inference.CacheWarmResult{
		Blocks: refs,
		Stats:  service.statsLocked(),
		Labels: labels,
	}, nil
}

// ClearCache clears all refs, or only refs whose metadata matches labels.
func (service *BlockCacheService) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	if err := cacheContextErr(ctx); err != nil {
		return inference.CacheStats{}, err
	}
	if service == nil {
		return inference.CacheStats{}, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheStats{}, err
	}
	if len(labels) == 0 {
		service.blocks = map[string]inference.CacheBlockRef{}
		service.hits = 0
		service.misses = 0
		service.cleared++
		if err := service.clearDiskLocked(); err != nil {
			return inference.CacheStats{}, err
		}
		if service.cfg.ClearRuntime != nil {
			service.cfg.ClearRuntime()
		}
		return service.statsLocked(), nil
	}
	for id, ref := range service.blocks {
		if blockRefMatchesLabels(ref, labels) {
			if err := service.removeDiskBlockLocked(ref.ID); err != nil {
				return inference.CacheStats{}, err
			}
			delete(service.blocks, id)
			service.cleared++
		}
	}
	return service.statsLocked(), nil
}

func (service *BlockCacheService) requestTokens(req inference.CacheWarmRequest) ([]int32, error) {
	if len(req.Tokens) > 0 {
		return append([]int32(nil), req.Tokens...), nil
	}
	if core.Trim(req.Prompt) == "" {
		return nil, nil
	}
	if service.cfg.Tokenize == nil {
		return nil, core.NewError("mlx: cache warm prompt requires tokenizer")
	}
	tokens, err := service.cfg.Tokenize(req.Prompt)
	if err != nil {
		return nil, err
	}
	return append([]int32(nil), tokens...), nil
}

func (service *BlockCacheService) blockRefs(req inference.CacheWarmRequest, tokens []int32, labels map[string]string) []inference.CacheBlockRef {
	blockSize := service.cfg.BlockSize
	if blockSize <= 0 {
		blockSize = DefaultCacheBlockSize
	}
	modelHash := firstNonEmptyString(service.cfg.ModelHash, req.Model.Hash, req.Model.ID)
	adapterHash := firstNonEmptyString(service.cfg.AdapterHash, req.Adapter.Hash)
	tokenizerHash := firstNonEmptyString(service.cfg.TokenizerHash, req.Labels["tokenizer_hash"])
	refs := make([]inference.CacheBlockRef, 0, (len(tokens)+blockSize-1)/blockSize)
	for start := 0; start < len(tokens); start += blockSize {
		end := start + blockSize
		if end > len(tokens) {
			end = len(tokens)
		}
		refLabels := cloneBlockCacheLabels(labels)
		refLabels["block_index"] = core.Sprintf("%d", len(refs))
		refLabels["prefix_tokens"] = core.Sprintf("%d", end)
		ref := inference.CacheBlockRef{
			ID:            blockCacheID(modelHash, adapterHash, tokenizerHash, req.Mode, tokens[:end]),
			Kind:          "prefix",
			ModelHash:     modelHash,
			AdapterHash:   adapterHash,
			TokenizerHash: tokenizerHash,
			TokenStart:    start,
			TokenCount:    end - start,
			SizeBytes:     uint64(end-start) * 4,
			Encoding:      "token-prefix/int32",
			Labels:        refLabels,
		}
		ref = service.withDiskLabels(ref)
		refs = append(refs, ref)
	}
	return refs
}

func (service *BlockCacheService) compatibilityLabels(req inference.CacheWarmRequest) map[string]string {
	labels := cloneBlockCacheLabels(req.Labels)
	labels["cache_mode"] = blockCacheMode
	labels["block_size"] = core.Sprintf("%d", service.cfg.BlockSize)
	labels["model_match"] = boolLabel(cacheIdentityMatches(service.cfg.ModelHash, firstNonEmptyString(req.Model.Hash, req.Model.ID)))
	labels["adapter_match"] = boolLabel(cacheIdentityMatches(service.cfg.AdapterHash, req.Adapter.Hash))
	labels["tokenizer_match"] = boolLabel(cacheIdentityMatches(service.cfg.TokenizerHash, req.Labels["tokenizer_hash"]))
	return labels
}

func (service *BlockCacheService) statsLocked() inference.CacheStats {
	stats := inference.CacheStats{
		Blocks:    len(service.blocks),
		Hits:      service.hits,
		Misses:    service.misses,
		Evictions: service.evictions,
		CacheMode: blockCacheMode,
		Labels: map[string]string{
			"block_size": core.Sprintf("%d", service.cfg.BlockSize),
			"cleared":    core.Sprintf("%d", service.cleared),
		},
	}
	if service.diskEnabled() {
		stats.DiskBytes = service.diskBytesLocked()
		stats.Labels["disk_path"] = service.cfg.DiskPath
		stats.Labels["disk_blocks"] = core.Sprintf("%d", len(core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json"))))
		stats.Labels["disk_corrupt"] = core.Sprintf("%d", service.diskCorrupt)
	}
	if service.memvidEnabled() {
		stats.Labels["cold_store"] = "memvid"
	}
	for _, ref := range service.blocks {
		stats.MemoryBytes += ref.SizeBytes
	}
	total := service.hits + service.misses
	if total > 0 {
		stats.HitRate = float64(service.hits) / float64(total)
	}
	return stats
}

func (service *BlockCacheService) diskEnabled() bool {
	return service != nil && core.Trim(service.cfg.DiskPath) != ""
}

func (service *BlockCacheService) memvidEnabled() bool {
	return service != nil && service.cfg.MemvidStore != nil
}

func (service *BlockCacheService) withDiskLabels(ref inference.CacheBlockRef) inference.CacheBlockRef {
	if !service.diskEnabled() || ref.ID == "" {
		return ref
	}
	labels := cloneBlockCacheLabels(ref.Labels)
	labels["disk"] = "true"
	labels["disk_path"] = service.diskBlockPath(ref.ID)
	ref.Labels = labels
	return ref
}

func (service *BlockCacheService) ensureDiskLoadedLocked() error {
	if !service.diskEnabled() || service.diskLoaded {
		return nil
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return core.E("BlockCacheService.ensureDiskLoaded", "create disk cache directory", blockCacheResultError(result))
	}
	for _, path := range core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json")) {
		record, ok := service.readDiskRecord(path)
		if !ok {
			service.quarantineDiskBlock(path)
			continue
		}
		if !service.diskRecordCompatible(record) {
			continue
		}
		ref := service.withDiskLabels(record.Ref)
		if record.MemvidRef != nil {
			ref = withMemvidLabels(ref, *record.MemvidRef)
		}
		service.blocks[record.Ref.ID] = ref
	}
	service.diskLoaded = true
	return nil
}

func (service *BlockCacheService) readDiskRecord(path string) (blockCacheDiskRecord, bool) {
	read := core.ReadFile(path)
	if !read.OK {
		return blockCacheDiskRecord{}, false
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return blockCacheDiskRecord{}, false
	}
	var record blockCacheDiskRecord
	result := core.JSONUnmarshal(data, &record)
	if !result.OK || record.Version != blockCacheDiskVersion || record.Ref.ID == "" {
		return blockCacheDiskRecord{}, false
	}
	return record, true
}

func (service *BlockCacheService) diskRecordCompatible(record blockCacheDiskRecord) bool {
	if record.Ref.ID == "" {
		return false
	}
	if !cacheIdentityMatches(service.cfg.ModelHash, record.Ref.ModelHash) {
		return false
	}
	if !cacheIdentityMatches(service.cfg.AdapterHash, record.Ref.AdapterHash) {
		return false
	}
	return cacheIdentityMatches(service.cfg.TokenizerHash, record.Ref.TokenizerHash)
}

func (service *BlockCacheService) writeDiskBlockLocked(ctx context.Context, ref inference.CacheBlockRef, tokens []int32) (inference.CacheBlockRef, error) {
	if !service.diskEnabled() {
		return ref, nil
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return inference.CacheBlockRef{}, core.E("BlockCacheService.writeDiskBlock", "create disk cache directory", blockCacheResultError(result))
	}
	var memvidRef *memvid.ChunkRef
	if service.memvidEnabled() {
		written, err := service.writeMemvidBlock(ctx, ref, tokens)
		if err != nil {
			return inference.CacheBlockRef{}, err
		}
		memvidRef = &written
		ref = withMemvidLabels(ref, written)
	}
	record := blockCacheDiskRecord{
		Version:   blockCacheDiskVersion,
		Ref:       service.withDiskLabels(ref),
		MemvidRef: memvidRef,
	}
	if memvidRef == nil {
		record.Tokens = append([]int32(nil), tokens...)
	}
	data := core.JSONMarshal(record)
	if !data.OK {
		return inference.CacheBlockRef{}, core.E("BlockCacheService.writeDiskBlock", "marshal disk cache record", blockCacheResultError(data))
	}
	write := core.WriteFile(service.diskBlockPath(ref.ID), data.Value.([]byte), 0o600)
	if !write.OK {
		return inference.CacheBlockRef{}, core.E("BlockCacheService.writeDiskBlock", "write disk cache record", blockCacheResultError(write))
	}
	return record.Ref, nil
}

func (service *BlockCacheService) writeMemvidBlock(ctx context.Context, ref inference.CacheBlockRef, tokens []int32) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if service == nil || service.cfg.MemvidStore == nil {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid store is nil")
	}
	payload := blockCacheMemvidPayload{
		Version:       blockCacheDiskVersion,
		BlockID:       ref.ID,
		Ref:           ref,
		Tokens:        append([]int32(nil), tokens...),
		Encoding:      ref.Encoding,
		CacheMode:     blockCacheMode,
		PayloadFormat: "token-prefix/int32-json",
	}
	chunk, err := service.cfg.MemvidStore.Put(ctx, core.JSONMarshalString(payload), memvid.PutOptions{
		URI:   "mlx://cache/block/" + ref.ID,
		Title: "go-mlx block cache " + ref.ID,
		Kind:  "kv-block-prefix",
		Track: blockCacheMode,
		Tags: map[string]string{
			"block_id":       ref.ID,
			"model_hash":     ref.ModelHash,
			"adapter_hash":   ref.AdapterHash,
			"tokenizer_hash": ref.TokenizerHash,
			"encoding":       ref.Encoding,
		},
		Labels: []string{"go-mlx", "block-cache", blockCacheMode},
	})
	if err != nil {
		return memvid.ChunkRef{}, core.E("BlockCacheService.writeMemvidBlock", "write memvid payload", err)
	}
	return chunk, nil
}

func withMemvidLabels(ref inference.CacheBlockRef, chunk memvid.ChunkRef) inference.CacheBlockRef {
	labels := cloneBlockCacheLabels(ref.Labels)
	labels["cold_store"] = "memvid"
	labels["memvid_chunk_id"] = core.Itoa(chunk.ChunkID)
	if chunk.Codec != "" {
		labels["memvid_codec"] = chunk.Codec
	}
	if chunk.Segment != "" {
		labels["memvid_segment"] = chunk.Segment
	}
	if chunk.HasFrameOffset {
		labels["memvid_frame_offset"] = core.FormatUint(chunk.FrameOffset, 10)
	}
	ref.Labels = labels
	return ref
}

func (service *BlockCacheService) clearDiskLocked() error {
	if !service.diskEnabled() {
		return nil
	}
	if result := core.RemoveAll(service.cfg.DiskPath); !result.OK {
		return core.E("BlockCacheService.clearDisk", "remove disk cache directory", blockCacheResultError(result))
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return core.E("BlockCacheService.clearDisk", "recreate disk cache directory", blockCacheResultError(result))
	}
	return nil
}

func (service *BlockCacheService) removeDiskBlockLocked(id string) error {
	if !service.diskEnabled() || id == "" {
		return nil
	}
	result := core.Remove(service.diskBlockPath(id))
	if result.OK {
		return nil
	}
	err := blockCacheResultError(result)
	if err != nil && core.IsNotExist(err) {
		return nil
	}
	return core.E("BlockCacheService.removeDiskBlock", "remove disk cache record", err)
}

func (service *BlockCacheService) quarantineDiskBlock(path string) {
	service.evictions++
	service.diskCorrupt++
	_ = core.Remove(path)
}

func (service *BlockCacheService) diskBytesLocked() uint64 {
	if !service.diskEnabled() {
		return 0
	}
	var total uint64
	for _, path := range core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json")) {
		stat := core.Stat(path)
		if stat.OK {
			if info, ok := stat.Value.(core.FsFileInfo); ok && info.Size() > 0 {
				total += uint64(info.Size())
				continue
			}
		}
		read := core.ReadFile(path)
		if read.OK {
			if data, ok := read.Value.([]byte); ok {
				total += uint64(len(data))
			}
		}
	}
	return total
}

func (service *BlockCacheService) diskBlockPath(id string) string {
	return core.PathJoin(service.cfg.DiskPath, id+".json")
}

func blockCacheID(modelHash, adapterHash, tokenizerHash, mode string, prefix []int32) string {
	payload := struct {
		ModelHash     string  `json:"model_hash,omitempty"`
		AdapterHash   string  `json:"adapter_hash,omitempty"`
		TokenizerHash string  `json:"tokenizer_hash,omitempty"`
		Mode          string  `json:"mode,omitempty"`
		Tokens        []int32 `json:"tokens,omitempty"`
	}{
		ModelHash:     modelHash,
		AdapterHash:   adapterHash,
		TokenizerHash: tokenizerHash,
		Mode:          firstNonEmptyString(mode, blockCacheMode),
		Tokens:        append([]int32(nil), prefix...),
	}
	return core.SHA256HexString(core.JSONMarshalString(payload))
}

func coreHashModelParts(parts ...any) string {
	return core.SHA256HexString(core.JSONMarshalString(parts))
}

func blockRefMatchesLabels(ref inference.CacheBlockRef, labels map[string]string) bool {
	for key, want := range labels {
		switch key {
		case "model_hash":
			if ref.ModelHash != want {
				return false
			}
		case "adapter_hash":
			if ref.AdapterHash != want {
				return false
			}
		case "tokenizer_hash":
			if ref.TokenizerHash != want {
				return false
			}
		default:
			if ref.Labels[key] != want {
				return false
			}
		}
	}
	return true
}

func cacheIdentityMatches(actual, requested string) bool {
	if actual == "" || requested == "" {
		return true
	}
	return actual == requested
}

func boolLabel(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func cacheContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func cloneBlockCacheLabels(input map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range input {
		out[key] = value
	}
	return out
}

func cloneCacheBlockRef(ref inference.CacheBlockRef) inference.CacheBlockRef {
	ref.Labels = cloneBlockCacheLabels(ref.Labels)
	return ref
}

func sortCacheBlockRefs(entries []inference.CacheBlockRef) {
	for i := 1; i < len(entries); i++ {
		current := entries[i]
		j := i - 1
		for j >= 0 && cacheBlockRefLess(current, entries[j]) {
			entries[j+1] = entries[j]
			j--
		}
		entries[j+1] = current
	}
}

func cacheBlockRefLess(a, b inference.CacheBlockRef) bool {
	if a.TokenStart != b.TokenStart {
		return a.TokenStart < b.TokenStart
	}
	return a.ID < b.ID
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func blockCacheResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	if result.OK {
		return nil
	}
	if message := result.Error(); message != "" {
		return core.NewError(message)
	}
	return core.NewError("unknown block cache result error")
}
