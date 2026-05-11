// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
)

func TestBlockCacheService_Good_StablePrefixBlocksAndStats(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{
		BlockSize:     3,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	})

	first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}})
	if err != nil {
		t.Fatalf("WarmCache(first) error = %v", err)
	}
	if len(first.Blocks) != 3 {
		t.Fatalf("blocks = %+v, want 3 prefix blocks", first.Blocks)
	}
	if first.Blocks[0].ID == "" || first.Blocks[0].ID == first.Blocks[1].ID {
		t.Fatalf("block IDs = %+v, want stable distinct IDs", first.Blocks)
	}
	if first.Blocks[0].TokenStart != 0 || first.Blocks[0].TokenCount != 3 || first.Blocks[2].TokenStart != 6 || first.Blocks[2].TokenCount != 1 {
		t.Fatalf("blocks = %+v, want chunked token ranges", first.Blocks)
	}

	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}})
	if err != nil {
		t.Fatalf("WarmCache(second) error = %v", err)
	}
	for i := range first.Blocks {
		if first.Blocks[i].ID != second.Blocks[i].ID {
			t.Fatalf("block %d ID changed: %q != %q", i, first.Blocks[i].ID, second.Blocks[i].ID)
		}
	}
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 3 || stats.Hits != 3 || stats.Misses != 3 || stats.HitRate != 0.5 {
		t.Fatalf("stats = %+v, want 3 blocks, 3 hits, 3 misses, 0.5 hit rate", stats)
	}
}

func TestBlockCacheService_Good_WarmPromptUsesTokenizerAndWarmer(t *testing.T) {
	var warmedPrompt string
	service := NewBlockCacheService(BlockCacheConfig{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		TokenizerHash: "sha256:tokenizer",
		Tokenize: func(prompt string) ([]int32, error) {
			if prompt != "hello" {
				t.Fatalf("tokenized prompt = %q, want hello", prompt)
			}
			return []int32{10, 11, 12}, nil
		},
		WarmPrompt: func(_ context.Context, prompt string) error {
			warmedPrompt = prompt
			return nil
		},
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"})
	if err != nil {
		t.Fatalf("WarmCache(prompt) error = %v", err)
	}
	if warmedPrompt != "hello" {
		t.Fatalf("warmed prompt = %q, want hello", warmedPrompt)
	}
	if len(result.Blocks) != 2 || result.Blocks[0].TokenCount != 2 || result.Blocks[1].TokenCount != 1 {
		t.Fatalf("blocks = %+v, want tokenized prompt blocks", result.Blocks)
	}
}

func TestBlockCacheService_Good_CompatibilityLabels(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{
		BlockSize:     2,
		ModelHash:     "sha256:model-a",
		AdapterHash:   "sha256:adapter-a",
		TokenizerHash: "sha256:tokenizer-a",
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Model:   inference.ModelIdentity{Hash: "sha256:model-b"},
		Adapter: inference.AdapterIdentity{Hash: "sha256:adapter-b"},
		Labels:  map[string]string{"tokenizer_hash": "sha256:tokenizer-b"},
		Tokens:  []int32{1, 2},
	})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	if result.Labels["model_match"] != "false" || result.Labels["adapter_match"] != "false" || result.Labels["tokenizer_match"] != "false" {
		t.Fatalf("labels = %+v, want mismatch labels", result.Labels)
	}
	if result.Blocks[0].Labels["adapter_match"] != "false" {
		t.Fatalf("block labels = %+v, want adapter mismatch", result.Blocks[0].Labels)
	}
}

func TestBlockCacheService_Good_CacheEntriesFiltersAndClonesRefs(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{BlockSize: 2, ModelHash: "sha256:model"})
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2, 3},
	}); err != nil {
		t.Fatalf("WarmCache(alpha) error = %v", err)
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "beta"},
		Tokens: []int32{4, 5},
	}); err != nil {
		t.Fatalf("WarmCache(beta) error = %v", err)
	}

	entries, err := service.CacheEntries(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		t.Fatalf("CacheEntries(alpha) error = %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %+v, want two alpha prefix blocks", entries)
	}
	if entries[0].TokenStart != 0 || entries[1].TokenStart != 2 {
		t.Fatalf("entries = %+v, want deterministic token order", entries)
	}
	for _, ref := range entries {
		if ref.Labels["tenant"] != "alpha" {
			t.Fatalf("entry labels = %+v, want alpha tenant", ref.Labels)
		}
	}

	entries[0].Labels["tenant"] = "mutated"
	again, err := service.CacheEntries(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		t.Fatalf("CacheEntries(alpha again) error = %v", err)
	}
	if again[0].Labels["tenant"] != "alpha" {
		t.Fatalf("entry labels were not cloned: %+v", again[0].Labels)
	}
}

func TestBlockCacheService_Good_ClearCache(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{BlockSize: 2, ModelHash: "sha256:model"})
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4}}); err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}

	stats, err := service.ClearCache(context.Background(), nil)
	if err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if stats.Blocks != 0 {
		t.Fatalf("ClearCache stats = %+v, want zero blocks", stats)
	}
}

func TestBlockCacheService_Good_DefaultDiskPathUsesEnv(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	t.Setenv(BlockCacheDiskPathEnv, diskPath)

	if got := DefaultBlockCacheDiskPath(); got != diskPath {
		t.Fatalf("DefaultBlockCacheDiskPath() = %q, want %q", got, diskPath)
	}
}

func TestBlockCacheService_Good_DiskBackedBlocksSurviveRestart(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	cfg := BlockCacheConfig{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
	}
	first := NewBlockCacheService(cfg)
	result, err := first.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(first) error = %v", err)
	}
	if len(result.Blocks) != 3 {
		t.Fatalf("blocks = %+v, want 3 persisted prefix blocks", result.Blocks)
	}
	for _, ref := range result.Blocks {
		if ref.Labels["disk"] != "true" || ref.Labels["disk_path"] == "" {
			t.Fatalf("block labels = %+v, want disk metadata", ref.Labels)
		}
		if stat := core.Stat(ref.Labels["disk_path"]); !stat.OK {
			t.Fatalf("persisted block %q was not written: %s", ref.Labels["disk_path"], stat.Error())
		}
	}
	if result.Stats.DiskBytes == 0 {
		t.Fatalf("warm stats = %+v, want disk bytes", result.Stats)
	}

	second := NewBlockCacheService(cfg)
	stats, err := second.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats(second) error = %v", err)
	}
	if stats.Blocks != 3 || stats.DiskBytes == 0 {
		t.Fatalf("second stats = %+v, want persisted blocks and disk bytes", stats)
	}
	hit, err := second.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(second) error = %v", err)
	}
	if hit.Stats.Hits != 3 || hit.Stats.Misses != 0 || hit.Stats.HitRate != 1 {
		t.Fatalf("second warm stats = %+v, want persisted block hits", hit.Stats)
	}
}

func TestBlockCacheService_Good_MemvidColdStoreRecordsPayload(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	store := memvid.NewInMemoryStore(nil)
	service := NewBlockCacheService(BlockCacheConfig{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
		MemvidStore:   store,
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	if len(result.Blocks) != 2 {
		t.Fatalf("blocks = %+v, want two memvid-backed blocks", result.Blocks)
	}
	ref := result.Blocks[0]
	if ref.Labels["cold_store"] != "memvid" || ref.Labels["memvid_chunk_id"] == "" || ref.Labels["memvid_codec"] != memvid.CodecMemory {
		t.Fatalf("block labels = %+v, want memvid cold-store labels", ref.Labels)
	}
	chunkIDResult := core.Atoi(ref.Labels["memvid_chunk_id"])
	if !chunkIDResult.OK {
		t.Fatalf("memvid chunk id %q did not parse: %s", ref.Labels["memvid_chunk_id"], chunkIDResult.Error())
	}
	chunk, err := memvid.Resolve(context.Background(), store, chunkIDResult.Value.(int))
	if err != nil {
		t.Fatalf("Resolve(memvid chunk) error = %v", err)
	}
	if !core.Contains(chunk.Text, `"block_id":"`+ref.ID+`"`) || !core.Contains(chunk.Text, `"tokens":[1,2]`) {
		t.Fatalf("memvid chunk = %s, want block payload", chunk.Text)
	}

	second := NewBlockCacheService(BlockCacheConfig{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
		MemvidStore:   store,
	})
	stats, err := second.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats(second) error = %v", err)
	}
	if stats.Blocks != 2 || stats.Labels["cold_store"] != "memvid" {
		t.Fatalf("second stats = %+v, want memvid-backed persisted blocks", stats)
	}
}

func TestBlockCacheService_Bad_CorruptDiskBlockIsIgnored(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	corruptPath := core.PathJoin(diskPath, "broken.json")
	if result := core.WriteFile(corruptPath, []byte("{broken"), 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}

	service := NewBlockCacheService(BlockCacheConfig{BlockSize: 2, DiskPath: diskPath})
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 || stats.Evictions != 1 || stats.Labels["disk_corrupt"] != "1" {
		t.Fatalf("stats = %+v, want corrupt record ignored and counted", stats)
	}
	if stat := core.Stat(corruptPath); stat.OK {
		t.Fatalf("corrupt cache record still exists at %s", corruptPath)
	}
}

func TestBlockCacheService_Good_ClearCacheRemovesDiskBlocks(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	service := NewBlockCacheService(BlockCacheConfig{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4}})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	var diskFiles []string
	for _, ref := range result.Blocks {
		diskFiles = append(diskFiles, ref.Labels["disk_path"])
	}

	stats, err := service.ClearCache(context.Background(), nil)
	if err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if stats.Blocks != 0 || stats.DiskBytes != 0 {
		t.Fatalf("ClearCache stats = %+v, want no persisted blocks", stats)
	}
	for _, path := range diskFiles {
		if stat := core.Stat(path); stat.OK {
			t.Fatalf("persisted block still exists at %s", path)
		}
	}
}

func TestBlockCacheService_Good_ClearCacheWithLabelsRemovesOnlyMatchingBlocks(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	service := NewBlockCacheService(BlockCacheConfig{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
	alpha, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2, 3},
	})
	if err != nil {
		t.Fatalf("WarmCache(alpha) error = %v", err)
	}
	beta, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "beta"},
		Tokens: []int32{4, 5},
	})
	if err != nil {
		t.Fatalf("WarmCache(beta) error = %v", err)
	}

	stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		t.Fatalf("ClearCache(alpha) error = %v", err)
	}
	if stats.Blocks != 1 || stats.Labels["cleared"] != "2" {
		t.Fatalf("ClearCache(alpha) stats = %+v, want one beta block remaining and two clears", stats)
	}
	for _, ref := range alpha.Blocks {
		if stat := core.Stat(ref.Labels["disk_path"]); stat.OK {
			t.Fatalf("alpha disk block still exists at %s", ref.Labels["disk_path"])
		}
	}
	if stat := core.Stat(beta.Blocks[0].Labels["disk_path"]); !stat.OK {
		t.Fatalf("beta disk block was removed: %s", beta.Blocks[0].Labels["disk_path"])
	}
	entries, err := service.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries() error = %v", err)
	}
	if len(entries) != 1 || entries[0].Labels["tenant"] != "beta" {
		t.Fatalf("remaining entries = %+v, want only beta", entries)
	}
}

func TestBlockCacheService_Bad_InputAndContextErrors(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (*BlockCacheService)(nil).CacheStats(context.Background()); err == nil {
		t.Fatal("CacheStats(nil service) error = nil")
	}
	if _, err := (*BlockCacheService)(nil).CacheEntries(context.Background(), nil); err == nil {
		t.Fatal("CacheEntries(nil service) error = nil")
	}
	if _, err := (*BlockCacheService)(nil).WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(nil service) error = nil")
	}
	if _, err := (*BlockCacheService)(nil).ClearCache(context.Background(), nil); err == nil {
		t.Fatal("ClearCache(nil service) error = nil")
	}
	service := NewBlockCacheService(BlockCacheConfig{})
	if _, err := service.CacheStats(cancelled); err == nil {
		t.Fatal("CacheStats(cancelled) error = nil")
	}
	if _, err := service.CacheEntries(cancelled, nil); err == nil {
		t.Fatal("CacheEntries(cancelled) error = nil")
	}
	if _, err := service.WarmCache(cancelled, inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(cancelled) error = nil")
	}
	if _, err := service.ClearCache(cancelled, nil); err == nil {
		t.Fatal("ClearCache(cancelled) error = nil")
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{}); err == nil {
		t.Fatal("WarmCache(empty request) error = nil")
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(prompt without tokenizer) error = nil")
	}
	tokenizerErr := NewBlockCacheService(BlockCacheConfig{
		Tokenize: func(string) ([]int32, error) {
			return nil, core.NewError("tokenize failed")
		},
	})
	if _, err := tokenizerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(tokenizer error) error = nil")
	}
	warmerErr := NewBlockCacheService(BlockCacheConfig{
		Tokenize: func(string) ([]int32, error) { return []int32{1}, nil },
		WarmPrompt: func(context.Context, string) error {
			return core.NewError("warm failed")
		},
	})
	if _, err := warmerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(warmer error) error = nil")
	}
	memvidErr := NewBlockCacheService(BlockCacheConfig{
		DiskPath:    core.PathJoin(t.TempDir(), "blocks"),
		MemvidStore: failingMemvidWriter{},
	})
	if _, err := memvidErr.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(memvid write error) error = nil")
	}
}

func TestBlockCacheService_Bad_IncompatibleDiskRecordIsIgnored(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	record := blockCacheDiskRecord{
		Version: blockCacheDiskVersion,
		Ref: inference.CacheBlockRef{
			ID:            "incompatible",
			ModelHash:     "sha256:other-model",
			AdapterHash:   "sha256:adapter",
			TokenizerHash: "sha256:tokenizer",
		},
	}
	if data := core.JSONMarshal(record); !data.OK {
		t.Fatalf("JSONMarshal(record) error = %s", data.Error())
	} else if result := core.WriteFile(core.PathJoin(diskPath, "incompatible.json"), data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("WriteFile(record) error = %s", result.Error())
	}

	service := NewBlockCacheService(BlockCacheConfig{
		DiskPath:      diskPath,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	})
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 || stats.Evictions != 0 || stats.Labels["disk_corrupt"] != "0" {
		t.Fatalf("stats = %+v, want incompatible record ignored without corruption", stats)
	}
}

func TestBlockCacheHelpers_Good(t *testing.T) {
	if got := coreHashModelParts("model", 4); got == "" {
		t.Fatal("coreHashModelParts() returned empty hash")
	}
	if !blockRefMatchesLabels(inference.CacheBlockRef{ModelHash: "m", AdapterHash: "a", TokenizerHash: "t", Labels: map[string]string{"tenant": "alpha"}}, map[string]string{
		"model_hash":     "m",
		"adapter_hash":   "a",
		"tokenizer_hash": "t",
		"tenant":         "alpha",
	}) {
		t.Fatal("blockRefMatchesLabels() returned false for matching labels")
	}
	if blockRefMatchesLabels(inference.CacheBlockRef{ModelHash: "m"}, map[string]string{"model_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels() returned true for model mismatch")
	}
	if cacheIdentityMatches("actual", "requested") {
		t.Fatal("cacheIdentityMatches() returned true for mismatch")
	}
	if boolLabel(true) != "true" || boolLabel(false) != "false" {
		t.Fatal("boolLabel() returned unexpected text")
	}
	if got := firstNonEmptyString("", "  ", "value"); got != "value" {
		t.Fatalf("firstNonEmptyString() = %q, want value", got)
	}
	labels := map[string]string{"a": "b"}
	cloned := cloneBlockCacheLabels(labels)
	cloned["a"] = "changed"
	if labels["a"] != "b" {
		t.Fatalf("cloneBlockCacheLabels mutated source = %+v", labels)
	}
	refs := []inference.CacheBlockRef{
		{ID: "b", TokenStart: 2},
		{ID: "a", TokenStart: 0},
	}
	sortCacheBlockRefs(refs)
	if refs[0].ID != "a" || !cacheBlockRefLess(refs[0], refs[1]) {
		t.Fatalf("sorted refs = %+v, want token order", refs)
	}
	if err := blockCacheResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("blockCacheResultError(OK) = %v", err)
	}
	if err := blockCacheResultError(core.Result{Value: core.NewError("explicit")}); err == nil || err.Error() != "explicit" {
		t.Fatalf("blockCacheResultError(error) = %v", err)
	}
	if err := blockCacheResultError(core.Result{}); err == nil {
		t.Fatal("blockCacheResultError(empty) = nil")
	}
}
