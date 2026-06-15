// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

// failingStateWriter is a test stub that always errors on Put. Used to
// exercise the State-write failure path inside blockcache.WarmCache.
type failingStateWriter struct{}

func (failingStateWriter) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, context.Canceled
}

func TestService_Good_StablePrefixBlocksAndStats(t *testing.T) {
	service := New(Config{
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

func TestService_Good_WarmPromptUsesTokenizerAndWarmer(t *testing.T) {
	var warmedPrompt string
	service := New(Config{
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

func TestService_Good_CompatibilityLabels(t *testing.T) {
	service := New(Config{
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

func TestService_Good_CacheEntriesFiltersAndClonesRefs(t *testing.T) {
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
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

func TestService_Good_ClearCache(t *testing.T) {
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
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

func TestService_Good_DiskBackedBlocksSurviveRestart(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	cfg := Config{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
	}
	first := New(cfg)
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

	second := New(cfg)
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

func TestService_Good_StateColdStoreRecordsPayload(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	store := state.NewInMemoryStore(nil)
	service := New(Config{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
		StateStore:    store,
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	if len(result.Blocks) != 2 {
		t.Fatalf("blocks = %+v, want two state-backed blocks", result.Blocks)
	}
	ref := result.Blocks[0]
	if ref.Labels["cold_store"] != "state" || ref.Labels["state_chunk_id"] == "" || ref.Labels["state_codec"] != state.CodecMemory {
		t.Fatalf("block labels = %+v, want State cold-store labels", ref.Labels)
	}
	chunkIDResult := core.Atoi(ref.Labels["state_chunk_id"])
	if !chunkIDResult.OK {
		t.Fatalf("State chunk id %q did not parse: %s", ref.Labels["state_chunk_id"], chunkIDResult.Error())
	}
	chunk, err := state.Resolve(context.Background(), store, chunkIDResult.Value.(int))
	if err != nil {
		t.Fatalf("Resolve(State chunk) error = %v", err)
	}
	if !core.Contains(chunk.Text, `"block_id":"`+ref.ID+`"`) || !core.Contains(chunk.Text, `"tokens":[1,2]`) {
		t.Fatalf("State chunk = %s, want block payload", chunk.Text)
	}

	second := New(Config{
		BlockSize:     2,
		ModelHash:     "sha256:model",
		TokenizerHash: "sha256:tokenizer",
		DiskPath:      diskPath,
		StateStore:    store,
	})
	stats, err := second.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats(second) error = %v", err)
	}
	if stats.Blocks != 2 || stats.Labels["cold_store"] != "state" {
		t.Fatalf("second stats = %+v, want state-backed persisted blocks", stats)
	}
}

func TestService_Bad_CorruptDiskBlockIsIgnored(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	corruptPath := core.PathJoin(diskPath, "broken.json")
	if result := core.WriteFile(corruptPath, []byte("{broken"), 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}

	service := New(Config{BlockSize: 2, DiskPath: diskPath})
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

func TestService_Good_ClearCacheRemovesDiskBlocks(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
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

func TestService_Good_ClearCacheWithLabelsRemovesOnlyMatchingBlocks(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
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

func TestService_Bad_InputAndContextErrors(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (*Service)(nil).CacheStats(context.Background()); err == nil {
		t.Fatal("CacheStats(nil service) error = nil")
	}
	if _, err := (*Service)(nil).CacheEntries(context.Background(), nil); err == nil {
		t.Fatal("CacheEntries(nil service) error = nil")
	}
	if _, err := (*Service)(nil).WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(nil service) error = nil")
	}
	if _, err := (*Service)(nil).ClearCache(context.Background(), nil); err == nil {
		t.Fatal("ClearCache(nil service) error = nil")
	}
	service := New(Config{})
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
	tokenizerErr := New(Config{
		Tokenize: func(string) ([]int32, error) {
			return nil, core.NewError("tokenize failed")
		},
	})
	if _, err := tokenizerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(tokenizer error) error = nil")
	}
	warmerErr := New(Config{
		Tokenize: func(string) ([]int32, error) { return []int32{1}, nil },
		WarmPrompt: func(context.Context, string) error {
			return core.NewError("warm failed")
		},
	})
	if _, err := warmerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(warmer error) error = nil")
	}
	memvidErr := New(Config{
		DiskPath:   core.PathJoin(t.TempDir(), "blocks"),
		StateStore: failingStateWriter{},
	})
	if _, err := memvidErr.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(State write error) error = nil")
	}
}

func TestService_Bad_IncompatibleDiskRecordIsIgnored(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	record := diskRecord{
		Version: diskVersion,
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

	service := New(Config{
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
	if got := HashModelParts("model", 4); got == "" {
		t.Fatal("HashModelParts() returned empty hash")
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
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v", err)
	}
	if err := resultError(core.Result{Value: core.NewError("explicit")}); err == nil || err.Error() != "explicit" {
		t.Fatalf("resultError(error) = %v", err)
	}
	if err := resultError(core.Result{}); err == nil {
		t.Fatal("resultError(empty) = nil")
	}
}

// TestBlockCache_Ugly_EdgeIdentitiesAndLabels covers the awkward-but-real
// corners the happy-path tests skip: the ID tie-break when two refs share a
// token start, the hasher buffer growing past its pooled 256-byte default for
// an over-long identity header, the prefix-token label falling back to Itoa
// beyond the pre-rendered cache cap, and the nil-context fast path. All are
// reachable with synthetic inputs — no model, no disk.
func TestBlockCache_Ugly_EdgeIdentitiesAndLabels(t *testing.T) {
	// cacheBlockRefLess tie-break: equal TokenStart falls through to the ID
	// comparison. The happy-path sort test only exercises the TokenStart
	// branch, so this pins the secondary key.
	if !cacheBlockRefLess(
		inference.CacheBlockRef{TokenStart: 4, ID: "aaa"},
		inference.CacheBlockRef{TokenStart: 4, ID: "bbb"},
	) {
		t.Fatal("cacheBlockRefLess(equal start, aaa<bbb) = false, want true")
	}
	if cacheBlockRefLess(
		inference.CacheBlockRef{TokenStart: 4, ID: "bbb"},
		inference.CacheBlockRef{TokenStart: 4, ID: "aaa"},
	) {
		t.Fatal("cacheBlockRefLess(equal start, bbb<aaa) = true, want false")
	}

	// Hasher buffer-grow: a header longer than the pooled 256-byte default
	// (16 length-prefix bytes + the four identity strings) forces
	// acquireBlockCacheHasher to grow scratch.buf. The resulting ID must
	// still be stable across repeated warms — exercising the grow path and
	// confirming the grown buffer is reused cleanly.
	longHash := "sha256:" + core.Repeat("ab", 200) // ~407-byte header
	longService := New(Config{
		BlockSize:     4,
		ModelHash:     longHash,
		AdapterHash:   longHash,
		TokenizerHash: longHash,
	})
	first, err := longService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(long header, first) error = %v", err)
	}
	second, err := longService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(long header, second) error = %v", err)
	}
	if len(first.Blocks) == 0 || first.Blocks[0].ID == "" {
		t.Fatalf("WarmCache(long header) blocks = %+v, want stable IDs", first.Blocks)
	}
	for i := range first.Blocks {
		if first.Blocks[i].ID != second.Blocks[i].ID {
			t.Fatalf("long-header block %d ID changed across warms: %q != %q", i, first.Blocks[i].ID, second.Blocks[i].ID)
		}
	}

	// prefixTokenLabel beyond the pre-rendered cap: with BlockSize 1 the
	// 33rd aligned end (33) sits past prefixTokenLabelCacheSize (32), so the
	// label is produced by the Itoa fallback rather than the cached slice.
	capService := New(Config{BlockSize: 1, ModelHash: "sha256:model"})
	tokens := make([]int32, prefixTokenLabelCacheSize+2) // 34 single-token blocks
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	capResult, err := capService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	if err != nil {
		t.Fatalf("WarmCache(beyond cap) error = %v", err)
	}
	if got := len(capResult.Blocks); got != prefixTokenLabelCacheSize+2 {
		t.Fatalf("beyond-cap blocks = %d, want %d", got, prefixTokenLabelCacheSize+2)
	}
	lastLabel := capResult.Blocks[len(capResult.Blocks)-1].Labels["prefix_tokens"]
	if lastLabel != core.Itoa(prefixTokenLabelCacheSize+2) {
		t.Fatalf("beyond-cap prefix_tokens = %q, want %q", lastLabel, core.Itoa(prefixTokenLabelCacheSize+2))
	}
	// In-cap aligned end still returns the pre-rendered string (same value,
	// different code path) — both arms produce identical decimals.
	if got := capResult.Blocks[0].Labels["prefix_tokens"]; got != "1" {
		t.Fatalf("in-cap prefix_tokens = %q, want 1", got)
	}

	// Nil context is the documented fast path: cacheContextErr returns nil
	// and WarmCache substitutes context.Background internally.
	nilCtxService := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	if _, err := nilCtxService.WarmCache(nil, inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}}); err != nil {
		t.Fatalf("WarmCache(nil ctx) error = %v, want nil", err)
	}
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	if _, err := nilCtxService.CacheStats(nil); err != nil {
		t.Fatalf("CacheStats(nil ctx) error = %v, want nil", err)
	}
}

// TestService_Good_ConcurrentWarmIsRaceFreeAndStable exercises the
// lock-free hashing path under contention. blockRefs runs before
// WarmCache takes service.mu, so concurrent warms run blockRefs (and
// its package-level sha256/encode-buffer pool) concurrently — and
// blockCacheID shares that same pool. This test fans out many
// goroutines, each warming a distinct token set in a loop, and asserts
// every goroutine's block IDs equal a serially-computed baseline. Run
// under -race it demonstrates the pooled scratch carries no shared
// mutable state across goroutines (a pool unit is held only between
// acquire and the deferred release, never across a yield), and that the
// recycling is byte-identical to the unpooled per-call form.
func TestService_Good_ConcurrentWarmIsRaceFreeAndStable(t *testing.T) {
	const (
		goroutines = 16
		iterations = 40
	)
	cfg := Config{
		BlockSize:     4,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	}
	// Distinct token set per goroutine so concurrent blockRefs calls
	// hash different content through the shared pool — any cross-call
	// state bleed would surface as a wrong, non-deterministic ID.
	tokenSets := make([][]int32, goroutines)
	wantIDs := make([][]string, goroutines)
	for g := range tokenSets {
		tokens := make([]int32, 10+g) // 3 blocks at size 4, last partial
		for i := range tokens {
			tokens[i] = int32(g*1000 + i + 1)
		}
		tokenSets[g] = tokens
		// Serial baseline computed on a private service — the
		// reference IDs the concurrent path must reproduce exactly.
		baseline, err := New(cfg).WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
		if err != nil {
			t.Fatalf("baseline WarmCache(g=%d) error = %v", g, err)
		}
		ids := make([]string, len(baseline.Blocks))
		for i, ref := range baseline.Blocks {
			ids[i] = ref.ID
		}
		wantIDs[g] = ids
	}

	var wg sync.WaitGroup
	errs := make(chan error, goroutines*iterations)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			// Fresh service per goroutine: isolates the in-memory
			// block map (the shared surface under test is the
			// package-level hasher pool, not Service state).
			service := New(cfg)
			tokens := tokenSets[g]
			for it := 0; it < iterations; it++ {
				result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
				if err != nil {
					errs <- core.NewError("WarmCache error in goroutine")
					return
				}
				if len(result.Blocks) != len(wantIDs[g]) {
					errs <- core.NewError("block count mismatch under concurrency")
					return
				}
				for i, ref := range result.Blocks {
					if ref.ID != wantIDs[g][i] {
						errs <- core.NewError("block ID mismatch under concurrency: pool leaked state")
						return
					}
				}
				// Also drive blockCacheID directly — it shares the
				// same pool and is the other lock-free caller. The
				// full-prefix ID must equal the final block's ID:
				// blockRefs emits the final boundary digest over all
				// tokens, and the header here uses the same empty mode
				// the WarmCache request left unset (req.Mode == "").
				if id := blockCacheID(cfg.ModelHash, cfg.AdapterHash, cfg.TokenizerHash, "", tokens); id != wantIDs[g][len(wantIDs[g])-1] {
					errs <- core.NewError("blockCacheID(full prefix) != final block ID under concurrency")
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}
