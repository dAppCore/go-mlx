// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

func TestStateBundle_SaveLoad_Good(t *testing.T) {
	coverageTokens := "StateBundle SaveLoad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := stateBundleTestSnapshot()
	tokenizerPath := core.PathJoin(t.TempDir(), "tokenizer.json")
	if result := core.WriteFile(tokenizerPath, []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]}}`), 0o600); !result.OK {
		t.Fatalf("WriteFile tokenizer: %s", result.Error())
	}
	tokenizerHash, err := StateBundleFileHash(tokenizerPath)
	if err != nil {
		t.Fatalf("StateBundleFileHash() error = %v", err)
	}
	bundle, err := NewStateBundle(snapshot, StateBundleOptions{
		Model:     "gemma4-e4b",
		ModelPath: "/models/gemma4",
		ModelInfo: ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			VocabSize:     262144,
			QuantBits:     4,
			ContextLength: 131072,
		},
		Prompt: "stable context",
		Tokenizer: StateBundleTokenizer{
			Kind:         "hf-tokenizer-json",
			Path:         tokenizerPath,
			Version:      "tokenizers-v1",
			Hash:         tokenizerHash,
			VocabSize:    262144,
			BOS:          2,
			EOS:          1,
			ChatTemplate: "<start_of_turn>model\n",
		},
		Runtime: StateBundleRuntime{
			Name:     "go-mlx",
			Version:  "dev",
			Platform: "darwin/arm64",
		},
		Adapter: StateBundleAdapter{
			Name:       "domain-lora",
			Path:       "/adapters/domain",
			Rank:       8,
			Alpha:      16,
			TargetKeys: []string{"q_proj", "v_proj"},
		},
		Sampler: GenerateConfig{
			MaxTokens:     32,
			Temperature:   0.2,
			TopK:          4,
			RepeatPenalty: 1.1,
		},
		MemvidRefs: []memvid.ChunkRef{{
			ChunkID:        42,
			FrameOffset:    7,
			HasFrameOffset: true,
			Codec:          memvid.CodecQRVideo,
			Segment:        "/tmp/trace.mp4",
		}},
		Refs: []StateBundleRef{{
			Kind: "kv",
			URI:  "file:///tmp/session.kvbin",
			Hash: "sha256:kv",
		}},
		Meta: map[string]string{"suite": "beta"},
	})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}
	snapshot.Tokens[0] = 99
	path := core.PathJoin(t.TempDir(), "state.bundle.json")

	if err := bundle.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := LoadStateBundle(path)

	if err != nil {
		t.Fatalf("LoadStateBundle() error = %v", err)
	}
	if loaded.Version != StateBundleVersion || loaded.Kind != StateBundleKind {
		t.Fatalf("loaded bundle version/kind = %d/%q", loaded.Version, loaded.Kind)
	}
	if loaded.Model.Name != "gemma4-e4b" || loaded.Model.Path != "/models/gemma4" || loaded.Model.Architecture != "gemma4_text" {
		t.Fatalf("loaded model = %+v", loaded.Model)
	}
	if loaded.Model.VocabSize != 262144 || loaded.Model.QuantBits != 4 || loaded.Model.ContextLength != 131072 {
		t.Fatalf("loaded model metadata = %+v", loaded.Model)
	}
	if loaded.Prompt.Text != "stable context" || loaded.Prompt.Hash == "" {
		t.Fatalf("loaded prompt = %+v", loaded.Prompt)
	}
	if loaded.Tokenizer.Path != tokenizerPath || loaded.Tokenizer.Hash != tokenizerHash || loaded.Tokenizer.ChatTemplateHash == "" {
		t.Fatalf("loaded tokenizer = %+v", loaded.Tokenizer)
	}
	if loaded.Runtime.Name != "go-mlx" || loaded.Runtime.Version != "dev" {
		t.Fatalf("loaded runtime = %+v", loaded.Runtime)
	}
	if loaded.Adapter.Name != "domain-lora" || loaded.Adapter.Path != "/adapters/domain" || loaded.Adapter.Hash == "" || loaded.Adapter.Rank != 8 {
		t.Fatalf("loaded adapter = %+v", loaded.Adapter)
	}
	if loaded.Sampler.MaxTokens != 32 || loaded.Sampler.TopK != 4 {
		t.Fatalf("loaded sampler = %+v", loaded.Sampler)
	}
	if loaded.KV == nil || loaded.KV.Tokens[0] != 1 || loaded.KVHash == "" {
		t.Fatalf("loaded KV = %+v hash=%q", loaded.KV, loaded.KVHash)
	}
	if loaded.Analysis == nil || loaded.SAMI == nil || loaded.SAMI.Architecture != "gemma4_text" {
		t.Fatalf("loaded analysis/SAMI = %+v/%+v", loaded.Analysis, loaded.SAMI)
	}
	if len(loaded.Refs) != 2 || loaded.Refs[1].Kind != StateBundleRefMemvid || loaded.Refs[1].Memvid.ChunkID != 42 {
		t.Fatalf("loaded refs = %+v", loaded.Refs)
	}
	if loaded.Meta["suite"] != "beta" {
		t.Fatalf("loaded meta = %+v", loaded.Meta)
	}
}

func TestStateBundle_Bad(t *testing.T) {
	_, err := NewStateBundle(nil, StateBundleOptions{})

	if err == nil {
		t.Fatal("NewStateBundle(nil) error = nil, want nil snapshot error")
	}
}

func TestStateBundleMemvidSnapshot_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	snapshot := stateBundleTestSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), store, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		KVHash:  hash,
		Refs: []StateBundleRef{{
			Kind:   StateBundleRefMemvid,
			URI:    stateMemvidURI(ref),
			Memvid: ref,
		}},
	}

	loaded, err := bundle.SnapshotFromMemvid(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded snapshot = %+v, want %+v", loaded, snapshot)
	}
}

func TestStateBundleMemvidSnapshot_Good_AllowsFrameZero(t *testing.T) {
	source := memvid.NewInMemoryStore(nil)
	snapshot := stateBundleTestSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), source, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	chunk, err := memvid.Resolve(context.Background(), source, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	store := memvid.NewInMemoryStoreWithManifest(map[int]string{0: chunk.Text}, map[int]memvid.ChunkRef{0: {
		ChunkID:        0,
		FrameOffset:    0,
		HasFrameOffset: true,
		Codec:          memvid.CodecQRVideo,
		Segment:        "/tmp/session.mp4",
	}})
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		KVHash:  hash,
		Refs: []StateBundleRef{{
			Kind: StateBundleRefMemvid,
			URI:  "memvid:///tmp/session.mp4#chunk=0",
			Memvid: memvid.ChunkRef{
				ChunkID:        0,
				FrameOffset:    0,
				HasFrameOffset: true,
				Codec:          memvid.CodecQRVideo,
				Segment:        "/tmp/session.mp4",
			},
		}},
	}

	loaded, err := bundle.SnapshotFromMemvid(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid(frame zero) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded token offset = %d, want %d", loaded.TokenOffset, snapshot.TokenOffset)
	}
}

func TestStateBundleSnapshot_Good_ClonesEmbeddedAndLoadsKVPath(t *testing.T) {
	snapshot := stateBundleTestSnapshot()
	bundle, err := NewStateBundle(snapshot, StateBundleOptions{Prompt: "persisted"})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}

	first, err := bundle.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	first.Tokens[0] = 99
	second, err := bundle.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot() second error = %v", err)
	}
	if second.Tokens[0] != 1 {
		t.Fatalf("Snapshot() returned shared tokens = %v, want defensive clone", second.Tokens)
	}

	kvPath := core.PathJoin(t.TempDir(), "state.kvbin")
	if err := snapshot.Save(kvPath); err != nil {
		t.Fatalf("kv.Snapshot.Save() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	pathBundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		KVPath:  kvPath,
		KVHash:  hash,
	}
	loaded, err := pathBundle.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot(KVPath) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded path snapshot = %+v, want %+v", loaded, snapshot)
	}

	pathBundle.KVHash = "bad-hash"
	if _, err := pathBundle.Snapshot(); err == nil {
		t.Fatal("Snapshot(KVPath hash mismatch) error = nil")
	}
}

func TestStateBundleValidationAndCompatibility_Bad(t *testing.T) {
	snapshot := stateBundleTestSnapshot()
	bundle, err := NewStateBundle(snapshot, StateBundleOptions{
		ModelInfo: ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    1,
		},
		Adapter: StateBundleAdapter{
			Name:  "domain",
			Path:  "/adapters/domain",
			Hash:  "adapter-hash",
			Rank:  8,
			Alpha: 16,
		},
	})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}

	if err := CheckStateBundleCompatibility(ModelInfo{
		Architecture: "gemma4_text",
		NumLayers:    1,
		Adapter: lora.AdapterInfo{
			Name:  "domain",
			Path:  "/adapters/domain",
			Hash:  "adapter-hash",
			Rank:  8,
			Alpha: 16,
		},
	}, bundle); err != nil {
		t.Fatalf("CheckStateBundleCompatibility(good) error = %v", err)
	}
	for name, bad := range map[string]*StateBundle{
		"nil kv": {
			Version: StateBundleVersion,
			Kind:    StateBundleKind,
		},
		"version": {
			Version: StateBundleVersion + 1,
			Kind:    StateBundleKind,
			KV:      snapshot.Clone(),
		},
		"kind": {
			Version: StateBundleVersion,
			Kind:    "wrong",
			KV:      snapshot.Clone(),
		},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("%s Validate() error = nil", name)
		}
	}
	hashMismatch := *bundle
	hashMismatch.KV = bundle.KV.Clone()
	hashMismatch.KV.Tokens[0] = 99
	if err := hashMismatch.Validate(); err == nil {
		t.Fatal("Validate(hash mismatch) error = nil")
	}
	if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "llama", NumLayers: 1}, bundle); err == nil {
		t.Fatal("CheckStateBundleCompatibility(architecture mismatch) error = nil")
	}
	if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2}, bundle); err == nil {
		t.Fatal("CheckStateBundleCompatibility(layer mismatch) error = nil")
	}
	if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}, bundle); err == nil {
		t.Fatal("CheckStateBundleCompatibility(missing adapter) error = nil")
	}
	for name, adapter := range map[string]lora.AdapterInfo{
		"hash":  {Path: "/adapters/domain", Hash: "wrong", Rank: 8, Alpha: 16},
		"path":  {Path: "/other/domain", Rank: 8, Alpha: 16},
		"rank":  {Path: "/adapters/domain", Rank: 4, Alpha: 16},
		"alpha": {Path: "/adapters/domain", Rank: 8, Alpha: 8},
	} {
		if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1, Adapter: adapter}, bundle); err == nil {
			t.Fatalf("CheckStateBundleCompatibility(%s mismatch) error = nil", name)
		}
	}
}

func TestStateBundleAdapterFromModelInfo_Good(t *testing.T) {
	info := ModelInfo{
		Adapter: lora.AdapterInfo{
			Name:       "active",
			Path:       "/adapters/active",
			Hash:       "active-hash",
			Rank:       4,
			Alpha:      8,
			Scale:      2,
			TargetKeys: []string{"q_proj"},
		},
	}
	bundle, err := NewStateBundle(stateBundleTestSnapshot(), StateBundleOptions{ModelInfo: info})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}
	info.Adapter.TargetKeys[0] = "mutated"

	if bundle.Adapter.Name != "active" || bundle.Adapter.Path != "/adapters/active" || bundle.Adapter.Hash != "active-hash" {
		t.Fatalf("bundle adapter = %+v, want active adapter identity", bundle.Adapter)
	}
	if len(bundle.Adapter.TargetKeys) != 1 || bundle.Adapter.TargetKeys[0] != "q_proj" {
		t.Fatalf("bundle adapter targets = %v, want defensive copy", bundle.Adapter.TargetKeys)
	}
}

func TestStateBundleSnapshot_Bad(t *testing.T) {
	if _, err := (*StateBundle)(nil).Snapshot(); err == nil {
		t.Fatal("Snapshot(nil bundle) error = nil")
	}
	if _, err := (&StateBundle{Version: StateBundleVersion, Kind: StateBundleKind}).Snapshot(); err == nil {
		t.Fatal("Snapshot(no KV) error = nil")
	}
	if _, err := (*StateBundle)(nil).SnapshotFromMemvid(context.Background(), memvid.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromMemvid(nil bundle) error = nil")
	}
	if _, err := (&StateBundle{Version: StateBundleVersion, Kind: StateBundleKind}).SnapshotFromMemvid(nil, memvid.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromMemvid(no ref) error = nil")
	}

	store := memvid.NewInMemoryStore(nil)
	ref, err := stateBundleTestSnapshot().SaveMemvid(context.Background(), store, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		KVHash:  "bad-hash",
		Refs: []StateBundleRef{{
			Kind:   StateBundleRefMemvid,
			Memvid: ref,
		}},
	}
	if _, err := bundle.SnapshotFromMemvid(context.Background(), store); err == nil {
		t.Fatal("SnapshotFromMemvid(hash mismatch) error = nil")
	}
}

func TestStateBundleResultError_Good(t *testing.T) {
	if err := stateBundleResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("stateBundleResultError(OK) = %v", err)
	}
	if err := stateBundleResultError(core.Result{Value: core.NewError("explicit")}); err == nil || err.Error() != "explicit" {
		t.Fatalf("stateBundleResultError(error) = %v", err)
	}
	if err := stateBundleResultError(core.Result{Value: "text"}); err == nil || err.Error() != "text" {
		t.Fatalf("stateBundleResultError(string) = %v", err)
	}
	if err := stateBundleResultError(core.Result{}); err == nil {
		t.Fatal("stateBundleResultError(empty) = nil")
	}
}

func TestStateBundle_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.bundle.json")
	if result := core.WriteFile(path, []byte("{"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := LoadStateBundle(path)

	if err == nil {
		t.Fatal("LoadStateBundle() error = nil, want corrupt bundle error")
	}
}

func stateBundleTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}
