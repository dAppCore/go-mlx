// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/lora"
)

func bundleTestSnapshot() *kv.Snapshot {
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

func TestNew_SaveLoad_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	tokenizerPath := core.PathJoin(t.TempDir(), "tokenizer.json")
	if result := core.WriteFile(tokenizerPath, []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]}}`), 0o600); !result.OK {
		t.Fatalf("WriteFile tokenizer: %s", result.Error())
	}
	tokenizerHash, err := FileHash(tokenizerPath)
	if err != nil {
		t.Fatalf("FileHash() error = %v", err)
	}
	b, err := New(snapshot, Options{
		Model:     "gemma4-e4b",
		ModelPath: "/models/gemma4",
		Source: ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			VocabSize:     262144,
			QuantBits:     4,
			ContextLength: 131072,
		},
		Prompt: "stable context",
		Tokenizer: Tokenizer{
			Kind: "hf-tokenizer-json", Path: tokenizerPath, Version: "tokenizers-v1",
			Hash: tokenizerHash, VocabSize: 262144, BOS: 2, EOS: 1,
			ChatTemplate: "<start_of_turn>model\n",
		},
		Runtime: Runtime{Name: "go-mlx", Version: "dev", Platform: "darwin/arm64"},
		Adapter: Adapter{
			Name: "domain-lora", Path: "/adapters/domain",
			Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"},
		},
		Sampler: Sampler{MaxTokens: 32, Temperature: 0.2, TopK: 4, RepeatPenalty: 1.1},
		StateRefs: []state.ChunkRef{{
			ChunkID: 42, FrameOffset: 7, HasFrameOffset: true,
			Codec: state.CodecQRVideo, Segment: "/tmp/trace.mp4",
		}},
		Refs: []Ref{{Kind: "kv", URI: "file:///tmp/session.kvbin", Hash: "sha256:kv"}},
		Meta: map[string]string{"suite": "beta"},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	snapshot.Tokens[0] = 99
	path := core.PathJoin(t.TempDir(), "state.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Version != Version || loaded.Kind != Kind {
		t.Fatalf("loaded version/kind = %d/%q", loaded.Version, loaded.Kind)
	}
	if loaded.Model.Name != "gemma4-e4b" || loaded.Model.Architecture != "gemma4_text" {
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
	if loaded.Adapter.Name != "domain-lora" || loaded.Adapter.Hash == "" || loaded.Adapter.Rank != 8 {
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
	if len(loaded.Refs) != 2 || loaded.Refs[1].Kind != RefState || loaded.Refs[1].State.ChunkID != 42 {
		t.Fatalf("loaded refs = %+v", loaded.Refs)
	}
	if loaded.Meta["suite"] != "beta" {
		t.Fatalf("loaded meta = %+v", loaded.Meta)
	}
}

func TestNew_NilSnapshot_Bad(t *testing.T) {
	if _, err := New(nil, Options{}); err == nil {
		t.Fatal("New(nil) error = nil, want nil snapshot error")
	}
}

func TestSnapshotFromState_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := bundleTestSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{Kind: RefState, URI: StateURI(ref), State: ref}},
	}
	loaded, err := b.SnapshotFromState(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromState() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded snapshot = %+v, want %+v", loaded, snapshot)
	}
}

func TestSnapshotFromMemvid_AllowsFrameZero_Good(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := bundleTestSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), source, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	chunk, err := state.Resolve(context.Background(), source, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	store := state.NewInMemoryStoreWithManifest(map[int]string{0: chunk.Text}, map[int]state.ChunkRef{0: {
		ChunkID: 0, FrameOffset: 0, HasFrameOffset: true,
		Codec: state.CodecQRVideo, Segment: "/tmp/session.mp4",
	}})
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{
			Kind: RefMemvid, URI: "memvid:///tmp/session.mp4#chunk=0",
			Memvid: state.ChunkRef{
				ChunkID: 0, FrameOffset: 0, HasFrameOffset: true,
				Codec: state.CodecQRVideo, Segment: "/tmp/session.mp4",
			},
		}},
	}
	loaded, err := b.SnapshotFromMemvid(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid(frame zero) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded token offset = %d, want %d", loaded.TokenOffset, snapshot.TokenOffset)
	}
}

func TestSnapshot_ClonesEmbeddedAndLoadsKVPath_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	b, err := New(snapshot, Options{Prompt: "persisted"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	first, err := b.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	first.Tokens[0] = 99
	second, err := b.Snapshot()
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
	pathBundle := &Bundle{Version: Version, Kind: Kind, KVPath: kvPath, KVHash: hash}
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

func TestValidateAndCheckCompatibility_Bad(t *testing.T) {
	snapshot := bundleTestSnapshot()
	b, err := New(snapshot, Options{
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Adapter: Adapter{
			Name: "domain", Path: "/adapters/domain", Hash: "adapter-hash",
			Rank: 8, Alpha: 16,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := CheckCompatibility(ModelInfo{
		Architecture: "gemma4_text", NumLayers: 1,
		Adapter: lora.AdapterInfo{Name: "domain", Path: "/adapters/domain", Hash: "adapter-hash", Rank: 8, Alpha: 16},
	}, b); err != nil {
		t.Fatalf("CheckCompatibility(good) error = %v", err)
	}
	for name, bad := range map[string]*Bundle{
		"nil kv":  {Version: Version, Kind: Kind},
		"version": {Version: Version + 1, Kind: Kind, KV: snapshot.Clone()},
		"kind":    {Version: Version, Kind: "wrong", KV: snapshot.Clone()},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("%s Validate() error = nil", name)
		}
	}
	hashMismatch := *b
	hashMismatch.KV = b.KV.Clone()
	hashMismatch.KV.Tokens[0] = 99
	if err := hashMismatch.Validate(); err == nil {
		t.Fatal("Validate(hash mismatch) error = nil")
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "llama", NumLayers: 1}, b); err == nil {
		t.Fatal("CheckCompatibility(architecture mismatch) error = nil")
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2}, b); err == nil {
		t.Fatal("CheckCompatibility(layer mismatch) error = nil")
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}, b); err == nil {
		t.Fatal("CheckCompatibility(missing adapter) error = nil")
	}
	for name, adapter := range map[string]lora.AdapterInfo{
		"hash":  {Path: "/adapters/domain", Hash: "wrong", Rank: 8, Alpha: 16},
		"path":  {Path: "/other/domain", Rank: 8, Alpha: 16},
		"rank":  {Path: "/adapters/domain", Rank: 4, Alpha: 16},
		"alpha": {Path: "/adapters/domain", Rank: 8, Alpha: 8},
	} {
		if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1, Adapter: adapter}, b); err == nil {
			t.Fatalf("CheckCompatibility(%s mismatch) error = nil", name)
		}
	}
}

func TestAdapterFromModelInfo_Good(t *testing.T) {
	info := ModelInfo{
		Adapter: lora.AdapterInfo{
			Name: "active", Path: "/adapters/active", Hash: "active-hash",
			Rank: 4, Alpha: 8, Scale: 2, TargetKeys: []string{"q_proj"},
		},
	}
	b, err := New(bundleTestSnapshot(), Options{Source: info})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	info.Adapter.TargetKeys[0] = "mutated"
	if b.Adapter.Name != "active" || b.Adapter.Path != "/adapters/active" || b.Adapter.Hash != "active-hash" {
		t.Fatalf("bundle adapter = %+v, want active adapter identity", b.Adapter)
	}
	if len(b.Adapter.TargetKeys) != 1 || b.Adapter.TargetKeys[0] != "q_proj" {
		t.Fatalf("bundle adapter targets = %v, want defensive copy", b.Adapter.TargetKeys)
	}
}

func TestSnapshot_NilAndMissingKV_Bad(t *testing.T) {
	if _, err := (*Bundle)(nil).Snapshot(); err == nil {
		t.Fatal("Snapshot(nil bundle) error = nil")
	}
	if _, err := (&Bundle{Version: Version, Kind: Kind}).Snapshot(); err == nil {
		t.Fatal("Snapshot(no KV) error = nil")
	}
	if _, err := (*Bundle)(nil).SnapshotFromState(context.Background(), state.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromState(nil bundle) error = nil")
	}
	if _, err := (&Bundle{Version: Version, Kind: Kind}).SnapshotFromState(nil, state.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromState(no ref) error = nil")
	}
	store := state.NewInMemoryStore(nil)
	ref, err := bundleTestSnapshot().SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: "bad-hash",
		Refs: []Ref{{Kind: RefState, State: ref}},
	}
	if _, err := b.SnapshotFromState(context.Background(), store); err == nil {
		t.Fatal("SnapshotFromState(hash mismatch) error = nil")
	}
}

func TestLoad_CorruptJSON_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.bundle.json")
	if result := core.WriteFile(path, []byte("{"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	if _, err := Load(path); err == nil {
		t.Fatal("Load() error = nil, want corrupt bundle error")
	}
}

func TestNormaliseTokenizer_FillsHashes_Good(t *testing.T) {
	in := Tokenizer{Path: "/tok.json", ChatTemplate: "<bos>"}
	out := NormaliseTokenizer(in)
	if out.Hash == "" || out.ChatTemplateHash == "" {
		t.Fatalf("NormaliseTokenizer left hashes empty: %+v", out)
	}
}

func TestAdapterEmpty_GoodBad(t *testing.T) {
	if !AdapterEmpty(Adapter{}) {
		t.Fatal("AdapterEmpty(zero) = false")
	}
	if AdapterEmpty(Adapter{Name: "x"}) {
		t.Fatal("AdapterEmpty(name set) = true")
	}
	if AdapterEmpty(Adapter{TargetKeys: []string{"q_proj"}}) {
		t.Fatal("AdapterEmpty(targets set) = true")
	}
}

func TestAdapterFromInfoRoundTrip_Good(t *testing.T) {
	src := lora.AdapterInfo{
		Name: "v1", Path: "/v1.safetensors", Hash: "abc",
		Rank: 8, Alpha: 16, Scale: 2, TargetKeys: []string{"q_proj", "v_proj"},
	}
	round := AdapterToInfo(AdapterFromInfo(src))
	if round.Name != src.Name || round.Rank != src.Rank ||
		len(round.TargetKeys) != 2 || round.TargetKeys[1] != "v_proj" {
		t.Fatalf("round-trip = %+v, want %+v", round, src)
	}
	src.TargetKeys[0] = "mutated"
	if round.TargetKeys[0] == "mutated" {
		t.Fatal("AdapterFromInfo did not clone TargetKeys")
	}
}

func TestHashString_EmptyReturnsEmpty_Ugly(t *testing.T) {
	if HashString("") != "" {
		t.Fatal("HashString(\"\") returned non-empty")
	}
	if HashString("hello") == "" {
		t.Fatal("HashString(non-empty) returned empty")
	}
}

func TestFileHash_RoundTrip_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "f.txt")
	if result := core.WriteFile(path, []byte("hello"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	h1, err := FileHash(path)
	if err != nil {
		t.Fatalf("FileHash() error = %v", err)
	}
	h2, err := FileHash(path)
	if err != nil {
		t.Fatalf("FileHash() second error = %v", err)
	}
	if h1 != h2 || h1 == "" {
		t.Fatalf("FileHash not stable: %q vs %q", h1, h2)
	}
}

func TestFileHash_MissingFile_Bad(t *testing.T) {
	if _, err := FileHash(core.PathJoin(t.TempDir(), "missing")); err == nil {
		t.Fatal("FileHash(missing) error = nil")
	}
}

// TestFileHash_StreamMatchesBufferLoad_Good — bit-exact parity check
// against the legacy `core.ReadFile + core.SHA256Hex` path. The
// streaming variant in FileHash MUST produce the same digest for any
// file content, otherwise bundle metadata round-trips silently
// regress across the version that flipped the impl.
func TestFileHash_StreamMatchesBufferLoad_Good(t *testing.T) {
	sizes := []int{
		0,             // empty file — boundary
		1,             // single byte — sub-block
		63,            // sub-SHA256-block
		64,            // exactly one SHA256 block
		65,            // one block + remainder
		1024,          // 1KB — small tokenizer
		32*1024 - 1,   // just under stdlib io.Copy default scratch
		32 * 1024,     // exactly stdlib io.Copy default scratch
		32*1024 + 1,   // straddle stdlib scratch boundary
		256 * 1024,    // 256KB
		1024 * 1024,   // 1MB — representative tokenizer.json
		3*1024*1024 + 7, // 3MB + 7 — non-aligned LoRA-scale
	}
	for _, n := range sizes {
		path := core.PathJoin(t.TempDir(), "f.bin")
		data := make([]byte, n)
		for i := range data {
			data[i] = byte(i * 31)
		}
		if result := core.WriteFile(path, data, 0o600); !result.OK {
			t.Fatalf("WriteFile(%d): %s", n, result.Error())
		}
		streamed, err := FileHash(path)
		if err != nil {
			t.Fatalf("FileHash(%d): %v", n, err)
		}
		expected := core.SHA256Hex(data)
		if streamed != expected {
			t.Fatalf("FileHash(%d) parity mismatch:\n  stream=%q\n  buffer=%q", n, streamed, expected)
		}
	}
}

func TestStateURI_BothShapes_Good(t *testing.T) {
	withSeg := StateURI(state.ChunkRef{ChunkID: 5, Segment: "/tmp/x.mp4"})
	withoutSeg := StateURI(state.ChunkRef{ChunkID: 7})
	if withSeg != "state:///tmp/x.mp4#chunk=5" {
		t.Fatalf("with-segment URI = %q", withSeg)
	}
	if withoutSeg != "state://chunk/7" {
		t.Fatalf("without-segment URI = %q", withoutSeg)
	}
}

func TestSAMIFromKV_NilSnapshot_Ugly(t *testing.T) {
	got := SAMIFromKV(nil, nil, SAMIOptions{})
	if got.Architecture != "" || got.NumLayers != 0 || len(got.LayerCoherence) != 0 || len(got.LayerCrossAlignment) != 0 {
		t.Fatalf("SAMIFromKV(nil) = %+v, want zero", got)
	}
}

func TestSAMIFromKV_BuildsLayerArrays_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	sami := SAMIFromKV(snapshot, nil, SAMIOptions{Model: "m", Prompt: "p"})
	if sami.Architecture != "gemma4_text" || sami.NumLayers != 1 {
		t.Fatalf("SAMI = %+v", sami)
	}
	if len(sami.LayerCoherence) != 1 || len(sami.LayerCrossAlignment) != 1 {
		t.Fatalf("SAMI layer arrays = coherence:%d cross:%d", len(sami.LayerCoherence), len(sami.LayerCrossAlignment))
	}
}
