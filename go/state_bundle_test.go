// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/lora"
)

// These tests cover the mlx-root state_bundle.go shim. The canonical
// algorithmic coverage lives in go-mlx/go/bundle/bundle_test.go; here
// we exercise the boundary converters + legacy alias surface.

func TestStateBundle_AliasMatchesBundle_Good(t *testing.T) {
	// Type aliases are identical types in Go's type system, so this
	// assignment compiles only if the alias is wired through.
	var b *StateBundle = &bundle.Bundle{Version: bundle.Version, Kind: bundle.Kind, KV: stateBundleTestSnapshot()}
	if b.Kind != StateBundleKind || b.Version != StateBundleVersion {
		t.Fatalf("alias constants disagree: kind=%q version=%d", b.Kind, b.Version)
	}
}

func TestNewStateBundle_ConvertsModelInfoAndSampler_Good(t *testing.T) {
	snapshot := stateBundleTestSnapshot()
	b, err := NewStateBundle(snapshot, StateBundleOptions{
		Model:     "gemma4-e4b",
		ModelPath: "/models/gemma4",
		ModelInfo: ModelInfo{
			Architecture: "gemma4_text", VocabSize: 262144, NumLayers: 1,
			QuantBits: 4, ContextLength: 131072,
			Adapter: lora.AdapterInfo{Name: "a", Path: "/p", Hash: "h", Rank: 8},
		},
		Prompt: "p",
		Sampler: GenerateConfig{
			MaxTokens: 32, Temperature: 0.2, TopK: 4,
			StopTokens: []int32{1, 2}, RepeatPenalty: 1.1,
		},
	})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}
	if b.Model.Architecture != "gemma4_text" || b.Model.VocabSize != 262144 || b.Model.NumLayers != 1 {
		t.Fatalf("model = %+v", b.Model)
	}
	if b.Sampler.MaxTokens != 32 || b.Sampler.Temperature != 0.2 || b.Sampler.TopK != 4 || b.Sampler.RepeatPenalty != 1.1 {
		t.Fatalf("sampler = %+v", b.Sampler)
	}
	if len(b.Sampler.StopTokens) != 2 {
		t.Fatalf("stop tokens lost: %v", b.Sampler.StopTokens)
	}
	if b.Adapter.Name != "a" || b.Adapter.Path != "/p" || b.Adapter.Hash != "h" || b.Adapter.Rank != 8 {
		t.Fatalf("adapter (from ModelInfo) = %+v", b.Adapter)
	}
}

func TestNewStateBundle_NilSnapshot_Bad(t *testing.T) {
	if _, err := NewStateBundle(nil, StateBundleOptions{}); err == nil {
		t.Fatal("NewStateBundle(nil) error = nil")
	}
}

func TestStateSamplerFromGenerateConfig_ClonesStopTokens_Good(t *testing.T) {
	stops := []int32{1, 2}
	out := stateSamplerFromGenerateConfig(GenerateConfig{MaxTokens: 4, StopTokens: stops})
	stops[0] = 99
	if out.StopTokens[0] == 99 {
		t.Fatal("stateSamplerFromGenerateConfig did not clone StopTokens")
	}
	if out.MaxTokens != 4 {
		t.Fatalf("MaxTokens = %d", out.MaxTokens)
	}
}

func TestModelInfoToBundle_FieldByField_Good(t *testing.T) {
	in := ModelInfo{
		Architecture: "qwen3", VocabSize: 151936, NumLayers: 28, HiddenSize: 2048,
		QuantBits: 4, QuantGroup: 32, ContextLength: 32768,
		Adapter: lora.AdapterInfo{Name: "v1", Rank: 8, TargetKeys: []string{"q_proj"}},
	}
	out := modelInfoToBundle(in)
	if out.Architecture != in.Architecture || out.NumLayers != in.NumLayers ||
		out.HiddenSize != in.HiddenSize || out.ContextLength != in.ContextLength {
		t.Fatalf("scalar copy lost: %+v vs %+v", out, in)
	}
	if out.Adapter.Name != "v1" || out.Adapter.Rank != 8 || len(out.Adapter.TargetKeys) != 1 {
		t.Fatalf("adapter copy lost: %+v", out.Adapter)
	}
}

func TestCheckStateBundleCompatibility_Good(t *testing.T) {
	b, err := NewStateBundle(stateBundleTestSnapshot(), StateBundleOptions{
		ModelInfo: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}
	if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}, b); err != nil {
		t.Fatalf("CheckStateBundleCompatibility(good) error = %v", err)
	}
	if err := CheckStateBundleCompatibility(ModelInfo{Architecture: "llama", NumLayers: 1}, b); err == nil {
		t.Fatal("CheckStateBundleCompatibility(bad arch) error = nil")
	}
}

func TestStateBundleFileHash_RoundTrip_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "f")
	if result := core.WriteFile(path, []byte("hi"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	h, err := StateBundleFileHash(path)
	if err != nil {
		t.Fatalf("StateBundleFileHash() error = %v", err)
	}
	if h == "" {
		t.Fatal("StateBundleFileHash returned empty")
	}
}

func TestLoadStateBundle_RoundTripsViaBundle_Good(t *testing.T) {
	b, err := NewStateBundle(stateBundleTestSnapshot(), StateBundleOptions{Prompt: "p"})
	if err != nil {
		t.Fatalf("NewStateBundle() error = %v", err)
	}
	path := core.PathJoin(t.TempDir(), "state.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := LoadStateBundle(path)
	if err != nil {
		t.Fatalf("LoadStateBundle() error = %v", err)
	}
	if loaded.Kind != StateBundleKind || loaded.Prompt.Text != "p" {
		t.Fatalf("loaded = %+v", loaded)
	}
}

func TestStateBundleSnapshot_MemvidShimRoute_Good(t *testing.T) {
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
	b := &StateBundle{
		Version: StateBundleVersion, Kind: StateBundleKind, KVHash: hash,
		Refs: []StateBundleRef{{Kind: StateBundleRefMemvid, URI: stateMemvidURI(ref), Memvid: ref}},
	}
	loaded, err := b.SnapshotFromMemvid(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture {
		t.Fatalf("loaded architecture = %q", loaded.Architecture)
	}
}

func TestStateBundleTokenizerHelper_FillsHashes_Good(t *testing.T) {
	out := stateBundleTokenizer(StateBundleTokenizer{Path: "/tok", ChatTemplate: "<bos>"})
	if out.Hash == "" || out.ChatTemplateHash == "" {
		t.Fatalf("stateBundleTokenizer left hashes empty: %+v", out)
	}
}

func TestStateHashHelper_Empty_Ugly(t *testing.T) {
	if stateHash("") != "" {
		t.Fatal("stateHash(\"\") returned non-empty")
	}
	if stateHash("x") == "" {
		t.Fatal("stateHash(x) returned empty")
	}
}
