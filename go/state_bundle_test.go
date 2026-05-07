// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
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

func stateBundleTestSnapshot() *KVSnapshot {
	return &KVSnapshot{
		Version:       KVSnapshotVersion,
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
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}
