// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestKVSnapshot_Clone_Good(t *testing.T) {
	snapshot := &KVSnapshot{
		Version:      KVSnapshotVersion,
		Tokens:       []int32{1, 2},
		Generated:    []int32{2},
		TokenOffset:  4,
		Architecture: "gemma4_text",
		LogitShape:   []int32{1, 1, 3},
		Logits:       []float32{0.1, 0.2, 0.7},
		Layers: []KVLayerSnapshot{{
			Layer: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	cloned := snapshot.Clone()
	cloned.Tokens[0] = 99
	cloned.Generated[0] = 88
	cloned.Logits[0] = 0.9
	cloned.LogitShape[0] = 9
	cloned.Layers[0].Heads[0].Key[0] = 88

	if snapshot.Tokens[0] != 1 || snapshot.Generated[0] != 2 || snapshot.Logits[0] != 0.1 || snapshot.LogitShape[0] != 1 || snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Clone() returned aliased snapshot data")
	}
}

func TestKVSnapshot_SaveLoadRestorable_Good(t *testing.T) {
	coverageTokens := "KVSnapshot SaveLoadRestorable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := &KVSnapshot{
		Version:       KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{11, 12},
		Generated:     []int32{12},
		TokenOffset:   9,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 4},
		Logits:        []float32{0.1, 0.2, 0.3, 0.4},
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "restorable.kvbin")

	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := LoadKVSnapshot(path)

	if err != nil {
		t.Fatalf("LoadKVSnapshot() error = %v", err)
	}
	if loaded.Version != KVSnapshotVersion || loaded.TokenOffset != 9 || loaded.Generated[0] != 12 {
		t.Fatalf("loaded version/offset/generated = %d/%d/%v", loaded.Version, loaded.TokenOffset, loaded.Generated)
	}
	if len(loaded.LogitShape) != 3 || loaded.LogitShape[2] != 4 || len(loaded.Logits) != 4 || loaded.Logits[3] != 0.4 {
		t.Fatalf("loaded logits = shape %v values %v", loaded.LogitShape, loaded.Logits)
	}
}

func TestKVSnapshot_Head_Ugly(t *testing.T) {
	snapshot := &KVSnapshot{
		Layers: []KVLayerSnapshot{{
			Layer: 7,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1},
				Value: []float32{2},
			}},
		}},
	}

	if _, ok := snapshot.Head(0, 0); ok {
		t.Fatal("Head(0, 0) ok = true for sparse layer 7")
	}
	if head, ok := snapshot.Head(7, 0); !ok || head.Key[0] != 1 || head.Value[0] != 2 {
		t.Fatalf("Head(7, 0) = %+v/%v, want sparse layer data", head, ok)
	}
}

func TestKVSnapshot_Clone_Bad(t *testing.T) {
	var snapshot *KVSnapshot

	if snapshot.Clone() != nil {
		t.Fatal("Clone() on nil snapshot returned non-nil")
	}
}

func TestKVSnapshot_Clone_Ugly(t *testing.T) {
	snapshot := &KVSnapshot{
		Layers: []KVLayerSnapshot{{Layer: 7}},
	}

	cloned := snapshot.Clone()

	if len(cloned.Layers) != 1 || cloned.Layers[0].Layer != 7 || cloned.Layers[0].Heads != nil {
		t.Fatalf("Clone() sparse layer = %+v, want preserved sparse metadata", cloned.Layers)
	}
}

func TestKVSnapshot_Save_Bad(t *testing.T) {
	var snapshot *KVSnapshot

	if err := snapshot.Save(core.PathJoin(t.TempDir(), "nil.kvbin")); err == nil {
		t.Fatal("Save() error = nil, want nil snapshot error")
	}
}

func TestLoadKVSnapshot_Bad(t *testing.T) {
	_, err := LoadKVSnapshot(core.PathJoin(t.TempDir(), "missing.kvbin"))

	if err == nil {
		t.Fatal("LoadKVSnapshot() error = nil, want missing file error")
	}
}

func TestLoadKVSnapshot_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.kvbin")
	if result := core.WriteFile(path, []byte("not-a-kv-snapshot"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := LoadKVSnapshot(path)

	if err == nil {
		t.Fatal("LoadKVSnapshot() error = nil, want corrupt file error")
	}
}
