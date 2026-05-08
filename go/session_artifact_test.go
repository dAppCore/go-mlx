// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

func TestSAMIFromKV_Good(t *testing.T) {
	snapshot := sessionArtifactTestSnapshot()
	analysis := &KVAnalysis{
		MeanKeyCoherence:    0.8,
		MeanValueCoherence:  0.6,
		MeanCrossAlignment:  0.5,
		MeanHeadEntropy:     0.4,
		PhaseLockScore:      0.9,
		JointCollapseCount:  1,
		LayerKeyCoherence:   []float64{0.7, 0.9},
		LayerValueCoherence: []float64{0.5, 0.7},
		LayerCrossAlignment: []float64{0.25},
	}

	got := SAMIFromKV(snapshot, analysis, SAMIOptions{Model: "lem-gemma", Prompt: "trace me"})

	if got.Model != "lem-gemma" || got.Prompt != "trace me" || got.Architecture != "gemma4_text" {
		t.Fatalf("SAMI identity = %+v", got)
	}
	if got.NumLayers != 2 || got.NumHeads != 1 || got.SeqLen != 2 || got.HeadDim != 2 {
		t.Fatalf("SAMI shape = %+v", got)
	}
	if got.MeanCoherence != 0.7 {
		t.Fatalf("MeanCoherence = %f, want 0.7", got.MeanCoherence)
	}
	if len(got.LayerCoherence) != got.NumLayers || len(got.LayerCrossAlignment) != got.NumLayers {
		t.Fatalf("layer lengths = %d/%d, want %d", len(got.LayerCoherence), len(got.LayerCrossAlignment), got.NumLayers)
	}
	if got.LayerCoherence[0] != 0.6 || got.LayerCrossAlignment[1] != 0.5 {
		t.Fatalf("layer metrics = %+v / %+v", got.LayerCoherence, got.LayerCrossAlignment)
	}
	if got.Composite <= 0 || got.Composite > 100 {
		t.Fatalf("Composite = %f, want 0..100", got.Composite)
	}
}

func TestSAMIFromKV_Bad(t *testing.T) {
	got := SAMIFromKV(nil, nil, SAMIOptions{})

	if got.NumLayers != 0 || got.Composite != 0 {
		t.Fatalf("nil SAMI result = %+v, want zero shape", got)
	}
}

func TestSAMIFromKV_Ugly(t *testing.T) {
	snapshot := sessionArtifactTestSnapshot()
	analysis := &KVAnalysis{
		MeanKeyCoherence:       2,
		MeanValueCoherence:     -1,
		MeanCrossAlignment:     3,
		MeanHeadEntropy:        -2,
		PhaseLockScore:         4,
		LayerKeyCoherence:      []float64{2},
		LayerValueCoherence:    []float64{-1},
		LayerCrossAlignment:    nil,
		JointCollapseCount:     99,
		SharedCacheLayerGroups: map[int][]int{},
	}

	got := SAMIFromKV(snapshot, analysis, SAMIOptions{})

	if got.MeanCoherence != 0.5 || got.MeanCrossAlignment != 1 || got.MeanHeadEntropy != 0 || got.PhaseLockScore != 1 {
		t.Fatalf("clamped means = %+v", got)
	}
	if got.JointCollapseCount != got.NumLayers {
		t.Fatalf("JointCollapseCount = %d, want %d", got.JointCollapseCount, got.NumLayers)
	}
}

func TestExportSessionArtifacts_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	path := core.PathJoin(t.TempDir(), "state.kvbin")

	artifact, err := ExportSessionArtifacts(context.Background(), sessionArtifactTestSnapshot(), SessionArtifactOptions{
		Model:  "lem-gemma",
		Prompt: "trace me",
		KVPath: path,
		Store:  store,
		URI:    "mlx://session/lem-gemma/trace",
		Title:  "LEM Gemma trace",
		Tags:   map[string]string{"arch": "gemma4_text"},
	})

	if err != nil {
		t.Fatalf("ExportSessionArtifacts() error = %v", err)
	}
	if artifact.KVPath != path {
		t.Fatalf("KVPath = %q, want %q", artifact.KVPath, path)
	}
	if artifact.ChunkRef.Codec != memvid.CodecMemory || artifact.ChunkRef.ChunkID == 0 {
		t.Fatalf("ChunkRef = %#v, want memory chunk", artifact.ChunkRef)
	}
	if artifact.SAMI.Model != "lem-gemma" || len(artifact.Features) != len(KVFeatureLabels()) {
		t.Fatalf("artifact = %+v", artifact)
	}
	if _, err := LoadKVSnapshot(path); err != nil {
		t.Fatalf("LoadKVSnapshot() error = %v", err)
	}
	chunk, err := store.Resolve(context.Background(), artifact.ChunkRef.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if !core.Contains(chunk.Text, `"sami"`) || !core.Contains(chunk.Text, `"feature_labels"`) {
		t.Fatalf("artifact chunk text = %q", chunk.Text)
	}
}

func TestExportSessionArtifacts_Bad(t *testing.T) {
	_, err := ExportSessionArtifacts(context.Background(), nil, SessionArtifactOptions{})

	if err == nil {
		t.Fatal("expected nil snapshot error")
	}
}

func TestExportSessionArtifacts_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := ExportSessionArtifacts(ctx, sessionArtifactTestSnapshot(), SessionArtifactOptions{})

	if !core.Is(err, context.Canceled) {
		t.Fatalf("ExportSessionArtifacts() error = %v, want context.Canceled", err)
	}
}

func sessionArtifactTestSnapshot() *KVSnapshot {
	return &KVSnapshot{
		Version:       KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		Layers: []KVLayerSnapshot{
			{
				Layer:      0,
				CacheIndex: 0,
				Heads: []KVHeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			},
			{
				Layer:      1,
				CacheIndex: 1,
				Heads: []KVHeadSnapshot{{
					Key:   []float32{1, 1, 0, 0},
					Value: []float32{0, 0, 1, 1},
				}},
			},
		},
	}
}
