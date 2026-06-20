// SPDX-Licence-Identifier: EUPL-1.2

package kv

import "testing"

// makeKVAnalysisZeroSnapshot builds a snapshot whose head vectors are all
// zero, driving the degenerate-norm guards in the analysis math (zero sums,
// zero entropy, zero anchor components).
func makeKVAnalysisZeroSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Architecture: "test",
		Tokens:       make([]int32, seqLen),
		NumLayers:    layers,
		NumHeads:     heads,
		SeqLen:       seqLen,
		HeadDim:      headDim,
		Layers:       make([]LayerSnapshot, layers),
	}
	for layer := range layers {
		snapshot.Layers[layer] = LayerSnapshot{Layer: layer, CacheIndex: layer, Heads: make([]HeadSnapshot, heads)}
		for h := range heads {
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{
				Key:   make([]float32, seqLen*headDim),
				Value: make([]float32, seqLen*headDim),
			}
		}
	}
	return snapshot
}

// TestAnalysisCover_ZeroVectors_MultiHead drives the all-zero degenerate
// branches of the multi-head analysis path: zero norms, zero entropy, zero
// position sums. Analyze must not panic and returns a populated result.
func TestAnalysisCover_ZeroVectors_MultiHead(t *testing.T) {
	result := Analyze(makeKVAnalysisZeroSnapshot(4, 8, 4, 4))
	if result == nil {
		t.Fatal("Analyze(zero vectors) = nil")
	}
}

// TestAnalysisCover_ZeroVectors_GQA drives the GQA analysis path (single KV
// head) over all-zero vectors, covering the GQA-side degenerate branches.
func TestAnalysisCover_ZeroVectors_GQA(t *testing.T) {
	result := Analyze(makeKVAnalysisZeroSnapshot(4, 1, 4, 4))
	if result == nil {
		t.Fatal("Analyze(zero GQA) = nil")
	}
	if !result.GQA {
		t.Fatal("GQA = false, want true for single KV head")
	}
}

// TestAnalysisCover_SinglePosition drives the count==0 / pairs==0 guards: a
// single-token snapshot has no position pairs to differentiate, so the
// per-position coherence accumulators stay empty.
func TestAnalysisCover_SinglePosition(t *testing.T) {
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 8, 1, 4)); result == nil {
		t.Fatal("Analyze(single position, multi-head) = nil")
	}
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 1, 1, 4)); result == nil {
		t.Fatal("Analyze(single position, GQA) = nil")
	}
}

// TestAnalysisCover_ShortHeadVectors drives the start >= len(head) guard in the
// entropy walk: a head whose backing slice is shorter than seqLen*headDim, so a
// later position's window starts past the end of the data.
func TestAnalysisCover_ShortHeadVectors(t *testing.T) {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Architecture: "test",
		Tokens:       make([]int32, 4),
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      4,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				// Only two positions' worth of data for a seqLen-4 head → the
				// entropy/position walk runs off the end and the guard fires.
				Key:   make([]float32, 2*4),
				Value: make([]float32, 2*4),
			}},
		}},
	}
	if result := Analyze(snapshot); result == nil {
		t.Fatal("Analyze(short head vectors) = nil")
	}
}
