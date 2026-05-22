// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"math"
	"testing"
)

func TestAnalyzeKV_Coherent_Good(t *testing.T) {
	snapshot := makeKVAnalysisCoherentSnapshot(4, 8, 4, 4)

	result := Analyze(snapshot)

	if result.GQA {
		t.Fatal("GQA = true, want false for 8 heads")
	}
	if result.MeanKeyCoherence < 0.9 {
		t.Fatalf("MeanKeyCoherence = %.3f, want high coherence", result.MeanKeyCoherence)
	}
	if result.MeanValueCoherence < 0.9 {
		t.Fatalf("MeanValueCoherence = %.3f, want high coherence", result.MeanValueCoherence)
	}
	if result.MeanKVCoupling < 0.9 {
		t.Fatalf("MeanKVCoupling = %.3f, want high K/V coupling", result.MeanKVCoupling)
	}
	if result.PhaseLockScore < 0.9 {
		t.Fatalf("PhaseLockScore = %.3f, want high phase lock", result.PhaseLockScore)
	}
	if result.JointCollapseCount != 0 {
		t.Fatalf("JointCollapseCount = %d, want 0", result.JointCollapseCount)
	}
}

func TestAnalyzeKV_Orthogonal_Bad(t *testing.T) {
	snapshot := makeKVAnalysisOrthogonalSnapshot(4, 8, 4, 8)

	result := Analyze(snapshot)

	if result.GQA {
		t.Fatal("GQA = true, want false for 8 heads")
	}
	if result.MeanKeyCoherence > 0.3 {
		t.Fatalf("MeanKeyCoherence = %.3f, want low coherence for orthogonal heads", result.MeanKeyCoherence)
	}
	if result.MeanValueCoherence > 0.3 {
		t.Fatalf("MeanValueCoherence = %.3f, want low coherence for orthogonal heads", result.MeanValueCoherence)
	}
}

func TestAnalyzeKV_GQA_Ugly(t *testing.T) {
	snapshot := makeKVAnalysisCoherentSnapshot(4, 1, 4, 4)

	result := Analyze(snapshot)

	if !result.GQA {
		t.Fatal("GQA = false, want true for single KV head")
	}
	if result.MeanKeyCoherence > 0.1 {
		t.Fatalf("MeanKeyCoherence = %.3f, want low position differentiation for identical positions", result.MeanKeyCoherence)
	}
	if len(result.LayerCrossAlignment) != 3 {
		t.Fatalf("LayerCrossAlignment len = %d, want 3", len(result.LayerCrossAlignment))
	}
}

func TestKVAnalysis_Composite_Good(t *testing.T) {
	result := &Analysis{
		MeanKeyCoherence:       1,
		MeanValueCoherence:     1,
		MeanCrossAlignment:     1,
		MeanHeadEntropy:        1,
		PhaseLockScore:         1,
		MeanKVCoupling:         1,
		JointCollapseCount:     0,
		LayerKeyCoherence:      []float64{1, 1},
		LayerValueCoherence:    []float64{1, 1},
		LayerCrossAlignment:    []float64{1},
		LayerKVCoupling:        []float64{1, 1},
		SharedCacheLayerGroups: map[int][]int{0: {0, 1}},
	}

	score := result.Composite()

	if score != 10000 {
		t.Fatalf("Composite() = %d, want 10000", score)
	}
}

func TestKVAnalysis_Composite_Bad(t *testing.T) {
	result := &Analysis{JointCollapseCount: 10}

	score := result.Composite()

	if score != 0 {
		t.Fatalf("Composite() = %d, want 0", score)
	}
}

func TestKVFeatures_Ugly(t *testing.T) {
	features := Features(nil)
	labels := FeatureLabels()

	if len(features) != 7 {
		t.Fatalf("Features(nil) len = %d, want 7", len(features))
	}
	if len(labels) != len(features) {
		t.Fatalf("FeatureLabels len = %d, want %d", len(labels), len(features))
	}
	for _, value := range features {
		if value != 0 {
			t.Fatalf("Features(nil) contains %f, want zeros", value)
		}
	}
}

func TestKVFeatures_Good(t *testing.T) {
	result := &Analysis{
		MeanKeyCoherence:   0.1,
		MeanValueCoherence: 0.2,
		MeanCrossAlignment: 0.3,
		MeanHeadEntropy:    0.4,
		PhaseLockScore:     0.5,
		MeanKVCoupling:     0.6,
		JointCollapseCount: 1,
	}

	features := Features(result)

	if len(features) != 7 {
		t.Fatalf("Features len = %d, want 7", len(features))
	}
	if features[0] != 0.1 || features[5] != 0.6 || math.Abs(features[6]-0.8) > 1e-6 {
		t.Fatalf("Features = %v, want ordered K/V metrics", features)
	}
}

func TestKVFeatureLabels_Good(t *testing.T) {
	labels := FeatureLabels()

	if len(labels) != 7 {
		t.Fatalf("FeatureLabels len = %d, want 7", len(labels))
	}
	if labels[0] != "key_coherence" || labels[5] != "kv_coupling" {
		t.Fatalf("FeatureLabels = %v, want stable K/V axis labels", labels)
	}
}

func TestKVAnalysisCosine32_Good(t *testing.T) {
	got := kvAnalysisCosine32([]float32{1, 0, 0}, []float32{1, 0, 0})

	if math.Abs(got-1) > 1e-6 {
		t.Fatalf("kvAnalysisCosine32 = %f, want 1", got)
	}
}

func TestKVAnalysisCosine32_Bad(t *testing.T) {
	got := kvAnalysisCosine32([]float32{1, 0, 0}, []float32{0, 1, 0})

	if math.Abs(got) > 1e-6 {
		t.Fatalf("kvAnalysisCosine32 = %f, want 0 for orthogonal vectors", got)
	}
}

func TestKVAnalysisHeadEntropy_Ugly(t *testing.T) {
	got := kvAnalysisHeadEntropy([]float32{1, 0, 1, 0}, 2, 2, nil)

	if math.Abs(got-1) > 1e-6 {
		t.Fatalf("kvAnalysisHeadEntropy = %f, want 1 for balanced magnitudes", got)
	}
}

func makeKVAnalysisCoherentSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
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
	head := make([]float32, seqLen*headDim)
	for pos := range seqLen {
		head[pos*headDim] = 1
	}
	for layer := range layers {
		snapshot.Layers[layer] = LayerSnapshot{
			Layer:      layer,
			CacheIndex: layer,
			Heads:      make([]HeadSnapshot, heads),
		}
		for h := range heads {
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{
				Key:   append([]float32(nil), head...),
				Value: append([]float32(nil), head...),
			}
		}
	}
	return snapshot
}

func makeKVAnalysisOrthogonalSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
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
		snapshot.Layers[layer] = LayerSnapshot{
			Layer:      layer,
			CacheIndex: layer,
			Heads:      make([]HeadSnapshot, heads),
		}
		for h := range heads {
			key := make([]float32, seqLen*headDim)
			value := make([]float32, seqLen*headDim)
			for pos := range seqLen {
				key[pos*headDim+h%headDim] = 1
				value[pos*headDim+(heads-h-1)%headDim] = 1
			}
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{Key: key, Value: value}
		}
	}
	return snapshot
}
