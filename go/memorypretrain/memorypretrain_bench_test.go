// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import "testing"

var memoryPretrainBenchSink []Retrieval
var memoryPretrainBenchVectorSink []float32

func BenchmarkBank_Retrieve_LeafCluster(b *testing.B) {
	blocks := make([]Block, 256)
	for i := range blocks {
		axis := i % 4
		embedding := make([]float32, 16)
		embedding[axis] = 1
		embedding[(axis+i)%16] += 0.1
		blocks[i] = Block{ID: "block", Embedding: embedding}
	}
	bank, err := BuildBank(blocks, BuildConfig{BranchingFactor: 4, MaxDepth: 3, MinClusterSize: 8})
	if err != nil {
		b.Fatalf("BuildBank() error = %v", err)
	}
	query := make([]float32, 16)
	query[0] = 1
	scratch := make([]Retrieval, 0, 64)
	b.ReportAllocs()
	for b.Loop() {
		memoryPretrainBenchSink, err = bank.RetrieveInto(scratch, query, 8)
		if err != nil {
			b.Fatalf("Retrieve() error = %v", err)
		}
	}
}

func BenchmarkBank_InjectAdditive_LeafCluster(b *testing.B) {
	blocks := make([]Block, 256)
	for i := range blocks {
		axis := i % 4
		embedding := make([]float32, 16)
		embedding[axis] = 1
		embedding[(axis+i)%16] += 0.1
		blocks[i] = Block{ID: "block", Embedding: embedding}
	}
	bank, err := BuildBank(blocks, BuildConfig{BranchingFactor: 4, MaxDepth: 3, MinClusterSize: 8})
	if err != nil {
		b.Fatalf("BuildBank() error = %v", err)
	}
	query := make([]float32, 16)
	query[0] = 1
	hidden := make([]float32, 16)
	scratch := make([]Retrieval, 0, 64)
	dst := make([]float32, 0, 16)
	cfg := InjectionConfig{TopK: 8, Scale: 0.25, PositiveScoresOnly: true}
	b.ReportAllocs()
	for b.Loop() {
		memoryPretrainBenchVectorSink, memoryPretrainBenchSink, _, err = bank.InjectAdditive(dst, hidden, query, scratch, cfg)
		if err != nil {
			b.Fatalf("InjectAdditive() error = %v", err)
		}
	}
}
