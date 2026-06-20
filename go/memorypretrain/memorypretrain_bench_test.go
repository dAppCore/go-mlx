// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

var memoryPretrainBenchSink []Retrieval
var memoryPretrainBenchVectorSink []float32
var memoryPretrainBenchBankSink *Bank

func benchBuildBlocks(n, dim int) []Block {
	blocks := make([]Block, n)
	for i := range blocks {
		embedding := make([]float32, dim)
		axis := i % dim
		embedding[axis] = 1
		embedding[(axis+i)%dim] += 0.1
		embedding[(axis+2*i)%dim] += 0.01
		blocks[i] = Block{ID: "block", Embedding: embedding}
	}
	return blocks
}

func BenchmarkBuildBank(b *testing.B) {
	blocks := benchBuildBlocks(2000, 16)
	cfg := BuildConfig{BranchingFactor: 8, MaxDepth: 3, MinClusterSize: 8}
	b.ReportAllocs()
	for b.Loop() {
		bank, err := BuildBank(blocks, cfg)
		if err != nil {
			b.Fatalf("BuildBank() error = %v", err)
		}
		memoryPretrainBenchBankSink = bank
	}
}

var memoryPretrainBenchFFNSink *FFNMemoryBank
var memoryPretrainBenchStringSink string
var memoryPretrainBenchReportSink ClusterIDJSONLReport

func BenchmarkNewFFNMemoryBank(b *testing.B) {
	cfg := FFNMemoryConfig{
		HiddenSize:      16,
		Layers:          4,
		MemoryLevels:    []string{"1", "2", "3"},
		FFNMemoryTokens: []int{4, 8, 16},
		NumClusters:     []int{8, 4, 2},
	}
	b.ReportAllocs()
	for b.Loop() {
		bank, err := NewFFNMemoryBank(cfg)
		if err != nil {
			b.Fatalf("NewFFNMemoryBank() error = %v", err)
		}
		memoryPretrainBenchFFNSink = bank
	}
}

func benchClusterIDJSONL(rows int) string {
	var sb strings.Builder
	for i := 0; i < rows; i++ {
		sb.WriteString(`{"id":"r`)
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString(`","context":"Go memory planning row `)
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString(`"}` + "\n")
	}
	return sb.String()
}

func BenchmarkAddClusterIDsToJSONL_Generic(b *testing.B) {
	raw := benchClusterIDJSONL(256)
	cfg := ClusterIDJSONLConfig{TaskType: ClusterIDTaskLanguageModeling, ClusterCounts: []int{3, 5}}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		out, report, err := AddClusterIDsToJSONL(ctx, raw, nil, nil, cfg)
		if err != nil {
			b.Fatalf("AddClusterIDsToJSONL() error = %v", err)
		}
		memoryPretrainBenchStringSink = out
		memoryPretrainBenchReportSink = report
	}
}

func BenchmarkAddClusterIDsToJSONL_Learned(b *testing.B) {
	router, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		b.Fatalf("BuildBank() error = %v", err)
	}
	raw := benchClusterIDJSONL(256)
	cfg := ClusterIDJSONLConfig{TaskType: ClusterIDTaskLanguageModeling, ClusterCounts: []int{2, 5}}
	embedder := EmbedFunc(func(_ context.Context, _ string) ([]float32, error) {
		return []float32{1, 0}, nil
	})
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		out, report, err := AddClusterIDsToJSONL(ctx, raw, embedder, router, cfg)
		if err != nil {
			b.Fatalf("AddClusterIDsToJSONL() error = %v", err)
		}
		memoryPretrainBenchStringSink = out
		memoryPretrainBenchReportSink = report
	}
}

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
