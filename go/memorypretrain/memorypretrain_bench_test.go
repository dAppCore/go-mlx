// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
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

// --- Outer artifact-build + corpus-load path benches (AX-11) ---------------
//
// Benches the outer offline pipeline + corpus/bank load functions the prior
// memorypretrain sweep left un-benched: BuildMemoryPretrainingArtifacts,
// BuildBankFromCorpus, LoadCorpusRecordsJSONL{,File}, LoadBank,
// LoadFFNMemoryBank. The inner tree (BuildBank/buildNode) and NewFFNMemoryBank
// already have benches above; the outer frames are measured here.

var (
	memoryPretrainBenchRecordSink   []CorpusRecord
	memoryPretrainBenchArtifactSink *MemoryPretrainingArtifacts
)

// benchCorpusDim is the embedding width used by the build-path fixtures. Small
// enough to keep the synthetic embedder cheap, wide enough that the
// hierarchical KMeans build does real work.
const benchCorpusDim = 16

// benchCorpusRecords builds n deterministic corpus records with id, text, and a
// two-key meta map, varied enough that the routed clusters are non-trivial.
func benchCorpusRecords(n int) []CorpusRecord {
	records := make([]CorpusRecord, n)
	for i := range records {
		records[i] = CorpusRecord{
			ID:   "rec-" + strconv.Itoa(i),
			Text: "memory planning corpus row " + strconv.Itoa(i),
			Meta: map[string]string{
				"source": "bench",
				"shard":  strconv.Itoa(i % 8),
			},
		}
	}
	return records
}

// benchCorpusEmbedder maps text to a deterministic unit-ish vector keyed off a
// cheap rolling hash so the build forms several clusters without a model.
func benchCorpusEmbedder() EmbedFunc {
	return EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		vec := make([]float32, benchCorpusDim)
		var h uint32 = 2166136261
		for i := 0; i < len(text); i++ {
			h ^= uint32(text[i])
			h *= 16777619
		}
		axis := int(h) % benchCorpusDim
		if axis < 0 {
			axis += benchCorpusDim
		}
		vec[axis] = 1
		vec[(axis+1)%benchCorpusDim] = 0.25
		vec[(axis+2)%benchCorpusDim] = 0.05
		return vec, nil
	})
}

// benchCorpusJSONL renders n corpus records as a JSONL document matching the
// id/text/meta shape LoadCorpusRecordsJSONL parses.
func benchCorpusJSONL(n int) string {
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sb.WriteString(`{"id":"rec-`)
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString(`","text":"memory planning corpus row `)
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString(`","meta":{"source":"bench","shard":"`)
		sb.WriteString(strconv.Itoa(i % 8))
		sb.WriteString(`"}}` + "\n")
	}
	return sb.String()
}

func benchArtifactFFNConfig() FFNMemoryConfig {
	return FFNMemoryConfig{
		HiddenSize:       benchCorpusDim,
		Layers:           4,
		MemoryLevels:     []string{"1", "2", "3"},
		FFNMemoryTokens:  []int{4, 8, 16},
		NumClusters:      []int{8, 4, 2},
		AddedGenericSize: 1,
	}
}

func BenchmarkLoadCorpusRecordsJSONL(b *testing.B) {
	raw := benchCorpusJSONL(2000)
	b.ReportAllocs()
	for b.Loop() {
		records, err := LoadCorpusRecordsJSONL(raw)
		if err != nil {
			b.Fatalf("LoadCorpusRecordsJSONL() error = %v", err)
		}
		memoryPretrainBenchRecordSink = records
	}
}

func BenchmarkLoadCorpusRecordsJSONLFile(b *testing.B) {
	raw := benchCorpusJSONL(2000)
	path := core.PathJoin(b.TempDir(), "corpus.jsonl")
	if result := core.WriteFile(path, []byte(raw), 0o644); !result.OK {
		b.Fatalf("WriteFile() error = %v", result.Value)
	}
	b.ReportAllocs()
	for b.Loop() {
		records, err := LoadCorpusRecordsJSONLFile(path)
		if err != nil {
			b.Fatalf("LoadCorpusRecordsJSONLFile() error = %v", err)
		}
		memoryPretrainBenchRecordSink = records
	}
}

func BenchmarkBuildBankFromCorpus(b *testing.B) {
	records := benchCorpusRecords(512)
	embedder := benchCorpusEmbedder()
	cfg := BuildConfig{BranchingFactor: 8, MaxDepth: 3, MinClusterSize: 8}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		bank, err := BuildBankFromCorpus(ctx, embedder, records, cfg)
		if err != nil {
			b.Fatalf("BuildBankFromCorpus() error = %v", err)
		}
		memoryPretrainBenchBankSink = bank
	}
}

func BenchmarkBuildMemoryPretrainingArtifacts(b *testing.B) {
	records := benchCorpusRecords(512)
	embedder := benchCorpusEmbedder()
	// Empty Router/FFNMemory/ClusterID paths: pure in-memory pipeline, no Save,
	// no JSONL enrichment.
	cfg := MemoryPretrainingArtifactConfig{
		Build:     BuildConfig{BranchingFactor: 8, MaxDepth: 3, MinClusterSize: 8},
		FFNMemory: benchArtifactFFNConfig(),
	}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		artifacts, err := BuildMemoryPretrainingArtifacts(ctx, embedder, records, cfg)
		if err != nil {
			b.Fatalf("BuildMemoryPretrainingArtifacts() error = %v", err)
		}
		memoryPretrainBenchArtifactSink = artifacts
	}
}

func BenchmarkLoadBank(b *testing.B) {
	bank, err := BuildBankFromCorpus(context.Background(), benchCorpusEmbedder(), benchCorpusRecords(512), BuildConfig{BranchingFactor: 8, MaxDepth: 3, MinClusterSize: 8})
	if err != nil {
		b.Fatalf("BuildBankFromCorpus() error = %v", err)
	}
	path := core.PathJoin(b.TempDir(), "bank.json")
	if err := SaveBank(path, bank); err != nil {
		b.Fatalf("SaveBank() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		loaded, err := LoadBank(path)
		if err != nil {
			b.Fatalf("LoadBank() error = %v", err)
		}
		memoryPretrainBenchBankSink = loaded
	}
}

func BenchmarkLoadFFNMemoryBank(b *testing.B) {
	bank, err := NewFFNMemoryBank(benchArtifactFFNConfig())
	if err != nil {
		b.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	path := core.PathJoin(b.TempDir(), "ffn.json")
	if err := SaveFFNMemoryBank(path, bank); err != nil {
		b.Fatalf("SaveFFNMemoryBank() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		loaded, err := LoadFFNMemoryBank(path)
		if err != nil {
			b.Fatalf("LoadFFNMemoryBank() error = %v", err)
		}
		memoryPretrainBenchFFNSink = loaded
	}
}
