// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestBuildMemoryPretrainingArtifacts_BuildsSavesAndEnriches_Good(t *testing.T) {
	dir := t.TempDir()
	routerPath := core.PathJoin(dir, "memory", "router.json")
	ffnPath := core.PathJoin(dir, "memory", "ffn.json")
	inputPath := core.PathJoin(dir, "tasks", "input.jsonl")
	outputPath := core.PathJoin(dir, "tasks", "clustered.jsonl")
	if result := core.MkdirAll(core.PathDir(inputPath), 0o755); !result.OK {
		t.Fatalf("MkdirAll(input dir) error = %v", result.Value)
	}
	writeFile(t, inputPath, `{"context":"Go memory planning"}`+"\n")
	embedCalls := 0
	embedder := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		embedCalls++
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})

	artifacts, err := BuildMemoryPretrainingArtifacts(context.Background(), embedder, []CorpusRecord{
		{ID: "go-1", Text: "Go memory planning"},
		{ID: "go-2", Text: "Go cgo bridge"},
		{ID: "poem-1", Text: "winter proof poem"},
		{ID: "poem-2", Text: "autumn prayer"},
	}, MemoryPretrainingArtifactConfig{
		RouterPath:    routerPath,
		FFNMemoryPath: ffnPath,
		Build:         BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
		FFNMemory: FFNMemoryConfig{
			HiddenSize:      2,
			Layers:          2,
			MemoryLevels:    []string{"1"},
			FFNMemoryTokens: []int{1},
		},
		ClusterIDInputPath:  inputPath,
		ClusterIDOutputPath: outputPath,
		ClusterIDJSONL:      ClusterIDJSONLConfig{TaskType: ClusterIDTaskLanguageModeling},
	})
	if err != nil {
		t.Fatalf("BuildMemoryPretrainingArtifacts() error = %v", err)
	}
	if artifacts.Router == nil || artifacts.FFNMemory == nil || artifacts.Report == nil {
		t.Fatalf("artifacts = %+v, want router, FFN memory, and report", artifacts)
	}
	if artifacts.FFNMemory.Config.NumClusters[0] != 2 {
		t.Fatalf("FFN num clusters = %+v, want derived router cluster count", artifacts.FFNMemory.Config.NumClusters)
	}
	if artifacts.Report.CorpusRecords != 4 || artifacts.Report.RouterNodes == 0 || artifacts.Report.FFNMemoryLayers != 2 {
		t.Fatalf("report = %+v, want corpus, router, and FFN layer counts", artifacts.Report)
	}
	if artifacts.Report.ClusterIDReport == nil || artifacts.Report.ClusterIDReport.LearnedRows != 1 {
		t.Fatalf("cluster report = %+v, want one learned clustered row", artifacts.Report.ClusterIDReport)
	}
	if embedCalls != 5 {
		t.Fatalf("embed calls = %d, want four corpus records plus one clustered JSONL row", embedCalls)
	}
	if _, err := LoadBank(routerPath); err != nil {
		t.Fatalf("LoadBank(routerPath) error = %v", err)
	}
	if _, err := LoadFFNMemoryBank(ffnPath); err != nil {
		t.Fatalf("LoadFFNMemoryBank(ffnPath) error = %v", err)
	}
	read := core.ReadFile(outputPath)
	if !read.OK {
		t.Fatalf("ReadFile(outputPath) error = %v", read.Value)
	}
	if got := core.AsString(read.Value.([]byte)); !strings.Contains(got, `"cluster_ids":[0]`) {
		t.Fatalf("clustered JSONL = %s, want learned cluster IDs", got)
	}
}

func TestBuildMemoryPretrainingArtifacts_ClusterIDsMatchFFNMemoryLevels_Good(t *testing.T) {
	dir := t.TempDir()
	inputPath := core.PathJoin(dir, "tasks", "input.jsonl")
	outputPath := core.PathJoin(dir, "tasks", "clustered.jsonl")
	if result := core.MkdirAll(core.PathDir(inputPath), 0o755); !result.OK {
		t.Fatalf("MkdirAll(input dir) error = %v", result.Value)
	}
	writeFile(t, inputPath, `{"context":"Go memory planning"}`+"\n")
	embedder := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})

	artifacts, err := BuildMemoryPretrainingArtifacts(context.Background(), embedder, []CorpusRecord{
		{ID: "go-1", Text: "Go memory planning"},
		{ID: "go-2", Text: "Go cgo bridge"},
		{ID: "poem-1", Text: "winter proof poem"},
		{ID: "poem-2", Text: "autumn prayer"},
	}, MemoryPretrainingArtifactConfig{
		Build: BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
		FFNMemory: FFNMemoryConfig{
			HiddenSize:      2,
			Layers:          1,
			MemoryLevels:    []string{"1", "2"},
			FFNMemoryTokens: []int{1, 1},
			NumClusters:     []int{2, 4},
		},
		ClusterIDInputPath:  inputPath,
		ClusterIDOutputPath: outputPath,
		ClusterIDJSONL:      ClusterIDJSONLConfig{TaskType: ClusterIDTaskLanguageModeling},
	})
	if err != nil {
		t.Fatalf("BuildMemoryPretrainingArtifacts() error = %v", err)
	}
	if artifacts.Report.ClusterIDReport == nil || artifacts.Report.ClusterIDReport.LearnedRows != 1 {
		t.Fatalf("cluster report = %+v, want one learned clustered row", artifacts.Report.ClusterIDReport)
	}
	read := core.ReadFile(outputPath)
	if !read.OK {
		t.Fatalf("ReadFile(outputPath) error = %v", read.Value)
	}
	if got := core.AsString(read.Value.([]byte)); !strings.Contains(got, `"cluster_ids":[0,4]`) {
		t.Fatalf("clustered JSONL = %s, want padded cluster IDs for both FFN memory levels", got)
	}
}

func TestBuildMemoryPretrainingArtifactsFromFiles_LoadsCorpusJSONL_Good(t *testing.T) {
	dir := t.TempDir()
	corpusPath := core.PathJoin(dir, "corpus", "records.jsonl")
	routerPath := core.PathJoin(dir, "memory", "router.json")
	ffnPath := core.PathJoin(dir, "memory", "ffn.json")
	if result := core.MkdirAll(core.PathDir(corpusPath), 0o755); !result.OK {
		t.Fatalf("MkdirAll(corpus dir) error = %v", result.Value)
	}
	writeFile(t, corpusPath,
		`{"id":"go-1","text":"Go memory planning","meta":{"source":"docs"}}`+"\n"+
			`{"id":"go-2","text":"Go cgo bridge","meta":{"source":"docs"}}`+"\n"+
			`{"id":"poem-1","text":"winter proof poem","meta":{"source":"creative"}}`+"\n"+
			`{"id":"poem-2","text":"autumn prayer","meta":{"source":"creative"}}`+"\n")
	embedder := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})

	artifacts, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), embedder, MemoryPretrainingArtifactConfig{
		CorpusPath:    corpusPath,
		RouterPath:    routerPath,
		FFNMemoryPath: ffnPath,
		Build:         BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
		FFNMemory: FFNMemoryConfig{
			HiddenSize:      2,
			Layers:          1,
			MemoryLevels:    []string{"1"},
			FFNMemoryTokens: []int{1},
		},
	})
	if err != nil {
		t.Fatalf("BuildMemoryPretrainingArtifactsFromFiles() error = %v", err)
	}
	if artifacts.Report.CorpusPath != corpusPath || artifacts.Report.CorpusRecords != 4 {
		t.Fatalf("report = %+v, want corpus path and record count", artifacts.Report)
	}
	if artifacts.Router.Blocks[0].ID != "go-1" || artifacts.Router.Blocks[0].Meta["source"] != "docs" {
		t.Fatalf("first router block = %+v, want corpus JSONL metadata", artifacts.Router.Blocks[0])
	}
	if _, err := LoadBank(routerPath); err != nil {
		t.Fatalf("LoadBank(routerPath) error = %v", err)
	}
	if _, err := LoadFFNMemoryBank(ffnPath); err != nil {
		t.Fatalf("LoadFFNMemoryBank(ffnPath) error = %v", err)
	}
}

func TestLoadCorpusRecordsJSONL_Validation_Bad(t *testing.T) {
	if _, err := LoadCorpusRecordsJSONL(""); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(empty) error = nil")
	}
	if _, err := LoadCorpusRecordsJSONL(`{"id":"x"}` + "\n"); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(missing text) error = nil")
	}
	if _, err := LoadCorpusRecordsJSONL(`{` + "\n"); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(bad json) error = nil")
	}
	if _, err := LoadCorpusRecordsJSONLFile(""); err == nil {
		t.Fatal("LoadCorpusRecordsJSONLFile(empty path) error = nil")
	}
	if _, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), MemoryPretrainingArtifactConfig{FFNMemory: FFNMemoryConfig{HiddenSize: 1, Layers: 1}}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifactsFromFiles(missing corpus path) error = nil")
	}
}

func TestBuildMemoryPretrainingArtifacts_Validation_Bad(t *testing.T) {
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), nil, []CorpusRecord{{Text: "x"}}, MemoryPretrainingArtifactConfig{}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(nil embedder) error = nil")
	}
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), nil, MemoryPretrainingArtifactConfig{}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(empty corpus) error = nil")
	}
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), []CorpusRecord{{Text: "x"}}, MemoryPretrainingArtifactConfig{FFNMemory: FFNMemoryConfig{Layers: 1}}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(missing hidden size) error = nil")
	}
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), []CorpusRecord{{Text: "x"}}, MemoryPretrainingArtifactConfig{
		FFNMemory:           FFNMemoryConfig{HiddenSize: 1, Layers: 1},
		ClusterIDInputPath:  "input.jsonl",
		ClusterIDOutputPath: "",
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(cluster input without output) error = nil")
	}
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), []CorpusRecord{{Text: "x"}}, MemoryPretrainingArtifactConfig{FFNMemory: FFNMemoryConfig{HiddenSize: 1}}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(missing layers) error = nil")
	}
}

// TestBuildMemoryPretrainingArtifacts_NilContext_Good proves a nil context is
// replaced with a background context rather than panicking.
func TestBuildMemoryPretrainingArtifacts_NilContext_Good(t *testing.T) {
	embedder := EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1, 0}, nil
	})
	artifacts, err := BuildMemoryPretrainingArtifacts(nil, embedder, []CorpusRecord{ //nolint:staticcheck // exercising the nil-context guard on purpose
		{ID: "a", Text: "alpha"},
		{ID: "b", Text: "beta"},
	}, MemoryPretrainingArtifactConfig{
		Build:     BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
		FFNMemory: FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}},
	})
	if err != nil {
		t.Fatalf("BuildMemoryPretrainingArtifacts(nil ctx) error = %v", err)
	}
	if artifacts.Router == nil || artifacts.FFNMemory == nil || artifacts.Report.CorpusRecords != 2 {
		t.Fatalf("artifacts = %+v, want a built router and FFN bank from a nil context", artifacts)
	}
}

// TestLoadCorpusRecordsJSONL_WhitespaceOnly_Ugly proves a body that trims to
// nothing per line still propagates the produced-no-rows error rather than
// returning an empty slice.
func TestLoadCorpusRecordsJSONL_WhitespaceOnly_Ugly(t *testing.T) {
	if _, err := LoadCorpusRecordsJSONL("   \n\t\n   "); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(whitespace-only rows) error = nil")
	}
}

// TestCorpusRecordMeta_TypeBranches_Ugly drives corpusRecordMeta across every
// path: a populated string map, a non-map value, an empty map, and a map whose
// values are all non-strings (which collapses back to nil).
func TestCorpusRecordMeta_TypeBranches_Ugly(t *testing.T) {
	got := corpusRecordMeta(map[string]any{"source": "docs", "n": 7})
	if len(got) != 1 || got["source"] != "docs" {
		t.Fatalf("corpusRecordMeta(string map) = %+v, want only the string-valued entry", got)
	}
	if got := corpusRecordMeta("not-a-map"); got != nil {
		t.Fatalf("corpusRecordMeta(non-map) = %+v, want nil", got)
	}
	if got := corpusRecordMeta(map[string]any{}); got != nil {
		t.Fatalf("corpusRecordMeta(empty map) = %+v, want nil", got)
	}
	if got := corpusRecordMeta(map[string]any{"n": 7, "b": true}); got != nil {
		t.Fatalf("corpusRecordMeta(no string values) = %+v, want nil", got)
	}
}

// TestRouterClusterCounts_Branches_Ugly covers the nil-bank guard and the
// per-level multiplication for a populated bank config.
func TestRouterClusterCounts_Branches_Ugly(t *testing.T) {
	if got := routerClusterCounts(nil); got != nil {
		t.Fatalf("routerClusterCounts(nil) = %+v, want nil", got)
	}
	bank := &Bank{Config: BuildConfig{BranchingFactor: 2, MaxDepth: 3, MinClusterSize: 1, KMeansIters: 1}}
	got := routerClusterCounts(bank)
	if len(got) != 3 || got[0] != 2 || got[1] != 4 || got[2] != 8 {
		t.Fatalf("routerClusterCounts(depth 3, branch 2) = %+v, want [2 4 8]", got)
	}
}
