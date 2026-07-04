// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

func artifactsCorpus() []CorpusRecord {
	return []CorpusRecord{
		{ID: "go-1", Text: "Go memory planning"},
		{ID: "go-2", Text: "Go cgo bridge"},
		{ID: "poem-1", Text: "winter proof poem"},
		{ID: "poem-2", Text: "autumn prayer"},
	}
}

func artifactsEmbedder() EmbedFunc {
	return EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_Good builds the full offline
// artefact set, persisting the router and FFN memory and enriching a JSONL task
// file with learned cluster IDs, and confirms the report, embed-call count, and
// padded cluster IDs across single- and multi-level FFN memory.
func TestArtifacts_BuildMemoryPretrainingArtifacts_Good(t *testing.T) {
	t.Run("builds saves and enriches", func(t *testing.T) {
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

		artifacts, err := BuildMemoryPretrainingArtifacts(context.Background(), embedder, artifactsCorpus(), MemoryPretrainingArtifactConfig{
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
	})

	t.Run("cluster ids match ffn memory levels", func(t *testing.T) {
		dir := t.TempDir()
		inputPath := core.PathJoin(dir, "tasks", "input.jsonl")
		outputPath := core.PathJoin(dir, "tasks", "clustered.jsonl")
		if result := core.MkdirAll(core.PathDir(inputPath), 0o755); !result.OK {
			t.Fatalf("MkdirAll(input dir) error = %v", result.Value)
		}
		writeFile(t, inputPath, `{"context":"Go memory planning"}`+"\n")

		artifacts, err := BuildMemoryPretrainingArtifacts(context.Background(), artifactsEmbedder(), artifactsCorpus(), MemoryPretrainingArtifactConfig{
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
	})
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_Bad drives each entry guard: nil
// embedder, empty corpus, missing FFN hidden size, missing FFN layers, and a
// cluster-ID input path supplied without an output path.
func TestArtifacts_BuildMemoryPretrainingArtifacts_Bad(t *testing.T) {
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
	}), []CorpusRecord{{Text: "x"}}, MemoryPretrainingArtifactConfig{FFNMemory: FFNMemoryConfig{HiddenSize: 1}}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(missing layers) error = nil")
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
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_Ugly proves a nil context is
// replaced with a background context rather than panicking, still producing a
// built router and FFN bank.
func TestArtifacts_BuildMemoryPretrainingArtifacts_Ugly(t *testing.T) {
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

// TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Good loads a corpus
// JSONL file, runs the offline builder, and confirms the report carries the
// corpus path, the cloned record metadata reaches the router, and both saved
// artefacts reload.
func TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Good(t *testing.T) {
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

	artifacts, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), artifactsEmbedder(), MemoryPretrainingArtifactConfig{
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

// TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Bad rejects a missing
// corpus path before any embedding work happens.
func TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Bad(t *testing.T) {
	if _, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), MemoryPretrainingArtifactConfig{FFNMemory: FFNMemoryConfig{HiddenSize: 1, Layers: 1}}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifactsFromFiles(missing corpus path) error = nil")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Ugly points the builder
// at a corpus path that does not exist, so the file read fails before the
// in-memory builder runs.
func TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_Ugly(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "does", "not", "exist.jsonl")
	if _, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), artifactsEmbedder(), MemoryPretrainingArtifactConfig{
		CorpusPath: missing,
		FFNMemory:  FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifactsFromFiles(missing corpus file) error = nil")
	}
}

// TestArtifacts_LoadCorpusRecordsJSONL_Good parses a multi-row JSONL corpus with
// id, text, and a string-valued meta object.
func TestArtifacts_LoadCorpusRecordsJSONL_Good(t *testing.T) {
	records, err := LoadCorpusRecordsJSONL(
		`{"id":"go","text":"Go memory planning","meta":{"source":"docs"}}` + "\n" +
			`{"id":"poem","text":"winter proof poem"}` + "\n")
	if err != nil {
		t.Fatalf("LoadCorpusRecordsJSONL() error = %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("LoadCorpusRecordsJSONL() = %+v, want two records", records)
	}
	if records[0].ID != "go" || records[0].Text != "Go memory planning" || records[0].Meta["source"] != "docs" {
		t.Fatalf("LoadCorpusRecordsJSONL()[0] = %+v, want parsed id/text/meta", records[0])
	}
	if records[1].Meta != nil {
		t.Fatalf("LoadCorpusRecordsJSONL()[1].Meta = %+v, want nil for the row without meta", records[1].Meta)
	}
}

// TestArtifacts_LoadCorpusRecordsJSONL_Bad rejects empty input, a row with no
// text, and an unparsable JSON row.
func TestArtifacts_LoadCorpusRecordsJSONL_Bad(t *testing.T) {
	if _, err := LoadCorpusRecordsJSONL(""); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(empty) error = nil")
	}
	if _, err := LoadCorpusRecordsJSONL(`{"id":"x"}` + "\n"); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(missing text) error = nil")
	}
	if _, err := LoadCorpusRecordsJSONL(`{` + "\n"); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(bad json) error = nil")
	}
}

// TestArtifacts_LoadCorpusRecordsJSONL_Ugly proves input that trims to nothing on
// every line still propagates the produced-no-rows error rather than returning an
// empty slice.
func TestArtifacts_LoadCorpusRecordsJSONL_Ugly(t *testing.T) {
	if _, err := LoadCorpusRecordsJSONL("   \n\t\n   "); err == nil {
		t.Fatal("LoadCorpusRecordsJSONL(whitespace-only rows) error = nil")
	}
}

// TestArtifacts_LoadCorpusRecordsJSONLFile_Good reads corpus records from a JSONL
// file on disk and parses them through the shared string loader.
func TestArtifacts_LoadCorpusRecordsJSONLFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "records.jsonl")
	writeFile(t, path, `{"id":"go","text":"Go memory planning"}`+"\n")
	records, err := LoadCorpusRecordsJSONLFile(path)
	if err != nil {
		t.Fatalf("LoadCorpusRecordsJSONLFile() error = %v", err)
	}
	if len(records) != 1 || records[0].ID != "go" || records[0].Text != "Go memory planning" {
		t.Fatalf("LoadCorpusRecordsJSONLFile() = %+v, want the file's single record", records)
	}
}

// TestArtifacts_LoadCorpusRecordsJSONLFile_Bad rejects an empty path before any
// read.
func TestArtifacts_LoadCorpusRecordsJSONLFile_Bad(t *testing.T) {
	if _, err := LoadCorpusRecordsJSONLFile(""); err == nil {
		t.Fatal("LoadCorpusRecordsJSONLFile(empty path) error = nil")
	}
}

// TestArtifacts_LoadCorpusRecordsJSONLFile_Ugly reads a path that does not exist,
// so the file read fails before parsing.
func TestArtifacts_LoadCorpusRecordsJSONLFile_Ugly(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "missing.jsonl")
	if _, err := LoadCorpusRecordsJSONLFile(missing); err == nil {
		t.Fatal("LoadCorpusRecordsJSONLFile(missing file) error = nil")
	}
}

// TestArtifacts_corpusRecordMeta_Ugly drives corpusRecordMeta across every path:
// a populated string map, a non-map value, an empty map, and a map whose values
// are all non-strings (which collapses back to nil).
func TestArtifacts_corpusRecordMeta_Ugly(t *testing.T) {
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

// TestArtifacts_routerClusterCounts_Ugly covers the nil-bank guard and the
// per-level multiplication for a populated bank config.
func TestArtifacts_routerClusterCounts_Ugly(t *testing.T) {
	if got := routerClusterCounts(nil); got != nil {
		t.Fatalf("routerClusterCounts(nil) = %+v, want nil", got)
	}
	bank := &Bank{Config: BuildConfig{BranchingFactor: 2, MaxDepth: 3, MinClusterSize: 1, KMeansIters: 1}}
	got := routerClusterCounts(bank)
	if len(got) != 3 || got[0] != 2 || got[1] != 4 || got[2] != 8 {
		t.Fatalf("routerClusterCounts(depth 3, branch 2) = %+v, want [2 4 8]", got)
	}
}
