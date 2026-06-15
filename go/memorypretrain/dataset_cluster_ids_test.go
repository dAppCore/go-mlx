// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestAddClusterIDsToJSONL_LearnedMultipleChoice_Good(t *testing.T) {
	router, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	raw := `{"id":"a","query":"Go memory planning","choices":["Go","poem"]}` + "\n"
	got, report, err := AddClusterIDsToJSONL(context.Background(), raw, EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if text != "Go memory planning" {
			t.Fatalf("embed text = %q, want query field", text)
		}
		return []float32{1, 0}, nil
	}), router, ClusterIDJSONLConfig{TaskType: ClusterIDTaskMultipleChoice})
	if err != nil {
		t.Fatalf("AddClusterIDsToJSONL() error = %v", err)
	}
	ids, err := router.ClusterIDs([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterIDs() error = %v", err)
	}
	if report.Rows != 1 || report.LearnedRows != 1 || report.GenericRows != 0 || report.SkippedRows != 0 {
		t.Fatalf("report = %+v, want one learned row", report)
	}
	var row map[string]any
	if result := core.JSONUnmarshalString(core.Trim(got), &row); !result.OK {
		t.Fatalf("JSONUnmarshalString(output): %v", result.Value)
	}
	gotIDs := row["cluster_ids"].([]any)
	if len(gotIDs) != 1 || int(gotIDs[0].(float64)) != ids[0] {
		t.Fatalf("cluster_ids = %+v, want %+v in row %s", gotIDs, ids, got)
	}
}

func TestAddClusterIDsToJSONL_LearnedPadsGenericLevels_Good(t *testing.T) {
	router, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	raw := `{"id":"a","context":"Go memory planning"}` + "\n"
	got, report, err := AddClusterIDsToJSONL(context.Background(), raw, EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		return []float32{1, 0}, nil
	}), router, ClusterIDJSONLConfig{
		TaskType:      ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{3, 5},
	})
	if err != nil {
		t.Fatalf("AddClusterIDsToJSONL() error = %v", err)
	}
	if report.Rows != 1 || report.LearnedRows != 1 {
		t.Fatalf("report = %+v, want one learned row", report)
	}
	if !strings.Contains(got, `"cluster_ids":[0,4]`) {
		t.Fatalf("clustered output = %s, want learned first level and generic fallback second level", got)
	}
}

func TestAddClusterIDsToJSONL_GenericAndSchema_Good(t *testing.T) {
	raw := `{"id":"schema","context_options":["alpha shared left","alpha shared right"],"continuation":"answer"}` + "\n"
	got, report, err := AddClusterIDsToJSONL(context.Background(), raw, nil, nil, ClusterIDJSONLConfig{
		TaskType:      ClusterIDTaskSchema,
		ClusterCounts: []int{3, 5},
	})
	if err != nil {
		t.Fatalf("AddClusterIDsToJSONL(generic) error = %v", err)
	}
	if report.Rows != 1 || report.GenericRows != 1 || report.LearnedRows != 0 {
		t.Fatalf("report = %+v, want one generic row", report)
	}
	if !strings.Contains(got, `"cluster_ids":[2,4]`) {
		t.Fatalf("generic output = %s, want last cluster IDs", got)
	}
}

func TestAddClusterIDsToJSONLFile_WritesOutput_Good(t *testing.T) {
	dir := t.TempDir()
	input := core.PathJoin(dir, "in.jsonl")
	output := core.PathJoin(dir, "nested", "out.jsonl")
	if result := core.WriteFile(input, []byte(`{"context":"x"}`+"\n"), 0o644); !result.OK {
		t.Fatalf("WriteFile(input): %v", result.Value)
	}
	report, err := AddClusterIDsToJSONLFile(context.Background(), input, output, nil, nil, ClusterIDJSONLConfig{
		TaskType:      ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{2},
	})
	if err != nil {
		t.Fatalf("AddClusterIDsToJSONLFile() error = %v", err)
	}
	if report.Rows != 1 || report.GenericRows != 1 {
		t.Fatalf("report = %+v, want one generic file row", report)
	}
	read := core.ReadFile(output)
	if !read.OK {
		t.Fatalf("ReadFile(output): %v", read.Value)
	}
	if got := core.AsString(read.Value.([]byte)); !strings.Contains(got, `"cluster_ids":[1]`) {
		t.Fatalf("output = %s, want generic cluster IDs", got)
	}
}

func TestAddClusterIDsToJSONLFile_Validation_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := AddClusterIDsToJSONLFile(context.Background(), "", core.PathJoin(dir, "out.jsonl"), nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(empty input path) error = nil")
	}
	if _, err := AddClusterIDsToJSONLFile(context.Background(), core.PathJoin(dir, "in.jsonl"), "", nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(empty output path) error = nil")
	}
	// A missing input file surfaces the read error before any enrichment.
	if _, err := AddClusterIDsToJSONLFile(context.Background(), core.PathJoin(dir, "missing.jsonl"), core.PathJoin(dir, "out.jsonl"), nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(missing input) error = nil")
	}
	// A readable input that produces no rows propagates the empty-corpus error
	// and writes no output.
	emptyInput := core.PathJoin(dir, "empty.jsonl")
	if result := core.WriteFile(emptyInput, []byte("\n\n"), 0o644); !result.OK {
		t.Fatalf("WriteFile(empty input): %v", result.Value)
	}
	if _, err := AddClusterIDsToJSONLFile(context.Background(), emptyInput, core.PathJoin(dir, "out.jsonl"), nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(no rows) error = nil")
	}
}

func TestAddClusterIDsToJSONL_Validation_Bad(t *testing.T) {
	if _, _, err := AddClusterIDsToJSONL(context.Background(), "", nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(empty raw) error = nil")
	}
	if _, _, err := AddClusterIDsToJSONL(context.Background(), `{"context":"x"}`+"\n", nil, nil, ClusterIDJSONLConfig{}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(generic without counts) error = nil")
	}
	router, err := BuildBank([]Block{{Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, _, err := AddClusterIDsToJSONL(context.Background(), `{"context":"x"}`+"\n", nil, router, ClusterIDJSONLConfig{}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(router without embedder) error = nil")
	}
	if _, _, err := AddClusterIDsToJSONL(context.Background(), `{"unknown":"x"}`+"\n", nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(no memory text) error = nil")
	}
}
