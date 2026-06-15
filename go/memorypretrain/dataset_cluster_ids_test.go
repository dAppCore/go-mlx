// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"errors"
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

// TestAddClusterIDsToJSONL_ContextCancelled_Ugly proves the per-row context
// check aborts enrichment before the first row is encoded.
func TestAddClusterIDsToJSONL_ContextCancelled_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := AddClusterIDsToJSONL(ctx, `{"context":"x"}`+"\n", nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{1}}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(cancelled context) error = nil")
	}
}

// TestAddClusterIDsToJSONL_NilContext_Good proves a nil context is replaced
// with a background context rather than panicking, and enrichment proceeds.
func TestAddClusterIDsToJSONL_NilContext_Good(t *testing.T) {
	out, report, err := AddClusterIDsToJSONL(nil, `{"context":"x"}`+"\n", nil, nil, ClusterIDJSONLConfig{ //nolint:staticcheck // exercising the nil-context guard on purpose
		TaskType:      ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{2},
	})
	if err != nil {
		t.Fatalf("AddClusterIDsToJSONL(nil ctx) error = %v", err)
	}
	if report.Rows != 1 || !strings.Contains(out, `"cluster_ids":[1]`) {
		t.Fatalf("AddClusterIDsToJSONL(nil ctx) out = %s report = %+v, want one generic row", out, report)
	}
}

// TestAddClusterIDsToJSONL_MalformedRow_Bad covers the per-row JSON parse-error
// branch: a non-empty line that is not valid JSON aborts the pass.
func TestAddClusterIDsToJSONL_MalformedRow_Bad(t *testing.T) {
	if _, _, err := AddClusterIDsToJSONL(context.Background(), `{not valid json}`+"\n", nil, nil, ClusterIDJSONLConfig{ClusterCounts: []int{2}}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(malformed row) error = nil")
	}
}

// TestAddClusterIDsToJSONL_LearnedPathErrors_Ugly drives the three learned-path
// error branches: the embedder failing, the router rejecting an embedding whose
// dimension mismatches, and the generic-fallback padding failing when the
// router has more hierarchy levels than the supplied cluster counts.
func TestAddClusterIDsToJSONL_LearnedPathErrors_Ugly(t *testing.T) {
	router, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	raw := `{"context":"x"}` + "\n"

	// The embedder errors: the embed-step failure is surfaced per row.
	failEmbed := EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, errors.New("embedder offline")
	})
	if _, _, err := AddClusterIDsToJSONL(context.Background(), raw, failEmbed, router, ClusterIDJSONLConfig{}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(embed error) error = nil")
	}

	// The embedding dimension mismatches the router: the routing step fails.
	wrongDim := EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1, 0, 0}, nil
	})
	if _, _, err := AddClusterIDsToJSONL(context.Background(), raw, wrongDim, router, ClusterIDJSONLConfig{}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(route error) error = nil")
	}

	// A two-level router padded against a single cluster count overflows the
	// generic-fallback padding.
	deepRouter, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0.9, 0.1}},
		{ID: "c", Embedding: []float32{0, 1}},
		{ID: "d", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank(deep router) error = %v", err)
	}
	goodEmbed := EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1, 0}, nil
	})
	if _, _, err := AddClusterIDsToJSONL(context.Background(), raw, goodEmbed, deepRouter, ClusterIDJSONLConfig{ClusterCounts: []int{2}}); err == nil {
		t.Fatal("AddClusterIDsToJSONL(pad overflow) error = nil")
	}
}

// TestClusterIDJSONLMemoryText_PerTaskType_Good drives clusterIDJSONLMemoryText
// across every task-type branch and the generic default, using synthetic rows
// and the normalised key set.
func TestClusterIDJSONLMemoryText_PerTaskType_Good(t *testing.T) {
	cfg := normaliseClusterIDJSONLConfig(ClusterIDJSONLConfig{})
	cases := []struct {
		name string
		task string
		row  map[string]any
		want string
	}{
		{
			name: "schema common substring plus continuation",
			task: ClusterIDTaskSchema,
			row: map[string]any{
				cfg.ChoicesKey:      []any{"alpha shared left", "alpha shared right"},
				cfg.ContinuationKey: "tail",
			},
			want: "alpha shared tail",
		},
		{
			name: "multiple choice prefers query",
			task: ClusterIDTaskMultipleChoice,
			row:  map[string]any{cfg.QueryKey: "the query", cfg.ContextKey: "context text"},
			want: "the query",
		},
		{
			name: "multiple choice falls back to context",
			task: ClusterIDTaskMultipleChoice,
			row:  map[string]any{cfg.ContextKey: "context text"},
			want: "context text",
		},
		{
			name: "generation reads context then text",
			task: ClusterIDTaskGenerationTaskWithAnswers,
			row:  map[string]any{cfg.TextField: "plain text"},
			want: "plain text",
		},
		{
			name: "unknown task type uses default first-string",
			task: "totally-unknown-task",
			row:  map[string]any{cfg.ContextKey: "default ctx"},
			want: "default ctx",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			taskCfg := cfg
			taskCfg.TaskType = tc.task
			if got := clusterIDJSONLMemoryText(tc.row, taskCfg); got != tc.want {
				t.Fatalf("clusterIDJSONLMemoryText() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestStringField_TypeBranches_Ugly exercises every type path in stringField:
// plain string, first element of a []any, and the guarded zero-value returns for
// nil rows, empty keys, missing keys, empty lists, and non-string list heads.
func TestStringField_TypeBranches_Ugly(t *testing.T) {
	row := map[string]any{
		"plain":        "  spaced  ",
		"list":         []any{"  first  ", "second"},
		"empty-list":   []any{},
		"non-string":   []any{42, "later"},
		"wrong-scalar": 42,
	}
	if got := stringField(row, "plain"); got != "spaced" {
		t.Fatalf("stringField(plain) = %q, want trimmed string", got)
	}
	if got := stringField(row, "list"); got != "first" {
		t.Fatalf("stringField(list) = %q, want trimmed first element", got)
	}
	if got := stringField(nil, "plain"); got != "" {
		t.Fatalf("stringField(nil row) = %q, want empty", got)
	}
	if got := stringField(row, ""); got != "" {
		t.Fatalf("stringField(empty key) = %q, want empty", got)
	}
	if got := stringField(row, "missing"); got != "" {
		t.Fatalf("stringField(missing key) = %q, want empty", got)
	}
	if got := stringField(row, "empty-list"); got != "" {
		t.Fatalf("stringField(empty list) = %q, want empty", got)
	}
	if got := stringField(row, "non-string"); got != "" {
		t.Fatalf("stringField(non-string head) = %q, want empty", got)
	}
	if got := stringField(row, "wrong-scalar"); got != "" {
		t.Fatalf("stringField(non-string scalar) = %q, want empty", got)
	}
}

// TestStringListField_TypeBranches_Ugly exercises every type path in
// stringListField: []any (mixed, blanks dropped), []string (copied), a lone
// string, a blank string, and the missing-key zero return.
func TestStringListField_TypeBranches_Ugly(t *testing.T) {
	row := map[string]any{
		"any-list":    []any{" a ", "", "b", 7},
		"string-list": []string{"x", "y"},
		"single":      "  lone  ",
		"blank":       "   ",
		"scalar":      99,
	}
	if got := stringListField(row, "any-list"); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("stringListField(any-list) = %+v, want trimmed non-blank strings", got)
	}
	if got := stringListField(row, "string-list"); len(got) != 2 || got[0] != "x" || got[1] != "y" {
		t.Fatalf("stringListField(string-list) = %+v, want copied slice", got)
	}
	if got := stringListField(row, "single"); len(got) != 1 || got[0] != "lone" {
		t.Fatalf("stringListField(single string) = %+v, want one trimmed entry", got)
	}
	if got := stringListField(row, "blank"); got != nil {
		t.Fatalf("stringListField(blank string) = %+v, want nil", got)
	}
	if got := stringListField(row, "missing"); got != nil {
		t.Fatalf("stringListField(missing key) = %+v, want nil", got)
	}
	if got := stringListField(row, "scalar"); got != nil {
		t.Fatalf("stringListField(non-string scalar) = %+v, want nil", got)
	}
}

// TestCommonStringPair_Boundaries_Ugly covers the longest-common-substring
// helper at each guard: empty, single value, a pair sharing a >=5 run, and a
// pair whose longest run is too short to qualify.
func TestCommonStringPair_Boundaries_Ugly(t *testing.T) {
	if got := commonStringPair(nil); got != "" {
		t.Fatalf("commonStringPair(nil) = %q, want empty", got)
	}
	if got := commonStringPair([]string{"only"}); got != "only" {
		t.Fatalf("commonStringPair(single) = %q, want the single value", got)
	}
	// The longest common substring is the shared prefix, trimmed of the trailing
	// space that precedes the diverging "left"/"right".
	if got := commonStringPair([]string{"the shared core left", "the shared core right"}); got != "the shared core" {
		t.Fatalf("commonStringPair(shared run) = %q, want the longest common substring", got)
	}
	if got := commonStringPair([]string{"abcd", "wxyz"}); got != "" {
		t.Fatalf("commonStringPair(short run) = %q, want empty for runs under 5", got)
	}
}
