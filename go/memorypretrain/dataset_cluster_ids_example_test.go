// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/mlx/memorypretrain"
)

// ExampleAddClusterIDsToJSONL enriches a JSONL row with generic-memory cluster
// IDs when no learned router is supplied.
func ExampleAddClusterIDsToJSONL() {
	raw := `{"id":"a","context":"Go memory planning"}` + "\n"
	out, report, err := memorypretrain.AddClusterIDsToJSONL(context.Background(), raw, nil, nil, memorypretrain.ClusterIDJSONLConfig{
		TaskType:      memorypretrain.ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{3},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(report.Rows, report.GenericRows)
	fmt.Println(strings.Contains(out, `"cluster_ids":[2]`))
	// Output:
	// 1 1
	// true
}

// ExampleAddClusterIDsToJSONLFile reads an input JSONL file, enriches each row
// with generic-memory cluster IDs, and writes the result to an output file.
func ExampleAddClusterIDsToJSONLFile() {
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-jsonl-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	input := core.PathJoin(dir, "in.jsonl")
	output := core.PathJoin(dir, "out.jsonl")
	if result := core.WriteFile(input, []byte(`{"id":"a","context":"Go memory planning"}`+"\n"), 0o644); !result.OK {
		fmt.Println("error:", result.Value)
		return
	}
	report, err := memorypretrain.AddClusterIDsToJSONLFile(context.Background(), input, output, nil, nil, memorypretrain.ClusterIDJSONLConfig{
		TaskType:      memorypretrain.ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{3},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	read := core.ReadFile(output)
	if !read.OK {
		fmt.Println("error:", read.Value)
		return
	}
	fmt.Println(report.Rows, strings.Contains(core.AsString(read.Value.([]byte)), `"cluster_ids":[2]`))
	// Output: 1 true
}
