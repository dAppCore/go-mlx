// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"
	"strings"

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
