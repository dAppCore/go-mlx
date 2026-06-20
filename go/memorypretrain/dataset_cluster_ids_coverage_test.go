// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// TestDatasetClusterIds_AddClusterIDsToJSONLFile_WriteFault_Ugly drives the two
// output-persistence faults inside AddClusterIDsToJSONLFile that the happy path
// never reaches. The generic (router-less) enrichment of a valid input row
// succeeds, then writing the result fails: once because the output path's parent
// is an existing regular file (MkdirAll fails) and once because the output path
// is an existing directory (WriteFile fails).
func TestDatasetClusterIds_AddClusterIDsToJSONLFile_WriteFault_Ugly(t *testing.T) {
	dir := t.TempDir()
	input := core.PathJoin(dir, "input.jsonl")
	writeFile(t, input, `{"text":"Go memory planning"}`+"\n")
	cfg := ClusterIDJSONLConfig{ClusterCounts: []int{1}}

	// Output parent is a regular file, so the directory-create guard fails.
	fileParent := core.PathJoin(dir, "afile")
	writeFile(t, fileParent, "x")
	if _, err := AddClusterIDsToJSONLFile(context.Background(), input, core.PathJoin(fileParent, "out.jsonl"), nil, nil, cfg); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(output parent is a file) error = nil, want mkdir failure")
	}

	// Output path is an existing directory, so the write fails after enrichment.
	if _, err := AddClusterIDsToJSONLFile(context.Background(), input, dir, nil, nil, cfg); err == nil {
		t.Fatal("AddClusterIDsToJSONLFile(output path is a directory) error = nil, want write failure")
	}
}
