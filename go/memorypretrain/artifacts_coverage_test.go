// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// TestArtifacts_BuildMemoryPretrainingArtifacts_RouterBuildError_Ugly forces the
// router build to fail inside BuildMemoryPretrainingArtifacts: the embedder
// returns an error, so BuildBankFromCorpus surfaces it before any FFN allocation
// or persistence.
func TestArtifacts_BuildMemoryPretrainingArtifacts_RouterBuildError_Ugly(t *testing.T) {
	failEmbed := EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, core.NewError("embed boom")
	})
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), failEmbed, artifactsCorpus(), MemoryPretrainingArtifactConfig{
		Build:     BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		FFNMemory: FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(embedder error) error = nil, want router build failure")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_FFNBankError_Ugly forces the FFN
// memory allocation to fail: the config passes the artefact-level hidden-size and
// layer guards but supplies mismatched level/token counts, so NewFFNMemoryBank
// rejects it after the router has already been built.
func TestArtifacts_BuildMemoryPretrainingArtifacts_FFNBankError_Ugly(t *testing.T) {
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), artifactsEmbedder(), artifactsCorpus(), MemoryPretrainingArtifactConfig{
		Build: BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		FFNMemory: FFNMemoryConfig{
			HiddenSize:      2,
			Layers:          1,
			MemoryLevels:    []string{"1"},
			FFNMemoryTokens: []int{1, 2},
			NumClusters:     []int{2},
		},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(mismatched FFN level counts) error = nil, want FFN bank failure")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_SaveRouterError_Ugly forces the
// router save to fail: RouterPath points under an existing regular file, so
// SaveBank cannot create the bank directory.
func TestArtifacts_BuildMemoryPretrainingArtifacts_SaveRouterError_Ugly(t *testing.T) {
	dir := t.TempDir()
	fileParent := core.PathJoin(dir, "afile")
	writeFile(t, fileParent, "x")
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), artifactsEmbedder(), artifactsCorpus(), MemoryPretrainingArtifactConfig{
		RouterPath: core.PathJoin(fileParent, "router.json"),
		Build:      BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		FFNMemory:  FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(unwritable router path) error = nil, want save failure")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_SaveFFNError_Ugly forces only the
// FFN save to fail: the router path is writable so the router persists, but the
// FFN memory path points under an existing regular file, so SaveFFNMemoryBank
// fails after the router save has succeeded.
func TestArtifacts_BuildMemoryPretrainingArtifacts_SaveFFNError_Ugly(t *testing.T) {
	dir := t.TempDir()
	fileParent := core.PathJoin(dir, "afile")
	writeFile(t, fileParent, "x")
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), artifactsEmbedder(), artifactsCorpus(), MemoryPretrainingArtifactConfig{
		RouterPath:    core.PathJoin(dir, "ok", "router.json"),
		FFNMemoryPath: core.PathJoin(fileParent, "ffn.json"),
		Build:         BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		FFNMemory:     FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(unwritable FFN path) error = nil, want save failure")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifacts_ClusterIDError_Ugly forces the
// cluster-ID enrichment to fail: a cluster-ID input path is supplied alongside an
// output path, but the input file does not exist, so AddClusterIDsToJSONLFile
// fails after the router and FFN bank have been built (no artefact paths set, so
// the saves are skipped and the failure isolates the enrichment branch).
func TestArtifacts_BuildMemoryPretrainingArtifacts_ClusterIDError_Ugly(t *testing.T) {
	dir := t.TempDir()
	if _, err := BuildMemoryPretrainingArtifacts(context.Background(), artifactsEmbedder(), artifactsCorpus(), MemoryPretrainingArtifactConfig{
		Build:               BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		FFNMemory:           FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}},
		ClusterIDInputPath:  core.PathJoin(dir, "missing-input.jsonl"),
		ClusterIDOutputPath: core.PathJoin(dir, "out.jsonl"),
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifacts(missing cluster-ID input) error = nil, want enrichment failure")
	}
}

// TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_BuildError_Ugly loads a
// valid corpus file but supplies a config that makes the in-memory build fail
// (no FFN hidden size), exercising the error-propagation return after the corpus
// has loaded but before the report's corpus path is stamped.
func TestArtifacts_BuildMemoryPretrainingArtifactsFromFiles_BuildError_Ugly(t *testing.T) {
	dir := t.TempDir()
	corpusPath := core.PathJoin(dir, "records.jsonl")
	writeFile(t, corpusPath, `{"id":"go","text":"Go memory planning"}`+"\n")
	if _, err := BuildMemoryPretrainingArtifactsFromFiles(context.Background(), artifactsEmbedder(), MemoryPretrainingArtifactConfig{
		CorpusPath: corpusPath,
		FFNMemory:  FFNMemoryConfig{Layers: 1},
	}); err == nil {
		t.Fatal("BuildMemoryPretrainingArtifactsFromFiles(build error) error = nil, want propagated build failure")
	}
}
