// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/safetensors"
)

// ---------------------------------------------------------------------------
// merge.go — prepare() source == output guard.
// ---------------------------------------------------------------------------

// TestModelMerge_PrepareSourceRootEqualsOutputBad gives a source pack a Root
// equal to the (empty) output directory while its WeightFiles live elsewhere.
// ensureEmptyDestination passes (the output dir holds no weights), so the loop
// reaches samePathResolved(pack.Root, output) which fires — the
// errOutputSameAsSource leg. (A normal source dir holds model.safetensors and
// would trip ensureEmptyDestination first, so the weight files are relocated.)
func TestModelMerge_PrepareSourceRootEqualsOutputBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	// The output directory: empty (no weight files), so ensureEmptyDestination
	// returns clean and the per-source same-path check is reached.
	output := core.PathJoin(t.TempDir(), "merged-same-as-source")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	// Real weight bytes live in a separate directory; the pack's Root is the
	// output path itself.
	weightDir := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	collidingPack := mp.ModelPack{
		Root:          output,
		Path:          output,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{core.PathJoin(weightDir, "model.safetensors")},
		TokenizerPath: core.PathJoin(weightDir, "tokenizer.json"),
		Architecture:  "qwen3",
	}

	_, err := Packs(context.Background(), Options{
		OutputPath:             output,
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: collidingPack},
		},
	})
	if err == nil {
		t.Fatal("Packs(source root == output) error = nil, want same-path failure")
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — writeMergedSafetensors non-chunked branch.
//
// The non-chunked (readTensorValues) branch runs only for methods that are
// neither Linear nor SLERP — which prepare() rejects, so it is reachable only by
// calling writeMergedSafetensors directly. These cover its merge/copy/mismatch
// legs.
// ---------------------------------------------------------------------------

// TestModelMerge_WriteMergedNonChunkedUnsupportedMethodBad supplies complete,
// matching tensors with an unsupported method. readTensorValues succeeds, then
// mergeTensorValues hits its default leg and returns the unsupported-method
// error — the non-chunked complete-case merge-error return.
func TestModelMerge_WriteMergedNonChunkedUnsupportedMethodBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{2}, []float32{3, 4})
	out := core.PathJoin(t.TempDir(), "nonchunked-unsupported.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, Method("average"), 0,
		[]Source{{Weight: 1}, {Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(non-chunked unsupported method) error = nil, want merge failure")
	}
}

// TestModelMerge_WriteMergedNonChunkedMismatchCopyGood supplies incomplete
// (shape-mismatched) tensors with an unsupported method and allowMismatch = true.
// The non-chunked branch copies values[0] and writes it via writeFloat32Values —
// the allow-mismatch copy leg and the trailing write of the non-chunked path.
func TestModelMerge_WriteMergedNonChunkedMismatchCopyGood(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	out := core.PathJoin(t.TempDir(), "nonchunked-copy.safetensors")

	merged, copied, skipped, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, Method("average"), 0,
		[]Source{{Weight: 1}, {Weight: 1}}, true,
	)
	if err != nil {
		t.Fatalf("writeMergedSafetensors(non-chunked copy) error = %v", err)
	}
	if merged != 0 {
		t.Fatalf("MergedTensors = %d, want 0", merged)
	}
	if copied != 1 {
		t.Fatalf("copied = %d, want 1", copied)
	}
	if len(skipped) != 1 || skipped[0] != name {
		t.Fatalf("skipped = %v, want [%q]", skipped, name)
	}
	// The copied tensor (values[0] = left) was written through writeFloat32Values.
	if stat := core.Stat(out); !stat.OK {
		t.Fatal("merged output file was not written")
	}
}

// TestModelMerge_WriteMergedNonChunkedMismatchNoCopyBad supplies incomplete
// tensors with an unsupported method and allowMismatch = false. The non-chunked
// branch falls to its default leg and returns the tensor-mismatch error.
func TestModelMerge_WriteMergedNonChunkedMismatchNoCopyBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	out := core.PathJoin(t.TempDir(), "nonchunked-nocopy.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, Method("average"), 0,
		[]Source{{Weight: 1}, {Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(non-chunked mismatch, no copy) error = nil, want mismatch failure")
	}
}

// TestModelMerge_WriteMergedChunkedCopyErrorBad drives the chunked allow-mismatch
// copy leg with a broken refs[0]: the shape-mismatched pair leaves refs
// incomplete but non-empty, so the loop calls WriteRefFloat32Chunks on refs[0],
// which fails reading the truncated tensor — the chunked-copy error return.
func TestModelMerge_WriteMergedChunkedCopyErrorBad(t *testing.T) {
	name := "model.norm.weight"
	// refs[0] is broken (claims 64 elements over a 1-element file). It is the
	// shape-defining ref, so the second (different-shape) tensor is skipped and
	// refs = [broken]; the copy path then reads it and fails.
	broken := brokenReaderRef(t, name)
	leftIdx := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: broken},
	}
	right := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	out := core.PathJoin(t.TempDir(), "chunked-copy-err.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{leftIdx, right}, MethodLinear, 0,
		[]Source{{Weight: 1}, {Weight: 1}}, true,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(chunked copy broken ref) error = nil, want copy failure")
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — cached chunk-writer default chunk size + SLERP scan errors.
// ---------------------------------------------------------------------------

// TestModelMerge_WriteLinearChunksCachedDefaultChunkGood passes chunkElements = 0
// to writeLinearChunksCached with valid refs so the <=0 fallback to the package
// default chunk size runs and the write completes cleanly.
func TestModelMerge_WriteLinearChunksCachedDefaultChunkGood(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{5, 6, 7, 8})
	refs := []safetensors.TensorRef{left.Tensors[name], right.Tensors[name]}

	cache := newFileCache()
	defer cache.close()
	out := openOSFile(t, "cached-default-chunk.bin")
	defer out.Close()
	if err := writeLinearChunksCached(context.Background(), out, cache, refs,
		[]float64{0.5, 0.5}, 0, &writeScratch{}); err != nil {
		t.Fatalf("writeLinearChunksCached(default chunk) error = %v", err)
	}
}

// TestModelMerge_WriteSLERPChunksCachedScanErrorBad gives writeSLERPChunksCached
// a same-element broken-ref pair so cache.readers opens fine but the SLERP
// dot/norm scan fails on the truncated read — the scan-error return.
func TestModelMerge_WriteSLERPChunksCachedScanErrorBad(t *testing.T) {
	name := "model.norm.weight"
	left := brokenReaderRef(t, name)
	right := brokenReaderRef(t, name)
	refs := []safetensors.TensorRef{left, right}

	cache := newFileCache()
	defer cache.close()
	out := openOSFile(t, "slerp-cached-scan-err.bin")
	defer out.Close()
	if err := writeSLERPChunksCached(context.Background(), out, cache, refs, 0.5,
		8, &writeScratch{}); err == nil {
		t.Fatal("writeSLERPChunksCached(broken refs) error = nil, want scan failure")
	}
}

// TestModelMerge_WriteSLERPChunksScanErrorBad gives writeSLERPChunks a
// same-element broken-ref pair so OpenReaders succeeds but the shared SLERP
// dot/norm scan fails on the truncated read — the scan-error return after the
// readers open (distinct from the not-two-refs / element-mismatch legs).
func TestModelMerge_WriteSLERPChunksScanErrorBad(t *testing.T) {
	name := "model.norm.weight"
	left := brokenReaderRef(t, name)
	right := brokenReaderRef(t, name)
	out := openOSFile(t, "slerp-scan-err.bin")
	defer out.Close()
	if err := writeSLERPChunks(context.Background(), out,
		[]safetensors.TensorRef{left, right}, 0.5, 8); err == nil {
		t.Fatal("writeSLERPChunks(broken refs) error = nil, want scan failure")
	}
}

// TestModelMerge_SlerpChunkedWeightsNearParallelGood feeds slerpChunkedWeights
// two nearly-identical non-zero tensors. The dot/norm scan yields |cosTheta| >
// 0.9995, so the chunked SLERP weight path takes the near-parallel linear
// fallback (1-t, t) rather than the sin-ratio interpolation — the chunked
// counterpart of the in-memory near-parallel fallback.
func TestModelMerge_SlerpChunkedWeightsNearParallelGood(t *testing.T) {
	name := "model.embed_tokens.weight"
	left := indexSinglePackTensor(t, name, []int{3}, []float32{1, 2, 3})
	// A near-identical, definitely non-zero vector: cosine ~ 1.
	right := indexSinglePackTensor(t, name, []int{3}, []float32{1.0001, 2.0001, 3.0001})
	refs := []safetensors.TensorRef{left.Tensors[name], right.Tensors[name]}

	weights, err := slerpChunkedWeights(context.Background(), refs, 0.25, 8)
	if err != nil {
		t.Fatalf("slerpChunkedWeights(near parallel) error = %v", err)
	}
	// Near-parallel fallback returns the linear pair (1-t, t) = (0.75, 0.25).
	if len(weights) != 2 || weights[0] != 0.75 || weights[1] != 0.25 {
		t.Fatalf("slerpChunkedWeights(near parallel) = %v, want [0.75 0.25]", weights)
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — writeProvenance write-error leg.
// ---------------------------------------------------------------------------

// TestModelMerge_WriteProvenanceWriteFailBad points writeProvenance at a path
// whose parent is a regular file so the marshal succeeds but core.WriteFile
// fails — the write-error return (distinct from the marshal-error leg the NaN-T
// test covers).
func TestModelMerge_WriteProvenanceWriteFailBad(t *testing.T) {
	notDir := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, notDir, "x")
	badPath := core.PathJoin(notDir, "adapter_provenance.json")

	err := writeProvenance(badPath, Provenance{
		Version: 1,
		Method:  MethodLinear,
		Sources: []Source{{Weight: 0.5}, {Weight: 0.5}},
	})
	if err == nil {
		t.Fatal("writeProvenance(unwritable path) error = nil, want write failure")
	}
}
