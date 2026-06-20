// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// ---------------------------------------------------------------------------
// merge.go — prepare() source-validation legs.
// ---------------------------------------------------------------------------

// TestModelMerge_PrepareSourceFormatNotSafetensorsBad gives a source pack a
// non-safetensors format. prepare() rejects it inside the per-source loop
// (pack.Format != ModelPackFormatSafetensors) before any weights are read —
// the otherwise-unreached errMergeNeedsSafetensors leg.
func TestModelMerge_PrepareSourceFormatNotSafetensorsBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	rightPack := testPack(right)
	rightPack.Format = mp.ModelPackFormatGGUF

	_, err := Packs(context.Background(), Options{
		OutputPath:             core.PathJoin(t.TempDir(), "merged-bad-format"),
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: rightPack},
		},
	})
	if err == nil {
		t.Fatal("Packs(non-safetensors source) error = nil, want format failure")
	}
}

// TestModelMerge_PrepareSourceSameAsOutputBad sets a source pack's Root equal
// to the (absolute) output path so prepare()'s samePathResolved guard rejects
// the merge — the errOutputSameAsSource leg.
func TestModelMerge_PrepareSourceSameAsOutputBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// Output path is the right source's own directory. prepare resolves both
	// to absolute and the samePathResolved check fires.
	_, err := Packs(context.Background(), Options{
		OutputPath:             right,
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(output == source root) error = nil, want same-path failure")
	}
}

// TestModelMerge_PrepareMkdirAllFailsBad makes the output's parent directory
// read-only after prepare's ensureEmptyDestination stat (which sees the target
// as not-existing and returns clean) so the subsequent core.MkdirAll fails —
// the create-merged-directory error leg. chmod-000 does not deny root, so the
// injection is skipped there.
func TestModelMerge_PrepareMkdirAllFailsBad(t *testing.T) {
	if isRoot(t) {
		t.Skip("chmod does not deny root; skipping mkdir-failure injection")
	}
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	parent := t.TempDir()
	// Read+execute but not write: stat of <parent>/merged still works (returns
	// not-exist), but MkdirAll cannot create the child.
	if result := core.Chmod(parent, 0o555); !result.OK {
		t.Fatalf("chmod parent 0555: %v", result.Value)
	}
	t.Cleanup(func() { core.Chmod(parent, 0o755) })

	_, err := Packs(context.Background(), Options{
		OutputPath:             core.PathJoin(parent, "merged"),
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(read-only output parent) error = nil, want mkdir failure")
	}
}

// ---------------------------------------------------------------------------
// merge.go — ensureEmptyDestination occupied/clean legs.
// ---------------------------------------------------------------------------

// TestModelMerge_EnsureEmptyDestinationGGUFOccupiedBad pre-creates the output
// directory holding only a .gguf file (no .safetensors). The safetensors glob
// is empty so the check falls through to the .gguf glob, which is non-empty —
// the second errOutputHasWeights leg the safetensors-occupied tests skip.
func TestModelMerge_EnsureEmptyDestinationGGUFOccupiedBad(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "occupied-gguf")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(output, "weights.gguf"), "fake gguf payload")

	if err := ensureEmptyDestination(output); err == nil {
		t.Fatal("ensureEmptyDestination(dir with .gguf) error = nil, want occupied failure")
	}
}

// TestModelMerge_EnsureEmptyDestinationSafetensorsOccupiedBad pre-creates the
// output directory holding a .safetensors file — the first occupied leg, driven
// directly so its return is observed independently of the .gguf path.
func TestModelMerge_EnsureEmptyDestinationSafetensorsOccupiedBad(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "occupied-st")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(output, "model.safetensors"), "fake")

	if err := ensureEmptyDestination(output); err == nil {
		t.Fatal("ensureEmptyDestination(dir with .safetensors) error = nil, want occupied failure")
	}
}

// TestModelMerge_EnsureEmptyDestinationExistingEmptyGood points the check at an
// existing directory that holds no weight files. Both globs are empty, so the
// function returns nil — the clean-existing-directory leg (the final return)
// that the not-exist and occupied tests never reach.
func TestModelMerge_EnsureEmptyDestinationExistingEmptyGood(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "empty-existing")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	// A non-weight file present — must not trip either weight glob.
	writeModelPackFile(t, core.PathJoin(output, "config.json"), `{"ok":true}`)

	if err := ensureEmptyDestination(output); err != nil {
		t.Fatalf("ensureEmptyDestination(empty existing dir) error = %v, want nil", err)
	}
}

// ---------------------------------------------------------------------------
// merge.go — validatePackCompatibility per-source tokenizer hash leg.
// ---------------------------------------------------------------------------

// TestModelMerge_ValidatePackCompatibilitySourceTokenizerHashFailBad makes the
// *base* tokenizer readable but a *non-base* source tokenizer missing. The base
// hash succeeds (baseHashErr is nil), then hashFile(pack.TokenizerPath) for the
// second source fails — the per-source "hash tokenizer" error return, distinct
// from the base-tokenizer leg the chmod test covers.
func TestModelMerge_ValidatePackCompatibilitySourceTokenizerHashFailBad(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	sourcePack := testPack(source)
	// Point the second source's tokenizer at a path that does not exist. The
	// base tokenizer is intact, so the base hash loads cleanly first.
	sourcePack.TokenizerPath = core.PathJoin(t.TempDir(), "no-tokenizer.json")

	err := validatePackCompatibility(
		[]mp.ModelPack{testPack(base), sourcePack},
		Options{}, // AllowTokenizerMismatch defaults to false → hashes are compared.
	)
	if err == nil {
		t.Fatal("validatePackCompatibility(missing source tokenizer) error = nil, want hash failure")
	}
}

// ---------------------------------------------------------------------------
// merge.go — validateTensorIndexes allow-mismatch continue legs.
// ---------------------------------------------------------------------------

// TestModelMerge_ValidateTensorIndexesAllowMismatchGood drives both
// allow-mismatch continue branches: a base tensor missing from the source, and
// a base tensor present at a different shape. With allowMismatch = true neither
// returns an error — the two continue legs the strict (false) tests bypass.
func TestModelMerge_ValidateTensorIndexesAllowMismatchGood(t *testing.T) {
	base := safetensors.Index{
		Names: []string{"a", "b"},
		Tensors: map[string]safetensors.TensorRef{
			"a": {Name: "a", Shape: []uint64{2}},
			"b": {Name: "b", Shape: []uint64{2}},
		},
	}
	// Source is missing "a" entirely and carries "b" at a mismatched shape.
	source := safetensors.Index{
		Names: []string{"b"},
		Tensors: map[string]safetensors.TensorRef{
			"b": {Name: "b", Shape: []uint64{4}},
		},
	}
	if err := validateTensorIndexes([]safetensors.Index{base, source}, true); err != nil {
		t.Fatalf("validateTensorIndexes(allow mismatch) error = %v, want nil", err)
	}
}

// ---------------------------------------------------------------------------
// merge.go — equalFold right-hand uppercase fold.
// ---------------------------------------------------------------------------

// TestModelMerge_EqualFoldRightUppercaseGood folds an uppercase character in the
// second argument. The package's only caller passes a lowercase literal as b, so
// the cb-uppercase branch is reached only by a direct call with uppercase on the
// right — matching equality must still hold under ASCII case folding.
func TestModelMerge_EqualFoldRightUppercaseGood(t *testing.T) {
	if !equalFold("readme", "README") {
		t.Fatal(`equalFold("readme", "README") = false, want true (right-hand fold)`)
	}
	if equalFold("readme", "READMX") {
		t.Fatal(`equalFold("readme", "READMX") = true, want false`)
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — readTensorRefsInto shape-mismatch continue.
// ---------------------------------------------------------------------------

// TestModelMerge_ReadTensorRefsIntoShapeMismatchGood feeds two indexes whose
// tensors share a name but differ in shape. readTensorRefsInto takes the
// shape-mismatch branch (complete = false, skip the ref) — the leg the
// matched/absent ReadTensorRefs tests never hit.
func TestModelMerge_ReadTensorRefsIntoShapeMismatchGood(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})

	refs, complete, err := readTensorRefsInto([]safetensors.Index{left, right}, name, nil)
	if err != nil {
		t.Fatalf("readTensorRefsInto(shape mismatch) error = %v", err)
	}
	if complete {
		t.Fatal("readTensorRefsInto(shape mismatch) complete = true, want false")
	}
	// Only the first (shape-defining) ref is retained; the mismatched one is skipped.
	if len(refs) != 1 {
		t.Fatalf("readTensorRefsInto(shape mismatch) len(refs) = %d, want 1", len(refs))
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — writeMergedSafetensors driven directly.
//
// writeMergedSafetensors is otherwise reached only through Packs, which always
// supplies a valid output path and a Linear/SLERP method, leaving its
// create/weights/switch error legs uncovered. These white-box calls reach them
// directly. (The header-marshal and post-create file.Write error legs are
// defensive checks on a freshly-created file / a float-free header that cannot
// fail without fault injection, so they are intentionally left, matching the
// package's existing unreachable-leg precedent.)
// ---------------------------------------------------------------------------

// TestModelMerge_WriteMergedSafetensorsCreateFailBad points the output path at a
// location whose parent is a regular file, so the internal core.Create fails —
// the create-error return leg.
func TestModelMerge_WriteMergedSafetensorsCreateFailBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{2}, []float32{3, 4})
	// Parent is a file: <file>/out.safetensors cannot be created.
	notDir := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, notDir, "x")
	badPath := core.PathJoin(notDir, "out.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), badPath,
		[]safetensors.Index{left, right}, MethodLinear, 0,
		[]Source{{}, {}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(uncreatable path) error = nil, want create failure")
	}
}

// TestModelMerge_WriteMergedSafetensorsWeightNotFiniteBad supplies a source with
// a non-finite weight so normalizedWeights rejects it after the header is
// written — the normalizedWeights error return inside writeMergedSafetensors.
func TestModelMerge_WriteMergedSafetensorsWeightNotFiniteBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{2}, []float32{3, 4})
	out := core.PathJoin(t.TempDir(), "weights-not-finite.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, MethodLinear, 0,
		// A +Inf weight: normalizedWeights returns errMergeWeightNotFinite.
		[]Source{{Weight: math.Inf(1)}, {Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(non-finite weight) error = nil, want normalize failure")
	}
}

// TestModelMerge_WriteMergedSafetensorsCachedWriteErrorBad supplies a complete
// (same-shape) ref pair whose backing files claim more elements than they hold,
// so the cached linear writer fails mid-chunk — the writeLinearChunksCached
// error return in the complete-case switch leg.
func TestModelMerge_WriteMergedSafetensorsCachedWriteErrorBad(t *testing.T) {
	name := "model.norm.weight"
	// Two broken refs at the SAME claimed element count → complete = true, so the
	// merged write attempts the cached linear path and fails on the truncated read.
	left := brokenReaderRef(t, name)
	right := brokenReaderRef(t, name)
	leftIdx := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: left},
	}
	rightIdx := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: right},
	}
	out := core.PathJoin(t.TempDir(), "cached-write-err.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{leftIdx, rightIdx}, MethodLinear, 0,
		[]Source{{Weight: 1}, {Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(broken complete refs) error = nil, want cached write failure")
	}
}

// TestModelMerge_WriteMergedSafetensorsMismatchNoCopyBad gives the second index
// a different shape for the shared tensor with allowMismatch = false. The refs
// are incomplete and copying is disallowed, so the loop hits the default leg —
// the "model merge tensor mismatch" error.
func TestModelMerge_WriteMergedSafetensorsMismatchNoCopyBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	out := core.PathJoin(t.TempDir(), "mismatch-nocopy.safetensors")

	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, MethodLinear, 0,
		[]Source{{Weight: 1}, {Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(mismatch, no copy) error = nil, want mismatch failure")
	}
}

// TestModelMerge_WriteMergedSafetensorsMismatchCopyGood gives the second index a
// different shape for the shared tensor with allowMismatch = true. The refs are
// incomplete but at least one is present, so the loop copies refs[0] via
// WriteRefFloat32Chunks and records the name as skipped — the allow-mismatch
// copy leg.
func TestModelMerge_WriteMergedSafetensorsMismatchCopyGood(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	out := core.PathJoin(t.TempDir(), "mismatch-copy.safetensors")

	merged, copied, skipped, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{left, right}, MethodLinear, 0,
		[]Source{{Weight: 1}, {Weight: 1}}, true,
	)
	if err != nil {
		t.Fatalf("writeMergedSafetensors(mismatch, copy) error = %v", err)
	}
	if merged != 0 {
		t.Fatalf("MergedTensors = %d, want 0 (mismatch copied not merged)", merged)
	}
	if copied != 1 {
		t.Fatalf("copied = %d, want 1", copied)
	}
	if len(skipped) != 1 || skipped[0] != name {
		t.Fatalf("skipped = %v, want [%q]", skipped, name)
	}
}

// TestModelMerge_WriteMergedSafetensorsNonChunkedReadErrorBad calls
// writeMergedSafetensors with a method that is neither Linear nor SLERP, so the
// loop takes the readTensorValues path (the non-chunked branch Packs never
// reaches). A broken ref makes ReadRefValues fail — the readTensorValues error
// return. (Reserved methods are rejected by prepare(); this direct call is the
// only way to exercise the non-chunked value path's read-error leg.)
func TestModelMerge_WriteMergedSafetensorsNonChunkedReadErrorBad(t *testing.T) {
	name := "model.norm.weight"
	broken := brokenReaderRef(t, name)
	idx := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: broken},
	}
	out := core.PathJoin(t.TempDir(), "nonchunked-read-err.safetensors")

	// A single index with a non-chunked method drives readTensorValues, which
	// fails decoding the truncated tensor.
	_, _, _, err := writeMergedSafetensors(
		context.Background(), out,
		[]safetensors.Index{idx}, Method("average"), 0,
		[]Source{{Weight: 1}}, false,
	)
	if err == nil {
		t.Fatal("writeMergedSafetensors(non-chunked broken ref) error = nil, want read failure")
	}
}

// ---------------------------------------------------------------------------
// compare.go — appendTensorDelta unchanged-skip.
// ---------------------------------------------------------------------------

// TestCompare_AppendTensorDeltaUnchangedSkippedGood appends an unchanged delta
// with IncludeUnchanged = false. The delta must be dropped — the
// unchanged-and-not-included early return the max-reports test does not reach
// (it only feeds Changed deltas).
func TestCompare_AppendTensorDeltaUnchangedSkippedGood(t *testing.T) {
	result := &CompareResult{}
	opts := CompareOptions{IncludeUnchanged: false}
	appendTensorDelta(result, opts, TensorDelta{Name: "a", Status: CompareStatusUnchanged})
	if len(result.Tensors) != 0 {
		t.Fatalf("unchanged delta retained: len = %d, want 0", len(result.Tensors))
	}
	// A changed delta under the same options is still recorded (sanity: the skip
	// is status-specific, not a blanket drop).
	appendTensorDelta(result, opts, TensorDelta{Name: "b", Status: CompareStatusChanged})
	if len(result.Tensors) != 1 {
		t.Fatalf("changed delta dropped: len = %d, want 1", len(result.Tensors))
	}
}
