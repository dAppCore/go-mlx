// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// indexSinglePackTensor is a small local helper that writes a one-tensor F32
// safetensors file and returns its safetensors.Index. The coverage tests below
// drive the private chunk/merge helpers directly with these indexes (the
// idiomatic unit-call style already used across this package's tests).
func indexSinglePackTensor(t *testing.T, name string, shape []int, data []float32) safetensors.Index {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "tensor.safetensors")
	writeTestSafetensorsF32(t, path, []safetensorTestTensor{{Name: name, Shape: shape, Data: data}})
	index, err := safetensors.IndexFiles([]string{path})
	if err != nil {
		t.Fatalf("index %q: %v", name, err)
	}
	return index
}

// brokenReaderRef returns a TensorRef whose Path points at a real (but tiny)
// file while claiming far more elements than the file holds. A reader built
// over it via NewFileReader opens fine but fails inside ReadFloat32ChunkInto
// with a truncated-chunk read — the seam every chunked write/scan path uses
// to surface a read error.
func brokenReaderRef(t *testing.T, name string) safetensors.TensorRef {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "tiny.safetensors")
	// A single real element on disk, but the ref claims 64 — the reader's
	// ReadAt at the claimed length overruns the file payload.
	writeTestSafetensorsF32(t, path, []safetensorTestTensor{{Name: name, Shape: []int{1}, Data: []float32{1}}})
	index, err := safetensors.IndexFiles([]string{path})
	if err != nil {
		t.Fatalf("index broken ref: %v", err)
	}
	ref := index.Tensors[name]
	ref.Elements = 64
	return ref
}

// openOSFile creates and returns an empty *core.OSFile for chunk-write target
// arguments. The caller closes it.
func openOSFile(t *testing.T, name string) *core.OSFile {
	t.Helper()
	created := core.Create(core.PathJoin(t.TempDir(), name))
	if !created.OK {
		t.Fatalf("create %s: %v", name, created.Value)
	}
	return created.Value.(*core.OSFile)
}

// ---------------------------------------------------------------------------
// merge.go — Packs wrapper-only legs.
// ---------------------------------------------------------------------------

// TestModelMerge_PacksNilContextGood drives Packs with a nil context, covering
// the ctx == nil guard that swaps in context.Background(). The merge otherwise
// succeeds.
func TestModelMerge_PacksNilContextGood(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	//nolint:staticcheck // SA1012: passing nil context is the branch under test.
	result, err := Packs(nil, Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-nil-ctx"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err != nil {
		t.Fatalf("Packs(nil ctx) error = %v", err)
	}
	if result.MergedTensors != 1 {
		t.Fatalf("MergedTensors = %d, want 1", result.MergedTensors)
	}
}

// TestModelMerge_PacksIndexSourcesErrorBad points a source's WeightFiles at a
// missing model.safetensors. prepare() validates only Format/Root/tokenizer
// (never reads the weights), so the merge passes preparation and fails inside
// indexSources — covering Packs's indexSources error return.
func TestModelMerge_PacksIndexSourcesErrorBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// A pack whose declared weight file does not exist on disk. The tokenizer
	// is present so the (allowed-mismatch) compatibility step never reads it.
	rightPack := testPack(right)
	rightPack.WeightFiles = []string{core.PathJoin(right, "does-not-exist.safetensors")}

	_, err := Packs(context.Background(), Options{
		OutputPath:             core.PathJoin(t.TempDir(), "merged-index-fail"),
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: rightPack},
		},
	})
	if err == nil {
		t.Fatal("Packs(missing weight file) error = nil, want index failure")
	}
}

// TestModelMerge_PacksWriteProvenanceErrorBad merges with T = NaN. prepare's
// range guard (T < 0 || T > 1) is false for NaN, so the value flows through to
// writeProvenance where json.Marshal rejects the non-finite float — covering
// both Packs's writeProvenance error return and writeProvenance's own marshal
// error leg. (Linear merge ignores T, so the weights still produce.)
func TestModelMerge_PacksWriteProvenanceErrorBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-nan-t"),
		Method:     MethodLinear,
		T:          math.NaN(),
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(NaN t) error = nil, want provenance marshal failure")
	}
}

// ---------------------------------------------------------------------------
// merge.go — small predicate folds.
// ---------------------------------------------------------------------------

// TestModelMerge_HasSuffixFoldGood covers hasSuffixFold's case-folding match,
// length short-circuit, and the mismatch leg the suffix-not-matched path takes.
func TestModelMerge_HasSuffixFoldGood(t *testing.T) {
	if !hasSuffixFold("MODEL.SAFETENSORS", ".safetensors") {
		t.Fatal("hasSuffixFold(upper, .safetensors) = false, want true")
	}
	if !hasSuffixFold("weights.gguf", ".gguf") {
		t.Fatal("hasSuffixFold(.gguf) = false, want true")
	}
	// Shorter than the suffix short-circuits to false.
	if hasSuffixFold(".g", ".gguf") {
		t.Fatal("hasSuffixFold(shorter than suffix) = true, want false")
	}
	// Same length, last byte differs — the per-byte mismatch leg.
	if hasSuffixFold("model.safetensorx", ".safetensors") {
		t.Fatal("hasSuffixFold(non-matching suffix) = true, want false")
	}
}

// TestModelMerge_SameUint64SliceGood covers sameUint64Slice's equal, length
// mismatch, and element-mismatch legs.
func TestModelMerge_SameUint64SliceGood(t *testing.T) {
	if !sameUint64Slice([]uint64{1, 2, 3}, []uint64{1, 2, 3}) {
		t.Fatal("sameUint64Slice(equal) = false, want true")
	}
	if sameUint64Slice([]uint64{1, 2}, []uint64{1, 2, 3}) {
		t.Fatal("sameUint64Slice(length mismatch) = true, want false")
	}
	if sameUint64Slice([]uint64{1, 2, 3}, []uint64{1, 9, 3}) {
		t.Fatal("sameUint64Slice(element mismatch) = true, want false")
	}
}

// TestModelMerge_EqualFoldMismatchGood covers equalFold's per-byte mismatch
// leg (same length, differing folded bytes) — the case-fold match and length
// short-circuit are already exercised elsewhere.
func TestModelMerge_EqualFoldMismatchGood(t *testing.T) {
	if !equalFold("ABC", "abc") {
		t.Fatal("equalFold(case fold) = false, want true")
	}
	if equalFold("abc", "abd") {
		t.Fatal("equalFold(byte mismatch) = true, want false")
	}
}

// ---------------------------------------------------------------------------
// merge_write.go — readTensorValues legs.
// ---------------------------------------------------------------------------

// TestModelMerge_ReadTensorValuesIncompleteGood drives readTensorValues across
// the two not-complete legs: a name absent from the second index, and a name
// present in both but with a mismatched shape. Both return complete == false
// without error.
func TestModelMerge_ReadTensorValuesIncompleteGood(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})

	// Leg 1: name missing from the second index.
	missing := indexSinglePackTensor(t, "other.weight", []int{2}, []float32{3, 4})
	_, complete, err := readTensorValues([]safetensors.Index{left, missing}, name)
	if err != nil {
		t.Fatalf("readTensorValues(absent) error = %v", err)
	}
	if complete {
		t.Fatal("readTensorValues(absent) complete = true, want false")
	}

	// Leg 2: name present in both indexes but with a different shape.
	differShape := indexSinglePackTensor(t, name, []int{3}, []float32{3, 4, 5})
	_, complete, err = readTensorValues([]safetensors.Index{left, differShape}, name)
	if err != nil {
		t.Fatalf("readTensorValues(shape mismatch) error = %v", err)
	}
	if complete {
		t.Fatal("readTensorValues(shape mismatch) complete = true, want false")
	}
}

// TestModelMerge_ReadTensorValuesReadErrorBad makes the second index point at a
// ref claiming more elements than its file holds, so readTensorValues's
// ReadRefValues call fails — the read-error return leg.
func TestModelMerge_ReadTensorValuesReadErrorBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	broken := brokenReaderRef(t, name)
	brokenIndex := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: broken},
	}
	// Base shape must match the broken ref's shape so readTensorValues reaches
	// the ReadRefValues call rather than bailing on a shape mismatch first.
	base := safetensors.Index{
		Names:   []string{name},
		Tensors: map[string]safetensors.TensorRef{name: {Name: name, Shape: broken.Shape, Elements: broken.Elements, DType: "F32"}},
	}
	_, _, err := readTensorValues([]safetensors.Index{base, brokenIndex}, name)
	if err == nil {
		t.Fatal("readTensorValues(read error) error = nil, want read failure")
	}
	_ = left
}

// ---------------------------------------------------------------------------
// merge_write.go — chunk write/scan error + default legs.
// ---------------------------------------------------------------------------

// TestModelMerge_WriteLinearChunksDefaultsAndErrorsBad covers writeLinearChunks'
// chunkElements<=0 default, the element-count mismatch leg, the OpenReaders
// failure leg, and a successful default-chunk write.
func TestModelMerge_WriteLinearChunksDefaultsAndErrorsBad(t *testing.T) {
	name := "model.norm.weight"
	left := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	right := indexSinglePackTensor(t, name, []int{4}, []float32{5, 6, 7, 8})
	refs := []safetensors.TensorRef{left.Tensors[name], right.Tensors[name]}

	// Element-count mismatch between refs.
	if err := writeLinearChunks(context.Background(), openOSFile(t, "x.bin"),
		[]safetensors.TensorRef{{Elements: 1}, {Elements: 2}}, []float64{0.5, 0.5}, 2); err == nil {
		t.Fatal("writeLinearChunks(element mismatch) error = nil")
	}

	// OpenReaders fails when a ref points at a missing path.
	badRefs := []safetensors.TensorRef{
		{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), Elements: 4, DType: "F32"},
		{Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), Elements: 4, DType: "F32"},
	}
	if err := writeLinearChunks(context.Background(), openOSFile(t, "y.bin"), badRefs, []float64{0.5, 0.5}, 2); err == nil {
		t.Fatal("writeLinearChunks(open readers fail) error = nil")
	}

	// chunkElements <= 0 falls back to the package default and writes cleanly.
	out := openOSFile(t, "z.bin")
	if err := writeLinearChunks(context.Background(), out, refs, []float64{0.5, 0.5}, 0); err != nil {
		t.Fatalf("writeLinearChunks(default chunk) error = %v", err)
	}
	if err := out.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestModelMerge_WriteLinearChunksCachedErrorsBad covers writeLinearChunksCached's
// no-tensors, weight/source mismatch, element-count mismatch, and cache-open
// failure legs through the cached entry point.
func TestModelMerge_WriteLinearChunksCachedErrorsBad(t *testing.T) {
	cache := newFileCache()
	defer cache.close()
	scratch := &writeScratch{}

	if err := writeLinearChunksCached(context.Background(), openOSFile(t, "a.bin"), cache, nil, nil, 2, scratch); err == nil {
		t.Fatal("writeLinearChunksCached(no tensors) error = nil")
	}
	if err := writeLinearChunksCached(context.Background(), openOSFile(t, "b.bin"), cache,
		[]safetensors.TensorRef{{Elements: 1}}, nil, 2, scratch); err == nil {
		t.Fatal("writeLinearChunksCached(weight/source mismatch) error = nil")
	}
	if err := writeLinearChunksCached(context.Background(), openOSFile(t, "c.bin"), cache,
		[]safetensors.TensorRef{{Elements: 1}, {Elements: 2}}, []float64{0.5, 0.5}, 2, scratch); err == nil {
		t.Fatal("writeLinearChunksCached(element mismatch) error = nil")
	}
	// cache.readers fails on a missing path.
	badRefs := []safetensors.TensorRef{
		{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), Elements: 4, DType: "F32"},
		{Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), Elements: 4, DType: "F32"},
	}
	if err := writeLinearChunksCached(context.Background(), openOSFile(t, "d.bin"), cache, badRefs, []float64{0.5, 0.5}, 2, scratch); err == nil {
		t.Fatal("writeLinearChunksCached(cache open fail) error = nil")
	}
}

// TestModelMerge_WriteLinearChunksUsingReadErrorBad gives writeLinearChunksUsing
// a reader over a ref that claims more elements than its file holds, so the
// per-chunk ReadFloat32ChunkInto fails — the chunk-read error leg.
func TestModelMerge_WriteLinearChunksUsingReadErrorBad(t *testing.T) {
	broken := brokenReaderRef(t, "model.norm.weight")
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{broken})
	if err != nil {
		t.Fatalf("open broken reader: %v", err)
	}
	defer safetensors.CloseReaders(readers)
	out := openOSFile(t, "broken.bin")
	defer out.Close()
	if err := writeLinearChunksUsing(context.Background(), out, readers, broken.Elements, []float64{1}, 8, &writeScratch{}); err == nil {
		t.Fatal("writeLinearChunksUsing(read error) error = nil, want read failure")
	}
}

// TestModelMerge_WriteLinearChunksUsingCancelBad cancels the context before the
// chunk loop runs so writeLinearChunksUsing's per-chunk ctx.Err() guard aborts.
func TestModelMerge_WriteLinearChunksUsingCancelBad(t *testing.T) {
	name := "model.norm.weight"
	idx := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{idx.Tensors[name]})
	if err != nil {
		t.Fatalf("open reader: %v", err)
	}
	defer safetensors.CloseReaders(readers)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := openOSFile(t, "cancel.bin")
	defer out.Close()
	if err := writeLinearChunksUsing(ctx, out, readers, 4, []float64{1}, 2, &writeScratch{}); err == nil {
		t.Fatal("writeLinearChunksUsing(cancelled) error = nil")
	}
}

// TestModelMerge_WriteSLERPChunksErrorsBad covers writeSLERPChunks' two-ref
// requirement, element-count mismatch, chunkElements<=0 default, and the
// OpenReaders failure leg.
func TestModelMerge_WriteSLERPChunksErrorsBad(t *testing.T) {
	// Not exactly two refs.
	if err := writeSLERPChunks(context.Background(), openOSFile(t, "a.bin"),
		[]safetensors.TensorRef{{Elements: 1}}, 0.5, 2); err == nil {
		t.Fatal("writeSLERPChunks(one ref) error = nil")
	}
	// Element-count mismatch between the two refs.
	if err := writeSLERPChunks(context.Background(), openOSFile(t, "b.bin"),
		[]safetensors.TensorRef{{Elements: 1}, {Elements: 2}}, 0.5, 2); err == nil {
		t.Fatal("writeSLERPChunks(element mismatch) error = nil")
	}
	// chunkElements <= 0 default + OpenReaders failure (missing paths).
	badRefs := []safetensors.TensorRef{
		{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), Elements: 4, DType: "F32"},
		{Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), Elements: 4, DType: "F32"},
	}
	if err := writeSLERPChunks(context.Background(), openOSFile(t, "c.bin"), badRefs, 0.5, 0); err == nil {
		t.Fatal("writeSLERPChunks(default chunk, open fail) error = nil")
	}
}

// TestModelMerge_WriteSLERPChunksCachedErrorsBad covers writeSLERPChunksCached's
// two-ref requirement, element mismatch, chunkElements<=0 default, and the
// cache-open failure leg.
func TestModelMerge_WriteSLERPChunksCachedErrorsBad(t *testing.T) {
	cache := newFileCache()
	defer cache.close()
	scratch := &writeScratch{}

	if err := writeSLERPChunksCached(context.Background(), openOSFile(t, "a.bin"), cache,
		[]safetensors.TensorRef{{Elements: 1}}, 0.5, 2, scratch); err == nil {
		t.Fatal("writeSLERPChunksCached(one ref) error = nil")
	}
	if err := writeSLERPChunksCached(context.Background(), openOSFile(t, "b.bin"), cache,
		[]safetensors.TensorRef{{Elements: 1}, {Elements: 2}}, 0.5, 2, scratch); err == nil {
		t.Fatal("writeSLERPChunksCached(element mismatch) error = nil")
	}
	badRefs := []safetensors.TensorRef{
		{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), Elements: 4, DType: "F32"},
		{Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), Elements: 4, DType: "F32"},
	}
	// chunkElements <= 0 default path + cache-open failure on missing paths.
	if err := writeSLERPChunksCached(context.Background(), openOSFile(t, "c.bin"), cache, badRefs, 0.5, 0, scratch); err == nil {
		t.Fatal("writeSLERPChunksCached(default chunk, cache open fail) error = nil")
	}
}

// TestModelMerge_SLERPChunkedWeightsFromReadersErrorsBad covers the readers-already-open
// scan: the two-readers requirement, the chunkElements<=0 default, and a
// chunk-read failure from a reader over a too-large ref.
func TestModelMerge_SLERPChunkedWeightsFromReadersErrorsBad(t *testing.T) {
	// Not exactly two readers.
	if _, err := slerpChunkedWeightsFromReaders(context.Background(), nil, 4, 0.5, 4, nil, nil); err == nil {
		t.Fatal("slerpChunkedWeightsFromReaders(no readers) error = nil")
	}

	// Read failure on the first reader (claims more elements than the file has).
	broken := brokenReaderRef(t, "model.norm.weight")
	name2 := "model.norm.weight"
	good := indexSinglePackTensor(t, name2, []int{1}, []float32{1})
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{broken, good.Tensors[name2]})
	if err != nil {
		t.Fatalf("open readers: %v", err)
	}
	defer safetensors.CloseReaders(readers)
	// chunkElements <= 0 default applied; the first chunk read overruns.
	if _, err := slerpChunkedWeightsFromReaders(context.Background(), readers, broken.Elements, 0.5, 0, nil, nil); err == nil {
		t.Fatal("slerpChunkedWeightsFromReaders(read error) error = nil")
	}
}

// TestModelMerge_SLERPChunkedWeightsFromReadersSecondReadErrorBad makes the
// SECOND reader the broken one so the scan reaches readers[1].ReadFloat32ChunkInto
// before failing — the second-read error leg.
func TestModelMerge_SLERPChunkedWeightsFromReadersSecondReadErrorBad(t *testing.T) {
	name := "model.norm.weight"
	good := indexSinglePackTensor(t, name, []int{64}, make([]float32, 64))
	broken := brokenReaderRef(t, name)
	// Both readers must agree on element count for the scan loop; the good
	// reader has 64 real elements, the broken one claims 64 but holds 1.
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{good.Tensors[name], broken})
	if err != nil {
		t.Fatalf("open readers: %v", err)
	}
	defer safetensors.CloseReaders(readers)
	if _, err := slerpChunkedWeightsFromReaders(context.Background(), readers, 64, 0.5, 64, nil, nil); err == nil {
		t.Fatal("slerpChunkedWeightsFromReaders(second read error) error = nil")
	}
}

// TestModelMerge_SLERPChunkedWeightsFromReadersCancelBad cancels before the
// scan loop so its per-chunk ctx.Err() guard aborts.
func TestModelMerge_SLERPChunkedWeightsFromReadersCancelBad(t *testing.T) {
	name := "model.norm.weight"
	a := indexSinglePackTensor(t, name, []int{4}, []float32{1, 0, 0, 0})
	b := indexSinglePackTensor(t, name, []int{4}, []float32{0, 1, 0, 0})
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{a.Tensors[name], b.Tensors[name]})
	if err != nil {
		t.Fatalf("open readers: %v", err)
	}
	defer safetensors.CloseReaders(readers)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := slerpChunkedWeightsFromReaders(ctx, readers, 4, 0.5, 2, nil, nil); err == nil {
		t.Fatal("slerpChunkedWeightsFromReaders(cancelled) error = nil")
	}
}

// TestModelMerge_SlerpChunkedWeightsDefaultChunkGood covers slerpChunkedWeights'
// chunkElements<=0 default (the single-call wrapper), succeeding on orthogonal
// unit vectors.
func TestModelMerge_SlerpChunkedWeightsDefaultChunkGood(t *testing.T) {
	name := "model.embed_tokens.weight"
	a := indexSinglePackTensor(t, name, []int{2}, []float32{1, 0})
	b := indexSinglePackTensor(t, name, []int{2}, []float32{0, 1})
	weights, err := slerpChunkedWeights(context.Background(), []safetensors.TensorRef{a.Tensors[name], b.Tensors[name]}, 0.5, 0)
	if err != nil {
		t.Fatalf("slerpChunkedWeights(default chunk) error = %v", err)
	}
	want := math.Sin(math.Pi/4) / math.Sin(math.Pi/2)
	if math.Abs(weights[0]-want) > 1e-9 || math.Abs(weights[1]-want) > 1e-9 {
		t.Fatalf("weights = %v, want both %f", weights, want)
	}
}

// TestModelMerge_SlerpChunkedWeightsOpenReadersFailBad points the wrapper at a
// missing path so its OpenReaders fails.
func TestModelMerge_SlerpChunkedWeightsOpenReadersFailBad(t *testing.T) {
	refs := []safetensors.TensorRef{
		{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), Elements: 2, DType: "F32"},
		{Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), Elements: 2, DType: "F32"},
	}
	if _, err := slerpChunkedWeights(context.Background(), refs, 0.5, 4); err == nil {
		t.Fatal("slerpChunkedWeights(open readers fail) error = nil")
	}
}

// TestModelMerge_SlerpMergeNearParallelGood feeds slerpMerge two nearly
// identical vectors (cosTheta > 0.9995), driving the near-parallel branch that
// falls back to the linear weight pair rather than the sin-ratio interpolation.
func TestModelMerge_SlerpMergeNearParallelGood(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	// b is a tiny perturbation of a — the unit vectors are within the 0.9995
	// cosine threshold, so the merge uses the linear (1-t, t) fallback.
	b := []float32{1.0001, 2.0001, 3.0001, 4.0001}
	out, err := slerpMerge([][]float32{a, b}, 0.5)
	if err != nil {
		t.Fatalf("slerpMerge(near parallel) error = %v", err)
	}
	// Linear fallback with t = 0.5 averages the two vectors element-wise.
	for i := range out {
		want := 0.5*a[i] + 0.5*b[i]
		if math.Abs(float64(out[i]-want)) > 1e-4 {
			t.Fatalf("out[%d] = %f, want ~%f", i, out[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// merge_copy.go — copy/hash legs.
// ---------------------------------------------------------------------------

// TestModelMerge_CopyModelPackMetadataMissingDirGood points copyModelPackMetadata
// at a non-existent source directory: ReadDir fails and the function returns nil
// (a missing metadata dir is not fatal to a merge).
func TestModelMerge_CopyModelPackMetadataMissingDirGood(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "no-such-dir")
	out := t.TempDir()
	if err := copyModelPackMetadata(missing, out); err != nil {
		t.Fatalf("copyModelPackMetadata(missing source) error = %v, want nil", err)
	}
}

// TestModelMerge_CopyModelPackMetadataSkipsDirGood places a subdirectory whose
// name carries a metadata suffix in the source. copyModelPackMetadata must skip
// it (entry.IsDir() continue) and copy only the regular metadata file.
func TestModelMerge_CopyModelPackMetadataSkipsDirGood(t *testing.T) {
	src := t.TempDir()
	out := t.TempDir()
	// A directory whose name ends in .json — must be skipped, not copied.
	if result := core.MkdirAll(core.PathJoin(src, "nested.json"), 0o755); !result.OK {
		t.Fatalf("mkdir nested.json: %v", result.Value)
	}
	// A regular metadata file that should be copied.
	writeModelPackFile(t, core.PathJoin(src, "config.json"), `{"ok":true}`)
	if err := copyModelPackMetadata(src, out); err != nil {
		t.Fatalf("copyModelPackMetadata() error = %v", err)
	}
	if stat := core.Stat(core.PathJoin(out, "config.json")); !stat.OK {
		t.Fatal("config.json should have been copied")
	}
	if stat := core.Stat(core.PathJoin(out, "nested.json")); stat.OK {
		t.Fatal("directory named nested.json should have been skipped, not copied")
	}
}

// TestModelMerge_CopyModelPackLocalFileDestErrorBad points the destination at a
// path whose parent is not a directory, so copyModelPackLocalFile's destination
// open fails after the source opened cleanly.
func TestModelMerge_CopyModelPackLocalFileDestErrorBad(t *testing.T) {
	src := core.PathJoin(t.TempDir(), "source.txt")
	writeModelPackFile(t, src, "payload")
	// A regular file standing in for a directory: <file>/dest cannot be created.
	notDir := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, notDir, "x")
	dest := core.PathJoin(notDir, "dest.txt")
	if err := copyModelPackLocalFile(src, dest); err == nil {
		t.Fatal("copyModelPackLocalFile(bad dest) error = nil, want open failure")
	}
}

// TestModelMerge_CopyModelPackLocalFileSourceErrorBad points the source at a
// missing path so copyModelPackLocalFile's source open fails.
func TestModelMerge_CopyModelPackLocalFileSourceErrorBad(t *testing.T) {
	src := core.PathJoin(t.TempDir(), "missing-source.txt")
	dest := core.PathJoin(t.TempDir(), "dest.txt")
	if err := copyModelPackLocalFile(src, dest); err == nil {
		t.Fatal("copyModelPackLocalFile(missing source) error = nil, want open failure")
	}
}

// TestModelMerge_ModelPackCopyResultErrorGood covers modelPackCopyResultError's
// OK-result (nil) and non-error-value (fallback error) legs.
func TestModelMerge_ModelPackCopyResultErrorGood(t *testing.T) {
	if err := modelPackCopyResultError(core.Ok("fine")); err != nil {
		t.Fatalf("modelPackCopyResultError(ok) = %v, want nil", err)
	}
	if err := modelPackCopyResultError(core.Result{Value: "not an error", OK: false}); err == nil {
		t.Fatal("modelPackCopyResultError(non-error value) = nil, want fallback error")
	}
}

// TestModelMerge_HashFileMissingBad points hashFile at a missing path so its
// ReadFile fails — the read-error return leg (the non-byte-data leg is
// unreachable: a successful ReadFile always yields []byte).
func TestModelMerge_HashFileMissingBad(t *testing.T) {
	if _, err := hashFile(core.PathJoin(t.TempDir(), "absent.json")); err == nil {
		t.Fatal("hashFile(missing) error = nil, want read failure")
	}
}

// TestModelMerge_HashFileGood hashes a real file and confirms the digest is the
// stable SHA-256 hex of its bytes.
func TestModelMerge_HashFileGood(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "data.json")
	writeModelPackFile(t, path, "abc")
	got, err := hashFile(path)
	if err != nil {
		t.Fatalf("hashFile() error = %v", err)
	}
	want := core.SHA256Hex([]byte("abc"))
	if got != want {
		t.Fatalf("hashFile() = %q, want %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// compare.go — ComparePacks wrapper legs + helpers.
// ---------------------------------------------------------------------------

// TestCompare_ComparePacksNilContextGood drives ComparePacks with a nil context,
// covering the ctx == nil guard.
func TestCompare_ComparePacksNilContextGood(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 3}},
	})
	//nolint:staticcheck // SA1012: passing nil context is the branch under test.
	result, err := ComparePacks(nil, CompareOptions{Base: testPack(base), FineTuned: testPack(tuned)})
	if err != nil {
		t.Fatalf("ComparePacks(nil ctx) error = %v", err)
	}
	if result.ChangedTensors != 1 {
		t.Fatalf("ChangedTensors = %d, want 1", result.ChangedTensors)
	}
}

// TestCompare_ComparePacksCancelledBad cancels the context before ComparePacks
// runs so the early ctx.Err() guard rejects the comparison.
func TestCompare_ComparePacksCancelledBad(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 3}},
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ComparePacks(ctx, CompareOptions{Base: testPack(base), FineTuned: testPack(tuned)}); err == nil {
		t.Fatal("ComparePacks(cancelled) error = nil")
	}
}

// TestCompare_ComparePacksCancelDuringLoopBad lets the early ctx.Err() pass then
// trips the per-tensor loop guard, covering ComparePacks's in-loop cancellation
// return. The countingCancelContext returns nil for the first call (the early
// guard) and Canceled thereafter (the first loop iteration).
func TestCompare_ComparePacksCancelDuringLoopBad(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 3}},
	})
	ctx := &countingCancelContext{Context: context.Background(), cancelAfter: 1}
	if _, err := ComparePacks(ctx, CompareOptions{Base: testPack(base), FineTuned: testPack(tuned)}); err == nil {
		t.Fatal("ComparePacks(cancel during loop) error = nil")
	}
}

// TestCompare_ComparePacksTensorReadErrorBad truncates the tuned pack's weight
// file on disk after it is written: the safetensors header still declares the
// full tensor (so indexing succeeds), but the payload bytes are gone, so the
// per-tensor chunk read fails. This surfaces through ComparePacks's
// compareTensorRefs error return — the wrapper-only error leg.
func TestCompare_ComparePacksTensorReadErrorBad(t *testing.T) {
	name := "model.norm.weight"
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: name, Shape: []int{8}, Data: make([]float32, 8)},
	})
	tunedDir := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: name, Shape: []int{8}, Data: make([]float32, 8)},
	})
	// Truncate the tuned weight file to just its 8-byte length prefix + header,
	// dropping the tensor payload. The header still claims 8 elements.
	weightPath := core.PathJoin(tunedDir, "model.safetensors")
	read := core.ReadFile(weightPath)
	if !read.OK {
		t.Fatalf("read tuned weights: %v", read.Value)
	}
	raw := read.Value.([]byte)
	headerLen := int(binary.LittleEndian.Uint64(raw[:8]))
	truncated := raw[:8+headerLen] // header only, no payload
	if result := core.WriteFile(weightPath, truncated, 0o644); !result.OK {
		t.Fatalf("truncate tuned weights: %v", result.Value)
	}

	if _, err := ComparePacks(context.Background(), CompareOptions{
		Base:      testPack(base),
		FineTuned: testPack(tunedDir),
	}); err == nil {
		t.Fatal("ComparePacks(truncated tuned weights) error = nil, want read failure")
	}
}

// TestCompare_CompareTensorRefsErrorsBad drives compareTensorRefs directly: a
// shape mismatch (early return), a dtype mismatch (early return), the
// chunkElements<=0 default, a base reader open failure, a tuned reader open
// failure, and chunk-read failures on each side.
func TestCompare_CompareTensorRefsErrorsBad(t *testing.T) {
	ctx := context.Background()
	cache := newFileCache()
	defer cache.close()
	scratch := &compareScratch{}

	// Shape mismatch — returns a shape-mismatch delta without reading.
	delta, err := compareTensorRefs(ctx, cache, scratch, nil,
		safetensors.TensorRef{Name: "a", DType: "F32", Shape: []uint64{2}, Elements: 2},
		safetensors.TensorRef{Name: "a", DType: "F32", Shape: []uint64{3}, Elements: 3}, 0)
	if err != nil {
		t.Fatalf("compareTensorRefs(shape mismatch) error = %v", err)
	}
	if delta.Status != CompareStatusShapeMismatch {
		t.Fatalf("status = %q, want shape_mismatch", delta.Status)
	}

	// DType mismatch (same shape) — returns a dtype-mismatch delta. arena=nil
	// exercises the dualShapeClone branch of the clone selection.
	delta, err = compareTensorRefs(ctx, cache, scratch, nil,
		safetensors.TensorRef{Name: "a", DType: "F32", Shape: []uint64{2}, Elements: 2},
		safetensors.TensorRef{Name: "a", DType: "F16", Shape: []uint64{2}, Elements: 2}, 0)
	if err != nil {
		t.Fatalf("compareTensorRefs(dtype mismatch) error = %v", err)
	}
	if delta.Status != CompareStatusDTypeMismatch {
		t.Fatalf("status = %q, want dtype_mismatch", delta.Status)
	}

	// Base reader open failure: base path is missing (shape/dtype match so the
	// function reaches the reader open, and chunkElements<=0 takes the default).
	missingBase := safetensors.TensorRef{Name: "a", Path: core.PathJoin(t.TempDir(), "nope.safetensors"), DType: "F32", Shape: []uint64{2}, Elements: 2}
	name := "model.norm.weight"
	goodIdx := indexSinglePackTensor(t, name, []int{2}, []float32{1, 2})
	goodRef := goodIdx.Tensors[name]
	if _, err := compareTensorRefs(ctx, cache, scratch, nil, missingBase, missingBase, 0); err == nil {
		t.Fatal("compareTensorRefs(missing base) error = nil")
	}

	// Tuned reader open failure: base good, tuned missing (same shape/dtype).
	missingTuned := safetensors.TensorRef{Name: name, Path: core.PathJoin(t.TempDir(), "nope2.safetensors"), DType: "F32", Shape: goodRef.Shape, Elements: goodRef.Elements}
	if _, err := compareTensorRefs(ctx, cache, scratch, nil, goodRef, missingTuned, 4); err == nil {
		t.Fatal("compareTensorRefs(missing tuned) error = nil")
	}

	// Base chunk-read failure: base over-claims elements (file too small).
	brokenBase := brokenReaderRef(t, name)
	matchingTuned := safetensors.TensorRef{Name: name, Path: goodRef.Path, DType: "F32", Shape: brokenBase.Shape, Elements: brokenBase.Elements, DataStart: goodRef.DataStart}
	if _, err := compareTensorRefs(ctx, cache, scratch, nil, brokenBase, matchingTuned, 8); err == nil {
		t.Fatal("compareTensorRefs(base read error) error = nil")
	}

	// Tuned chunk-read failure: base good (64 real), tuned over-claims.
	bigName := "big.weight"
	bigIdx := indexSinglePackTensor(t, bigName, []int{64}, make([]float32, 64))
	bigRef := bigIdx.Tensors[bigName]
	brokenTuned := brokenReaderRef(t, name)
	brokenTuned2 := safetensors.TensorRef{Name: bigName, Path: brokenTuned.Path, DType: "F32", Shape: bigRef.Shape, Elements: 64, DataStart: brokenTuned.DataStart}
	if _, err := compareTensorRefs(ctx, cache, scratch, nil, bigRef, brokenTuned2, 64); err == nil {
		t.Fatal("compareTensorRefs(tuned read error) error = nil")
	}
}

// TestCompare_CompareTensorRefsCancelBad cancels before the chunk loop so
// compareTensorRefs's per-chunk ctx.Err() guard aborts.
func TestCompare_CompareTensorRefsCancelBad(t *testing.T) {
	name := "model.norm.weight"
	idx := indexSinglePackTensor(t, name, []int{4}, []float32{1, 2, 3, 4})
	ref := idx.Tensors[name]
	cache := newFileCache()
	defer cache.close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := compareTensorRefs(ctx, cache, &compareScratch{}, nil, ref, ref, 2); err == nil {
		t.Fatal("compareTensorRefs(cancelled) error = nil")
	}
}

// TestCompare_FileCacheReaderOpenErrorBad covers fileCache.reader's open-failure
// leg (and, via readers, its propagation) when a ref points at a missing path.
func TestCompare_FileCacheReaderOpenErrorBad(t *testing.T) {
	cache := newFileCache()
	defer cache.close()
	missing := safetensors.TensorRef{Path: core.PathJoin(t.TempDir(), "nope.safetensors"), DType: "F32", Elements: 1}
	if _, err := cache.reader(missing); err == nil {
		t.Fatal("cache.reader(missing) error = nil")
	}
	if _, err := cache.readers([]safetensors.TensorRef{missing}); err == nil {
		t.Fatal("cache.readers(missing) error = nil")
	}
}

// TestCompare_CloneUint64sGood covers cloneUint64s's empty (nil) and populated
// legs.
func TestCompare_CloneUint64sGood(t *testing.T) {
	if got := cloneUint64s(nil); got != nil {
		t.Fatalf("cloneUint64s(nil) = %v, want nil", got)
	}
	if got := cloneUint64s([]uint64{}); got != nil {
		t.Fatalf("cloneUint64s(empty) = %v, want nil", got)
	}
	got := cloneUint64s([]uint64{1, 2, 3})
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("cloneUint64s([1 2 3]) = %v", got)
	}
}

// TestCompare_ShapeArenaCloneGood covers shapeArena.clone's empty-src (nil),
// in-slab carve, and slab-exhausted fallback legs.
func TestCompare_ShapeArenaCloneGood(t *testing.T) {
	// Empty source returns nil regardless of slab state.
	arena := &shapeArena{slab: make([]uint64, 4)}
	if got := arena.clone(nil); got != nil {
		t.Fatalf("clone(nil) = %v, want nil", got)
	}
	// In-slab carve: a 2-element clone fits the 4-slot slab.
	got := arena.clone([]uint64{7, 8})
	if len(got) != 2 || got[0] != 7 || got[1] != 8 || cap(got) != 2 {
		t.Fatalf("clone([7 8]) = %v (cap %d)", got, cap(got))
	}
	// Slab exhausted: a 3-element clone no longer fits (cursor=2, slab=4), so
	// it falls back to a fresh allocation.
	got = arena.clone([]uint64{1, 2, 3})
	if len(got) != 3 || got[2] != 3 {
		t.Fatalf("clone([1 2 3]) fallback = %v", got)
	}
	// A nil arena receiver also takes the fallback path.
	var nilArena *shapeArena
	if got := nilArena.clone([]uint64{9}); len(got) != 1 || got[0] != 9 {
		t.Fatalf("(*shapeArena)(nil).clone([9]) = %v", got)
	}
}

// TestCompare_AppendTensorDeltaMaxReportsGood covers appendTensorDelta's
// MaxTensorReports cap (the early return once the report budget is full).
func TestCompare_AppendTensorDeltaMaxReportsGood(t *testing.T) {
	result := &CompareResult{Tensors: make([]TensorDelta, 0, 1)}
	opts := CompareOptions{MaxTensorReports: 1}
	appendTensorDelta(result, opts, TensorDelta{Name: "a", Status: CompareStatusChanged})
	if len(result.Tensors) != 1 {
		t.Fatalf("after first append: len = %d, want 1", len(result.Tensors))
	}
	// Budget full — the second delta must be dropped.
	appendTensorDelta(result, opts, TensorDelta{Name: "b", Status: CompareStatusChanged})
	if len(result.Tensors) != 1 {
		t.Fatalf("after capped append: len = %d, want 1", len(result.Tensors))
	}
}

// TestCompare_ValidateComparePackBad covers validateComparePack's missing-root,
// wrong-format, and missing-weight-files legs for both labels.
func TestCompare_ValidateComparePackBad(t *testing.T) {
	if err := validateComparePack("base", mp.ModelPack{}); err == nil {
		t.Fatal("validateComparePack(no root) error = nil")
	}
	if err := validateComparePack("base", mp.ModelPack{Root: "/x", Format: mp.ModelPackFormatGGUF}); err == nil {
		t.Fatal("validateComparePack(wrong format) error = nil")
	}
	if err := validateComparePack("fine-tuned", mp.ModelPack{Root: "/x", Format: mp.ModelPackFormatSafetensors}); err == nil {
		t.Fatal("validateComparePack(no weight files) error = nil")
	}
}
