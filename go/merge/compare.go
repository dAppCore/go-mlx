// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// CompareStatus classifies one tensor when comparing a base model pack against
// a fine-tuned pack.
type CompareStatus string

const (
	CompareStatusChanged        CompareStatus = "changed"
	CompareStatusUnchanged      CompareStatus = "unchanged"
	CompareStatusMissingInTuned CompareStatus = "missing_in_fine_tuned"
	CompareStatusExtraInTuned   CompareStatus = "extra_in_fine_tuned"
	CompareStatusShapeMismatch  CompareStatus = "shape_mismatch"
	CompareStatusDTypeMismatch  CompareStatus = "dtype_mismatch"
)

// CompareOptions configures a safetensors weight comparison.
type CompareOptions struct {
	Base             mp.ModelPack      `json:"base"`
	FineTuned        mp.ModelPack      `json:"fine_tuned"`
	IncludeUnchanged bool              `json:"include_unchanged,omitempty"`
	MaxTensorReports int               `json:"max_tensor_reports,omitempty"`
	Labels           map[string]string `json:"labels,omitempty"`
}

// TensorDelta reports per-tensor distance statistics between base and
// fine-tuned weights.
type TensorDelta struct {
	Name           string        `json:"name"`
	Status         CompareStatus `json:"status"`
	BaseDType      string        `json:"base_dtype,omitempty"`
	FineTunedDType string        `json:"fine_tuned_dtype,omitempty"`
	Shape          []uint64      `json:"shape,omitempty"`
	BaseShape      []uint64      `json:"base_shape,omitempty"`
	FineTunedShape []uint64      `json:"fine_tuned_shape,omitempty"`
	Elements       int           `json:"elements,omitempty"`
	MeanAbsDelta   float64       `json:"mean_abs_delta,omitempty"`
	RMSDelta       float64       `json:"rms_delta,omitempty"`
	MaxAbsDelta    float64       `json:"max_abs_delta,omitempty"`
	L2Delta        float64       `json:"l2_delta,omitempty"`
	Cosine         float64       `json:"cosine,omitempty"`
}

// CompareResult summarises base/fine-tuned tensor differences without loading
// either model through the runtime.
type CompareResult struct {
	Base               mp.ModelPack      `json:"base"`
	FineTuned          mp.ModelPack      `json:"fine_tuned"`
	TensorCount        int               `json:"tensor_count"`
	ComparedTensors    int               `json:"compared_tensors"`
	ChangedTensors     int               `json:"changed_tensors"`
	UnchangedTensors   int               `json:"unchanged_tensors"`
	MissingInFineTuned int               `json:"missing_in_fine_tuned"`
	ExtraInFineTuned   int               `json:"extra_in_fine_tuned"`
	ShapeMismatches    int               `json:"shape_mismatches"`
	DTypeMismatches    int               `json:"dtype_mismatches"`
	ElementsCompared   int               `json:"elements_compared"`
	MeanAbsDelta       float64           `json:"mean_abs_delta,omitempty"`
	RMSDelta           float64           `json:"rms_delta,omitempty"`
	MaxAbsDelta        float64           `json:"max_abs_delta,omitempty"`
	Tensors            []TensorDelta     `json:"tensors,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
}

// ComparePacks compares safetensors weights in a base model pack against a
// fine-tuned pack and returns aggregate plus per-tensor delta metrics.
func ComparePacks(ctx context.Context, opts CompareOptions) (*CompareResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateComparePack("base", opts.Base); err != nil {
		return nil, err
	}
	if err := validateComparePack("fine-tuned", opts.FineTuned); err != nil {
		return nil, err
	}
	baseIndex, err := safetensors.IndexFiles(opts.Base.WeightFiles)
	if err != nil {
		return nil, core.E("ComparePacks", "index base weights", err)
	}
	tunedIndex, err := safetensors.IndexFiles(opts.FineTuned.WeightFiles)
	if err != nil {
		return nil, core.E("ComparePacks", "index fine-tuned weights", err)
	}

	// Pre-size result.Tensors: it grows to at most len(baseIndex.Names)
	// entries (every base tensor either appears in tuned or not). Growing
	// through the default nil/zero-cap path costs N growslice walks for
	// large N.
	expectedTensors := len(baseIndex.Names)
	if opts.MaxTensorReports > 0 && opts.MaxTensorReports < expectedTensors {
		expectedTensors = opts.MaxTensorReports
	}
	result := &CompareResult{
		Base:      opts.Base,
		FineTuned: opts.FineTuned,
		Labels:    cloneCompareLabels(opts.Labels),
		Tensors:   make([]TensorDelta, 0, expectedTensors),
	}
	// One file cache for the whole compare pass — opens each shard once
	// and shares the handle across every tensor's reader instead of
	// re-opening base + tuned per tensor.
	cache := newFileCache()
	defer cache.close()
	// One scratch set for the whole pass — the per-tensor chunk reads grow
	// it to the largest tensor once, then reuse it.
	scratch := &compareScratch{}
	// One slab for every matched tensor's base + tuned shape clones. Pre-walk
	// the matched names to size it exactly (each compared tensor carves
	// len(base.Shape)+len(tuned.Shape) uint64s); the per-tensor make in
	// dualShapeClone was the dominant ComparePacks alloc count. Only the
	// matched path uses the arena — the rare missing/extra tensors keep their
	// own cloneUint64s.
	totalShapeDims := 0
	for _, name := range baseIndex.Names {
		baseRef := baseIndex.Tensors[name]
		if tunedRef, ok := tunedIndex.Tensors[name]; ok {
			totalShapeDims += len(baseRef.Shape) + len(tunedRef.Shape)
		}
	}
	var arena *shapeArena
	if totalShapeDims > 0 {
		arena = &shapeArena{slab: make([]uint64, totalShapeDims)}
	}
	acc := compareAccumulator{}
	for _, name := range baseIndex.Names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		baseRef := baseIndex.Tensors[name]
		tunedRef, ok := tunedIndex.Tensors[name]
		if !ok {
			result.MissingInFineTuned++
			appendTensorDelta(result, opts, TensorDelta{
				Name:      name,
				Status:    CompareStatusMissingInTuned,
				BaseDType: baseRef.DType,
				BaseShape: cloneUint64s(baseRef.Shape),
				Elements:  baseRef.Elements,
			})
			continue
		}
		delta, err := compareTensorRefs(ctx, cache, scratch, arena, baseRef, tunedRef, modelMergeTensorChunkElements)
		if err != nil {
			return nil, core.E("ComparePacks", "compare tensor "+name, err)
		}
		recordTensorDelta(result, &acc, opts, delta)
	}
	// Walk tunedIndex.Names once and consult baseIndex.Tensors to detect
	// extras — previously a separate tunedSeen map was built up during
	// the base loop just to filter this pass. baseIndex.Tensors is the
	// authoritative "name was present in base" lookup; using it directly
	// drops the tunedSeen map allocation + the per-base-match map insert.
	for _, name := range tunedIndex.Names {
		if _, ok := baseIndex.Tensors[name]; ok {
			continue
		}
		tunedRef := tunedIndex.Tensors[name]
		result.ExtraInFineTuned++
		appendTensorDelta(result, opts, TensorDelta{
			Name:           name,
			Status:         CompareStatusExtraInTuned,
			FineTunedDType: tunedRef.DType,
			FineTunedShape: cloneUint64s(tunedRef.Shape),
			Elements:       tunedRef.Elements,
		})
	}
	result.TensorCount = result.ComparedTensors + result.MissingInFineTuned + result.ExtraInFineTuned + result.ShapeMismatches + result.DTypeMismatches
	if acc.elements > 0 {
		result.ElementsCompared = acc.elements
		result.MeanAbsDelta = acc.sumAbs / float64(acc.elements)
		result.RMSDelta = math.Sqrt(acc.sumSq / float64(acc.elements))
		result.MaxAbsDelta = acc.maxAbs
	}
	return result, nil
}

type compareAccumulator struct {
	elements int
	sumAbs   float64
	sumSq    float64
	maxAbs   float64
}

func validateComparePack(label string, pack mp.ModelPack) error {
	if pack.Root == "" {
		return core.NewError("mlx: " + label + " model pack root is required")
	}
	if pack.Format != mp.ModelPackFormatSafetensors {
		return core.NewError("mlx: " + label + " model comparison requires safetensors weights")
	}
	if len(pack.WeightFiles) == 0 {
		return core.NewError("mlx: " + label + " model comparison requires weight files")
	}
	return nil
}

// fileCache lazily opens safetensors shard files and caches the handle by
// path. Every tensor in a pack typically lives in the same shard, so
// ComparePacks previously paid an os.Open + the path→C-string
// syscall.ByteSliceFromString allocation for base AND tuned on every
// tensor (16 opens for an 8-tensor compare). The cache opens each distinct
// shard once and hands ReadAt-based readers (NewFileReader) over the shared
// handle, dropping the file-open allocs — the dominant ComparePacks cost by
// alloc_objects — to one per distinct shard. ReadAt is offset-addressed so
// many readers safely share one handle. The caller closes the cache once.
type fileCache struct {
	files map[string]*core.OSFile
	// readerBuf is reused by readers() so the merge write loop's
	// per-tensor reader slice is allocated once for the whole pass
	// instead of once per tensor.
	readerBuf []safetensors.TensorReader
}

func newFileCache() *fileCache {
	return &fileCache{files: make(map[string]*core.OSFile, 2)}
}

func (c *fileCache) reader(ref safetensors.TensorRef) (safetensors.TensorReader, error) {
	file, ok := c.files[ref.Path]
	if !ok {
		opened := core.Open(ref.Path)
		if !opened.OK {
			return safetensors.TensorReader{}, resultError(opened)
		}
		file = opened.Value.(*core.OSFile)
		c.files[ref.Path] = file
	}
	return safetensors.NewFileReader(file, ref)
}

// readers returns a TensorReader for each ref, all bound to cache-owned
// shard handles (opened once, shared via ReadAt). The returned slice
// aliases the cache's reusable readerBuf — it is valid only until the
// next readers() call, which the merge write loop honours (it consumes
// the readers fully within one tensor iteration before the next). Used
// by the merge write path so the source shards are opened once for the
// whole merge rather than once per tensor.
func (c *fileCache) readers(refs []safetensors.TensorRef) ([]safetensors.TensorReader, error) {
	buf := c.readerBuf[:0]
	if cap(buf) < len(refs) {
		buf = make([]safetensors.TensorReader, 0, len(refs))
	}
	for _, ref := range refs {
		reader, err := c.reader(ref)
		if err != nil {
			return nil, err
		}
		buf = append(buf, reader)
	}
	c.readerBuf = buf
	return buf, nil
}

func (c *fileCache) close() {
	for _, file := range c.files {
		_ = file.Close()
	}
}

// compareScratch holds the per-reader chunk buffers reused across every
// tensor of a ComparePacks pass. Each tensor's compareTensorRefs
// previously started from nil scratch, so the chunk read for base + tuned
// re-allocated a raw []byte and a decoded []float32 per tensor (the
// dominant ComparePacks B/op once the shard handles were cached). Owning
// the buffers at the pass level grows them to the largest tensor once and
// reuses them for every subsequent tensor. base and tuned keep separate
// buffers because both chunks are live simultaneously in the diff loop.
type compareScratch struct {
	rawBase     []byte
	rawTuned    []byte
	valuesBase  []float32
	valuesTuned []float32
}

func compareTensorRefs(ctx context.Context, cache *fileCache, scratch *compareScratch, arena *shapeArena, base, tuned safetensors.TensorRef, chunkElements int) (TensorDelta, error) {
	// Shape clones come from the pass-level arena when ComparePacks supplies
	// one (carving cap==len sub-slices from a single pre-sized slab), else
	// from dualShapeClone (one arena for base + tuned). TensorDelta carries
	// the BaseShape and FineTunedShape fields as independent sub-slices;
	// consumers never mutate either, so the shared backing array is safe.
	shapeMatch := sameUint64Slice(base.Shape, tuned.Shape) && base.Elements == tuned.Elements
	var baseShapeClone, tunedShapeClone []uint64
	if arena != nil {
		baseShapeClone = arena.clone(base.Shape)
		tunedShapeClone = arena.clone(tuned.Shape)
	} else {
		baseShapeClone, tunedShapeClone = dualShapeClone(base.Shape, tuned.Shape)
	}
	delta := TensorDelta{
		Name:           base.Name,
		BaseDType:      base.DType,
		FineTunedDType: tuned.DType,
		BaseShape:      baseShapeClone,
		FineTunedShape: tunedShapeClone,
		Elements:       base.Elements,
	}
	if !shapeMatch {
		delta.Status = CompareStatusShapeMismatch
		return delta, nil
	}
	// Reuse the base-shape clone for Shape — it's the same array of
	// uint64s and TensorDelta does not mutate either field.
	delta.Shape = baseShapeClone
	if base.DType != tuned.DType {
		delta.Status = CompareStatusDTypeMismatch
		return delta, nil
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	// Readers share cached shard handles (see fileCache) — no per-tensor
	// open, no per-tensor close. The cache owns the *core.OSFile lifetime
	// and ComparePacks closes it once for the whole pass.
	baseReader, err := cache.reader(base)
	if err != nil {
		return TensorDelta{}, err
	}
	tunedReader, err := cache.reader(tuned)
	if err != nil {
		return TensorDelta{}, err
	}
	readers := [2]safetensors.TensorReader{baseReader, tunedReader}

	var sumAbs float64
	var sumSq float64
	var maxAbs float64
	var dot float64
	var baseNorm float64
	var tunedNorm float64
	// base and tuned chunks are compared element-by-element in the same
	// iteration, so each reader keeps its own raw + values scratch (a
	// shared values buffer would let the tuned read overwrite the base
	// chunk before the diff loop runs). The buffers are owned by the
	// ComparePacks pass (compareScratch) and reused across chunks AND
	// across every tensor — replaces the fresh []byte + []float32 each
	// tensor's ReadFloat32Chunk allocated, the dominant ComparePacks B/op.
	for offset := 0; offset < base.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return TensorDelta{}, err
		}
		count := min(chunkElements, base.Elements-offset)
		var baseValues, tunedValues []float32
		var err error
		scratch.rawBase, scratch.valuesBase, baseValues, err = readers[0].ReadFloat32ChunkInto(offset, count, scratch.rawBase, scratch.valuesBase)
		if err != nil {
			return TensorDelta{}, err
		}
		scratch.rawTuned, scratch.valuesTuned, tunedValues, err = readers[1].ReadFloat32ChunkInto(offset, count, scratch.rawTuned, scratch.valuesTuned)
		if err != nil {
			return TensorDelta{}, err
		}
		for i := range baseValues {
			baseValue := float64(baseValues[i])
			tunedValue := float64(tunedValues[i])
			diff := tunedValue - baseValue
			abs := diff
			if abs < 0 {
				abs = -abs
			}
			sumAbs += abs
			sumSq += diff * diff
			// Inlined max — math.Max is NOT a compiler intrinsic on arm64
			// (it does explicit NaN handling) so it shows up as a function
			// call per element. For our domain (no NaNs reach this point;
			// the safetensors readers reject malformed data upstream) the
			// plain compare is correct and ~3x cheaper per iteration.
			if abs > maxAbs {
				maxAbs = abs
			}
			dot += baseValue * tunedValue
			baseNorm += baseValue * baseValue
			tunedNorm += tunedValue * tunedValue
		}
	}
	delta.MeanAbsDelta = sumAbs / float64(base.Elements)
	delta.RMSDelta = math.Sqrt(sumSq / float64(base.Elements))
	delta.MaxAbsDelta = maxAbs
	delta.L2Delta = math.Sqrt(sumSq)
	delta.Cosine = compareCosine(dot, baseNorm, tunedNorm)
	if maxAbs == 0 {
		delta.Status = CompareStatusUnchanged
	} else {
		delta.Status = CompareStatusChanged
	}
	return delta, nil
}

func recordTensorDelta(result *CompareResult, acc *compareAccumulator, opts CompareOptions, delta TensorDelta) {
	switch delta.Status {
	case CompareStatusChanged:
		result.ComparedTensors++
		result.ChangedTensors++
		acc.elements += delta.Elements
		acc.sumAbs += delta.MeanAbsDelta * float64(delta.Elements)
		acc.sumSq += delta.RMSDelta * delta.RMSDelta * float64(delta.Elements)
		// Inlined max — same reasoning as compareTensorRefs (math.Max is
		// not an intrinsic; the upstream tensor diff scan guarantees
		// finite values).
		if delta.MaxAbsDelta > acc.maxAbs {
			acc.maxAbs = delta.MaxAbsDelta
		}
	case CompareStatusUnchanged:
		result.ComparedTensors++
		result.UnchangedTensors++
		acc.elements += delta.Elements
	case CompareStatusShapeMismatch:
		result.ShapeMismatches++
	case CompareStatusDTypeMismatch:
		result.DTypeMismatches++
	}
	appendTensorDelta(result, opts, delta)
}

func appendTensorDelta(result *CompareResult, opts CompareOptions, delta TensorDelta) {
	if delta.Status == CompareStatusUnchanged && !opts.IncludeUnchanged {
		return
	}
	if opts.MaxTensorReports > 0 && len(result.Tensors) >= opts.MaxTensorReports {
		return
	}
	result.Tensors = append(result.Tensors, delta)
}

func compareCosine(dot, baseNorm, tunedNorm float64) float64 {
	switch {
	case baseNorm == 0 && tunedNorm == 0:
		return 1
	case baseNorm == 0 || tunedNorm == 0:
		return 0
	default:
		return clampFloat64(dot/(math.Sqrt(baseNorm)*math.Sqrt(tunedNorm)), -1, 1)
	}
}

func cloneCompareLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	// core.MapClone — substrate map-copy primitive; cuts the for-range loop
	// to a single call and lets the runtime pick the optimal bulk copy.
	return core.MapClone(labels)
}

func cloneUint64s(values []uint64) []uint64 {
	if len(values) == 0 {
		return nil
	}
	// core.SliceClone — exact-cap clone, no growslice over-allocation.
	return core.SliceClone(values)
}

// dualShapeClone allocates one arena for both base and tuned shape
// clones, returning two sub-slices that share the backing array. Both
// slices have cap == len so any caller-side append would re-alloc;
// since TensorDelta's shape fields are read-only after construction
// this is safe. Saves one alloc per compareTensorRefs call vs two
// separate cloneUint64s.
func dualShapeClone(base, tuned []uint64) ([]uint64, []uint64) {
	bn, tn := len(base), len(tuned)
	if bn == 0 && tn == 0 {
		return nil, nil
	}
	if bn == 0 {
		return nil, core.SliceClone(tuned)
	}
	if tn == 0 {
		return core.SliceClone(base), nil
	}
	arena := make([]uint64, bn+tn)
	copy(arena[:bn], base)
	copy(arena[bn:], tuned)
	return arena[:bn:bn], arena[bn : bn+tn : bn+tn]
}

// shapeArena is a pass-level bump allocator for the per-tensor base +
// tuned shape clones in ComparePacks's matched-tensor path. The clone
// dominated ComparePacks alloc_objects (one make([]uint64) per compared
// tensor); ComparePacks pre-walks the matched tensors, sizes one slab to
// the total dims, and carves cap==len sub-slices from it — N per-tensor
// allocs collapse to one for the whole pass (the buildMergedHeader slab
// pattern). Handouts always have cap==len so a consumer append re-allocs
// rather than corrupting a neighbour, and TensorDelta's shape fields are
// read-only after construction. If the pre-walk under-counts (it should
// not), carve falls back to a fresh make rather than panicking.
type shapeArena struct {
	slab   []uint64
	cursor int
}

// clone returns a cap==len copy of src carved from the arena slab, or a
// fresh allocation when the slab is exhausted (defensive) or src is empty.
func (a *shapeArena) clone(src []uint64) []uint64 {
	n := len(src)
	if n == 0 {
		return nil
	}
	if a == nil || a.cursor+n > len(a.slab) {
		return core.SliceClone(src)
	}
	out := a.slab[a.cursor : a.cursor+n : a.cursor+n]
	copy(out, src)
	a.cursor += n
	return out
}
