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

	// Pre-size both the result.Tensors slice and the tunedSeen tracker:
	// they each grow to at most len(baseIndex.Names) entries (every base
	// tensor either appears in tuned or not). Growing through the default
	// nil/zero-cap path costs N growslice/maphint walks for large N.
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
	tunedSeen := make(map[string]struct{}, len(baseIndex.Names))
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
		tunedSeen[name] = struct{}{}
		delta, err := compareTensorRefs(ctx, baseRef, tunedRef, modelMergeTensorChunkElements)
		if err != nil {
			return nil, core.E("ComparePacks", "compare tensor "+name, err)
		}
		recordTensorDelta(result, &acc, opts, delta)
	}
	for _, name := range tunedIndex.Names {
		if _, ok := tunedSeen[name]; ok {
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

func compareTensorRefs(ctx context.Context, base, tuned safetensors.TensorRef, chunkElements int) (TensorDelta, error) {
	// Single arena for the base + tuned shape clones — replaces the two
	// cloneUint64s allocations with one when both shapes are non-empty.
	// TensorDelta carries the BaseShape and FineTunedShape fields as
	// independent sub-slices sharing the arena's backing array; consumers
	// never mutate either, so aliasing is safe.
	shapeMatch := sameUint64Slice(base.Shape, tuned.Shape) && base.Elements == tuned.Elements
	baseShapeClone, tunedShapeClone := dualShapeClone(base.Shape, tuned.Shape)
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
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{base, tuned})
	if err != nil {
		return TensorDelta{}, err
	}
	defer safetensors.CloseReaders(readers)

	var sumAbs float64
	var sumSq float64
	var maxAbs float64
	var dot float64
	var baseNorm float64
	var tunedNorm float64
	for offset := 0; offset < base.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return TensorDelta{}, err
		}
		count := min(chunkElements, base.Elements-offset)
		baseValues, err := readers[0].ReadFloat32Chunk(offset, count)
		if err != nil {
			return TensorDelta{}, err
		}
		tunedValues, err := readers[1].ReadFloat32Chunk(offset, count)
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
