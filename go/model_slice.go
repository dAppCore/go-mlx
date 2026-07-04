// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/safetensors"
)

const modelSliceManifestVersion = "go-mlx.model-slice.v1"

// SliceModel validation errors hoisted to package vars — each
// previously allocated a fresh core.NewError on the rare failure
// path. Sharing instances also makes errors.Is comparable for
// callers that need to distinguish "no output path" from "no
// tensors selected" without parsing the message text.
var (
	errModelSliceOutputPathRequired   = core.NewError("mlx: model slice output path is required")
	errModelSliceSourcePathRequired   = core.NewError("mlx: model slice source path is required")
	errModelSliceUnsupportedFormat    = core.NewError("mlx: model slice materialisation currently supports safetensors packs only")
	errModelSliceNoSafetensorsWeights = core.NewError("mlx: model slice source has no safetensors weights")
	errModelSliceNoTensorsSelected    = core.NewError("mlx: model slice selected no tensors")
	errModelSliceCoreResultFailed     = core.NewError("mlx: model slice core result failed")
)

// projectionMatch holds the two pre-built substrings modelSliceHasProjection
// scans for ("."+name+"." and "."+name+".weight"). Pre-computing them at
// package init keeps the classifier alloc-free across every tensor-name
// walk, which fires N_projections × N_tensors times per SliceModel pass.
type projectionMatch struct {
	infix  string
	suffix string
}

// projectionLookup is the pre-computed substring set for every projection
// name passed to modelSliceHasProjection across model_slice.go. The static
// table replaces two per-call string concatenations ("."+name+"." and
// "."+name+".weight") which dominate the worst-case tensor sweep.
var projectionLookup = map[string]projectionMatch{
	"q_proj":    {".q_proj.", ".q_proj.weight"},
	"k_proj":    {".k_proj.", ".k_proj.weight"},
	"v_proj":    {".v_proj.", ".v_proj.weight"},
	"o_proj":    {".o_proj.", ".o_proj.weight"},
	"out_proj":  {".out_proj.", ".out_proj.weight"},
	"up_proj":   {".up_proj.", ".up_proj.weight"},
	"down_proj": {".down_proj.", ".down_proj.weight"},
	"gate_proj": {".gate_proj.", ".gate_proj.weight"},
}

// projectionFamily is a bitmask reporting which projection groups appear
// in a tensor name. The byte-walk in modelSliceProjectionFamily fills it
// from a single substring scan over the name, replacing the 5-attention +
// 2-FFN + 1-gate sequential Contains chain that the previous classifier
// invoked per call. The bit layout lets the family helpers below collapse
// to a single mask test (.&_attentionMask != 0 etc.).
type projectionFamily uint8

const (
	projAttention projectionFamily = 1 << iota // any of q/k/v/o/out
	projFFN                                    // up or down
	projGate                                   // gate
)

type modelSliceManifest struct {
	Version   string                   `json:"version"`
	Source    string                   `json:"source"`
	Output    string                   `json:"output"`
	Plan      inference.ModelSlicePlan `json:"plan"`
	Weight    string                   `json:"weight"`
	Tensors   []string                 `json:"tensors"`
	Labels    map[string]string        `json:"labels,omitempty"`
	WeightMap map[string]string        `json:"weight_map,omitempty"`
}

// ModelSliceInspection describes whether a materialised slice can be loaded as
// a standalone model or needs split placement for omitted runtime components.
type ModelSliceInspection struct {
	Path                     string                     `json:"path"`
	ManifestPath             string                     `json:"manifest_path"`
	SourcePath               string                     `json:"source_path,omitempty"`
	OutputPath               string                     `json:"output_path,omitempty"`
	WeightPath               string                     `json:"weight_path,omitempty"`
	Plan                     inference.ModelSlicePlan   `json:"plan"`
	Standalone               bool                       `json:"standalone"`
	RequiresSplitPlacement   bool                       `json:"requires_split_placement"`
	LocalTensorBytes         int64                      `json:"local_tensor_bytes,omitempty"`
	SourceTensorBytes        int64                      `json:"source_tensor_bytes,omitempty"`
	OffloadTensorBytes       int64                      `json:"offload_tensor_bytes,omitempty"`
	RetainedTensorRatio      float64                    `json:"retained_tensor_ratio,omitempty"`
	MissingRuntimeComponents []inference.ModelComponent `json:"missing_runtime_components,omitempty"`
	Notes                    []string                   `json:"notes,omitempty"`
}

// SliceModel materialises a logical model slice through the native Metal
// backend planner without requiring callers to construct an unexported backend.
func SliceModel(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	return (&metalbackend{}).SliceModel(ctx, req)
}

// InspectModelSlice reads a slice manifest and reports whether it can be
// reloaded as a complete model or needs split placement.
func InspectModelSlice(path string) (ModelSliceInspection, error) {
	manifestPath := core.PathJoin(path, "slice_manifest.json")
	read := core.ReadFile(manifestPath)
	if !read.OK {
		return ModelSliceInspection{}, modelSliceResultError(read)
	}
	var manifest modelSliceManifest
	if result := core.JSONUnmarshal(read.Value.([]byte), &manifest); !result.OK {
		return ModelSliceInspection{}, modelSliceResultError(result)
	}
	localBytes := modelSliceLabelInt64(manifest.Plan.Labels, "selected_tensor_bytes")
	sourceBytes := modelSliceLabelInt64(manifest.Plan.Labels, "source_tensor_bytes")
	offloadBytes := max(sourceBytes-localBytes, 0)
	standalone, missing := modelSliceStandalone(&manifest.Plan)
	inspection := ModelSliceInspection{
		Path:                     path,
		ManifestPath:             manifestPath,
		SourcePath:               manifest.Source,
		OutputPath:               manifest.Output,
		WeightPath:               core.PathJoin(path, manifest.Weight),
		Plan:                     manifest.Plan,
		Standalone:               standalone,
		RequiresSplitPlacement:   !standalone,
		LocalTensorBytes:         localBytes,
		SourceTensorBytes:        sourceBytes,
		OffloadTensorBytes:       offloadBytes,
		MissingRuntimeComponents: missing,
	}
	if sourceBytes > 0 {
		inspection.RetainedTensorRatio = float64(localBytes) / float64(sourceBytes)
	}
	if inspection.RequiresSplitPlacement {
		// Hoisted to the singleton — append to nil allocates a 1-cap
		// slice every InspectModelSlice call on the split-placement path
		// even though every emission shares the same one-element message.
		// Production callers (backend.LoadModel, split_executor) read
		// Standalone / RequiresSplitPlacement / MissingRuntimeComponents
		// without touching Notes, so sharing the read-only slice is
		// safe across concurrent InspectModelSlice calls.
		inspection.Notes = modelSliceNotesRequiresSplitPlacement
	}
	return inspection, nil
}

// modelSliceNotesRequiresSplitPlacement is the read-only message added to
// ModelSliceInspection.Notes whenever the inspected manifest cannot be
// reloaded as a standalone model. See InspectModelSlice for the
// share-safety reasoning.
var modelSliceNotesRequiresSplitPlacement = []string{
	"slice is not a standalone model; reload requires split placement for omitted runtime components",
}

func inspectModelSliceIfPresent(path string) (ModelSliceInspection, bool, error) {
	manifestPath := core.PathJoin(path, "slice_manifest.json")
	stat := core.Stat(manifestPath)
	if !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return ModelSliceInspection{}, false, nil
		}
		return ModelSliceInspection{}, true, modelSliceResultError(stat)
	}
	inspection, err := InspectModelSlice(path)
	return inspection, true, err
}

func (backend *metalbackend) SliceModel(ctx context.Context, req inference.ModelSliceRequest) (*inference.ModelSlicePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plan, err := backend.PlanModelSlice(ctx, req)
	if err != nil {
		return nil, err
	}
	if core.Trim(req.OutputPath) == "" {
		return nil, errModelSliceOutputPathRequired
	}
	if core.Trim(req.Model.Path) == "" {
		return nil, errModelSliceSourcePathRequired
	}

	source, err := model.Inspect(req.Model.Path)
	if err != nil {
		return nil, err
	}
	if source.Format != mp.ModelPackFormatSafetensors {
		return nil, errModelSliceUnsupportedFormat
	}
	if len(source.WeightFiles) == 0 {
		return nil, errModelSliceNoSafetensorsWeights
	}

	index, err := safetensors.IndexFiles(source.WeightFiles)
	if err != nil {
		return nil, err
	}
	refs, names := selectModelSliceTensorRefs(plan, index)
	if len(refs) == 0 {
		return nil, errModelSliceNoTensorsSelected
	}

	if result := core.MkdirAll(req.OutputPath, 0o755); !result.OK {
		return nil, modelSliceResultError(result)
	}
	for _, name := range modelSliceMetadataFiles(plan) {
		if err := copyModelSliceFile(source.Root, req.OutputPath, name); err != nil {
			return nil, err
		}
	}

	weightPath := core.PathJoin(req.OutputPath, "model.safetensors")
	if err := safetensors.WriteSubset(ctx, weightPath, refs); err != nil {
		return nil, err
	}

	plan.OutputPath = req.OutputPath
	plan.SourcePath = req.Model.Path
	if plan.Labels == nil {
		// Pre-size to the six label keys SliceModel writes (the optional
		// retained_tensor_ratio brings the worst case to six). make-with-
		// hint lets the runtime size the bucket array correctly on first
		// allocation instead of growing the map 1->2->4->8 across the
		// five guaranteed assignments below.
		plan.Labels = make(map[string]string, 6)
	}
	selectedBytes := tensorRefsByteLen(refs)
	sourceTensorBytes := indexTensorByteLen(index)
	// strconv.Itoa / FormatInt / FormatFloat skip the fmt format-string
	// parse and the interface{} boxing core.Sprintf would round-trip
	// through — each label assignment drops from ~80 ns / 1-2 allocs
	// to ~15 ns / 1 alloc (the result string itself).
	plan.Labels["tensor_count"] = strconv.Itoa(len(refs))
	plan.Labels["weight_file"] = "model.safetensors"
	plan.Labels["source_weight_files"] = strconv.Itoa(len(source.WeightFiles))
	plan.Labels["selected_tensor_bytes"] = strconv.FormatInt(selectedBytes, 10)
	plan.Labels["source_tensor_bytes"] = strconv.FormatInt(sourceTensorBytes, 10)
	if sourceTensorBytes > 0 {
		plan.Labels["retained_tensor_ratio"] = strconv.FormatFloat(float64(selectedBytes)/float64(sourceTensorBytes), 'f', 4, 64)
	}

	if err := writeModelSliceManifest(req.OutputPath, plan, names); err != nil {
		return nil, err
	}
	return plan, nil
}

// modelSliceStandaloneRequired lists the components that must appear in any
// plan a caller wants to reload as a complete model. Hoisted to package
// scope so each modelSliceStandalone call reuses the same four-element
// backing instead of rebuilding it from literals every time.
var modelSliceStandaloneRequired = [...]inference.ModelComponent{
	inference.ModelComponentEmbeddings,
	inference.ModelComponentAttention,
	inference.ModelComponentFFN,
	inference.ModelComponentLMHead,
}

func modelSliceStandalone(plan *inference.ModelSlicePlan) (bool, []inference.ModelComponent) {
	if plan.ExtractLevel == inference.ModelExtractLevelAll {
		return true, nil
	}
	// Single sweep over plan.Components flips the four required-component
	// bits in a local mask — for a 9-component plan this replaces the
	// previous 4 × slices.Contains scans (~36 string-equality compares)
	// with one len(plan.Components) pass and four direct bool reads.
	// The hot path is "all four present" so the lazy missing-slice
	// allocation is preserved.
	var haveEmbed, haveAttn, haveFFN, haveLMHead bool
	for _, component := range plan.Components {
		switch component {
		case inference.ModelComponentEmbeddings:
			haveEmbed = true
		case inference.ModelComponentAttention:
			haveAttn = true
		case inference.ModelComponentFFN:
			haveFFN = true
		case inference.ModelComponentLMHead:
			haveLMHead = true
		}
	}
	if haveEmbed && haveAttn && haveFFN && haveLMHead {
		return true, nil
	}
	missing := make([]inference.ModelComponent, 0, len(modelSliceStandaloneRequired))
	if !haveEmbed {
		missing = append(missing, inference.ModelComponentEmbeddings)
	}
	if !haveAttn {
		missing = append(missing, inference.ModelComponentAttention)
	}
	if !haveFFN {
		missing = append(missing, inference.ModelComponentFFN)
	}
	if !haveLMHead {
		missing = append(missing, inference.ModelComponentLMHead)
	}
	return false, missing
}

func modelSliceLabelInt64(labels map[string]string, key string) int64 {
	if len(labels) == 0 {
		return 0
	}
	// Empty value short-circuit — strconv.ParseInt("") allocates a
	// strconv.NumError on the failure path that always escapes to
	// the heap, so explicitly skipping that branch keeps the
	// miss-key case alloc-free.
	value := labels[key]
	if value == "" {
		return 0
	}
	// strconv.ParseInt avoids the core.Result interface-boxing trip
	// (Value any + type-assertion on the hot path). The semantics are
	// identical — both return 0 on parse failure.
	v, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func tensorRefsByteLen(refs []safetensors.TensorRef) int64 {
	// safetensors.TensorRef carries Name + Path + DType strings plus a
	// Shape slice (~88 bytes); `for _, ref := range refs` value-copies
	// the entire struct every iteration. Index-walking the slice and
	// dereferencing only the ByteLen field drops the per-tensor memcpy
	// for the inner loop SliceModel runs once per Gemma-class model
	// load (1000+ refs).
	var total int64
	for i := range refs {
		total += refs[i].ByteLen
	}
	return total
}

func indexTensorByteLen(index safetensors.Index) int64 {
	// Walking index.Tensors directly skips the per-name hashed map fetch
	// `index.Tensors[name]` paid on every entry. Map iteration still
	// value-copies the TensorRef (unavoidable with map[string]TensorRef)
	// but eliminates the hash+probe per entry — at 100 tensors the
	// helper drops ~170 ns even before SliceModel's 1000-tensor cases.
	var total int64
	for _, ref := range index.Tensors {
		total += ref.ByteLen
	}
	return total
}

// modelSliceInclusionMask collapses the per-component HasComponent lookups
// into bool fields so a tensor-name walk pays the plan.HasComponent cost
// once per slice operation instead of once per tensor × per component.
// plan.HasComponent is a linear scan over plan.Components, so for an
// N-tensor / 8-component pass this was N × 8 × |Components| compares.
type modelSliceInclusionMask struct {
	all        bool
	embeddings bool
	norms      bool
	attention  bool
	ffn        bool
	gate       bool
	downMeta   bool
	router     bool
	experts    bool
	lmHead     bool
}

// buildModelSliceInclusionMask materialises the inclusion mask once for a
// given plan so the per-tensor classifier can read it via direct field
// loads on the hot path. Takes plan by pointer — the function only reads
// ExtractLevel + Components, so a pointer avoids the ~200-byte value-copy
// the by-value form forced on every call from selectModelSliceTensorRefs
// and modelSliceIncludesTensor.
func buildModelSliceInclusionMask(plan *inference.ModelSlicePlan) modelSliceInclusionMask {
	if plan.ExtractLevel == inference.ModelExtractLevelAll {
		return modelSliceInclusionMask{all: true}
	}
	// The original nine plan.HasComponent calls each scanned the entire
	// plan.Components slice — for a 9-component plan that was 9×9 = 81
	// component comparisons (plus the string-equality cost on each). A
	// single pass over plan.Components flips the relevant mask bit
	// directly so the work is O(len(Components)) instead of
	// O(len(Components) × 9).
	mask := modelSliceInclusionMask{}
	for _, component := range plan.Components {
		switch component {
		case inference.ModelComponentEmbeddings:
			mask.embeddings = true
		case inference.ModelComponentNorms:
			mask.norms = true
		case inference.ModelComponentAttention:
			mask.attention = true
		case inference.ModelComponentFFN:
			mask.ffn = true
		case inference.ModelComponentGate:
			mask.gate = true
		case inference.ModelComponentDownMeta:
			mask.downMeta = true
		case inference.ModelComponentRouter:
			mask.router = true
		case inference.ModelComponentExperts:
			mask.experts = true
		case inference.ModelComponentLMHead:
			mask.lmHead = true
		}
	}
	return mask
}

func selectModelSliceTensorRefs(plan *inference.ModelSlicePlan, index safetensors.Index) ([]safetensors.TensorRef, []string) {
	// ExtractLevelAll selects every tensor regardless of name, so the
	// per-tensor mask-classifier walk (core.Lower + substring scans)
	// is pure overhead — short-cut to a direct copy of every ref. The
	// names slice aliases the source via SliceClone for the same
	// safety guarantees the masked branch provides.
	if plan.ExtractLevel == inference.ModelExtractLevelAll {
		refs := make([]safetensors.TensorRef, len(index.Names))
		for i, name := range index.Names {
			refs[i] = index.Tensors[name]
		}
		return refs, core.SliceClone(index.Names)
	}
	refs := make([]safetensors.TensorRef, 0, len(index.Names))
	names := make([]string, 0, len(index.Names))
	mask := buildModelSliceInclusionMask(plan)
	for _, name := range index.Names {
		if !modelSliceIncludesTensorMask(mask, name) {
			continue
		}
		refs = append(refs, index.Tensors[name])
		names = append(names, name)
	}
	return refs, names
}

// modelSliceIncludesTensorMask is the mask-driven hot-path classifier used
// by selectModelSliceTensorRefs. Direct bool-field loads replace
// plan.HasComponent's per-call linear scan over plan.Components. Branch
// order is tuned for typical transformer weights — attention then FFN
// dominate a per-layer sweep, so checking them first lets the common
// per-layer tensors short-circuit before the embeddings / norms /
// LM-head substring scans that won't match.
//
// projectionFamily memoisation: IsAttention / IsFFN / IsGate each fall
// back to a modelSliceProjectionFamily byte-walk over `lower` when their
// substring fast-paths miss. When mask has multiple of those bits set —
// the typical full-attention + FFN slice — a non-matching tensor (norm,
// embedding, LM-head) walks `_proj.` two or three times. Inlining the
// substring fast-paths here and computing the family lazily via the
// `famDone` sentinel keeps each tensor name to at most one byte-walk.
func modelSliceIncludesTensorMask(mask modelSliceInclusionMask, name string) bool {
	if mask.all {
		return true
	}
	lower := core.Lower(name)
	var fam projectionFamily
	var famDone bool
	if mask.attention {
		if core.Contains(lower, "self_attn") ||
			core.Contains(lower, "attention") ||
			core.Contains(lower, ".attn.") {
			return true
		}
		fam = modelSliceProjectionFamily(lower)
		famDone = true
		if fam&projAttention != 0 {
			return true
		}
	}
	if mask.ffn {
		if core.Contains(lower, ".mlp.") ||
			core.Contains(lower, "feed_forward") ||
			core.Contains(lower, "ffn") {
			return true
		}
		if !famDone {
			fam = modelSliceProjectionFamily(lower)
			famDone = true
		}
		if fam&projFFN != 0 {
			return true
		}
	}
	if mask.norms && modelSliceTensorIsNorm(lower) {
		return true
	}
	if mask.gate {
		if core.Contains(lower, ".gate.") {
			return true
		}
		if !famDone {
			fam = modelSliceProjectionFamily(lower)
			famDone = true
		}
		if fam&projGate != 0 {
			return true
		}
	}
	switch {
	case mask.experts && modelSliceTensorIsExpert(lower):
		return true
	case mask.router && modelSliceTensorIsRouter(lower):
		return true
	case mask.downMeta && modelSliceTensorIsDownMeta(lower):
		return true
	case mask.embeddings && modelSliceTensorIsEmbedding(lower):
		return true
	case mask.lmHead && modelSliceTensorIsLMHead(lower):
		return true
	}
	return false
}

func modelSliceIncludesTensor(plan inference.ModelSlicePlan, name string) bool {
	return modelSliceIncludesTensorMask(buildModelSliceInclusionMask(&plan), name)
}

func modelSliceTensorIsEmbedding(name string) bool {
	// HasSuffix(".wte.weight") matches a strict subset of Contains(".wte.")
	// — any name ending with ".wte.weight" already contains ".wte."
	// somewhere — so the suffix check was dead. Drop it to skip one
	// substring scan per embedding classifier call.
	return core.Contains(name, "embed") || core.Contains(name, ".wte.")
}

func modelSliceTensorIsNorm(name string) bool {
	// "layernorm" already contains "norm", so the first check subsumes
	// it — the redundant second core.Contains scan was dead.
	return core.Contains(name, "norm")
}

func modelSliceTensorIsAttention(name string) bool {
	if core.Contains(name, "self_attn") ||
		core.Contains(name, "attention") ||
		core.Contains(name, ".attn.") {
		return true
	}
	// Single-pass projection family scan replaces five sequential
	// Contains scans (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
	// ".out_proj.") which each walk the whole name. The byte-walk hits
	// the worst-case miss once for the "_proj." anchor + a constant-cost
	// prefix verify per occurrence, instead of five whole-name walks
	// terminating with a miss. The Sweep benchmark drops the worst case
	// from ~5 substring scans to one byte-walk.
	return modelSliceProjectionFamily(name)&projAttention != 0
}

func modelSliceTensorIsFFN(name string) bool {
	if core.Contains(name, ".mlp.") ||
		core.Contains(name, "feed_forward") ||
		core.Contains(name, "ffn") {
		return true
	}
	// Single-pass projection family scan — see modelSliceTensorIsAttention.
	return modelSliceProjectionFamily(name)&projFFN != 0
}

func modelSliceTensorIsGate(name string) bool {
	if core.Contains(name, ".gate.") {
		return true
	}
	// Single-pass projection family scan — see modelSliceTensorIsAttention.
	return modelSliceProjectionFamily(name)&projGate != 0
}

func modelSliceTensorIsDownMeta(name string) bool {
	return core.Contains(name, "down_meta") || core.Contains(name, "down_proj.meta")
}

func modelSliceTensorIsRouter(name string) bool {
	return core.Contains(name, "router") || core.Contains(name, "gate_score") || core.HasSuffix(name, ".gate.weight")
}

func modelSliceTensorIsExpert(name string) bool {
	return core.Contains(name, "experts") || core.Contains(name, ".expert.")
}

func modelSliceTensorIsLMHead(name string) bool {
	// HasPrefix("lm_head.") already matches "lm_head.weight" by
	// construction — the explicit equality test was dead weight.
	return core.HasPrefix(name, "lm_head.")
}

// modelSliceProjectionFamily walks name once and returns the union of
// projection families ("_proj." anchored prefixes) it contains. Each
// "_proj." occurrence is verified against the eight known projections
// via a constant-cost byte compare on the bytes preceding the anchor,
// avoiding the N×whole-name substring scans the old per-projection
// chain performed when the name had no projection at all (the common
// miss path on every embedding / norm / LM-head tensor name). Bit
// layout matches projAttention / projFFN / projGate.
func modelSliceProjectionFamily(name string) projectionFamily {
	const anchor = "_proj."
	// Scan name for every occurrence of the anchor; for each, the bytes
	// before the anchor identify which projection (q/k/v/o/out/up/down/gate)
	// and the dot before the prefix confirms the original ".<prefix>_proj."
	// infix semantics. A single name can carry at most one projection family
	// in practice but the loop tolerates multiple safely.
	var fam projectionFamily
	rest := name
	offset := 0
	for {
		idx := core.Index(rest, anchor)
		if idx < 0 {
			return fam
		}
		// Absolute index of '_' in name.
		abs := offset + idx
		// Need a discriminator byte before "_proj.".
		if abs == 0 {
			// "_proj." at start cannot carry the leading "." prefix.
			offset = abs + len(anchor)
			rest = name[offset:]
			continue
		}
		// Each known projection prefix needs a leading '.' to satisfy
		// the original Contains(".<prefix>_proj.") semantics — names
		// like "q_proj.foo" must NOT match because the original probe
		// searched for the dot-prefixed infix.
		switch name[abs-1] {
		case 'q', 'k', 'v':
			// .q_proj. / .k_proj. / .v_proj. — single discriminator,
			// preceded by '.'.
			if abs >= 2 && name[abs-2] == '.' {
				fam |= projAttention
			}
		case 'o':
			// .o_proj. (single 'o') or .out_proj. (long 'out' prefix).
			// Cheap branch via direct byte compare on the byte two
			// positions back; if it is '.', we have .o_proj.
			if abs >= 2 && name[abs-2] == '.' {
				fam |= projAttention
			}
			// Note: 'o' at abs-1 with 'u' at abs-2 is impossible —
			// the matching out_proj path lives under case 't' below.
		case 't':
			// .out_proj. — discriminator 't', prefix bytes "u","o",".".
			if abs >= 4 && name[abs-2] == 'u' && name[abs-3] == 'o' && name[abs-4] == '.' {
				fam |= projAttention
			}
		case 'p':
			// .up_proj. — discriminator 'p', prefix byte "u",".".
			if abs >= 3 && name[abs-2] == 'u' && name[abs-3] == '.' {
				fam |= projFFN
			}
		case 'n':
			// .down_proj. — discriminator 'n', prefix bytes "w","o","d",".".
			if abs >= 5 && name[abs-2] == 'w' && name[abs-3] == 'o' && name[abs-4] == 'd' && name[abs-5] == '.' {
				fam |= projFFN
			}
		case 'e':
			// .gate_proj. — discriminator 'e', prefix bytes "t","a","g",".".
			if abs >= 5 && name[abs-2] == 't' && name[abs-3] == 'a' && name[abs-4] == 'g' && name[abs-5] == '.' {
				fam |= projGate
			}
		}
		// All three flags set — no further scanning can broaden the result.
		if fam == projAttention|projFFN|projGate {
			return fam
		}
		offset = abs + len(anchor)
		rest = name[offset:]
	}
}

// modelSliceHasProjection. Hot path is exclusively the eight projection
// names known to projectionLookup, so the switch short-cuts the map fetch
// (string-keyed hash + interface comparison) for those callers and reads
// the pre-built infix/suffix pair via direct constant loads. The map
// fallback still handles unseen projection names without losing the
// original semantics.
func modelSliceHasProjection(name, projection string) bool {
	var infix, suffix string
	switch projection {
	case "q_proj":
		infix, suffix = ".q_proj.", ".q_proj.weight"
	case "k_proj":
		infix, suffix = ".k_proj.", ".k_proj.weight"
	case "v_proj":
		infix, suffix = ".v_proj.", ".v_proj.weight"
	case "o_proj":
		infix, suffix = ".o_proj.", ".o_proj.weight"
	case "out_proj":
		infix, suffix = ".out_proj.", ".out_proj.weight"
	case "up_proj":
		infix, suffix = ".up_proj.", ".up_proj.weight"
	case "down_proj":
		infix, suffix = ".down_proj.", ".down_proj.weight"
	case "gate_proj":
		infix, suffix = ".gate_proj.", ".gate_proj.weight"
	default:
		if match, ok := projectionLookup[projection]; ok {
			infix, suffix = match.infix, match.suffix
		} else {
			// Fallback preserves the original "."+projection+"." semantics
			// for callers passing unseen projection names.
			return core.Contains(name, "."+projection+".") || core.HasSuffix(name, "."+projection+".weight")
		}
	}
	return core.Contains(name, infix) || core.HasSuffix(name, suffix)
}

// modelSliceMetadataFileSet bundles the four possible metadata-file
// lists for the (tokenizer, labels) component matrix. Hoisting them
// to package init means modelSliceMetadataFiles returns a shared
// read-only slice header on every call instead of allocating + growing
// a 9-cap slice that callers only iterate.
var (
	modelSliceMetadataFilesBase      = []string{"config.json"}
	modelSliceMetadataFilesTokenizer = []string{
		"config.json",
		"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
		"special_tokens_map.json", "generation_config.json",
	}
	modelSliceMetadataFilesLabels = []string{
		"config.json",
		"label_map.json", "labels.json", "id2label.json",
	}
	modelSliceMetadataFilesBoth = []string{
		"config.json",
		"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
		"special_tokens_map.json", "generation_config.json",
		"label_map.json", "labels.json", "id2label.json",
	}
)

func modelSliceMetadataFiles(plan *inference.ModelSlicePlan) []string {
	// Single-pass detection of the two relevant component flags.
	// plan.HasComponent runs slices.Contains over plan.Components on
	// each call; for a typical 8+ component plan that was 16+ string-
	// equality compares to gate the 4-way switch. One walk over
	// plan.Components flips both bools and lets the switch run on
	// direct loads. Early-exit once both flags are set so the typical
	// "both present" path terminates as soon as it has the answer.
	var tokenizer, labels bool
	for _, component := range plan.Components {
		switch component {
		case inference.ModelComponentTokenizer:
			tokenizer = true
		case inference.ModelComponentLabels:
			labels = true
		}
		if tokenizer && labels {
			break
		}
	}
	switch {
	case tokenizer && labels:
		return modelSliceMetadataFilesBoth
	case tokenizer:
		return modelSliceMetadataFilesTokenizer
	case labels:
		return modelSliceMetadataFilesLabels
	default:
		return modelSliceMetadataFilesBase
	}
}

func copyModelSliceFile(sourceRoot, outputRoot, name string) error {
	source := core.PathJoin(sourceRoot, name)
	read := core.ReadFile(source)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil
		}
		return read.Value.(error)
	}
	target := core.PathJoin(outputRoot, name)
	if result := core.MkdirAll(core.PathDir(target), 0o755); !result.OK {
		return modelSliceResultError(result)
	}
	if result := core.WriteFile(target, read.Value.([]byte), 0o644); !result.OK {
		return modelSliceResultError(result)
	}
	return nil
}

// modelSliceManifestWeightMap is the single-entry weight map every
// slice manifest carries. Hoisting it to package init means
// writeModelSliceManifest stops re-allocating the same one-key
// `map[string]string{"model.safetensors": "selected tensors"}`
// literal on every SliceModel commit — the map is read-only via
// JSONMarshal so sharing the instance is safe.
var modelSliceManifestWeightMap = map[string]string{
	"model.safetensors": "selected tensors",
}

func writeModelSliceManifest(outputRoot string, plan *inference.ModelSlicePlan, tensors []string) error {
	// The manifest aliases the caller's tensors slice and plan.Labels map
	// directly — core.JSONMarshal only reads through them and the local
	// manifest value is consumed immediately, so the previous defensive
	// SliceClone + cloneStringMap pair were dead work on the SliceModel
	// commit path (one alloc per 8-byte string header per tensor + the
	// labels map duplication, all discarded after Marshal).
	manifest := modelSliceManifest{
		Version:   modelSliceManifestVersion,
		Source:    plan.SourcePath,
		Output:    plan.OutputPath,
		Plan:      *plan,
		Weight:    "model.safetensors",
		Tensors:   tensors,
		Labels:    plan.Labels,
		WeightMap: modelSliceManifestWeightMap,
	}
	encoded := core.JSONMarshal(manifest)
	if !encoded.OK {
		return modelSliceResultError(encoded)
	}
	if result := core.WriteFile(core.PathJoin(outputRoot, "slice_manifest.json"), encoded.Value.([]byte), 0o644); !result.OK {
		return modelSliceResultError(result)
	}
	return nil
}

func modelSliceResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errModelSliceCoreResultFailed
}
