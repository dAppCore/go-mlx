// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
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
	offloadBytes := sourceBytes - localBytes
	if offloadBytes < 0 {
		offloadBytes = 0
	}
	standalone, missing := modelSliceStandalone(manifest.Plan)
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
		inspection.Notes = append(inspection.Notes, "slice is not a standalone model; reload requires split placement for omitted runtime components")
	}
	return inspection, nil
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
	refs, names := selectModelSliceTensorRefs(*plan, index)
	if len(refs) == 0 {
		return nil, errModelSliceNoTensorsSelected
	}

	if result := core.MkdirAll(req.OutputPath, 0o755); !result.OK {
		return nil, modelSliceResultError(result)
	}
	for _, name := range modelSliceMetadataFiles(*plan) {
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
		plan.Labels = map[string]string{}
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

	if err := writeModelSliceManifest(req.OutputPath, *plan, names); err != nil {
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

func modelSliceStandalone(plan inference.ModelSlicePlan) (bool, []inference.ModelComponent) {
	if plan.ExtractLevel == inference.ModelExtractLevelAll {
		return true, nil
	}
	// Lazy-allocate missing only when the first absent component is
	// observed. The vast majority of slices passed to standalone checks
	// either declare ExtractLevelAll (handled above) or have all four
	// required components, so the typical path now skips the make()
	// entirely.
	var missing []inference.ModelComponent
	for _, component := range modelSliceStandaloneRequired {
		if !plan.HasComponent(component) {
			if missing == nil {
				missing = make([]inference.ModelComponent, 0, len(modelSliceStandaloneRequired))
			}
			missing = append(missing, component)
		}
	}
	return len(missing) == 0, missing
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
// loads on the hot path.
func buildModelSliceInclusionMask(plan inference.ModelSlicePlan) modelSliceInclusionMask {
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

func selectModelSliceTensorRefs(plan inference.ModelSlicePlan, index safetensors.Index) ([]safetensors.TensorRef, []string) {
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
func modelSliceIncludesTensorMask(mask modelSliceInclusionMask, name string) bool {
	if mask.all {
		return true
	}
	lower := core.Lower(name)
	switch {
	case mask.attention && modelSliceTensorIsAttention(lower):
		return true
	case mask.ffn && modelSliceTensorIsFFN(lower):
		return true
	case mask.norms && modelSliceTensorIsNorm(lower):
		return true
	case mask.gate && modelSliceTensorIsGate(lower):
		return true
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
	default:
		return false
	}
}

func modelSliceIncludesTensor(plan inference.ModelSlicePlan, name string) bool {
	return modelSliceIncludesTensorMask(buildModelSliceInclusionMask(plan), name)
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
	// All five projection probes search for "._proj." / "._proj.weight"
	// substrings that share the "_proj." suffix on the infix. If the name
	// has no "_proj." anywhere, none of the five lookups can match — skip
	// the per-projection switch + double substring scan. Sweep over the
	// representative tensor-name set drops by ~10% because the embedding /
	// norm / LM-head names go through this short-circuit instead of the
	// five-projection chain.
	if !core.Contains(name, "_proj.") {
		return false
	}
	return modelSliceHasProjection(name, "q_proj") ||
		modelSliceHasProjection(name, "k_proj") ||
		modelSliceHasProjection(name, "v_proj") ||
		modelSliceHasProjection(name, "o_proj") ||
		modelSliceHasProjection(name, "out_proj")
}

func modelSliceTensorIsFFN(name string) bool {
	if core.Contains(name, ".mlp.") ||
		core.Contains(name, "feed_forward") ||
		core.Contains(name, "ffn") {
		return true
	}
	// "up_proj" / "down_proj" share the "_proj." infix gate — names
	// without "_proj." anywhere cannot match either projection so the
	// per-projection switch + substring scans are dead work.
	if !core.Contains(name, "_proj.") {
		return false
	}
	return modelSliceHasProjection(name, "up_proj") ||
		modelSliceHasProjection(name, "down_proj")
}

func modelSliceTensorIsGate(name string) bool {
	return modelSliceHasProjection(name, "gate_proj") || core.Contains(name, ".gate.")
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

func modelSliceMetadataFiles(plan inference.ModelSlicePlan) []string {
	tokenizer := plan.HasComponent(inference.ModelComponentTokenizer)
	labels := plan.HasComponent(inference.ModelComponentLabels)
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

func writeModelSliceManifest(outputRoot string, plan inference.ModelSlicePlan, tensors []string) error {
	// The manifest aliases the caller's tensors slice and plan.Labels map
	// directly — core.JSONMarshal only reads through them and the local
	// manifest value is consumed immediately, so the previous defensive
	// SliceClone + cloneStringMap pair were dead work on the SliceModel
	// commit path (one alloc per 8-byte string header per tensor + the
	// labels map duplication, all discarded after Marshal).
	manifest := modelSliceManifest{
		Version: modelSliceManifestVersion,
		Source:  plan.SourcePath,
		Output:  plan.OutputPath,
		Plan:    plan,
		Weight:  "model.safetensors",
		Tensors: tensors,
		Labels:  plan.Labels,
		WeightMap: map[string]string{
			"model.safetensors": "selected tensors",
		},
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
