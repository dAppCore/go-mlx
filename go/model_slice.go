// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"slices"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

const modelSliceManifestVersion = "go-mlx.model-slice.v1"

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
		return nil, core.NewError("mlx: model slice output path is required")
	}
	if core.Trim(req.Model.Path) == "" {
		return nil, core.NewError("mlx: model slice source path is required")
	}

	source, err := model.Inspect(req.Model.Path)
	if err != nil {
		return nil, err
	}
	if source.Format != mp.ModelPackFormatSafetensors {
		return nil, core.NewError("mlx: model slice materialisation currently supports safetensors packs only")
	}
	if len(source.WeightFiles) == 0 {
		return nil, core.NewError("mlx: model slice source has no safetensors weights")
	}

	index, err := safetensors.IndexFiles(source.WeightFiles)
	if err != nil {
		return nil, err
	}
	refs, names := selectModelSliceTensorRefs(*plan, index)
	if len(refs) == 0 {
		return nil, core.NewError("mlx: model slice selected no tensors")
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
	parsed := core.ParseInt(labels[key], 10, 64)
	if !parsed.OK {
		return 0
	}
	return parsed.Value.(int64)
}

func tensorRefsByteLen(refs []safetensors.TensorRef) int64 {
	var total int64
	for _, ref := range refs {
		total += ref.ByteLen
	}
	return total
}

func indexTensorByteLen(index safetensors.Index) int64 {
	var total int64
	for _, name := range index.Names {
		total += index.Tensors[name].ByteLen
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
	return modelSliceInclusionMask{
		embeddings: plan.HasComponent(inference.ModelComponentEmbeddings),
		norms:      plan.HasComponent(inference.ModelComponentNorms),
		attention:  plan.HasComponent(inference.ModelComponentAttention),
		ffn:        plan.HasComponent(inference.ModelComponentFFN),
		gate:       plan.HasComponent(inference.ModelComponentGate),
		downMeta:   plan.HasComponent(inference.ModelComponentDownMeta),
		router:     plan.HasComponent(inference.ModelComponentRouter),
		experts:    plan.HasComponent(inference.ModelComponentExperts),
		lmHead:     plan.HasComponent(inference.ModelComponentLMHead),
	}
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
// plan.HasComponent's per-call linear scan over plan.Components.
func modelSliceIncludesTensorMask(mask modelSliceInclusionMask, name string) bool {
	if mask.all {
		return true
	}
	lower := core.Lower(name)
	switch {
	case mask.embeddings && modelSliceTensorIsEmbedding(lower):
		return true
	case mask.norms && modelSliceTensorIsNorm(lower):
		return true
	case mask.attention && modelSliceTensorIsAttention(lower):
		return true
	case mask.ffn && modelSliceTensorIsFFN(lower):
		return true
	case mask.gate && modelSliceTensorIsGate(lower):
		return true
	case mask.downMeta && modelSliceTensorIsDownMeta(lower):
		return true
	case mask.router && modelSliceTensorIsRouter(lower):
		return true
	case mask.experts && modelSliceTensorIsExpert(lower):
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
	return core.Contains(name, "embed") || core.Contains(name, ".wte.") || core.HasSuffix(name, ".wte.weight")
}

func modelSliceTensorIsNorm(name string) bool {
	// "layernorm" already contains "norm", so the first check subsumes
	// it — the redundant second core.Contains scan was dead.
	return core.Contains(name, "norm")
}

func modelSliceTensorIsAttention(name string) bool {
	return core.Contains(name, "self_attn") ||
		core.Contains(name, "attention") ||
		core.Contains(name, ".attn.") ||
		modelSliceHasProjection(name, "q_proj") ||
		modelSliceHasProjection(name, "k_proj") ||
		modelSliceHasProjection(name, "v_proj") ||
		modelSliceHasProjection(name, "o_proj") ||
		modelSliceHasProjection(name, "out_proj")
}

func modelSliceTensorIsFFN(name string) bool {
	return core.Contains(name, ".mlp.") ||
		core.Contains(name, "feed_forward") ||
		core.Contains(name, "ffn") ||
		modelSliceHasProjection(name, "up_proj") ||
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

func modelSliceHasProjection(name, projection string) bool {
	if match, ok := projectionLookup[projection]; ok {
		return core.Contains(name, match.infix) || core.HasSuffix(name, match.suffix)
	}
	// Fallback for callers passing unseen projection names — preserves the
	// original "."+projection+"." semantics without the lookup table.
	return core.Contains(name, "."+projection+".") || core.HasSuffix(name, "."+projection+".weight")
}

func modelSliceMetadataFiles(plan inference.ModelSlicePlan) []string {
	files := []string{"config.json"}
	if plan.HasComponent(inference.ModelComponentTokenizer) {
		files = append(files, "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json")
	}
	if plan.HasComponent(inference.ModelComponentLabels) {
		files = append(files, "label_map.json", "labels.json", "id2label.json")
	}
	return files
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
	manifest := modelSliceManifest{
		Version: modelSliceManifestVersion,
		Source:  plan.SourcePath,
		Output:  plan.OutputPath,
		Plan:    plan,
		Weight:  "model.safetensors",
		Tensors: slices.Clone(tensors),
		Labels:  cloneStringMap(plan.Labels),
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
	return core.NewError("mlx: model slice core result failed")
}
