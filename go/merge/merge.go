// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// Method names the tensor merge algorithm.
type Method string

const (
	MethodLinear Method = "linear"
	MethodSLERP  Method = "slerp"
	MethodTIES   Method = "ties"
	MethodDARE   Method = "dare"

	ProvenanceFile                = "model_merge_provenance.json"
	modelMergeOutputWeights       = "model.safetensors"
	modelMergeTensorChunkElements = 1 << 20
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Sharing instances also makes errors.Is comparable for
// callers distinguishing "no tensors" from "len mismatch" without
// parsing message text.
var (
	errSLERPLenMismatch        = core.NewError("mlx: tensor length mismatch during SLERP merge")
	errSLERPNeedTwoTensors     = core.NewError("mlx: SLERP tensor merge requires exactly two tensors")
	errLinearLenMismatch       = core.NewError("mlx: tensor length mismatch during linear merge")
	errNoTensors               = core.NewError("mlx: no tensors to merge")
	errOutputHasWeights        = core.NewError("mlx: merged output path already contains model weights")
	errPackMetadataCopy        = core.NewError("model pack metadata copy failed")
	errWeightsSourceCount      = core.NewError("mlx: tensor merge weights do not match source count")
	errSLERPNeedTwoReaders     = core.NewError("mlx: SLERP tensor merge requires exactly two readers")
	errSLERPNeedTwoSources     = core.NewError("mlx: SLERP model merge requires exactly two sources")
	errTokenizerMismatch       = core.NewError("mlx: model merge tokenizer mismatch")
	errMergeTOutOfRange        = core.NewError("mlx: model merge t must be between 0 and 1")
	errMergeWeightsSumZero     = core.NewError("mlx: model merge source weights sum to zero")
	errMergeWeightNotFinite    = core.NewError("mlx: model merge source weight must be finite")
	errMergeSourcePackRequired = core.NewError("mlx: model merge source pack is required")
	errMergeNeedTwoSources     = core.NewError("mlx: model merge requires at least two sources")
	errMergeNeedsSafetensors   = core.NewError("mlx: model merge currently requires safetensors source weights")
	errOutputSameAsSource      = core.NewError("mlx: merged output path must differ from source model path")
	errOutputNotPackDir        = core.NewError("mlx: merged output path must be a model-pack directory")
	errOutputPathRequired      = core.NewError("mlx: merged model output path is required")
	errReadNonByteData         = core.NewError("merge: read file returned non-byte data")
	errCoreResultFailed        = core.NewError("core result failed")
)

// Source identifies a pre-validated model pack participating in a merge.
// Callers run mlx.ValidateModelPack on each source before invoking merge.Packs.
type Source struct {
	Pack   mp.ModelPack `json:"pack"`
	Weight float64      `json:"weight,omitempty"`
}

// Options configures local model-pack tensor merging.
type Options struct {
	Sources                   []Source          `json:"sources"`
	OutputPath                string            `json:"output_path"`
	Method                    Method            `json:"method,omitempty"`
	T                         float64           `json:"t,omitempty"`
	AllowArchitectureMismatch bool              `json:"allow_architecture_mismatch,omitempty"`
	AllowTokenizerMismatch    bool              `json:"allow_tokenizer_mismatch,omitempty"`
	AllowTensorMismatch       bool              `json:"allow_tensor_mismatch,omitempty"`
	Labels                    map[string]string `json:"labels,omitempty"`
}

// Result reports the paths of the generated merged model pack and its
// per-tensor counts. Callers re-validate via mlx.ValidateModelPack(OutputPath)
// when they need a populated pack.ModelPack.
type Result struct {
	OutputPath     string         `json:"output_path"`
	WeightPath     string         `json:"weight_path"`
	ProvenancePath string         `json:"provenance_path"`
	Method         Method         `json:"method"`
	T              float64        `json:"t,omitempty"`
	Sources        []mp.ModelPack `json:"sources"`
	TensorCount    int            `json:"tensor_count"`
	MergedTensors  int            `json:"merged_tensors"`
	CopiedTensors  int            `json:"copied_tensors,omitempty"`
	SkippedTensors []string       `json:"skipped_tensors,omitempty"`
}

// Provenance records how a merged pack was produced.
type Provenance struct {
	Version        int               `json:"version"`
	Method         Method            `json:"method"`
	T              float64           `json:"t,omitempty"`
	Sources        []Source          `json:"sources"`
	SourcePacks    []mp.ModelPack    `json:"source_packs"`
	OutputWeight   string            `json:"output_weight"`
	MergedTensors  int               `json:"merged_tensors"`
	CopiedTensors  int               `json:"copied_tensors,omitempty"`
	SkippedTensors []string          `json:"skipped_tensors,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
}

type prepared struct {
	Method  Method
	T       float64
	Sources []Source
	Packs   []mp.ModelPack
	Output  string
}

// Packs merges compatible local safetensors model packs and writes a loadable pack.
func Packs(ctx context.Context, opts Options) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepare(ctx, opts)
	if err != nil {
		return nil, err
	}

	indexes, err := indexSources(prepared.Packs)
	if err != nil {
		return nil, err
	}
	if err := validateTensorIndexes(indexes, opts.AllowTensorMismatch); err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(prepared.Output, modelMergeOutputWeights)
	merged, copied, skipped, err := writeMergedSafetensors(ctx, weightPath, indexes, prepared.Method, prepared.T, prepared.Sources, opts.AllowTensorMismatch)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prepared.Output, ProvenanceFile)
	if err := writeProvenance(provenancePath, Provenance{
		Version:        1,
		Method:         prepared.Method,
		T:              prepared.T,
		Sources:        prepared.Sources,
		SourcePacks:    prepared.Packs,
		OutputWeight:   core.PathBase(weightPath),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
		Labels:         opts.Labels,
	}); err != nil {
		return nil, err
	}

	return &Result{
		OutputPath:     prepared.Output,
		WeightPath:     weightPath,
		ProvenancePath: provenancePath,
		Method:         prepared.Method,
		T:              prepared.T,
		Sources:        prepared.Packs,
		TensorCount:    len(indexes[0].Names),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
	}, nil
}

func prepare(ctx context.Context, opts Options) (prepared, error) {
	if err := ctx.Err(); err != nil {
		return prepared{}, err
	}
	if len(opts.Sources) < 2 {
		return prepared{}, errMergeNeedTwoSources
	}
	if opts.OutputPath == "" {
		return prepared{}, errOutputPathRequired
	}
	// hasSuffixFold replaces core.Lower(opts.OutputPath) which allocated a
	// full copy of the (potentially long) output path string just to test
	// two short suffixes.
	if hasSuffixFold(opts.OutputPath, ".safetensors") || hasSuffixFold(opts.OutputPath, ".gguf") {
		return prepared{}, errOutputNotPackDir
	}

	method := opts.Method
	if method == "" {
		method = MethodLinear
	}
	switch method {
	case MethodLinear, MethodSLERP:
	case MethodTIES, MethodDARE:
		return prepared{}, core.NewError("mlx: model merge method " + string(method) + " is reserved as a future sparse-merge hook and is not implemented yet")
	default:
		return prepared{}, core.NewError("mlx: unsupported model merge method: " + string(method))
	}
	if method == MethodSLERP && len(opts.Sources) != 2 {
		return prepared{}, errSLERPNeedTwoSources
	}
	if opts.T < 0 || opts.T > 1 {
		return prepared{}, errMergeTOutOfRange
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if err := ensureEmptyDestination(output); err != nil {
		return prepared{}, err
	}

	packs := make([]mp.ModelPack, 0, len(opts.Sources))
	normalizedSources := make([]Source, 0, len(opts.Sources))
	for _, source := range opts.Sources {
		pack := source.Pack
		if pack.Root == "" {
			return prepared{}, errMergeSourcePackRequired
		}
		if pack.Format != mp.ModelPackFormatSafetensors {
			return prepared{}, errMergeNeedsSafetensors
		}
		if samePathResolved(pack.Root, output) {
			return prepared{}, errOutputSameAsSource
		}
		packs = append(packs, pack)
		normalizedSources = append(normalizedSources, source)
	}

	if err := validatePackCompatibility(packs, opts); err != nil {
		return prepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return prepared{}, core.E("Packs", "create merged model directory", resultError(result))
	}
	if err := copyModelPackMetadata(packs[0].Root, output); err != nil {
		return prepared{}, err
	}

	return prepared{
		Method:  method,
		T:       opts.T,
		Sources: normalizedSources,
		Packs:   packs,
		Output:  output,
	}, nil
}

func ensureEmptyDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("Packs", "inspect output path", resultError(stat))
	}
	// Check the two glob patterns independently — the previous append form
	// always allocated a combined slice even when the first pattern was
	// already non-empty. Short-circuit on the first non-empty pattern.
	if len(core.PathGlob(core.PathJoin(output, "*.safetensors"))) > 0 {
		return errOutputHasWeights
	}
	if len(core.PathGlob(core.PathJoin(output, "*.gguf"))) > 0 {
		return errOutputHasWeights
	}
	return nil
}

func validatePackCompatibility(packs []mp.ModelPack, opts Options) error {
	base := packs[0]
	// Hash the base tokenizer once up front, lazily — only if we actually
	// need it (any non-AllowTokenizerMismatch source). Previously the
	// inner loop re-read + re-hashed the base file once per source pack,
	// turning an O(1) check into O(N) IO + crypto for the N-source case.
	var baseHash string
	var baseHashErr error
	baseHashLoaded := opts.AllowTokenizerMismatch
	for i := 1; i < len(packs); i++ {
		pack := packs[i]
		if !opts.AllowArchitectureMismatch && pack.Architecture != base.Architecture {
			// core.Concat is ~4x cheaper than core.Sprintf for fixed-string
			// composition. Architecture names are short identifiers; the fmt
			// machinery is pure overhead here.
			return core.NewError(core.Concat(
				"mlx: model merge architecture mismatch: ",
				base.Architecture,
				" vs ",
				pack.Architecture,
			))
		}
		if opts.AllowTokenizerMismatch {
			continue
		}
		if !baseHashLoaded {
			baseHash, baseHashErr = hashFile(base.TokenizerPath)
			baseHashLoaded = true
		}
		if baseHashErr != nil {
			return core.E("Packs", "hash base tokenizer", baseHashErr)
		}
		hash, err := hashFile(pack.TokenizerPath)
		if err != nil {
			return core.E("Packs", "hash tokenizer", err)
		}
		if hash != baseHash {
			return errTokenizerMismatch
		}
	}
	return nil
}

func indexSources(packs []mp.ModelPack) ([]safetensors.Index, error) {
	indexes := make([]safetensors.Index, 0, len(packs))
	for _, pack := range packs {
		index, err := safetensors.IndexFiles(pack.WeightFiles)
		if err != nil {
			return nil, err
		}
		indexes = append(indexes, index)
	}
	return indexes, nil
}

func validateTensorIndexes(indexes []safetensors.Index, allowMismatch bool) error {
	base := indexes[0]
	for i := 1; i < len(indexes); i++ {
		index := indexes[i]
		for _, name := range base.Names {
			ref, ok := index.Tensors[name]
			if !ok {
				if allowMismatch {
					continue
				}
				return core.NewError("mlx: model merge tensor missing from source: " + name)
			}
			// baseRef is only needed when we actually compare shapes — lift
			// the lookup inside the if-ok branch. Saves one map probe per
			// matched-name iteration (the dominant path).
			baseRef := base.Tensors[name]
			if !sameUint64Slice(baseRef.Shape, ref.Shape) {
				if allowMismatch {
					continue
				}
				return core.NewError("mlx: model merge tensor shape mismatch: " + name)
			}
		}
		if allowMismatch {
			continue
		}
		for _, name := range index.Names {
			if _, ok := base.Tensors[name]; !ok {
				return core.NewError("mlx: model merge extra tensor in source: " + name)
			}
		}
	}
	return nil
}

// hasSuffixFold reports whether s ends with suffix using ASCII case
// folding. Suffix is required to be lowercase. Pure scan, no allocations —
// replaces the core.Lower(s) + core.HasSuffix pattern that always allocated
// a lowered copy of s regardless of input.
func hasSuffixFold(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	off := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		c := s[off+i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
}

func sameUint64Slice(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func clampFloat64(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errCoreResultFailed
}

// equalFold is len-prefixed ASCII case-insensitive equality. Zero allocations.
func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// containsFold reports whether s contains substr using ASCII case folding.
// substr is required to be lowercase. Zero allocations.
func containsFold(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(substr) > len(s) {
		return false
	}
	last := len(s) - len(substr)
outer:
	for i := 0; i <= last; i++ {
		for j := 0; j < len(substr); j++ {
			c := s[i+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != substr[j] {
				continue outer
			}
		}
		return true
	}
	return false
}
