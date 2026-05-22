// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/pack"
	"slices"
)

const (
	// FuseProvenanceFile is the basename written into fused model packs.
	FuseProvenanceFile = "adapter_provenance.json"
	fuseOutputWeights  = "model.safetensors"
)

// Sentinel errors returned by fuse validation and orchestration paths.
// Hoisted to package vars so each guard returns the shared instance
// instead of allocating a fresh *core.Err per call — relevant both for
// the always-fired validation guards in prepareFuse and the per-fuse
// integrity checks downstream.
var (
	errFuseSourceRootRequired   = core.NewError("mlx: source pack root is required")
	errFuseAdapterPathRequired  = core.NewError("mlx: LoRA adapter path is required")
	errFuseOutputPathRequired   = core.NewError("mlx: fused model output path is required")
	errFuseOutputNotPackDir     = core.NewError("mlx: fused output path must be a model-pack directory")
	errFuseRequiresSafetensors  = core.NewError("mlx: LoRA pack fusion currently requires safetensors base weights")
	errFuseRankRequired         = core.NewError("mlx: LoRA adapter rank is required for fusion")
	errFuseScaleRequired        = core.NewError("mlx: LoRA adapter scale is required for fusion")
	errFuseOutputSameAsSource   = core.NewError("mlx: fused output path must differ from source model path")
	errFuseOutputContainsWeight = core.NewError("mlx: fused output path already contains model weights")
	errFuseNoAdapterSafetensors = core.NewError("mlx: no adapter safetensors found")
	errFuseNoLoRATensorPairs    = core.NewError("mlx: no LoRA tensor pairs found")
	errFuseNoBaseWeightFiles    = core.NewError("mlx: no base weight files available for LoRA fusion")
)

// FuseOptions configures pack-level LoRA fusion.
//
// SourcePack must be a validated, safetensors-format model pack; callers
// validate via mlx.ValidateModelPack before invoking lora.FuseIntoPack.
// Splitting validation out of the lora package keeps lora free of the
// mlx-root cycle.
type FuseOptions struct {
	SourcePack  pack.ModelPack    `json:"source_pack"`
	AdapterPath string            `json:"adapter_path"`
	OutputPath  string            `json:"output_path"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// FuseResult reports the paths and identity of a fused model pack.
//
// Callers re-validate the output via mlx.ValidateModelPack(OutputPath)
// when they need the populated pack.ModelPack for downstream use.
type FuseResult struct {
	OutputPath      string      `json:"output_path"`
	WeightPath      string      `json:"weight_path"`
	WeightFiles     []string    `json:"weight_files,omitempty"`
	ProvenancePath  string      `json:"provenance_path"`
	Adapter         AdapterInfo `json:"adapter"`
	FusedWeights    int         `json:"fused_weights"`
	FusedWeightKeys []string    `json:"fused_weight_keys,omitempty"`
}

// FuseProvenance records how a fused pack was produced. Written into
// adapter_provenance.json next to the fused weights.
type FuseProvenance struct {
	Version         int               `json:"version"`
	SourceModel     pack.ModelPack    `json:"source_model"`
	Adapter         AdapterInfo       `json:"adapter"`
	OutputWeight    string            `json:"output_weight"`
	OutputWeights   []string          `json:"output_weights,omitempty"`
	FusedWeightKeys []string          `json:"fused_weight_keys"`
	Labels          map[string]string `json:"labels,omitempty"`
}

type fusePrepared struct {
	Model   pack.ModelPack
	Adapter AdapterInfo
	Output  string
}

func prepareFuse(ctx context.Context, opts FuseOptions) (fusePrepared, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return fusePrepared{}, err
	}
	if opts.SourcePack.Root == "" {
		return fusePrepared{}, errFuseSourceRootRequired
	}
	if opts.AdapterPath == "" {
		return fusePrepared{}, errFuseAdapterPathRequired
	}
	if opts.OutputPath == "" {
		return fusePrepared{}, errFuseOutputPathRequired
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") || core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") {
		return fusePrepared{}, errFuseOutputNotPackDir
	}
	if opts.SourcePack.Format != pack.ModelPackFormatSafetensors {
		return fusePrepared{}, errFuseRequiresSafetensors
	}

	adapter, err := Inspect(opts.AdapterPath, opts.AdapterPath)
	if err != nil {
		return fusePrepared{}, core.E("lora.FuseIntoPack", "inspect LoRA adapter", err)
	}
	if adapter.Rank <= 0 {
		return fusePrepared{}, errFuseRankRequired
	}
	if adapter.Scale == 0 && adapter.Alpha == 0 {
		adapter.Alpha = float32(adapter.Rank) * 2
		adapter.Scale = adapter.Alpha / float32(adapter.Rank)
	}
	if adapter.Scale == 0 {
		return fusePrepared{}, errFuseScaleRequired
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(opts.SourcePack.Root, output) {
		return fusePrepared{}, errFuseOutputSameAsSource
	}
	if err := ensureEmptyFuseWeightDestination(output); err != nil {
		return fusePrepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return fusePrepared{}, core.E("lora.FuseIntoPack", "create fused model directory", resultError(result))
	}
	if err := copyModelPackMetadata(opts.SourcePack.Root, output); err != nil {
		return fusePrepared{}, err
	}

	return fusePrepared{
		Model:   opts.SourcePack,
		Adapter: adapter,
		Output:  output,
	}, nil
}

func ensureEmptyFuseWeightDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("lora.FuseIntoPack", "inspect output path", resultError(stat))
	}
	weights := append(core.PathGlob(core.PathJoin(output, "*.safetensors")), core.PathGlob(core.PathJoin(output, "*.gguf"))...)
	if len(weights) > 0 {
		return errFuseOutputContainsWeight
	}
	return nil
}

func samePath(a, b string) bool {
	// Fast path: identical strings cannot resolve to different absolutes,
	// so skip the two PathAbs round-trips when the raw inputs already
	// match. The fuse-self-fuse guard in prepareFuse fires this once per
	// call and the SameAbsolute bench covers the equality path.
	if a == b {
		return true
	}
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	patterns := []string{"*.json", "*.model", "*.txt"}
	seen := map[string]struct{}{}
	for _, pattern := range patterns {
		for _, sourcePath := range core.PathGlob(core.PathJoin(sourceRoot, pattern)) {
			name := core.PathBase(sourcePath)
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			if isModelWeightMetadataCopySkip(name) {
				continue
			}
			if err := copyLocalFile(sourcePath, core.PathJoin(outputRoot, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func isModelWeightMetadataCopySkip(name string) bool {
	// Contains(".safetensors") is a strict superset of HasSuffix(".safetensors"):
	// any name ending in .safetensors necessarily contains the substring. The
	// previous HasSuffix terms were dead under the OR — drop them and let the
	// Contains checks carry both the suffix and the .safetensors.index.json
	// case the copy filter is meant to skip.
	lower := core.Lower(name)
	return lower == FuseProvenanceFile ||
		core.Contains(lower, ".safetensors") ||
		core.Contains(lower, ".gguf")
}

func copyLocalFile(sourcePath, destinationPath string) error {
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return core.E("lora.FuseIntoPack", "read "+sourcePath, resultError(read))
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return core.E("lora.FuseIntoPack", "write "+destinationPath, resultError(result))
	}
	return nil
}

func fuseAdapterWeightFiles(path string) ([]string, error) {
	if core.HasSuffix(core.Lower(path), ".safetensors") {
		return []string{path}, nil
	}
	matches := core.PathGlob(core.PathJoin(path, "*.safetensors"))
	slices.Sort(matches)
	if len(matches) == 0 {
		return nil, errFuseNoAdapterSafetensors
	}
	return matches, nil
}

func fusePairName(weightName string) (string, string, bool) {
	for _, variant := range []struct {
		suffix string
		kind   string
	}{
		{suffix: ".lora_a.weight", kind: "a"},
		{suffix: ".lora_A.weight", kind: "a"},
		{suffix: ".lora_a", kind: "a"},
		{suffix: ".lora_A", kind: "a"},
		{suffix: ".lora_b.weight", kind: "b"},
		{suffix: ".lora_B.weight", kind: "b"},
		{suffix: ".lora_b", kind: "b"},
		{suffix: ".lora_B", kind: "b"},
	} {
		if core.HasSuffix(weightName, variant.suffix) {
			return core.TrimSuffix(weightName, variant.suffix), variant.kind, true
		}
	}
	return "", "", false
}

func fuseBaseWeightKey(pairName string) string {
	return pairName + ".weight"
}

func writeFuseProvenance(path string, provenance FuseProvenance) error {
	slices.Sort(provenance.FusedWeightKeys)
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("lora.FuseIntoPack", "marshal adapter provenance", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("lora.FuseIntoPack", "write adapter provenance", resultError(result))
	}
	return nil
}

type fusePair struct {
	MatrixA *metal.Array
	MatrixB *metal.Array
}

// FuseIntoPack merges a LoRA adapter into dense safetensors base weights
// and writes a go-mlx-loadable model pack. Callers validate
// opts.SourcePack with mlx.ValidateModelPack before invoking, and
// validate the OutputPath after the call returns.
//
//	src, err := mlx.ValidateModelPack(path)
//	res, err := lora.FuseIntoPack(ctx, lora.FuseOptions{SourcePack: src, AdapterPath: a, OutputPath: o})
//	out, err := mlx.ValidateModelPack(res.OutputPath)
func FuseIntoPack(ctx context.Context, opts FuseOptions) (*FuseResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepareFuse(ctx, opts)
	if err != nil {
		return nil, err
	}

	adapterWeights, err := loadFuseAdapterWeights(opts.AdapterPath)
	if err != nil {
		return nil, err
	}
	defer freeMetalMap(adapterWeights)

	pairs, err := buildFusePairs(adapterWeights)
	if err != nil {
		return nil, err
	}

	weightFiles, fusedKeys, err := fuseModelWeightFiles(ctx, prepared.Model.WeightFiles, prepared.Output, pairs, prepared.Adapter.Scale)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prepared.Output, FuseProvenanceFile)
	if err := writeFuseProvenance(provenancePath, FuseProvenance{
		Version:         1,
		SourceModel:     prepared.Model,
		Adapter:         prepared.Adapter,
		OutputWeight:    core.PathBase(weightFiles[0]),
		OutputWeights:   outputWeightFileNames(weightFiles),
		FusedWeightKeys: fusedKeys,
		Labels:          opts.Labels,
	}); err != nil {
		return nil, err
	}

	return &FuseResult{
		OutputPath:      prepared.Output,
		WeightPath:      weightFiles[0],
		WeightFiles:     weightFiles,
		ProvenancePath:  provenancePath,
		Adapter:         prepared.Adapter,
		FusedWeights:    len(fusedKeys),
		FusedWeightKeys: fusedKeys,
	}, nil
}

func loadFuseAdapterWeights(path string) (map[string]*metal.Array, error) {
	paths, err := fuseAdapterWeightFiles(path)
	if err != nil {
		return nil, err
	}
	weights := make(map[string]*metal.Array)
	for _, path := range paths {
		loaded, err := metal.LoadAllSafetensors(path)
		if err != nil {
			freeMetalMap(weights)
			return nil, core.E("lora.FuseIntoPack", "load adapter weights "+core.PathBase(path), err)
		}
		for name, tensor := range loaded {
			if previous := weights[name]; previous != nil {
				metal.Free(previous)
			}
			weights[name] = tensor
		}
	}
	return weights, nil
}

func buildFusePairs(weights map[string]*metal.Array) (map[string]fusePair, error) {
	pairs := make(map[string]fusePair)
	for name, tensor := range weights {
		pairName, suffix, ok := fusePairName(name)
		if !ok {
			continue
		}
		pair := pairs[pairName]
		switch suffix {
		case "a":
			pair.MatrixA = tensor
		case "b":
			pair.MatrixB = tensor
		}
		pairs[pairName] = pair
	}
	if len(pairs) == 0 {
		return nil, errFuseNoLoRATensorPairs
	}
	for name, pair := range pairs {
		if pair.MatrixA == nil || pair.MatrixB == nil {
			return nil, core.NewError("mlx: incomplete LoRA tensor pair: " + name)
		}
	}
	return pairs, nil
}

func fuseModelWeightFiles(ctx context.Context, sourceFiles []string, outputRoot string, pairs map[string]fusePair, scale float32) ([]string, []string, error) {
	if len(sourceFiles) == 0 {
		return nil, nil, errFuseNoBaseWeightFiles
	}

	fusedPairs := map[string]struct{}{}
	weightFiles := make([]string, 0, len(sourceFiles))
	fusedKeys := make([]string, 0, len(pairs))
	for _, sourceFile := range sourceFiles {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		baseWeights, err := metal.LoadAllSafetensors(sourceFile)
		if err != nil {
			return nil, nil, core.E("lora.FuseIntoPack", "load base weights "+core.PathBase(sourceFile), err)
		}

		shardFusedKeys, err := fuseWeightPairs(ctx, baseWeights, pairs, fusedPairs, scale)
		if err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, err
		}
		fusedKeys = append(fusedKeys, shardFusedKeys...)

		outputName := fuseOutputWeights
		if len(sourceFiles) > 1 {
			outputName = core.PathBase(sourceFile)
		}
		weightPath := core.PathJoin(outputRoot, outputName)
		if err := metal.SaveSafetensors(weightPath, baseWeights); err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, core.E("lora.FuseIntoPack", "save fused safetensors", err)
		}
		freeMetalMap(baseWeights)
		weightFiles = append(weightFiles, weightPath)
	}

	for name := range pairs {
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		return nil, nil, core.NewError("mlx: base weight not found for LoRA target: " + fuseBaseWeightKey(name))
	}
	return weightFiles, fusedKeys, nil
}

func fuseWeightPairs(ctx context.Context, baseWeights map[string]*metal.Array, pairs map[string]fusePair, fusedPairs map[string]struct{}, scale float32) ([]string, error) {
	names := make([]string, 0, len(pairs))
	for name := range pairs {
		names = append(names, name)
	}
	slices.Sort(names)

	fusedKeys := make([]string, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		baseKey := fuseBaseWeightKey(name)
		base := baseWeights[baseKey]
		if base == nil {
			continue
		}

		pair := pairs[name]
		delta := metal.Matmul(pair.MatrixB, pair.MatrixA)
		scaled := metal.MulScalar(delta, scale)
		fused := metal.Add(base, scaled)
		metal.Materialize(fused)
		metal.Free(delta, scaled, base)
		baseWeights[baseKey] = fused
		fusedKeys = append(fusedKeys, baseKey)
		fusedPairs[name] = struct{}{}
	}
	return fusedKeys, nil
}

func outputWeightFileNames(paths []string) []string {
	names := make([]string, 0, len(paths))
	for _, path := range paths {
		names = append(names, core.PathBase(path))
	}
	return names
}

func freeMetalMap(weights map[string]*metal.Array) {
	for _, tensor := range weights {
		metal.Free(tensor)
	}
}
