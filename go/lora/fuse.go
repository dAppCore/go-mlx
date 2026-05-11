// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/mlx/pack"
)

const (
	// FuseProvenanceFile is the basename written into fused model packs.
	FuseProvenanceFile = "adapter_provenance.json"
	fuseOutputWeights  = "model.safetensors"
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
		return fusePrepared{}, core.NewError("mlx: source pack root is required")
	}
	if opts.AdapterPath == "" {
		return fusePrepared{}, core.NewError("mlx: LoRA adapter path is required")
	}
	if opts.OutputPath == "" {
		return fusePrepared{}, core.NewError("mlx: fused model output path is required")
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") || core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") {
		return fusePrepared{}, core.NewError("mlx: fused output path must be a model-pack directory")
	}
	if opts.SourcePack.Format != pack.ModelPackFormatSafetensors {
		return fusePrepared{}, core.NewError("mlx: LoRA pack fusion currently requires safetensors base weights")
	}

	adapter, err := Inspect(opts.AdapterPath, opts.AdapterPath)
	if err != nil {
		return fusePrepared{}, core.E("lora.FuseIntoPack", "inspect LoRA adapter", err)
	}
	if adapter.Rank <= 0 {
		return fusePrepared{}, core.NewError("mlx: LoRA adapter rank is required for fusion")
	}
	if adapter.Scale == 0 && adapter.Alpha == 0 {
		adapter.Alpha = float32(adapter.Rank) * 2
		adapter.Scale = adapter.Alpha / float32(adapter.Rank)
	}
	if adapter.Scale == 0 {
		return fusePrepared{}, core.NewError("mlx: LoRA adapter scale is required for fusion")
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(opts.SourcePack.Root, output) {
		return fusePrepared{}, core.NewError("mlx: fused output path must differ from source model path")
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
		return core.NewError("mlx: fused output path already contains model weights")
	}
	return nil
}

func samePath(a, b string) bool {
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
	lower := core.Lower(name)
	return lower == FuseProvenanceFile ||
		core.Contains(lower, ".safetensors") ||
		core.Contains(lower, ".gguf") ||
		core.HasSuffix(lower, ".safetensors") ||
		core.HasSuffix(lower, ".gguf")
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
		return nil, core.NewError("mlx: no adapter safetensors found")
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
