// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
)

const (
	// LoRAFuseProvenanceFile is written into fused model packs.
	LoRAFuseProvenanceFile = "adapter_provenance.json"
	loRAFuseOutputWeights  = "model.safetensors"
)

// FuseLoRAOptions configures pack-level LoRA fusion.
type FuseLoRAOptions struct {
	ModelPath   string            `json:"model_path"`
	AdapterPath string            `json:"adapter_path"`
	OutputPath  string            `json:"output_path"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// FuseLoRAResult reports the generated model pack and adapter identity.
type FuseLoRAResult struct {
	OutputPath      string          `json:"output_path"`
	WeightPath      string          `json:"weight_path"`
	WeightFiles     []string        `json:"weight_files,omitempty"`
	ProvenancePath  string          `json:"provenance_path"`
	Pack            ModelPack       `json:"pack"`
	Adapter         lora.AdapterInfo `json:"adapter"`
	FusedWeights    int             `json:"fused_weights"`
	FusedWeightKeys []string        `json:"fused_weight_keys,omitempty"`
}

// LoRAFuseProvenance records how a fused pack was produced.
type LoRAFuseProvenance struct {
	Version         int               `json:"version"`
	SourceModel     ModelPack         `json:"source_model"`
	Adapter         lora.AdapterInfo   `json:"adapter"`
	OutputWeight    string            `json:"output_weight"`
	OutputWeights   []string          `json:"output_weights,omitempty"`
	FusedWeightKeys []string          `json:"fused_weight_keys"`
	Labels          map[string]string `json:"labels,omitempty"`
}

type loraFusePrepared struct {
	Model   ModelPack
	Adapter lora.AdapterInfo
	Output  string
}

func prepareLoRAFuse(ctx context.Context, opts FuseLoRAOptions) (loraFusePrepared, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return loraFusePrepared{}, err
	}
	if opts.ModelPath == "" {
		return loraFusePrepared{}, core.NewError("mlx: source model path is required")
	}
	if opts.AdapterPath == "" {
		return loraFusePrepared{}, core.NewError("mlx: LoRA adapter path is required")
	}
	if opts.OutputPath == "" {
		return loraFusePrepared{}, core.NewError("mlx: fused model output path is required")
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") || core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") {
		return loraFusePrepared{}, core.NewError("mlx: fused output path must be a model-pack directory")
	}

	model, err := ValidateModelPack(opts.ModelPath)
	if err != nil {
		return loraFusePrepared{}, core.E("FuseLoRAIntoModelPack", "validate source model pack", err)
	}
	if model.Format != ModelPackFormatSafetensors {
		return loraFusePrepared{}, core.NewError("mlx: LoRA pack fusion currently requires safetensors base weights")
	}

	adapter, err := lora.InspectAdapter(opts.AdapterPath)
	if err != nil {
		return loraFusePrepared{}, core.E("FuseLoRAIntoModelPack", "inspect LoRA adapter", err)
	}
	if adapter.Rank <= 0 {
		return loraFusePrepared{}, core.NewError("mlx: LoRA adapter rank is required for fusion")
	}
	if adapter.Scale == 0 && adapter.Alpha == 0 {
		adapter.Alpha = float32(adapter.Rank) * 2
		adapter.Scale = adapter.Alpha / float32(adapter.Rank)
	}
	if adapter.Scale == 0 {
		return loraFusePrepared{}, core.NewError("mlx: LoRA adapter scale is required for fusion")
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(model.Root, output) {
		return loraFusePrepared{}, core.NewError("mlx: fused output path must differ from source model path")
	}
	if err := ensureEmptyFuseWeightDestination(output); err != nil {
		return loraFusePrepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return loraFusePrepared{}, core.E("FuseLoRAIntoModelPack", "create fused model directory", loraAdapterResultError(result))
	}
	if err := copyModelPackMetadata(model.Root, output); err != nil {
		return loraFusePrepared{}, err
	}

	return loraFusePrepared{
		Model:   model,
		Adapter: adapter,
		Output:  output,
	}, nil
}

func ensureEmptyFuseWeightDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("FuseLoRAIntoModelPack", "inspect output path", loraAdapterResultError(stat))
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
	return lower == LoRAFuseProvenanceFile ||
		core.Contains(lower, ".safetensors") ||
		core.Contains(lower, ".gguf") ||
		core.HasSuffix(lower, ".safetensors") ||
		core.HasSuffix(lower, ".gguf")
}

func copyLocalFile(sourcePath, destinationPath string) error {
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return core.E("FuseLoRAIntoModelPack", "read "+sourcePath, loraAdapterResultError(read))
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return core.E("FuseLoRAIntoModelPack", "write "+destinationPath, loraAdapterResultError(result))
	}
	return nil
}

func loraFuseAdapterWeightFiles(path string) ([]string, error) {
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

func loraFusePairName(weightName string) (string, string, bool) {
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

func loraFuseBaseWeightKey(pairName string) string {
	return pairName + ".weight"
}

func writeLoRAFuseProvenance(path string, provenance LoRAFuseProvenance) error {
	slices.Sort(provenance.FusedWeightKeys)
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("FuseLoRAIntoModelPack", "marshal adapter provenance", loraAdapterResultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("FuseLoRAIntoModelPack", "write adapter provenance", loraAdapterResultError(result))
	}
	return nil
}

func loraAdapterResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
