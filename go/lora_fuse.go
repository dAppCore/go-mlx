// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
	modelinspect "dappco.re/go/mlx/model"
	"dappco.re/go/mlx/pack"
)

// ModelPack summarises whether a local model directory is natively loadable.
type ModelPack = pack.ModelPack

// ModelPackOption configures model-pack inspection.
type ModelPackOption = pack.ModelPackOption

// LoRAAdapterInfo is the reproducible identity for an adapter.
type LoRAAdapterInfo = lora.AdapterInfo

// LoRAFuseProvenance records how a fused model pack was produced.
type LoRAFuseProvenance = lora.FuseProvenance

// FuseLoRAOptions configures pack-level LoRA fusion through the root API.
type FuseLoRAOptions struct {
	ModelPath   string            `json:"model_path"`
	AdapterPath string            `json:"adapter_path"`
	OutputPath  string            `json:"output_path"`
	Labels      map[string]string `json:"labels,omitempty"`
	PackOptions []ModelPackOption `json:"-"`
}

// FuseLoRAResult reports the paths and identities of a fused model pack.
type FuseLoRAResult struct {
	OutputPath      string          `json:"output_path"`
	WeightPath      string          `json:"weight_path"`
	WeightFiles     []string        `json:"weight_files,omitempty"`
	ProvenancePath  string          `json:"provenance_path"`
	SourcePack      ModelPack       `json:"source_pack"`
	OutputPack      ModelPack       `json:"output_pack"`
	Adapter         LoRAAdapterInfo `json:"adapter"`
	FusedWeights    int             `json:"fused_weights"`
	FusedWeightKeys []string        `json:"fused_weight_keys,omitempty"`
}

// InspectModelPack validates local model metadata without loading tensors.
func InspectModelPack(modelPath string, opts ...ModelPackOption) (ModelPack, error) {
	return modelinspect.Inspect(modelPath, opts...)
}

// ValidateModelPack returns an error when model-pack inspection finds issues.
func ValidateModelPack(modelPath string, opts ...ModelPackOption) (ModelPack, error) {
	return modelinspect.Validate(modelPath, opts...)
}

// FuseLoRAIntoModelPack merges a LoRA adapter into a safetensors model pack
// and validates both the source and fused output through the shared model-pack
// inspector.
func FuseLoRAIntoModelPack(ctx context.Context, opts FuseLoRAOptions) (*FuseLoRAResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	source, err := ValidateModelPack(opts.ModelPath, opts.PackOptions...)
	if err != nil {
		return nil, core.E("mlx.FuseLoRAIntoModelPack", "validate source model pack", err)
	}
	fused, err := lora.FuseIntoPack(ctx, lora.FuseOptions{
		SourcePack:  source,
		AdapterPath: opts.AdapterPath,
		OutputPath:  opts.OutputPath,
		Labels:      opts.Labels,
	})
	if err != nil {
		return nil, core.E("mlx.FuseLoRAIntoModelPack", "fuse adapter", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	output, err := ValidateModelPack(fused.OutputPath, opts.PackOptions...)
	if err != nil {
		return nil, core.E("mlx.FuseLoRAIntoModelPack", "validate fused model pack", err)
	}
	return &FuseLoRAResult{
		OutputPath:      fused.OutputPath,
		WeightPath:      fused.WeightPath,
		WeightFiles:     fused.WeightFiles,
		ProvenancePath:  fused.ProvenancePath,
		SourcePack:      source,
		OutputPack:      output,
		Adapter:         fused.Adapter,
		FusedWeights:    fused.FusedWeights,
		FusedWeightKeys: fused.FusedWeightKeys,
	}, nil
}
