// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/hf"
	"dappco.re/go/mlx/memory"
)

// Legacy aliases — the canonical HuggingFace metadata + fit planner
// lives at dappco.re/go/mlx/hf/. mlx-root callers keep their existing
// HF* + HuggingFace* surface via these aliases.
type (
	HFModelSource                = hf.ModelSource
	HuggingFaceModelSourceConfig = hf.RemoteConfig
	HuggingFaceModelSource       = hf.RemoteSource
	HFModelFitConfig             = hf.FitConfig
	HFModelMetadata              = hf.ModelMetadata
	HFModelFile                  = hf.ModelFile
	HFModelConfig                = hf.ModelConfig
	HFQuantizationConfig         = hf.QuantizationConfig
	HFModelFitReport             = hf.FitReport
	HFModelFitPlan               = hf.FitPlan
	HFTrainingFit                = hf.TrainingFit
)

// Source constants forwarded from the hf package.
const (
	HFModelSourceRemote = hf.SourceRemote
	HFModelSourceLocal  = hf.SourceLocal
)

// NewHuggingFaceModelSource creates a network-backed HF metadata source.
//
//	source := mlx.NewHuggingFaceModelSource(mlx.HuggingFaceModelSourceConfig{...})
func NewHuggingFaceModelSource(cfg HuggingFaceModelSourceConfig) *HuggingFaceModelSource {
	return hf.NewRemoteSource(cfg)
}

// PlanHFModelFits discovers HF/local metadata and estimates local Apple
// fit. Auto-populates Device from the runtime metal probe when empty.
//
//	report, err := mlx.PlanHFModelFits(ctx, cfg)
func PlanHFModelFits(ctx context.Context, cfg HFModelFitConfig) (*HFModelFitReport, error) {
	if cfg.Device.MemorySize == 0 && cfg.Device.MaxRecommendedWorkingSetSize == 0 {
		info := GetDeviceInfo()
		cfg.Device = memory.DeviceInfo{
			Architecture:                 info.Architecture,
			MaxBufferLength:              info.MaxBufferLength,
			MaxRecommendedWorkingSetSize: info.MaxRecommendedWorkingSetSize,
			MemorySize:                   info.MemorySize,
		}
	}
	return hf.PlanFits(ctx, cfg)
}

// InferJANGFromHF inspects HF metadata + tags + filenames to derive a
// best-guess JANG quantization profile.
//
//	info := mlx.InferJANGFromHF(meta)
func InferJANGFromHF(meta HFModelMetadata) *jang.Info {
	return hf.InferJANG(meta)
}
