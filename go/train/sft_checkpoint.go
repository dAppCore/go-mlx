// SPDX-Licence-Identifier: EUPL-1.2

// Package train holds the native training machinery — SFT batch building,
// sequence packing, checkpoint metadata, the LoRA epoch loop, and the SSD
// (sampling-and-fine-tuning) pipeline with its code benchmark. The root mlx
// package aliases the exported types and keeps the Model-bound entry points
// (Model.TrainSFT / Model.RunSSD), which delegate here.
package train

import (
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/spine"
)

// NewSFTCheckpointMetadata captures the reproducible state for one checkpoint.
func NewSFTCheckpointMetadata(path string, model string, cfg SFTConfig, result *SFTResult, epoch int) SFTCheckpointMetadata {
	return newSFTMetadata(path, path, model, cfg, result, epoch)
}

// NewSFTArtifactMetadata captures the reproducible state for a final adapter artifact.
func NewSFTArtifactMetadata(path string, model string, cfg SFTConfig, result *SFTResult) SFTCheckpointMetadata {
	epoch := 0
	if result != nil {
		epoch = result.Epochs
	}
	return newSFTMetadata(path, path, model, cfg, result, epoch)
}

// SaveSFTCheckpointMetadata writes checkpoint metadata beside an adapter package.
func SaveSFTCheckpointMetadata(path string, meta SFTCheckpointMetadata) error {
	if path == "" {
		return core.NewError("mlx: SFT checkpoint metadata path is required")
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	if meta.Path == "" {
		meta.Path = path
	}
	metadataPath := sftCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("SFTCheckpointMetadata.Save", "ensure metadata dir", sftResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("SFTCheckpointMetadata.Save", "marshal metadata", sftResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("SFTCheckpointMetadata.Save", "write metadata", sftResultError(result))
	}
	return nil
}

// LoadSFTCheckpointMetadata reads checkpoint metadata written by SaveSFTCheckpointMetadata.
func LoadSFTCheckpointMetadata(path string) (*SFTCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: SFT checkpoint metadata path is required")
	}
	read := core.ReadFile(sftCheckpointMetadataPath(path))
	if !read.OK {
		return nil, sftResultError(read)
	}
	var meta SFTCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadSFTCheckpointMetadata", "parse metadata", sftResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	return &meta, nil
}

// ApplySFTResumeMetadata attaches optional checkpoint metadata from ResumePath to a result.
func ApplySFTResumeMetadata(result *SFTResult, cfg SFTConfig) error {
	if result == nil {
		return core.NewError("mlx: SFT result is nil")
	}
	if cfg.ResumePath == "" {
		return nil
	}
	result.ResumePath = cfg.ResumePath
	meta, err := loadSFTResumeMetadata(cfg.ResumePath)
	if err != nil {
		return err
	}
	result.ResumedFrom = meta
	return nil
}

func newSFTMetadata(path string, adapterPath string, model string, cfg SFTConfig, result *SFTResult, epoch int) SFTCheckpointMetadata {
	cfg = normalizeSFTConfig(cfg)
	step := 0
	optimizerStep := 0
	samples := 0
	loss := 0.0
	valLoss := 0.0
	if result != nil {
		step = result.Steps
		optimizerStep = result.OptimizerSteps
		if optimizerStep == 0 {
			optimizerStep = step
		}
		samples = result.Samples
		loss = result.LastLoss
		valLoss = result.LastValLoss
	}
	return SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      path,
		AdapterPath:               adapterPath,
		ResumePath:                cfg.ResumePath,
		Model:                     model,
		Step:                      step,
		OptimizerStep:             optimizerStep,
		Epoch:                     epoch,
		Samples:                   samples,
		Loss:                      loss,
		ValLoss:                   valLoss,
		LearningRate:              cfg.LearningRate,
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
		EffectiveBatchSize:        SFTEffectiveBatchSize(cfg),
		MaxSeqLen:                 cfg.MaxSeqLen,
		SequencePacking:           cfg.SequencePacking,
		EvalPrompts:               core.SliceClone(cfg.EvalPrompts),
		EvalTemperature:           cfg.EvalTemperature,
		LoRA:                      sftLoRAMetadata(cfg.LoRA),
		AdamW:                     sftAdamWMetadata(SFTAdamWConfig(cfg)),
		ScoreComposite:            sftScoreCompositeAt(result, step),
	}
}

// sftScoreCompositeAt annotates a checkpoint with the cascade's windowed
// composite at its step — 0 when the cascade isn't armed or hasn't scored.
func sftScoreCompositeAt(result *SFTResult, step int) float64 {
	if result == nil || result.cascade == nil {
		return 0
	}
	return result.cascade.compositeAt(step)
}

func sftLoRAMetadata(cfg spine.LoRAConfig) SFTLoRAMetadata {
	cfg = normalizeSFTLoRAConfig(cfg)
	return SFTLoRAMetadata{
		Rank:                 cfg.Rank,
		Alpha:                cfg.Alpha,
		Scale:                cfg.Scale,
		TargetKeys:           core.SliceClone(cfg.TargetKeys),
		TargetLayers:         core.SliceClone(cfg.TargetLayers),
		Lambda:               cfg.Lambda,
		DType:                cfg.DType.String(),
		AllowExtendedTargets: cfg.AllowExtendedTargets,
	}
}

func sftAdamWMetadata(cfg metal.AdamWConfig) SFTAdamWMetadata {
	return SFTAdamWMetadata{
		LearningRate: cfg.LearningRate,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Eps:          cfg.Eps,
		WeightDecay:  cfg.WeightDecay,
		PackedState:  cfg.PackedState,
	}
}

func SFTAdamWConfig(cfg SFTConfig) metal.AdamWConfig {
	cfg = normalizeSFTConfig(cfg)
	adam := metal.DefaultAdamWConfig()
	if cfg.AdamW.LearningRate != 0 || cfg.AdamW.LearningRateSet {
		adam.LearningRate = cfg.AdamW.LearningRate
	}
	if cfg.AdamW.Beta1 != 0 || cfg.AdamW.Beta1Set {
		adam.Beta1 = cfg.AdamW.Beta1
	}
	if cfg.AdamW.Beta2 != 0 || cfg.AdamW.Beta2Set {
		adam.Beta2 = cfg.AdamW.Beta2
	}
	if cfg.AdamW.Eps != 0 || cfg.AdamW.EpsSet {
		adam.Eps = cfg.AdamW.Eps
	}
	if cfg.AdamW.WeightDecay != 0 || cfg.AdamW.WeightDecaySet {
		adam.WeightDecay = cfg.AdamW.WeightDecay
	}
	if cfg.AdamW.PackedState || cfg.AdamW.PackedStateSet {
		adam.PackedState = cfg.AdamW.PackedState
	}
	if cfg.LearningRate != 0 {
		adam.LearningRate = cfg.LearningRate
	}
	return adam
}

func normalizeSFTLoRAConfig(cfg spine.LoRAConfig) spine.LoRAConfig {
	return sftLoRAConfigFromMetal(cfg, metal.NormalizeLoRAConfig(spine.ToMetalLoRAConfig(cfg)))
}

func normalizeSFTLoRAConfigForModel(cfg spine.LoRAConfig, info spine.ModelInfo) spine.LoRAConfig {
	if !profile.IsGemma4TargetArchitecture(info.Architecture) {
		return normalizeSFTLoRAConfig(cfg)
	}
	return sftLoRAConfigFromMetal(cfg, gemma4.NormalizeLoRA(spine.ToMetalLoRAConfig(cfg)))
}

func sftLoRAConfigFromMetal(source spine.LoRAConfig, cfg metal.LoRAConfig) spine.LoRAConfig {
	out := spine.LoRAConfigFromMetal(cfg)
	out.ProbeSink = source.ProbeSink
	return out
}

func loadSFTResumeMetadata(path string) (*SFTCheckpointMetadata, error) {
	read := core.ReadFile(sftCheckpointMetadataPath(path))
	if !read.OK {
		err := sftResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta SFTCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadSFTResumeMetadata", "parse metadata", sftResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	return &meta, nil
}

func sftCheckpointMetadataPath(path string) string {
	if core.HasSuffix(path, ".safetensors") {
		return core.PathJoin(core.PathDir(path), "sft_checkpoint.json")
	}
	return core.PathJoin(path, "sft_checkpoint.json")
}

// sftStepName renders the step-NNNNNN directory name used for SFT
// checkpoints — same output as fmt.Sprintf("step-%06d", step). Built
// with strconv.AppendInt so no fmt format-parser and no interface
// boxing of the int arg, with a pre-sized scratch buffer keeping the
// alloc count at one.
func sftStepName(step int) string {
	const prefix = "step-"
	const padTo = 6
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		digits := 1
		for n := step / 10; n > 0; n /= 10 {
			digits++
		}
		for i := digits; i < padTo; i++ {
			buf = append(buf, '0')
		}
	}
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
}
