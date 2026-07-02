// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/checkpoint"
)

func maybeSaveGRPOCheckpoint(ctx context.Context, runner GRPORunner, cfg GRPOConfig, result *GRPOResult, update *GRPOUpdate) error {
	if cfg.CheckpointDir == "" || cfg.CheckpointEvery <= 0 || result.Metrics.Steps%cfg.CheckpointEvery != 0 {
		return nil
	}
	path := core.PathJoin(cfg.CheckpointDir, grpoStepName(result.Metrics.Steps))
	meta := NewGRPOCheckpointMetadata(path, cfg, result, *update)
	if runner.SaveCheckpoint != nil {
		if err := runner.SaveCheckpoint(ctx, GRPOCheckpointContext{Path: path, Update: *update, Metadata: meta}); err != nil {
			return err
		}
	}
	if err := SaveGRPOCheckpointMetadata(path, meta); err != nil {
		return err
	}
	result.Checkpoints = append(result.Checkpoints, path)
	result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	return nil
}

// NewGRPOCheckpointMetadata captures reproducible experimental GRPO state.
func NewGRPOCheckpointMetadata(path string, cfg GRPOConfig, result *GRPOResult, update GRPOUpdate) GRPOCheckpointMetadata {
	cfg = normalizeGRPOConfig(cfg)
	meta := GRPOCheckpointMetadata{
		Version:       GRPOCheckpointMetadataVersion,
		Experimental:  true,
		Path:          path,
		ResumePath:    cfg.ResumePath,
		Step:          update.Step,
		Epoch:         update.Epoch,
		GroupSize:     cfg.GroupSize,
		RewardMean:    update.RewardMean,
		RewardStd:     update.RewardStd,
		KLMean:        update.KLMean,
		Loss:          update.Loss,
		KLCoefficient: cfg.KLCoefficient,
		LearningRate:  cfg.LearningRate,
	}
	if result != nil {
		meta.Samples = result.Metrics.Samples
		meta.Rollouts = result.Metrics.Rollouts
		meta.Policy = result.Policy
	}
	return meta
}

// SaveGRPOCheckpointMetadata writes checkpoint metadata beside policy
// artifacts. The marshal-and-write mechanics are the shared checkpoint
// engine (dappco.re/go/inference/checkpoint); only the Version/Path/
// Experimental defaulting and the sidecar filename are grpo's own.
func SaveGRPOCheckpointMetadata(path string, meta GRPOCheckpointMetadata) error {
	if path == "" {
		return core.NewError("mlx: experimental GRPO checkpoint metadata path is required")
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	meta.Experimental = true
	if meta.Path == "" {
		meta.Path = path
	}
	return checkpoint.Save(grpoCheckpointMetadataPath(path), meta)
}

// LoadGRPOCheckpointMetadata reads checkpoint metadata written by SaveGRPOCheckpointMetadata.
func LoadGRPOCheckpointMetadata(path string) (*GRPOCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: experimental GRPO checkpoint metadata path is required")
	}
	meta, err := checkpoint.Load[GRPOCheckpointMetadata](grpoCheckpointMetadataPath(path))
	if err != nil {
		return nil, err
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return meta, nil
}

// loadGRPOResumeMetadata reads checkpoint metadata for a resume path,
// using the shared checkpoint engine's soft-missing-file semantics: an
// absent sidecar returns (nil, nil) rather than an error, matching
// --resume's "start fresh" contract.
func loadGRPOResumeMetadata(path string) (*GRPOCheckpointMetadata, error) {
	meta, err := checkpoint.LoadResume[GRPOCheckpointMetadata](grpoCheckpointMetadataPath(path))
	if err != nil || meta == nil {
		return meta, err
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return meta, nil
}

func grpoCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "grpo_checkpoint.json")
}

// grpoStepName renders the step-NNNNNN directory name used for GRPO
// checkpoints — delegates to the shared checkpoint engine
// (dappco.re/go/inference/checkpoint) so grpo's copy of the zero-pad
// logic cannot drift from distill/train's.
func grpoStepName(step int) string {
	return checkpoint.FormatStepDir(step)
}
