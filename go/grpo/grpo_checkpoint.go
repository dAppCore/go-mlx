// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"strconv"

	core "dappco.re/go"
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

// SaveGRPOCheckpointMetadata writes checkpoint metadata beside policy artifacts.
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
	metadataPath := grpoCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("GRPOCheckpointMetadata.Save", "ensure metadata dir", grpoResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("GRPOCheckpointMetadata.Save", "marshal metadata", grpoResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("GRPOCheckpointMetadata.Save", "write metadata", grpoResultError(result))
	}
	return nil
}

// LoadGRPOCheckpointMetadata reads checkpoint metadata written by SaveGRPOCheckpointMetadata.
func LoadGRPOCheckpointMetadata(path string) (*GRPOCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: experimental GRPO checkpoint metadata path is required")
	}
	read := core.ReadFile(grpoCheckpointMetadataPath(path))
	if !read.OK {
		return nil, grpoResultError(read)
	}
	var meta GRPOCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadGRPOCheckpointMetadata", "parse metadata", grpoResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return &meta, nil
}

func loadGRPOResumeMetadata(path string) (*GRPOCheckpointMetadata, error) {
	read := core.ReadFile(grpoCheckpointMetadataPath(path))
	if !read.OK {
		err := grpoResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta GRPOCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadGRPOResumeMetadata", "parse metadata", grpoResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return &meta, nil
}

func grpoCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "grpo_checkpoint.json")
}

// grpoStepName renders the step-NNNNNN directory name used for GRPO
// checkpoints. Same output as fmt.Sprintf("step-%06d", step) — six-
// digit zero-pad below 1e6, untruncated digit count above. Built with
// strconv.AppendInt so no fmt format-parser + no interface-boxing of
// the int arg; pre-sized output keeps the alloc count at one.
func grpoStepName(step int) string {
	const prefix = "step-"
	const padTo = 6
	// Allocate room for the prefix plus enough digits — 20 covers the
	// max int64 width.
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		// Hand-rolled zero-pad — strconv.Itoa lacks a Printf-style
		// width modifier, so for the typical sub-1e5 range we count
		// leading zeros ourselves. Above 1e5 strconv emits the full
		// width naturally.
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
