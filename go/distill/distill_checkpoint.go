// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/checkpoint"
)

func maybeSaveDistillCheckpoint(ctx context.Context, runner DistillRunner, cfg DistillConfig, result *DistillResult, batch *DistillBatch, loss *DistillLoss) error {
	if cfg.CheckpointDir == "" || cfg.CheckpointEvery <= 0 || result.Metrics.Steps%cfg.CheckpointEvery != 0 {
		return nil
	}
	path := core.PathJoin(cfg.CheckpointDir, formatDistillStepDir(result.Metrics.Steps))
	meta := NewDistillCheckpointMetadata(path, cfg, result, *loss, batch.Epoch)
	if runner.SaveCheckpoint != nil {
		if err := runner.SaveCheckpoint(ctx, DistillCheckpointContext{
			Path:     path,
			Batch:    *batch,
			Loss:     *loss,
			Metadata: meta,
		}); err != nil {
			return err
		}
	}
	if err := SaveDistillCheckpointMetadata(path, meta); err != nil {
		return err
	}
	result.Checkpoints = append(result.Checkpoints, path)
	result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	return nil
}

// NewDistillCheckpointMetadata captures reproducible distillation state.
func NewDistillCheckpointMetadata(path string, cfg DistillConfig, result *DistillResult, loss DistillLoss, epoch int) DistillCheckpointMetadata {
	cfg = normalizeDistillConfig(cfg)
	meta := DistillCheckpointMetadata{
		Version:     DistillCheckpointMetadataVersion,
		Path:        path,
		ResumePath:  cfg.ResumePath,
		Epoch:       epoch,
		Temperature: cfg.Temperature,
		LossKind:    cfg.Loss,
		Batch:       cfg.Batch,
	}
	if result != nil {
		meta.Step = result.Metrics.Steps
		meta.Samples = result.Metrics.Samples
		meta.Tokens = result.Metrics.Tokens
		meta.Teacher = result.Teacher
		meta.Student = result.Student
		meta.TeacherCacheHits = result.Metrics.TeacherCacheHits
		meta.TeacherCacheMisses = result.Metrics.TeacherCacheMisses
	}
	meta.Loss = loss.Value
	meta.KL = loss.KL
	meta.SoftCrossEntropy = loss.SoftCrossEntropy
	meta.TeacherEntropy = loss.TeacherEntropy
	return meta
}

// SaveDistillCheckpointMetadata writes checkpoint metadata beside student
// artifacts. The marshal-and-write mechanics are the shared checkpoint
// engine (dappco.re/go/inference/checkpoint); only the Version/Path
// defaulting and the sidecar filename are distill's own.
func SaveDistillCheckpointMetadata(path string, meta DistillCheckpointMetadata) error {
	if path == "" {
		return errDistillCheckpointPath
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	if meta.Path == "" {
		meta.Path = path
	}
	return checkpoint.Save(distillCheckpointMetadataPath(path), meta)
}

// LoadDistillCheckpointMetadata reads checkpoint metadata written by SaveDistillCheckpointMetadata.
func LoadDistillCheckpointMetadata(path string) (*DistillCheckpointMetadata, error) {
	if path == "" {
		return nil, errDistillCheckpointPath
	}
	meta, err := checkpoint.Load[DistillCheckpointMetadata](distillCheckpointMetadataPath(path))
	if err != nil {
		return nil, err
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return meta, nil
}

// loadDistillResumeMetadata reads checkpoint metadata for a resume path,
// using the shared checkpoint engine's soft-missing-file semantics: an
// absent sidecar returns (nil, nil) rather than an error, matching
// --resume's "start fresh" contract.
func loadDistillResumeMetadata(path string) (*DistillCheckpointMetadata, error) {
	meta, err := checkpoint.LoadResume[DistillCheckpointMetadata](distillCheckpointMetadataPath(path))
	if err != nil || meta == nil {
		return meta, err
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return meta, nil
}

func distillCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "distill_checkpoint.json")
}

// formatDistillStepDir builds the "step-NNNNNN" checkpoint dirname —
// delegates to the shared checkpoint engine
// (dappco.re/go/inference/checkpoint) so distill's copy of the zero-pad
// logic cannot drift from grpo/train's.
func formatDistillStepDir(step int) string {
	return checkpoint.FormatStepDir(step)
}
