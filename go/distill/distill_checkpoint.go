// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"strconv"

	core "dappco.re/go"
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

// SaveDistillCheckpointMetadata writes checkpoint metadata beside student artifacts.
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
	metadataPath := distillCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("DistillCheckpointMetadata.Save", "ensure metadata dir", distillResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("DistillCheckpointMetadata.Save", "marshal metadata", distillResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("DistillCheckpointMetadata.Save", "write metadata", distillResultError(result))
	}
	return nil
}

// LoadDistillCheckpointMetadata reads checkpoint metadata written by SaveDistillCheckpointMetadata.
func LoadDistillCheckpointMetadata(path string) (*DistillCheckpointMetadata, error) {
	if path == "" {
		return nil, errDistillCheckpointPath
	}
	read := core.ReadFile(distillCheckpointMetadataPath(path))
	if !read.OK {
		return nil, distillResultError(read)
	}
	var meta DistillCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadDistillCheckpointMetadata", "parse metadata", distillResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return &meta, nil
}

func loadDistillResumeMetadata(path string) (*DistillCheckpointMetadata, error) {
	read := core.ReadFile(distillCheckpointMetadataPath(path))
	if !read.OK {
		err := distillResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta DistillCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadDistillResumeMetadata", "parse metadata", distillResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return &meta, nil
}

func distillCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "distill_checkpoint.json")
}

// formatDistillStepDir builds the "step-NNNNNN" checkpoint dirname using
// strconv.AppendInt with explicit zero padding, avoiding fmt's reflection
// path on the per-checkpoint hot loop. Digit count is computed in place
// instead of via a throwaway strconv.AppendInt(nil, ...) so the function
// allocates exactly once — the returned string itself.
func formatDistillStepDir(step int) string {
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
