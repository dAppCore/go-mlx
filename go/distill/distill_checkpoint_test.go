// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"testing"

	core "dappco.re/go"
)

// TestDistillCheckpoint_NewDistillCheckpointMetadata_Good shows
// NewDistillCheckpointMetadata captures reproducible distillation identity
// from a config, result and loss — version, path, temperature, loss kind,
// step counters and teacher/student model identity all land on the meta.
func TestDistillCheckpoint_NewDistillCheckpointMetadata_Good(t *testing.T) {
	result := &DistillResult{Teacher: ModelInfo{Architecture: "qwen3"}, Student: ModelInfo{Architecture: "qwen3"}}
	result.Metrics.Steps = 5
	result.Metrics.Samples = 12
	result.Metrics.Tokens = 40
	result.Metrics.TeacherCacheHits = 3
	result.Metrics.TeacherCacheMisses = 2

	meta := NewDistillCheckpointMetadata(
		"/tmp/ckpt/step-000005",
		DistillConfig{Temperature: 2, Loss: DistillLossKL, ResumePath: "/tmp/resume"},
		result,
		DistillLoss{Value: 0.5, KL: 0.5, SoftCrossEntropy: 0.7, TeacherEntropy: 0.2},
		3,
	)
	if meta.Version != DistillCheckpointMetadataVersion {
		t.Fatalf("version = %d, want %d", meta.Version, DistillCheckpointMetadataVersion)
	}
	if meta.Path != "/tmp/ckpt/step-000005" || meta.ResumePath != "/tmp/resume" {
		t.Fatalf("paths = %q / %q, want carried from arg + config", meta.Path, meta.ResumePath)
	}
	if meta.Epoch != 3 || meta.Temperature != 2 || meta.LossKind != DistillLossKL {
		t.Fatalf("meta = %+v, want epoch 3, temp 2, KL kind", meta)
	}
	if meta.Step != 5 || meta.Samples != 12 || meta.Tokens != 40 {
		t.Fatalf("counters = %+v, want metrics carried through", meta)
	}
	if meta.TeacherCacheHits != 3 || meta.TeacherCacheMisses != 2 {
		t.Fatalf("cache counters = %d/%d, want 3/2", meta.TeacherCacheHits, meta.TeacherCacheMisses)
	}
	if meta.Loss != 0.5 || meta.KL != 0.5 || meta.SoftCrossEntropy != 0.7 || meta.TeacherEntropy != 0.2 {
		t.Fatalf("loss components = %+v, want carried from DistillLoss", meta)
	}
	if meta.Teacher.Architecture != "qwen3" || meta.Student.Architecture != "qwen3" {
		t.Fatalf("model identity = %+v / %+v, want qwen3", meta.Teacher, meta.Student)
	}
}

// TestDistillCheckpoint_NewDistillCheckpointMetadata_Bad shows the
// degenerate config is repaired rather than propagated: a zero temperature
// is normalised to the package default instead of being recorded as zero.
func TestDistillCheckpoint_NewDistillCheckpointMetadata_Bad(t *testing.T) {
	meta := NewDistillCheckpointMetadata("/tmp/ckpt", DistillConfig{Temperature: 0}, nil, DistillLoss{}, 1)
	if meta.Temperature <= 0 {
		t.Fatalf("temperature = %v, want zero config normalised to a positive default", meta.Temperature)
	}
}

// TestDistillCheckpoint_NewDistillCheckpointMetadata_Ugly feeds
// NewDistillCheckpointMetadata a nil result. It must skip the result-derived
// fields (leaving them zero) rather than dereferencing the nil pointer.
func TestDistillCheckpoint_NewDistillCheckpointMetadata_Ugly(t *testing.T) {
	meta := NewDistillCheckpointMetadata("/tmp/ckpt", DistillConfig{Temperature: 1, Loss: DistillLossKL}, nil, DistillLoss{Value: 9}, 2)
	if meta.Step != 0 || meta.Samples != 0 || meta.Tokens != 0 {
		t.Fatalf("counters = %+v, want zero under nil result", meta)
	}
	if meta.Epoch != 2 || meta.Loss != 9 {
		t.Fatalf("meta = %+v, want epoch + loss still recorded under nil result", meta)
	}
}

// TestDistillCheckpoint_SaveDistillCheckpointMetadata_Good writes metadata
// beside a checkpoint and reads it back, confirming the round-trip preserves
// the step and stamps the current metadata version.
func TestDistillCheckpoint_SaveDistillCheckpointMetadata_Good(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "step-000001")
	if err := SaveDistillCheckpointMetadata(dir, DistillCheckpointMetadata{Step: 1, Loss: 0.25}); err != nil {
		t.Fatalf("SaveDistillCheckpointMetadata() error = %v", err)
	}
	meta, err := LoadDistillCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadDistillCheckpointMetadata() error = %v", err)
	}
	if meta.Step != 1 || meta.Loss != 0.25 {
		t.Fatalf("meta = %+v, want round-tripped step 1 / loss 0.25", meta)
	}
	if meta.Version != DistillCheckpointMetadataVersion {
		t.Fatalf("version = %d, want %d stamped on save", meta.Version, DistillCheckpointMetadataVersion)
	}
}

// TestDistillCheckpoint_SaveDistillCheckpointMetadata_Bad shows an empty
// path is rejected outright — Save cannot write a metadata sidecar without
// a checkpoint directory.
func TestDistillCheckpoint_SaveDistillCheckpointMetadata_Bad(t *testing.T) {
	if err := SaveDistillCheckpointMetadata("", DistillCheckpointMetadata{}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(empty) error = nil")
	}
}

// TestDistillCheckpoint_SaveDistillCheckpointMetadata_Ugly drives the two
// distinct filesystem failure arms: a metadata dir whose parent is a
// regular file (MkdirAll arm), and a metadata file path that is itself an
// existing directory (WriteFile arm).
func TestDistillCheckpoint_SaveDistillCheckpointMetadata_Ugly(t *testing.T) {
	t.Run("parent_is_file", func(t *testing.T) {
		// A metadata dir whose parent is a regular file cannot be created,
		// so the MkdirAll arm of SaveDistillCheckpointMetadata must error.
		fileAsParent := core.PathJoin(t.TempDir(), "not-a-dir")
		writeModelPackFile(t, fileAsParent, "x")
		target := core.PathJoin(fileAsParent, "child")
		if err := SaveDistillCheckpointMetadata(target, DistillCheckpointMetadata{Step: 1}); err == nil {
			t.Fatal("SaveDistillCheckpointMetadata(file-as-parent) error = nil")
		}
	})

	t.Run("metadata_file_path_is_dir", func(t *testing.T) {
		// The metadata dir is created fine, but the metadata FILE path is
		// itself an existing directory, so the WriteFile arm of Save errors
		// (distinct from the MkdirAll arm).
		dir := t.TempDir()
		if result := core.MkdirAll(distillCheckpointMetadataPath(dir), 0o755); !result.OK {
			t.Fatalf("setup mkdir: %v", result.Value)
		}
		if err := SaveDistillCheckpointMetadata(dir, DistillCheckpointMetadata{Step: 1}); err == nil {
			t.Fatal("SaveDistillCheckpointMetadata(file-path-is-dir) error = nil")
		}
	})
}

// TestDistillCheckpoint_LoadDistillCheckpointMetadata_Good shows Load reads
// what Save wrote and back-fills a missing version: metadata persisted
// without a version field loads with the current version stamped in.
func TestDistillCheckpoint_LoadDistillCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, distillCheckpointMetadataPath(dir), `{"step":3}`)
	meta, err := LoadDistillCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadDistillCheckpointMetadata() error = %v", err)
	}
	if meta.Step != 3 || meta.Version != DistillCheckpointMetadataVersion {
		t.Fatalf("meta = %+v, want step 3 and defaulted version", meta)
	}
}

// TestDistillCheckpoint_LoadDistillCheckpointMetadata_Bad shows Load rejects
// an empty path and a path with no metadata sidecar (a missing file is an
// error from Load — the resume path handles missing specially, Load does not).
func TestDistillCheckpoint_LoadDistillCheckpointMetadata_Bad(t *testing.T) {
	if _, err := LoadDistillCheckpointMetadata(""); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadDistillCheckpointMetadata(core.PathJoin(t.TempDir(), "absent")); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(missing file) error = nil")
	}
}

// TestDistillCheckpoint_LoadDistillCheckpointMetadata_Ugly feeds Load a
// sidecar containing malformed JSON. It must surface a parse error rather
// than returning a half-populated metadata struct.
func TestDistillCheckpoint_LoadDistillCheckpointMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, distillCheckpointMetadataPath(dir), "{not json")
	if _, err := LoadDistillCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(invalid JSON) error = nil")
	}
}

// TestDistillCheckpoint_FormatDistillStepDir_Good covers the step-dir
// formatter: steps below 100000 are zero-padded to six digits; steps at or
// above 100000 already exceed the pad width and use natural digits verbatim.
func TestDistillCheckpoint_FormatDistillStepDir_Good(t *testing.T) {
	if got := formatDistillStepDir(42); got != "step-000042" {
		t.Fatalf("formatDistillStepDir(42) = %q, want step-000042", got)
	}
	if got := formatDistillStepDir(123456); got != "step-123456" {
		t.Fatalf("formatDistillStepDir(123456) = %q, want unpadded step-123456", got)
	}
}
