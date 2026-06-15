// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

func TestDistillCheckpointMetadataErrors_Bad(t *testing.T) {
	if err := SaveDistillCheckpointMetadata("", DistillCheckpointMetadata{}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadDistillCheckpointMetadata(""); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadDistillCheckpointMetadata(core.PathJoin(t.TempDir(), "absent")); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(missing file) error = nil")
	}
	dir := t.TempDir()
	writeModelPackFile(t, distillCheckpointMetadataPath(dir), "{")
	if _, err := LoadDistillCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(invalid JSON) error = nil")
	}
	if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return nil, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return nil, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{ResumePath: dir}); err == nil {
		t.Fatal("RunKnowledgeDistillation(invalid resume metadata) error = nil")
	}
}

func TestSaveDistillCheckpointMetadata_UnwritablePath_Bad(t *testing.T) {
	// A metadata dir whose parent is a regular file cannot be created,
	// so the MkdirAll arm of SaveDistillCheckpointMetadata must error.
	fileAsParent := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, fileAsParent, "x")
	target := core.PathJoin(fileAsParent, "child")
	if err := SaveDistillCheckpointMetadata(target, DistillCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(file-as-parent) error = nil")
	}
}

func TestFormatDistillStepDir_LargeStepSkipsPad_Good(t *testing.T) {
	// Steps below 100000 are zero-padded to six digits; steps at or above
	// 100000 already exceed the pad width, so the padding block is skipped
	// and the natural digits are used verbatim.
	if got := formatDistillStepDir(42); got != "step-000042" {
		t.Fatalf("formatDistillStepDir(42) = %q, want step-000042", got)
	}
	if got := formatDistillStepDir(123456); got != "step-123456" {
		t.Fatalf("formatDistillStepDir(123456) = %q, want unpadded step-123456", got)
	}
}

func TestLoadDistillCheckpointMetadata_VersionDefaulted_Good(t *testing.T) {
	// Metadata written without a version field loads with the current
	// version stamped in (the zero-version default arm of Load).
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
