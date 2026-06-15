// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

func TestSaveLoadGRPOCheckpointMetadata_RoundTrip_Good(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "ckpt")
	meta := GRPOCheckpointMetadata{
		Step: 7, Epoch: 2, GroupSize: 4, RewardMean: 0.5, Loss: 1.25,
		Policy: ModelInfo{Architecture: "qwen3", VocabSize: 32},
	}
	if err := SaveGRPOCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveGRPOCheckpointMetadata() error = %v", err)
	}
	loaded, err := LoadGRPOCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
	}
	// Save backfills Version + Experimental even when the caller left them
	// zero, and the round-trip preserves the substantive fields.
	if loaded.Version != GRPOCheckpointMetadataVersion || !loaded.Experimental {
		t.Fatalf("loaded = %+v, want version + experimental backfilled", loaded)
	}
	if loaded.Step != 7 || loaded.GroupSize != 4 || loaded.Policy.Architecture != "qwen3" {
		t.Fatalf("loaded = %+v, want round-tripped fields", loaded)
	}
}

// TestGRPOStepName_ZeroPadAndOverflow_Good pins the checkpoint step
// directory naming to exact strings. Below 1e5 the name is zero-padded
// to six digits; at and above 1e6 the natural digit count is preserved
// (no truncation). This is the path that lays out checkpoint directories
// on disk, so the exact name is load-bearing.
func TestGRPOStepName_ZeroPadAndOverflow_Good(t *testing.T) {
	cases := []struct {
		step int
		want string
	}{
		{0, "step-000000"},
		{7, "step-000007"},
		{42, "step-000042"},
		{12345, "step-012345"},
		{100000, "step-100000"},
		{1234567, "step-1234567"},
	}
	for _, tc := range cases {
		if got := grpoStepName(tc.step); got != tc.want {
			t.Fatalf("grpoStepName(%d) = %q, want %q", tc.step, got, tc.want)
		}
	}
}

// TestLoadGRPOCheckpointMetadata_BackfillsVersion_Good covers the
// version-backfill branch in both the public Load and the internal
// resume loader: a sidecar written with version 0 reads back stamped at
// the current metadata version, and the resume path surfaces the same
// metadata through a full training run.
func TestLoadGRPOCheckpointMetadata_BackfillsVersion_Good(t *testing.T) {
	dir := t.TempDir()
	// Hand-write a sidecar with an explicit version 0 to drive the backfill.
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir), `{"version":0,"step":5,"group_size":2}`)

	loaded, err := LoadGRPOCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != GRPOCheckpointMetadataVersion || loaded.Step != 5 {
		t.Fatalf("loaded = %+v, want version backfilled to %d with step 5", loaded, GRPOCheckpointMetadataVersion)
	}

	// The resume loader (via RunGRPOReasoningTraining) backfills the same way.
	result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
		GroupSize:   1,
		ResumePath:  dir,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if err != nil {
		t.Fatalf("RunGRPOReasoningTraining(resume) error = %v", err)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Version != GRPOCheckpointMetadataVersion || result.ResumedFrom.Step != 5 {
		t.Fatalf("resumedFrom = %+v, want version-backfilled resume metadata at step 5", result.ResumedFrom)
	}
}
