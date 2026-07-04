// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

// TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Good covers the happy
// metadata-construction path: a config + update produce a sidecar stamped
// experimental at the current version, with the group size normalised and
// the step/epoch carried from the update.
func TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Good(t *testing.T) {
	result := &GRPOResult{
		Metrics: GRPOMetrics{Samples: 5, Rollouts: 20},
		Policy:  ModelInfo{Architecture: "qwen3", VocabSize: 32},
	}
	meta := NewGRPOCheckpointMetadata("runs/step-000010", GRPOConfig{
		GroupSize:     8,
		KLCoefficient: 0.1,
		LearningRate:  1e-4,
		ResumePath:    "runs/prev",
	}, result, GRPOUpdate{Step: 10, Epoch: 2, RewardMean: 0.7, RewardStd: 0.2, KLMean: 0.05, Loss: 1.5})
	if meta.Version != GRPOCheckpointMetadataVersion || !meta.Experimental {
		t.Fatalf("meta = %+v, want experimental at current version", meta)
	}
	if meta.Step != 10 || meta.Epoch != 2 || meta.GroupSize != 8 || meta.Path != "runs/step-000010" {
		t.Fatalf("meta = %+v, want update step/epoch + config group size", meta)
	}
	if meta.Samples != 5 || meta.Rollouts != 20 || meta.Policy.Architecture != "qwen3" {
		t.Fatalf("meta = %+v, want result metrics + policy carried through", meta)
	}
	if meta.RewardMean != 0.7 || meta.Loss != 1.5 || meta.LearningRate != 1e-4 || meta.ResumePath != "runs/prev" {
		t.Fatalf("meta = %+v, want update + config scalars carried", meta)
	}
}

// TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Bad covers the
// degenerate-config path: a zero/empty GRPOConfig must still normalise the
// group size to the default (4) rather than emitting a zero group size, so
// a checkpoint built from an unconfigured run is still well-formed.
func TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Bad(t *testing.T) {
	meta := NewGRPOCheckpointMetadata("p", GRPOConfig{}, nil, GRPOUpdate{})
	if meta.GroupSize != 4 {
		t.Fatalf("group size = %d, want normalised default 4 from empty config", meta.GroupSize)
	}
	if meta.Version != GRPOCheckpointMetadataVersion || !meta.Experimental {
		t.Fatalf("meta = %+v, want version + experimental stamped even for empty input", meta)
	}
	// A zero group size on the update side does not leak through — the
	// config default wins.
	if meta.Step != 0 || meta.Epoch != 0 {
		t.Fatalf("meta = %+v, want zero step/epoch for an empty update", meta)
	}
}

// TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Ugly covers the nil-result
// branch: when result is nil the constructor must skip the result-derived
// fields (Samples, Rollouts, Policy) without panicking and still produce a
// valid sidecar from the config + update alone.
func TestGrpoCheckpoint_NewGRPOCheckpointMetadata_Ugly(t *testing.T) {
	meta := NewGRPOCheckpointMetadata("runs/x", GRPOConfig{GroupSize: 2}, nil, GRPOUpdate{Step: 3, Epoch: 1})
	if meta.Samples != 0 || meta.Rollouts != 0 {
		t.Fatalf("meta = %+v, want zero result-derived counters for nil result", meta)
	}
	if meta.Policy.Architecture != "" || meta.Policy.VocabSize != 0 {
		t.Fatalf("meta.Policy = %+v, want zero policy for nil result", meta.Policy)
	}
	if meta.Step != 3 || meta.GroupSize != 2 || !meta.Experimental {
		t.Fatalf("meta = %+v, want update + config fields still populated", meta)
	}
}

// TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Good covers the happy
// round-trip: Save writes the sidecar (backfilling Version + Experimental
// even when the caller left them zero) and Load reads the substantive
// fields back faithfully.
func TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Good(t *testing.T) {
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

// TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Bad covers the rejected
// input: an empty path is a caller contract violation that must error
// before any filesystem write rather than writing a sidecar to a bogus
// location.
func TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Bad(t *testing.T) {
	if err := SaveGRPOCheckpointMetadata("", GRPOCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveGRPOCheckpointMetadata(empty path) error = nil, want path-required rejection")
	}
}

// TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Ugly covers the
// backfill-on-save edge: a metadata value left almost entirely zero (no
// Version, no Path, Experimental false) is still written as a valid
// experimental sidecar — Save stamps Version, forces Experimental true,
// and fills Path from the save target.
func TestGrpoCheckpoint_SaveGRPOCheckpointMetadata_Ugly(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "bare")
	if err := SaveGRPOCheckpointMetadata(dir, GRPOCheckpointMetadata{}); err != nil {
		t.Fatalf("SaveGRPOCheckpointMetadata(zero meta) error = %v", err)
	}
	loaded, err := LoadGRPOCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != GRPOCheckpointMetadataVersion || !loaded.Experimental {
		t.Fatalf("loaded = %+v, want Save to stamp version + experimental on zero meta", loaded)
	}
	if loaded.Path != dir {
		t.Fatalf("loaded.Path = %q, want Save to backfill the save target %q", loaded.Path, dir)
	}
}

// TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Good covers the
// version-backfill read path in both the public Load and the internal
// resume loader: a sidecar written with version 0 reads back stamped at
// the current metadata version, and the resume path surfaces the same
// metadata through a full training run.
func TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Good(t *testing.T) {
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

// TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Bad covers the rejected
// and corrupt read paths: an empty path errors before any read, a missing
// sidecar surfaces the read failure, and a syntactically invalid JSON
// sidecar fails to parse rather than returning a half-filled struct.
func TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Bad(t *testing.T) {
	if _, err := LoadGRPOCheckpointMetadata(""); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(empty path) error = nil, want path-required rejection")
	}
	// A directory with no sidecar at all surfaces the underlying read error.
	if _, err := LoadGRPOCheckpointMetadata(t.TempDir()); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(missing sidecar) error = nil, want read failure")
	}
	// A syntactically broken sidecar must fail to parse.
	dir := t.TempDir()
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir), "{")
	if _, err := LoadGRPOCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(invalid JSON) error = nil, want parse failure")
	}
}

// TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Ugly covers the
// awkward-but-valid read: a sidecar carrying an explicit version far ahead
// of the current one is preserved verbatim (the backfill only fires for
// version 0), so a newer-format checkpoint is not silently downgraded.
func TestGrpoCheckpoint_LoadGRPOCheckpointMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	// A non-zero version is preserved as-is; only version 0 triggers the
	// backfill to the current version.
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir), `{"version":99,"step":1,"experimental":true}`)
	loaded, err := LoadGRPOCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != 99 {
		t.Fatalf("loaded.Version = %d, want the explicit 99 preserved (no downgrade)", loaded.Version)
	}
	// An empty JSON object loads as a zero-but-version-stamped struct.
	dir2 := t.TempDir()
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir2), `{}`)
	empty, err := LoadGRPOCheckpointMetadata(dir2)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata(empty object) error = %v", err)
	}
	if empty.Version != GRPOCheckpointMetadataVersion || empty.Step != 0 {
		t.Fatalf("loaded = %+v, want version-0 object backfilled to current with zero step", empty)
	}
}

// TestGrpoCheckpoint_grpoStepName_Good pins the checkpoint step directory
// naming to exact strings. Below 1e5 the name is zero-padded to six
// digits; at and above 1e6 the natural digit count is preserved (no
// truncation). This is the path that lays out checkpoint directories on
// disk, so the exact name is load-bearing.
func TestGrpoCheckpoint_grpoStepName_Good(t *testing.T) {
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
