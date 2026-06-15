// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

func TestSFTAdapterArtifactMetadata_Good(t *testing.T) {
	result := &SFTResult{Steps: 3, Samples: 5, LastLoss: 0.25}
	cfg := normalizeSFTConfig(SFTConfig{
		SavePath:                  core.PathJoin(t.TempDir(), "adapter"),
		BatchSize:                 2,
		GradientAccumulationSteps: 4,
		LearningRate:              1e-4,
		EvalTemperature:           0.25,
		LoRA: spine.LoRAConfig{
			Rank:                 8,
			Alpha:                16,
			TargetKeys:           []string{"q_proj"},
			AllowExtendedTargets: true,
		},
	})

	meta := NewSFTArtifactMetadata(cfg.SavePath, "gemma4", cfg, result)
	if meta.Path != cfg.SavePath || meta.Step != 3 || meta.Samples != 5 {
		t.Fatalf("artifact metadata = %+v, want final adapter state", meta)
	}
	if meta.GradientAccumulationSteps != 4 || meta.EvalTemperature != 0.25 || meta.LoRA.Rank != 8 || !meta.LoRA.AllowExtendedTargets || meta.Model != "gemma4" {
		t.Fatalf("artifact metadata = %+v, want config attached", meta)
	}
}

func TestSFTAdamWConfig_UsesExplicitOptimizer_Bad(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{
		AdamW: metal.AdamWConfig{
			LearningRate:   3e-4,
			Beta1:          0.85,
			Beta2:          0.98,
			WeightDecay:    0,
			WeightDecaySet: true,
			PackedState:    false,
			PackedStateSet: true,
		},
	})

	adam := SFTAdamWConfig(cfg)
	if adam.LearningRate != 3e-4 || adam.Beta1 != 0.85 || adam.Beta2 != 0.98 || adam.WeightDecay != 0 || adam.PackedState {
		t.Fatalf("adam = %+v, want explicit optimizer config", adam)
	}
	meta := sftAdamWMetadata(adam)
	if meta.PackedState {
		t.Fatalf("adam metadata = %+v, want explicit packed-state setting", meta)
	}
}

// TestSFT_CheckpointMetadataRoundTrip_Good writes metadata beside an adapter
// path and reads it back, asserting the durable fields survive the JSON round
// trip. Drives sftCheckpointMetadataPath + sftResultError transitively. Uses
// t.TempDir() — no model, no network.
func TestSFT_CheckpointMetadataRoundTrip_Good(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	result := &SFTResult{Steps: 12, OptimizerSteps: 6, Epochs: 1, Samples: 48, LastLoss: 0.22}
	cfg := normalizeSFTConfig(SFTConfig{
		BatchSize:                 4,
		GradientAccumulationSteps: 2,
		LearningRate:              1e-4,
		MaxSeqLen:                 512,
	})

	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", cfg, result, 1)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}

	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("version = %d, want %d", loaded.Version, SFTCheckpointMetadataVersion)
	}
	if loaded.Model != "gemma4" || loaded.Step != 12 || loaded.OptimizerStep != 6 || loaded.Epoch != 1 || loaded.Samples != 48 {
		t.Fatalf("loaded core fields = %+v, want round-tripped run state", loaded)
	}
	if loaded.BatchSize != 4 || loaded.GradientAccumulationSteps != 2 || loaded.EffectiveBatchSize != 8 || loaded.MaxSeqLen != 512 {
		t.Fatalf("loaded config fields = %+v, want round-tripped config", loaded)
	}
	if loaded.Loss != 0.22 || loaded.LearningRate != 1e-4 {
		t.Fatalf("loaded scalars = loss %v lr %v, want 0.22 / 1e-4", loaded.Loss, loaded.LearningRate)
	}
}

// TestSFT_CheckpointMetadata_EmptyPathAndMissingFile_Bad asserts the loud
// failure modes: an empty path on either side, and a Load against a directory
// with no sidecar (a real read failure surfaced through sftResultError).
func TestSFT_CheckpointMetadata_EmptyPathAndMissingFile_Bad(t *testing.T) {
	if err := SaveSFTCheckpointMetadata("", SFTCheckpointMetadata{}); err == nil {
		t.Fatal("SaveSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
	if _, err := LoadSFTCheckpointMetadata(""); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
	// Load against a fresh empty dir → the sidecar does not exist → error.
	if _, err := LoadSFTCheckpointMetadata(core.PathJoin(t.TempDir(), "nope")); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(missing) error = nil, want read failure")
	}
}

// TestSFT_ApplySFTResumeMetadata_Good attaches resume metadata from a real
// saved checkpoint and asserts it lands on the result. Also covers the
// no-resume-path no-op and the nil-result rejection.
func TestSFT_ApplySFTResumeMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	resumePath := core.PathJoin(dir, "prev.safetensors")
	prev := NewSFTCheckpointMetadata(resumePath, "gemma4", normalizeSFTConfig(SFTConfig{BatchSize: 2}), &SFTResult{Steps: 7}, 1)
	if err := SaveSFTCheckpointMetadata(resumePath, prev); err != nil {
		t.Fatalf("seed SaveSFTCheckpointMetadata() error = %v", err)
	}

	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: resumePath}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata() error = %v", err)
	}
	if result.ResumePath != resumePath {
		t.Fatalf("ResumePath = %q, want %q", result.ResumePath, resumePath)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 7 || result.ResumedFrom.Model != "gemma4" {
		t.Fatalf("ResumedFrom = %+v, want the saved checkpoint", result.ResumedFrom)
	}

	// No resume path → no-op, no error, nothing attached.
	clean := &SFTResult{}
	if err := ApplySFTResumeMetadata(clean, SFTConfig{}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata(no path) error = %v", err)
	}
	if clean.ResumedFrom != nil || clean.ResumePath != "" {
		t.Fatalf("no-resume result mutated = %+v", clean)
	}
}

// TestSFT_ApplySFTResumeMetadata_NilResultAndMissing_Bad covers the nil-result
// rejection and the missing-sidecar tolerance: loadSFTResumeMetadata treats a
// non-existent resume sidecar as "nothing to resume" (nil, nil), not an error.
func TestSFT_ApplySFTResumeMetadata_NilResultAndMissing_Bad(t *testing.T) {
	if err := ApplySFTResumeMetadata(nil, SFTConfig{ResumePath: "x"}); err == nil {
		t.Fatal("ApplySFTResumeMetadata(nil result) error = nil, want rejection")
	}
	// Resume path set but no sidecar on disk → tolerated as no-op.
	result := &SFTResult{}
	missing := core.PathJoin(t.TempDir(), "absent.safetensors")
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: missing}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata(missing sidecar) error = %v, want tolerated no-op", err)
	}
	if result.ResumedFrom != nil {
		t.Fatalf("ResumedFrom = %+v, want nil (no sidecar to resume from)", result.ResumedFrom)
	}
}

// A malformed sidecar is a loud parse failure on both the checkpoint-load and
// the resume-load paths — corrupt metadata must never read as a clean resume.
func TestSFT_LoadMetadata_MalformedJSON_Bad(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	sidecar := core.PathJoin(dir, "sft_checkpoint.json")
	if w := core.WriteFile(sidecar, []byte("{not valid json"), 0o600); !w.OK {
		t.Fatalf("seed malformed sidecar: %v", w.Value)
	}
	if _, err := LoadSFTCheckpointMetadata(adapterPath); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(malformed) error = nil, want parse failure")
	}
	// loadSFTResumeMetadata distinguishes not-exist (tolerated) from a parse
	// error (loud): a present-but-corrupt sidecar is the loud case.
	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: adapterPath}); err == nil {
		t.Fatal("ApplySFTResumeMetadata(malformed) error = nil, want parse failure")
	}
}

// A sidecar written without a version reads back with the current version
// backfilled — older checkpoints stay loadable, tagged to the live schema.
func TestSFT_LoadCheckpointMetadata_VersionBackfill_Ugly(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	sidecar := core.PathJoin(dir, "sft_checkpoint.json")
	// version omitted → JSON zero value 0 on load.
	if w := core.WriteFile(sidecar, []byte(`{"model":"gemma4","step":3}`), 0o600); !w.OK {
		t.Fatalf("seed version-0 sidecar: %v", w.Value)
	}
	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("version = %d, want backfilled %d", loaded.Version, SFTCheckpointMetadataVersion)
	}
	if loaded.Model != "gemma4" || loaded.Step != 3 {
		t.Fatalf("loaded = %+v, want the seeded fields preserved", loaded)
	}
}

// --- SFTEffectiveBatchSize / newSFTBatchBuilder — the <=0 defaults ---

// TestSFT_StepName_Good asserts the zero-padded names matching
// fmt.Sprintf("step-%06d", step) across the padded range, and
// TestSFT_StepName_OverflowAndZero_Ugly the boundaries the padding branch
// guards (0, exactly 100000, and beyond — where padTo no longer applies).
func TestSFT_StepName_Good(t *testing.T) {
	cases := map[int]string{
		1:    "step-000001",
		42:   "step-000042",
		999:  "step-000999",
		1234: "step-001234",
	}
	for step, want := range cases {
		if got := sftStepName(step); got != want {
			t.Fatalf("sftStepName(%d) = %q, want %q", step, got, want)
		}
	}
}

func TestSFT_StepName_OverflowAndZero_Ugly(t *testing.T) {
	if got := sftStepName(0); got != "step-000000" {
		t.Fatalf("sftStepName(0) = %q, want step-000000", got)
	}
	// 99999 is the last value inside the zero-pad branch (step < 100000).
	if got := sftStepName(99999); got != "step-099999" {
		t.Fatalf("sftStepName(99999) = %q, want step-099999", got)
	}
	// 100000 and above print without leading pad — the width is the digit count.
	if got := sftStepName(100000); got != "step-100000" {
		t.Fatalf("sftStepName(100000) = %q, want step-100000", got)
	}
	if got := sftStepName(1234567); got != "step-1234567" {
		t.Fatalf("sftStepName(1234567) = %q, want step-1234567", got)
	}
}

// --- runSFTEvaluations — the capture-first + score-cascade wiring ---
//
// The eval pass uses the synthetic sftTestModel (seeded text, no weights), so
// the capture sidecar, the score cascade, and the probe sink all exercise
// without Metal. AX-11 governs perf benchmarking, not functional fakes.
