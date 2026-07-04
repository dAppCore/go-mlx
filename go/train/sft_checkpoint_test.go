// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// --- NewSFTCheckpointMetadata ---

// TestSftCheckpoint_NewSFTCheckpointMetadata_Good captures a checkpoint's
// reproducible state: the run scalars, the config, and the epoch all land in
// the returned metadata with the current schema version.
func TestSftCheckpoint_NewSFTCheckpointMetadata_Good(t *testing.T) {
	result := &SFTResult{Steps: 12, OptimizerSteps: 6, Epochs: 1, Samples: 48, LastLoss: 0.22}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, MaxSeqLen: 512})
	meta := NewSFTCheckpointMetadata("ckpt.safetensors", "gemma4", cfg, result, 1)
	if meta.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("version = %d, want %d", meta.Version, SFTCheckpointMetadataVersion)
	}
	if meta.Path != "ckpt.safetensors" || meta.Model != "gemma4" || meta.Step != 12 || meta.OptimizerStep != 6 || meta.Epoch != 1 {
		t.Fatalf("metadata core = %+v, want run state attached", meta)
	}
	if meta.BatchSize != 4 || meta.GradientAccumulationSteps != 2 || meta.EffectiveBatchSize != 8 || meta.MaxSeqLen != 512 {
		t.Fatalf("metadata config = %+v, want config attached", meta)
	}
}

// TestSftCheckpoint_NewSFTCheckpointMetadata_Bad asserts a nil result does not
// panic: the metadata is still well-formed, with the run-derived scalars zeroed
// and the config-derived fields present.
func TestSftCheckpoint_NewSFTCheckpointMetadata_Bad(t *testing.T) {
	meta := NewSFTCheckpointMetadata("ckpt", "m", SFTConfig{BatchSize: 2}, nil, 0)
	if meta.Step != 0 || meta.OptimizerStep != 0 || meta.Samples != 0 || meta.Loss != 0 {
		t.Fatalf("nil-result metadata run scalars = %+v, want zeroes", meta)
	}
	if meta.Version != SFTCheckpointMetadataVersion || meta.BatchSize != 2 {
		t.Fatalf("nil-result metadata config = %+v, want version + batch size", meta)
	}
}

// TestSftCheckpoint_NewSFTCheckpointMetadata_Ugly asserts the OptimizerStep
// fallback: a result with Steps but no explicit OptimizerSteps reports the step
// count as the optimizer step (the degenerate equal-clock case).
func TestSftCheckpoint_NewSFTCheckpointMetadata_Ugly(t *testing.T) {
	meta := NewSFTCheckpointMetadata("ckpt", "m", SFTConfig{}, &SFTResult{Steps: 9}, 2)
	if meta.Step != 9 || meta.OptimizerStep != 9 {
		t.Fatalf("metadata step/optimizer = %d/%d, want 9/9 (fallback to Steps)", meta.Step, meta.OptimizerStep)
	}
	if meta.Epoch != 2 {
		t.Fatalf("metadata epoch = %d, want 2", meta.Epoch)
	}
}

// --- NewSFTArtifactMetadata ---

// TestSftCheckpoint_NewSFTArtifactMetadata_Good captures a final adapter's
// reproducible state: the run, the config (grad-accum, eval temperature, LoRA
// identity), and the model name all attach.
func TestSftCheckpoint_NewSFTArtifactMetadata_Good(t *testing.T) {
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

// TestSftCheckpoint_NewSFTArtifactMetadata_Bad asserts a nil result is tolerated:
// the artifact metadata derives its epoch from the (absent) run as 0 and stays
// well-formed rather than panicking.
func TestSftCheckpoint_NewSFTArtifactMetadata_Bad(t *testing.T) {
	meta := NewSFTArtifactMetadata("adapter", "m", SFTConfig{}, nil)
	if meta.Epoch != 0 || meta.Step != 0 {
		t.Fatalf("nil-result artifact = %+v, want zeroed epoch/step", meta)
	}
	if meta.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("nil-result artifact version = %d, want %d", meta.Version, SFTCheckpointMetadataVersion)
	}
}

// TestSftCheckpoint_NewSFTArtifactMetadata_Ugly asserts the artifact epoch
// tracks the run's Epochs count (the artifact is the end-of-run snapshot), not a
// caller-supplied epoch index.
func TestSftCheckpoint_NewSFTArtifactMetadata_Ugly(t *testing.T) {
	meta := NewSFTArtifactMetadata("adapter", "m", SFTConfig{}, &SFTResult{Epochs: 5, Steps: 100})
	if meta.Epoch != 5 {
		t.Fatalf("artifact epoch = %d, want 5 (final run epoch count)", meta.Epoch)
	}
	if meta.Step != 100 {
		t.Fatalf("artifact step = %d, want 100", meta.Step)
	}
}

// --- SaveSFTCheckpointMetadata ---

// TestSftCheckpoint_SaveSFTCheckpointMetadata_Good writes metadata beside an
// adapter path and confirms it reads back — the durable JSON sidecar.
func TestSftCheckpoint_SaveSFTCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", normalizeSFTConfig(SFTConfig{BatchSize: 4}), &SFTResult{Steps: 12}, 1)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		t.Fatalf("read-back LoadSFTCheckpointMetadata() error = %v", err)
	}
	if loaded.Model != "gemma4" || loaded.Step != 12 {
		t.Fatalf("read-back = %+v, want saved fields", loaded)
	}
}

// TestSftCheckpoint_SaveSFTCheckpointMetadata_Bad asserts the empty-path
// rejection — Save needs a path to derive the sidecar location.
func TestSftCheckpoint_SaveSFTCheckpointMetadata_Bad(t *testing.T) {
	if err := SaveSFTCheckpointMetadata("", SFTCheckpointMetadata{}); err == nil {
		t.Fatal("SaveSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
}

// TestSftCheckpoint_SaveSFTCheckpointMetadata_Ugly asserts Save backfills the
// zero-valued Version and Path: a metadata struct missing both is still written
// with the current schema version and the call's path.
func TestSftCheckpoint_SaveSFTCheckpointMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	// Version 0 and Path "" → both backfilled by Save.
	if err := SaveSFTCheckpointMetadata(adapterPath, SFTCheckpointMetadata{Model: "m", Step: 2}); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("version = %d, want backfilled %d", loaded.Version, SFTCheckpointMetadataVersion)
	}
	if loaded.Path != adapterPath {
		t.Fatalf("path = %q, want backfilled %q", loaded.Path, adapterPath)
	}
}

// --- LoadSFTCheckpointMetadata ---

// TestSftCheckpoint_LoadSFTCheckpointMetadata_Good writes then reads metadata
// and asserts the durable fields survive the JSON round trip.
func TestSftCheckpoint_LoadSFTCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	result := &SFTResult{Steps: 12, OptimizerSteps: 6, Epochs: 1, Samples: 48, LastLoss: 0.22}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, LearningRate: 1e-4, MaxSeqLen: 512})

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

// TestSftCheckpoint_LoadSFTCheckpointMetadata_Bad asserts the loud failures: an
// empty path, a directory with no sidecar (a real read failure), and a
// present-but-malformed sidecar (a parse failure) all error.
func TestSftCheckpoint_LoadSFTCheckpointMetadata_Bad(t *testing.T) {
	if _, err := LoadSFTCheckpointMetadata(""); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
	if _, err := LoadSFTCheckpointMetadata(core.PathJoin(t.TempDir(), "nope")); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(missing) error = nil, want read failure")
	}
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	sidecar := core.PathJoin(dir, "sft_checkpoint.json")
	if w := core.WriteFile(sidecar, []byte("{not valid json"), 0o600); !w.OK {
		t.Fatalf("seed malformed sidecar: %v", w.Value)
	}
	if _, err := LoadSFTCheckpointMetadata(adapterPath); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(malformed) error = nil, want parse failure")
	}
}

// TestSftCheckpoint_LoadSFTCheckpointMetadata_Ugly asserts the version backfill:
// a sidecar written without a version reads back tagged to the live schema, so
// older checkpoints stay loadable.
func TestSftCheckpoint_LoadSFTCheckpointMetadata_Ugly(t *testing.T) {
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

// --- ApplySFTResumeMetadata ---

// TestSftCheckpoint_ApplySFTResumeMetadata_Good attaches resume metadata from a
// real saved checkpoint and asserts it lands on the result, plus the
// no-resume-path no-op leaves a clean result untouched.
func TestSftCheckpoint_ApplySFTResumeMetadata_Good(t *testing.T) {
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

// TestSftCheckpoint_ApplySFTResumeMetadata_Bad asserts the nil-result rejection
// and the malformed-sidecar loud failure — a present-but-corrupt resume sidecar
// must never read as a clean resume.
func TestSftCheckpoint_ApplySFTResumeMetadata_Bad(t *testing.T) {
	if err := ApplySFTResumeMetadata(nil, SFTConfig{ResumePath: "x"}); err == nil {
		t.Fatal("ApplySFTResumeMetadata(nil result) error = nil, want rejection")
	}
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	sidecar := core.PathJoin(dir, "sft_checkpoint.json")
	if w := core.WriteFile(sidecar, []byte("{not valid json"), 0o600); !w.OK {
		t.Fatalf("seed malformed sidecar: %v", w.Value)
	}
	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: adapterPath}); err == nil {
		t.Fatal("ApplySFTResumeMetadata(malformed) error = nil, want parse failure")
	}
}

// TestSftCheckpoint_ApplySFTResumeMetadata_Ugly asserts the missing-sidecar
// tolerance: a resume path set but no sidecar on disk is treated as "nothing to
// resume" (nil, nil), not an error — the ResumePath is still recorded.
func TestSftCheckpoint_ApplySFTResumeMetadata_Ugly(t *testing.T) {
	result := &SFTResult{}
	missing := core.PathJoin(t.TempDir(), "absent.safetensors")
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: missing}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata(missing sidecar) error = %v, want tolerated no-op", err)
	}
	if result.ResumedFrom != nil {
		t.Fatalf("ResumedFrom = %+v, want nil (no sidecar to resume from)", result.ResumedFrom)
	}
	if result.ResumePath != missing {
		t.Fatalf("ResumePath = %q, want %q (recorded even without a sidecar)", result.ResumePath, missing)
	}
}

// --- SFTAdamWConfig ---

// TestSftCheckpoint_SFTAdamWConfig_Good asserts the default optimizer config is
// produced when nothing is overridden — the metal AdamW defaults flow through.
func TestSftCheckpoint_SFTAdamWConfig_Good(t *testing.T) {
	adam := SFTAdamWConfig(SFTConfig{})
	def := metal.DefaultAdamWConfig()
	if adam.Beta1 != def.Beta1 || adam.Beta2 != def.Beta2 || adam.Eps != def.Eps {
		t.Fatalf("adam betas/eps = %+v, want metal defaults %+v", adam, def)
	}
	// With no learning rate set, the normalised SFTConfig default (1e-5) drives it.
	if adam.LearningRate != 1e-5 {
		t.Fatalf("adam LR = %v, want 1e-5 (SFTConfig default)", adam.LearningRate)
	}
}

// TestSftCheckpoint_SFTAdamWConfig_Bad asserts the explicit-optimizer overrides
// win: each *Set flag forces its value through even when that value is the zero
// (e.g. WeightDecay 0 with WeightDecaySet, PackedState false with PackedStateSet).
func TestSftCheckpoint_SFTAdamWConfig_Bad(t *testing.T) {
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

// TestSftCheckpoint_SFTAdamWConfig_Ugly asserts the SFTConfig-level LearningRate
// takes final precedence over an AdamW-level learning rate — the outer scalar
// is the last word.
func TestSftCheckpoint_SFTAdamWConfig_Ugly(t *testing.T) {
	cfg := SFTConfig{
		LearningRate: 7e-4,
		AdamW:        metal.AdamWConfig{LearningRate: 1e-4, LearningRateSet: true},
	}
	adam := SFTAdamWConfig(cfg)
	if adam.LearningRate != 7e-4 {
		t.Fatalf("adam LR = %v, want 7e-4 (SFTConfig.LearningRate wins)", adam.LearningRate)
	}
}

// --- sftStepName — the step-NNNNNN checkpoint directory name (private) ---

// TestSftCheckpoint_StepName asserts the zero-padded names matching
// fmt.Sprintf("step-%06d", step) across the padded range and the boundaries the
// padding branch guards (0, exactly 100000, and beyond).
func TestSftCheckpoint_StepName(t *testing.T) {
	cases := map[int]string{
		0:       "step-000000",
		1:       "step-000001",
		42:      "step-000042",
		999:     "step-000999",
		1234:    "step-001234",
		99999:   "step-099999", // last value inside the zero-pad branch
		100000:  "step-100000", // padding no longer applies
		1234567: "step-1234567",
	}
	for step, want := range cases {
		if got := sftStepName(step); got != want {
			t.Fatalf("sftStepName(%d) = %q, want %q", step, got, want)
		}
	}
}
