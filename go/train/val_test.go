// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
)

// The validation lane end-to-end without Metal: armed with an injected
// loss function, the gate fires on cadence, every pass records a point,
// and the probe sink sees loss_type=val — the v0 schema's other curve.
func TestSFTValidation_CadenceAndRecording_Good(t *testing.T) {
	var events []probe.Event
	cfg := SFTConfig{ProbeSink: probe.SinkFunc(func(e probe.Event) { events = append(events, e) })}

	losses := []float64{2.0, 1.5, 1.0}
	calls := 0
	result := &SFTResult{}
	ArmSFTValidation(result, []SFTBatch{{}, {}}, 2, func(SFTBatch) (float64, bool) {
		v := losses[calls/2] // two batches per pass share one value
		calls++
		return v, true
	})

	// Baseline at step 0 — the curve starts before training moves anything.
	if err := RunSFTValidationPass(cfg, result); err != nil {
		t.Fatalf("baseline pass: %v", err)
	}
	// Steps 1..4 through the in-loop gate: passes at 2 and 4 only.
	for step := 1; step <= 4; step++ {
		result.Steps = step
		if err := maybeRunSFTValidation(cfg, result); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}

	if len(result.ValLosses) != 3 {
		t.Fatalf("val points = %d, want 3 (baseline + steps 2,4)", len(result.ValLosses))
	}
	wantSteps := []int{0, 2, 4}
	for i, p := range result.ValLosses {
		if p.Step != wantSteps[i] {
			t.Fatalf("point %d at step %d, want %d", i, p.Step, wantSteps[i])
		}
	}
	if result.LastValLoss != 1.0 {
		t.Fatalf("last val loss = %v, want 1.0", result.LastValLoss)
	}
	if result.ValLosses[0].Loss <= result.ValLosses[2].Loss {
		t.Fatalf("val curve did not descend: %v", result.ValLosses)
	}

	if len(events) != 3 {
		t.Fatalf("probe events = %d, want 3", len(events))
	}
	for _, e := range events {
		if e.Kind != probe.KindTraining || e.Training == nil {
			t.Fatalf("event shape = %+v", e)
		}
		if e.Training.LossType != probe.LossTypeVal {
			t.Fatalf("loss_type = %q, want %q", e.Training.LossType, probe.LossTypeVal)
		}
	}
}

// A loss failure must be loud — the operator armed the instrument, and a
// silent gap would forge the curve.
func TestSFTValidation_LossFailureIsLoud_Bad(t *testing.T) {
	result := &SFTResult{}
	ArmSFTValidation(result, []SFTBatch{{}}, 1, func(SFTBatch) (float64, bool) { return 0, false })
	if err := RunSFTValidationPass(SFTConfig{}, result); err == nil {
		t.Fatal("failed loss pass must error, not skip")
	}
}

// Unarmed and nil shapes no-op; arming requires all three pieces.
func TestSFTValidation_UnarmedNoOp_Ugly(t *testing.T) {
	if err := RunSFTValidationPass(SFTConfig{}, &SFTResult{}); err != nil {
		t.Fatalf("unarmed pass: %v", err)
	}
	if err := maybeRunSFTValidation(SFTConfig{}, nil); err != nil {
		t.Fatalf("nil result: %v", err)
	}
	result := &SFTResult{}
	ArmSFTValidation(result, nil, 1, func(SFTBatch) (float64, bool) { return 1, true })
	ArmSFTValidation(result, []SFTBatch{{}}, 1, nil)
	if len(result.valBatches) != 0 || result.valLossFn != nil {
		t.Fatal("partial arming must not install the lane")
	}
}

func TestSFTValEvery_Resolution_Good(t *testing.T) {
	if got := SFTValEvery(SFTConfig{ValEvery: 10, EvalEvery: 25}); got != 10 {
		t.Fatalf("explicit ValEvery = %d, want 10", got)
	}
	if got := SFTValEvery(SFTConfig{EvalEvery: 25}); got != 25 {
		t.Fatalf("fallback to EvalEvery = %d, want 25", got)
	}
	if got := SFTValEvery(SFTConfig{}); got != 0 {
		t.Fatalf("no cadence = %d, want 0 (baseline-only)", got)
	}
}

// Build refuses an empty validation set (a silent empty instrument is
// worse than none) and passes nil through when no set is configured.
func TestBuildSFTValidationBatches_EmptyAndNil_Bad(t *testing.T) {
	if batches, err := BuildSFTValidationBatches(nil, SFTConfig{}); err != nil || batches != nil {
		t.Fatalf("nil ValidData = (%v, %v), want (nil, nil)", batches, err)
	}
	cfg := SFTConfig{ValidData: dataset.NewSliceDataset(nil)}
	if _, err := BuildSFTValidationBatches(nil, cfg); err == nil {
		t.Fatal("empty validation set must error")
	}
}

// Checkpoint metadata carries the val loss measured under the same weights.
func TestSFTCheckpointMetadata_CarriesValLoss_Good(t *testing.T) {
	result := &SFTResult{Steps: 4, LastLoss: 0.9, LastValLoss: 1.1}
	meta := NewSFTCheckpointMetadata("ckpt", "m", SFTConfig{}, result, 1)
	if meta.ValLoss != 1.1 {
		t.Fatalf("metadata val_loss = %v, want 1.1", meta.ValLoss)
	}
}
