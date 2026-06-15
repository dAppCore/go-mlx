// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
)

// SFTValEvery resolves the validation cadence: explicit ValEvery wins.
func TestVal_SFTValEvery_Good(t *testing.T) {
	if got := SFTValEvery(SFTConfig{ValEvery: 10, EvalEvery: 25}); got != 10 {
		t.Fatalf("explicit ValEvery = %d, want 10", got)
	}
	if got := SFTValEvery(SFTConfig{ValEvery: 5}); got != 5 {
		t.Fatalf("ValEvery alone = %d, want 5", got)
	}
}

// SFTValEvery falls back to the eval cadence when ValEvery is unset — the
// probes and the val pass usually want the same clock. A zero EvalEvery with a
// zero ValEvery yields 0 (baseline-only), the degenerate fallback.
func TestVal_SFTValEvery_Bad(t *testing.T) {
	if got := SFTValEvery(SFTConfig{EvalEvery: 25}); got != 25 {
		t.Fatalf("fallback to EvalEvery = %d, want 25", got)
	}
	if got := SFTValEvery(SFTConfig{}); got != 0 {
		t.Fatalf("no cadence = %d, want 0 (baseline-only)", got)
	}
}

// A negative ValEvery is not "set" in the >0 sense, so SFTValEvery skips it and
// resolves the eval cadence instead; both negative leaves baseline-only.
func TestVal_SFTValEvery_Ugly(t *testing.T) {
	if got := SFTValEvery(SFTConfig{ValEvery: -1, EvalEvery: 7}); got != 7 {
		t.Fatalf("negative ValEvery = %d, want fallback 7", got)
	}
	if got := SFTValEvery(SFTConfig{ValEvery: -1, EvalEvery: -1}); got != -1 {
		t.Fatalf("both negative ValEvery = %d, want EvalEvery passthrough -1", got)
	}
}

// BuildSFTValidationBatches drains the fixed subset and tokenizes it, forcing
// sequence packing off so the subset stays sample-stable across passes.
func TestVal_BuildSFTValidationBatches_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})
	cfg := SFTConfig{ValidData: ds, ValidSamples: 2, SequencePacking: true}
	batches, err := BuildSFTValidationBatches(tok, cfg)
	if err != nil {
		t.Fatalf("BuildSFTValidationBatches() error = %v", err)
	}
	if len(batches) == 0 {
		t.Fatal("BuildSFTValidationBatches() built no batches from a non-empty set")
	}
	total := 0
	for _, b := range batches {
		total += len(b.Batch.Tokens)
	}
	if total != 2 {
		t.Fatalf("validation rows = %d, want 2 (packing forced off keeps samples discrete)", total)
	}
}

// Build refuses an empty validation set (a silent empty instrument is worse
// than none) and passes nil through when no set is configured.
func TestVal_BuildSFTValidationBatches_Bad(t *testing.T) {
	if batches, err := BuildSFTValidationBatches(nil, SFTConfig{}); err != nil || batches != nil {
		t.Fatalf("nil ValidData = (%v, %v), want (nil, nil)", batches, err)
	}
	cfg := SFTConfig{ValidData: dataset.NewSliceDataset(nil)}
	if _, err := BuildSFTValidationBatches(nil, cfg); err == nil {
		t.Fatal("empty validation set must error")
	}
}

// A non-positive ValidSamples floors to the 32-sample default cap: a set with
// fewer rows than the default drains fully without overrunning Next.
func TestVal_BuildSFTValidationBatches_Ugly(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})
	cfg := SFTConfig{ValidData: ds, ValidSamples: 0}
	batches, err := BuildSFTValidationBatches(tok, cfg)
	if err != nil {
		t.Fatalf("BuildSFTValidationBatches() error = %v", err)
	}
	total := 0
	for _, b := range batches {
		total += len(b.Batch.Tokens)
	}
	if total != 1 {
		t.Fatalf("rows = %d, want 1 (default cap drains the short set)", total)
	}
}

// ArmSFTValidation installs the validation lane end-to-end: the gate then fires
// on cadence, records a point per pass, and the probe sink sees loss_type=val.
func TestVal_ArmSFTValidation_Good(t *testing.T) {
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

// ArmSFTValidation refuses partial arming: a nil result, nil batches, or nil
// loss function must NOT install the lane (a half-armed instrument forges the
// curve). Each missing piece leaves the lane disarmed.
func TestVal_ArmSFTValidation_Bad(t *testing.T) {
	// Nil result is a tolerated no-op (no panic, nothing installed).
	ArmSFTValidation(nil, []SFTBatch{{}}, 1, func(SFTBatch) (float64, bool) { return 1, true })

	result := &SFTResult{}
	ArmSFTValidation(result, nil, 1, func(SFTBatch) (float64, bool) { return 1, true })
	if len(result.valBatches) != 0 || result.valLossFn != nil {
		t.Fatal("nil batches must not install the lane")
	}
	ArmSFTValidation(result, []SFTBatch{{}}, 1, nil)
	if len(result.valBatches) != 0 || result.valLossFn != nil {
		t.Fatal("nil lossFn must not install the lane")
	}
}

// Re-arming overwrites the prior lane wholesale: the second ArmSFTValidation
// replaces batches, cadence, and loss function — the instrument is re-fitted,
// not appended to.
func TestVal_ArmSFTValidation_Ugly(t *testing.T) {
	result := &SFTResult{}
	ArmSFTValidation(result, []SFTBatch{{}}, 1, func(SFTBatch) (float64, bool) { return 9, true })
	ArmSFTValidation(result, []SFTBatch{{}, {}}, 5, func(SFTBatch) (float64, bool) { return 2, true })
	if len(result.valBatches) != 2 || result.valEvery != 5 {
		t.Fatalf("re-arm batches=%d every=%d, want 2/5 (latest lane wins)", len(result.valBatches), result.valEvery)
	}
	if err := RunSFTValidationPass(SFTConfig{}, result); err != nil {
		t.Fatalf("pass after re-arm: %v", err)
	}
	if result.LastValLoss != 2 {
		t.Fatalf("last val loss = %v, want 2 (second lossFn active)", result.LastValLoss)
	}
}

// RunSFTValidationPass forwards the fixed subset and records the mean at the
// current step — the step-0 call is the run's baseline point.
func TestVal_RunSFTValidationPass_Good(t *testing.T) {
	result := &SFTResult{Steps: 6}
	ArmSFTValidation(result, []SFTBatch{{}, {}}, 1, func(SFTBatch) (float64, bool) { return 0.5, true })
	if err := RunSFTValidationPass(SFTConfig{}, result); err != nil {
		t.Fatalf("RunSFTValidationPass() error = %v", err)
	}
	if len(result.ValLosses) != 1 {
		t.Fatalf("val points = %d, want 1", len(result.ValLosses))
	}
	if result.ValLosses[0].Step != 6 || result.ValLosses[0].Loss != 0.5 {
		t.Fatalf("val point = %+v, want {Step:6 Loss:0.5}", result.ValLosses[0])
	}
	if result.LastValLoss != 0.5 {
		t.Fatalf("last val loss = %v, want 0.5", result.LastValLoss)
	}
}

// A loss failure must be loud — the operator armed the instrument, and a silent
// gap would forge the curve, so RunSFTValidationPass returns an error.
func TestVal_RunSFTValidationPass_Bad(t *testing.T) {
	result := &SFTResult{}
	ArmSFTValidation(result, []SFTBatch{{}}, 1, func(SFTBatch) (float64, bool) { return 0, false })
	if err := RunSFTValidationPass(SFTConfig{}, result); err == nil {
		t.Fatal("failed loss pass must error, not skip")
	}
}

// Unarmed and nil shapes no-op: RunSFTValidationPass on a fresh result and the
// in-loop gate on a nil result both return nil without recording.
func TestVal_RunSFTValidationPass_Ugly(t *testing.T) {
	if err := RunSFTValidationPass(SFTConfig{}, &SFTResult{}); err != nil {
		t.Fatalf("unarmed pass: %v", err)
	}
	if err := maybeRunSFTValidation(SFTConfig{}, nil); err != nil {
		t.Fatalf("nil result: %v", err)
	}
	// Off-cadence: armed every=3, step 4 is not a multiple → no recording.
	result := &SFTResult{Steps: 4}
	ArmSFTValidation(result, []SFTBatch{{}}, 3, func(SFTBatch) (float64, bool) { return 1, true })
	if err := maybeRunSFTValidation(SFTConfig{}, result); err != nil {
		t.Fatalf("off-cadence gate: %v", err)
	}
	if len(result.ValLosses) != 0 {
		t.Fatalf("off-cadence recorded %d points, want 0", len(result.ValLosses))
	}
}

// TestVal_CheckpointMetadataCarriesValLoss asserts the val curve threads through
// into the saved checkpoint sidecar: a result's LastValLoss lands in the
// metadata's ValLoss field. (Extra coverage of the val lane's persistence edge;
// the NewSFTCheckpointMetadata triplet itself lives in sft_checkpoint_test.go.)
func TestVal_CheckpointMetadataCarriesValLoss(t *testing.T) {
	result := &SFTResult{Steps: 4, LastLoss: 0.9, LastValLoss: 1.1}
	meta := NewSFTCheckpointMetadata("ckpt", "m", SFTConfig{}, result, 1)
	if meta.ValLoss != 1.1 {
		t.Fatalf("metadata val_loss = %v, want 1.1", meta.ValLoss)
	}
}
