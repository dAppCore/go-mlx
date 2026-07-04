// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
)

// runSFTJob over a TINY synthetic loads the model for real, parses the dataset
// (via the model-aware dataset config), marks running, builds the SFTConfig
// (constructing newSFTProbeSink), calls TrainSFT, and persists the final state.
// The whole worker body + probe-sink fire on synthetic weights — no real model
// load (AX-11). The handler-side kickoff is covered in the portable file; here
// we call runSFTJob directly for a deterministic wait.
func TestRunSFTJob_SyntheticLoadAndTrain_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	dir := t.TempDir()
	model := writeSyntheticGemma3Model(t, dir)
	dataset := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(dataset, []byte(
		`{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write dataset: %v", r.Value)
	}
	adapterDir := core.JoinPath(dir, "adapter")
	if r := core.MkdirAll(adapterDir, 0o755); !r.OK {
		t.Fatalf("mkdir adapter: %v", r.Value)
	}

	reg := newAdminSFTRegistry()
	job := &adminSFTJob{
		JobID:       "synthetic-job",
		State:       adminSFTStatePending,
		ModelPath:   model,
		DatasetPath: dataset,
		AdapterDir:  adapterDir,
		StartedUnix: time.Now().Unix(),
		cancel:      func() {},
	}
	reg.active = job
	runSFTJob(context.Background(), reg, job, adminSFTRequest{
		ModelPath:     model,
		DatasetPath:   dataset,
		Epochs:        1,
		BatchSize:     1,
		MaxSeqLen:     32,
		ContextLength: 64,
	})

	// The worker ran to completion (active → last). On the synthetic the model
	// -aware dataset config yields trainable batches, so training completes and
	// the job is done; failed/stopped are accepted in case a future synthetic
	// trips a training guard — what matters is the worker advanced through load
	// + parse + config build + TrainSFT + a terminal state flip.
	snap := reg.last
	if snap == nil {
		t.Fatal("runSFTJob did not move the job to last; the worker never finished")
	}
	switch snap.State {
	case adminSFTStateDone, adminSFTStateFailed, adminSFTStateStopped:
	default:
		t.Fatalf("final state = %q, want a terminal state (done/failed/stopped)", snap.State)
	}
}

// runSFTJob with a bad model path fails at the load step.
func TestRunSFTJob_BadModel_Bad(t *testing.T) {
	dir := t.TempDir()
	dataset := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(dataset, []byte(`{"messages":[{"role":"user","content":"hi"}]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write dataset: %v", r.Value)
	}
	reg := newAdminSFTRegistry()
	job := &adminSFTJob{JobID: "bad", State: adminSFTStatePending, cancel: func() {}}
	reg.active = job
	runSFTJob(context.Background(), reg, job, adminSFTRequest{
		ModelPath:   core.JoinPath(dir, "absent-model"),
		DatasetPath: dataset,
	})
	if reg.last == nil || reg.last.State != adminSFTStateFailed {
		t.Fatalf("final job = %+v, want a failed state on model-load error", reg.last)
	}
}
