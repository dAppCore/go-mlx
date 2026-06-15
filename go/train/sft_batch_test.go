// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// TestSFT_BuildSFTBatches_Good builds batches from a small prompt/response
// dataset and asserts the response-masked triple is correct: inputs/targets
// are V[0..n)/V[1..n+1) over prompt|response|EOS, the mask is 0 across the
// prompt region and 1 over the response+EOS region, and rows group into
// BatchSize batches. Drives the unexported batch builder + sftBatchFromExamples
// transitively (output asserted, not memory layout — the unsafe-slice share is
// a deliberate alloc optimisation).
func TestSFT_BuildSFTBatches_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})

	batches, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 2})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	// 3 rows at BatchSize 2 → batches of 2 + 1.
	if len(batches) != 2 {
		t.Fatalf("batches = %d, want 2 (2+1 at BatchSize 2)", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 2 || len(batches[1].Batch.Tokens) != 1 {
		t.Fatalf("batch sizes = %d/%d, want 2/1", len(batches[0].Batch.Tokens), len(batches[1].Batch.Tokens))
	}

	// Virtual sequence V = [100 101 102 103][500 501 502][EOS=9], len 8 → n=7.
	wantInputs := []int{100, 101, 102, 103, 500, 501, 502}
	wantTargets := []int{101, 102, 103, 500, 501, 502, 9}
	// promptLen=4 → mask 1 from index promptLen-1=3 onward.
	wantMask := []float32{0, 0, 0, 1, 1, 1, 1}

	row0 := batches[0]
	if !equalIntSlices(row0.Batch.Tokens[0], wantInputs) {
		t.Fatalf("inputs = %v, want %v", row0.Batch.Tokens[0], wantInputs)
	}
	if !equalIntSlices(row0.Targets[0], wantTargets) {
		t.Fatalf("targets = %v, want %v", row0.Targets[0], wantTargets)
	}
	if !equalFloat32Slices(row0.Batch.LossMask[0], wantMask) {
		t.Fatalf("mask = %v, want %v", row0.Batch.LossMask[0], wantMask)
	}
	if row0.Batch.Length[0] != len(wantInputs) {
		t.Fatalf("Length = %d, want %d", row0.Batch.Length[0], len(wantInputs))
	}
}

// TestSFT_BuildSFTTrainingBatches_GroupsByEffectiveBatch_Good asserts the
// runner-level entry point batches by the EFFECTIVE batch size (BatchSize ×
// GradientAccumulationSteps), not the raw BatchSize — that's the contract
// difference from BuildSFTBatches.
func TestSFT_BuildSFTTrainingBatches_GroupsByEffectiveBatch_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	rows := make([]dataset.Sample, 6)
	for i := range rows {
		rows[i] = dataset.Sample{Prompt: "prompt", Response: "response"}
	}
	ds := dataset.NewSliceDataset(rows)

	// BatchSize 2 × GradAccum 3 = effective 6 → all six rows in one batch.
	batches, err := BuildSFTTrainingBatches(tok, ds, SFTConfig{BatchSize: 2, GradientAccumulationSteps: 3})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches = %d, want 1 (effective batch size 6 holds all rows)", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 6 {
		t.Fatalf("batch rows = %d, want 6", len(batches[0].Batch.Tokens))
	}
}

// TestSFT_BuildSFTBatches_NilGuards_Bad asserts both entry points reject a nil
// tokenizer and a nil dataset rather than panicking.
func TestSFT_BuildSFTBatches_NilGuards_Bad(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	if _, err := BuildSFTBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil ds) error = nil, want rejection")
	}
	if _, err := BuildSFTTrainingBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTTrainingBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil ds) error = nil, want rejection")
	}
}

// TestSFT_BuildSFTBatches_SkipsUnusableRows_Ugly feeds rows that produce no
// training target — an empty prompt+response with NoEOS (virtual length < 2)
// and a response-only row — and asserts they are silently dropped, while a
// real row still lands. Exercises the usable==false skip in the build loop.
func TestSFT_BuildSFTBatches_SkipsUnusableRows_Ugly(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "", Response: ""},               // empty → unusable
		{Prompt: "prompt", Response: "response"}, // the one real row
		{Prompt: "", Response: ""},               // empty → unusable
	})

	// NoEOS removes the EOS token so the empty rows collapse below the
	// 2-token minimum and are dropped.
	batches, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 4, NoEOS: true})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	rows := 0
	for _, b := range batches {
		rows += len(b.Batch.Tokens)
	}
	if rows != 1 {
		t.Fatalf("usable rows = %d, want 1 (empty rows dropped)", rows)
	}
}

// --- SFTResult.Metrics — the dashboard summary ---

// A batch builder created with a non-positive size floors to 1, so it flushes
// one batch per added example rather than accumulating into an oversized (or
// zero-sized) batch. The public BuildSFTBatches normalises the size before it
// reaches the builder, so the floor is exercised directly here.
func TestNewSFTBatchBuilder_NonPositiveSizeFloorsToOne_Ugly(t *testing.T) {
	b := newSFTBatchBuilder(0)
	b.add(sftExample{inputs: []int{1}, targets: []int{2}, mask: []float32{1}})
	b.add(sftExample{inputs: []int{3}, targets: []int{4}, mask: []float32{1}})
	batches := b.finish()
	// Size floored to 1 → each add flushed its own single-row batch.
	if len(batches) != 2 {
		t.Fatalf("batches = %d, want 2 (size floored to 1, one row each)", len(batches))
	}
	for i, batch := range batches {
		if len(batch.Batch.Tokens) != 1 {
			t.Fatalf("batch %d rows = %d, want 1", i, len(batch.Batch.Tokens))
		}
	}

	// And the public path still produces batches with a misconfigured size —
	// the normaliser absorbs it, no rows dropped.
	tok := spine.NewTokenizer(exampleSFTTokenizer{})
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "hi", Response: "yes"}})
	built, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: -1})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	if len(built) != 1 {
		t.Fatalf("BuildSFTBatches batches = %d, want 1", len(built))
	}
}

// --- sftStepName — the step-NNNNNN checkpoint directory name ---
