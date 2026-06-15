// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// sftBatchTestTokenizer maps prompt/response/text strings to caller-chosen
// token IDs so a test can drive precise lengths without a model. Mirrors the
// shape of buildExampleTestTokenizer but with a fixed three-key vocab so the
// batch-building tests read clearly. (Distinct name — buildExampleTestTokenizer
// is owned by sft_buildexample_test.go.)
type sftBatchTestTokenizer struct {
	prompt   []int32
	response []int32
	text     []int32
	eos      int32
}

func (t sftBatchTestTokenizer) Encode(s string) []int32 {
	switch s {
	case "prompt":
		return append([]int32(nil), t.prompt...)
	case "response":
		return append([]int32(nil), t.response...)
	case "text":
		return append([]int32(nil), t.text...)
	}
	return nil
}

func (sftBatchTestTokenizer) Decode([]int32) string { return "" }

func (sftBatchTestTokenizer) DecodeOne(int32) string { return "" }

func (sftBatchTestTokenizer) TokenID(string) (int32, bool) { return 0, false }

func (sftBatchTestTokenizer) IDToken(int32) string { return "" }

func (sftBatchTestTokenizer) BOS() int32 { return 0 }

func (t sftBatchTestTokenizer) EOS() int32 { return t.eos }

func (sftBatchTestTokenizer) HasBOSToken() bool { return false }

func newSFTBatchTestTokenizer() *spine.Tokenizer {
	return spine.NewTokenizer(sftBatchTestTokenizer{
		prompt:   makeIDs(100, 4),
		response: makeIDs(500, 3),
		text:     makeIDs(900, 5),
		eos:      9,
	})
}

// TestSftBatch_BuildSFTBatches_Good builds batches from a small prompt/response
// dataset and asserts the response-masked triple is correct: inputs/targets are
// V[0..n)/V[1..n+1) over prompt|response|EOS, the mask is 0 across the prompt
// region and 1 over the response+EOS region, and rows group into BatchSize
// batches. Drives the unexported batch builder + sftBatchFromExamples
// transitively (output asserted, not memory layout — the unsafe-slice share is
// a deliberate alloc optimisation).
func TestSftBatch_BuildSFTBatches_Good(t *testing.T) {
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

// TestSftBatch_BuildSFTBatches_Bad asserts BuildSFTBatches rejects a nil
// tokenizer and a nil dataset rather than panicking.
func TestSftBatch_BuildSFTBatches_Bad(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	if _, err := BuildSFTBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil ds) error = nil, want rejection")
	}
}

// TestSftBatch_BuildSFTBatches_Ugly covers two degenerate-but-legal shapes via
// subtests: rows that produce no training target are silently dropped while a
// real row still lands (the usable==false skip), and a misconfigured BatchSize
// is absorbed by the normaliser so the unexported builder floors to one without
// dropping rows.
func TestSftBatch_BuildSFTBatches_Ugly(t *testing.T) {
	t.Run("SkipsUnusableRows", func(t *testing.T) {
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
	})

	t.Run("MisconfiguredSizeAbsorbed", func(t *testing.T) {
		// The unexported builder floors a non-positive size to 1 directly...
		b := newSFTBatchBuilder(0)
		b.add(sftExample{inputs: []int{1}, targets: []int{2}, mask: []float32{1}})
		b.add(sftExample{inputs: []int{3}, targets: []int{4}, mask: []float32{1}})
		built := b.finish()
		if len(built) != 2 {
			t.Fatalf("builder batches = %d, want 2 (size floored to 1, one row each)", len(built))
		}
		for i, batch := range built {
			if len(batch.Batch.Tokens) != 1 {
				t.Fatalf("builder batch %d rows = %d, want 1", i, len(batch.Batch.Tokens))
			}
		}

		// ...and the public BuildSFTBatches path absorbs a negative size via the
		// normaliser, no rows dropped.
		tok := spine.NewTokenizer(exampleSFTTokenizer{})
		ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "hi", Response: "yes"}})
		publicBuilt, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: -1})
		if err != nil {
			t.Fatalf("BuildSFTBatches() error = %v", err)
		}
		if len(publicBuilt) != 1 {
			t.Fatalf("BuildSFTBatches batches = %d, want 1", len(publicBuilt))
		}
	})
}

// TestSftBatch_BuildSFTTrainingBatches_Good asserts the runner-level entry point
// batches by the EFFECTIVE batch size (BatchSize × GradientAccumulationSteps),
// not the raw BatchSize — that's the contract difference from BuildSFTBatches.
func TestSftBatch_BuildSFTTrainingBatches_Good(t *testing.T) {
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

// TestSftBatch_BuildSFTTrainingBatches_Bad asserts the runner-level entry point
// rejects a nil tokenizer and a nil dataset rather than panicking.
func TestSftBatch_BuildSFTTrainingBatches_Bad(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	if _, err := BuildSFTTrainingBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTTrainingBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil ds) error = nil, want rejection")
	}
}

// TestSftBatch_BuildSFTTrainingBatches_Ugly drives the defaulting edges: a zero
// BatchSize and zero GradientAccumulationSteps both floor to 1 (effective 1, one
// row per batch), and unusable rows are skipped through the same build loop.
func TestSftBatch_BuildSFTTrainingBatches_Ugly(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "", Response: ""}, // unusable with NoEOS
		{Prompt: "prompt", Response: "response"},
	})

	// Zero BatchSize + zero GradAccum → effective 1 → one usable row per batch.
	batches, err := BuildSFTTrainingBatches(tok, ds, SFTConfig{NoEOS: true})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	rows := 0
	for _, b := range batches {
		rows += len(b.Batch.Tokens)
		if len(b.Batch.Tokens) != 1 {
			t.Fatalf("batch rows = %d, want 1 (effective batch floored to 1)", len(b.Batch.Tokens))
		}
	}
	if rows != 2 {
		t.Fatalf("usable rows = %d, want 2 (unusable middle row dropped)", rows)
	}
}
