// SPDX-Licence-Identifier: EUPL-1.2

// Pure-Go branch coverage for the SFT epoch machinery — the guard and error
// arms that need no real loss array. An empty (Model-nil) *metal.LoRAAdapter
// makes Step/StepAccumulated short-circuit to nil without touching the GPU, so
// the dispatch loop and the nil-loss error path are reachable in the default
// untagged test. The success paths that need a real loss live in the
// metal_runtime-tagged sft_epoch_metal_test.go.
package train

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// epochBranchModel satisfies train.Model with a configurable Generate result.
type epochBranchModel struct {
	genErr error
	gen    string
}

func (m epochBranchModel) ModelType() string    { return "epoch-branch" }
func (m epochBranchModel) Info() spine.ModelInfo { return spine.ModelInfo{} }
func (m epochBranchModel) Generate(string, ...spine.GenerateOption) (string, error) {
	return m.gen, m.genErr
}

// TestSftEpoch_AdapterStepDispatchArms covers both non-empty arms of
// sftAdapterStep: a single batch routes to Step, multiple batches build the
// metal batch/target slabs and route to StepAccumulated. The Model-nil adapter
// makes both return nil after the dispatch body runs — no GPU, just the arm.
func TestSftEpoch_AdapterStepDispatchArms(t *testing.T) {
	adapter := &metal.LoRAAdapter{}
	b1 := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	b2 := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{2}}, Length: []int{1}}, Targets: [][]int{{3}}}

	if loss := sftAdapterStep(adapter, []SFTBatch{b1}, nil); loss != nil {
		t.Fatalf("single-batch sftAdapterStep over Model-nil adapter = %v, want nil", loss)
	}
	if loss := sftAdapterStep(adapter, []SFTBatch{b1, b2}, nil); loss != nil {
		t.Fatalf("multi-batch sftAdapterStep over Model-nil adapter = %v, want nil", loss)
	}
}

// TestSftEpoch_RunSFTBatch_NilLoss drives the standalone runSFTBatch wrapper and,
// through it, runSFTBatchGroup's nil-loss error: a Model-nil adapter yields no
// loss, so the group reports the documented "nil loss" failure.
func TestSftEpoch_RunSFTBatch_NilLoss(t *testing.T) {
	adapter := &metal.LoRAAdapter{}
	batch := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1})
	result := &SFTResult{}
	err := runSFTBatch(context.Background(), epochBranchModel{}, batch, adapter, nil, cfg, result, 1)
	if err == nil {
		t.Fatal("runSFTBatch over Model-nil adapter error = nil, want nil-loss failure")
	}
	if result.Steps != 0 {
		t.Fatalf("Steps = %d, want 0 (no loss, no step)", result.Steps)
	}
}

// TestSftEpoch_RunSFTEvaluations_Branches covers the standalone eval helper's
// non-success arms: a nil context defaults to Background (no panic), a generate
// error aborts and surfaces, and an already-cancelled context aborts the prompt
// loop before any generate call.
func TestSftEpoch_RunSFTEvaluations_Branches(t *testing.T) {
	cfg := SFTConfig{EvalEvery: 1, EvalPrompts: []string{"a"}, EvalMaxTokens: 8}
	result := &SFTResult{Steps: 1}

	// nil context → defaults to context.Background, generate succeeds.
	if err := runSFTEvaluations(nil, epochBranchModel{gen: "ok"}, cfg, result); err != nil { //nolint:staticcheck // nil ctx is the branch under test
		t.Fatalf("runSFTEvaluations(nil ctx) error = %v, want nil", err)
	}
	if len(result.Evaluations) != 1 {
		t.Fatalf("evaluations = %d, want 1", len(result.Evaluations))
	}

	// Generate error surfaces.
	errModel := epochBranchModel{genErr: context.DeadlineExceeded}
	if err := runSFTEvaluations(context.Background(), errModel, cfg, &SFTResult{Steps: 1}); err == nil {
		t.Fatal("runSFTEvaluations generate-error = nil, want surfaced error")
	}

	// Cancelled context aborts the prompt loop before any generate.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runSFTEvaluations(cancelled, epochBranchModel{gen: "never"}, cfg, &SFTResult{Steps: 1}); err == nil {
		t.Fatal("runSFTEvaluations cancelled = nil, want context error")
	}
}

// TestSftEpoch_RunSFTDatasetEpoch_BuildError surfaces buildSFTExample's error
// mid-loop without a real loss: a tokenizer-less build fails on the first usable
// row, aborting the epoch before any gradient machinery.
func TestSftEpoch_RunSFTDatasetEpoch_BuildError(t *testing.T) {
	// buildSFTExample needs a valid tokenizer; a nil one makes it error,
	// aborting the epoch on the first usable row.
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1})
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "x", Response: "y"}})
	if err := RunSFTDatasetEpoch(context.Background(), epochBranchModel{}, nil, ds, &metal.LoRAAdapter{}, nil, cfg, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch over nil tokenizer error = nil, want buildSFTExample failure")
	}
}

// TestSftEpoch_StreamingPackerMidAddFlushError covers the streaming packer's
// add-time flush error arm: a second oversized row forces a flush mid-add, and
// when the emit callback fails the error surfaces out of add (not just finish).
func TestSftEpoch_StreamingPackerMidAddFlushError(t *testing.T) {
	wantErr := context.Canceled
	packer := newSFTStreamingPacker(3, func(sftExample) error { return wantErr })
	// First row fills the accumulator (len 2 ≤ maxSeqLen 3, no flush yet).
	if err := packer.add(sftExample{inputs: []int{1, 2}, targets: []int{2, 3}, mask: []float32{0, 1}}); err != nil {
		t.Fatalf("first add error = %v, want nil (under maxSeqLen)", err)
	}
	// Second row would push 2+2 > 3 → mid-add flush fires → emit fails.
	err := packer.add(sftExample{inputs: []int{3, 4}, targets: []int{4, 5}, mask: []float32{1, 1}})
	if err != wantErr { //nolint:errorlint // exact sentinel returned unwrapped
		t.Fatalf("mid-add flush error = %v, want %v", err, wantErr)
	}
}
