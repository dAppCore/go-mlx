// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"
)

// --- DistillationBatchLoss: in-call scratch regrow ---------------------------

// TestDistillBranch_DistillationBatchLoss_ScratchRegrow drives the in-call
// scratch-buffer regrow arm of DistillationBatchLoss (distill_loss.go else
// branch around line 285). The pooled float64 scratch is lazily allocated on
// the first cell at that cell's vocab size; a *later* cell in the same row
// with a strictly larger vocab finds the scratch pointer already set but its
// cap too small, taking the in-loop `make` regrow rather than the first-time
// pool Get. validateDistillLogitShapes checks teacher-vs-student vocab
// per-cell (not uniform across the row), so an ascending per-cell vocab is a
// legal input; the small cell must come first so the scratch starts undersized.
func TestDistillBranch_DistillationBatchLoss_ScratchRegrow(t *testing.T) {
	// One row, two cells: cell 0 vocab 2, cell 1 vocab 4. Student matches the
	// teacher cell-for-cell. Mask keeps both cells active so the inner loop
	// processes cell 1 after cell 0 has sized the scratch to 2.
	teacher := DistillLogits{{
		{0.0, 0.0},           // cell 0, vocab 2 -> scratch allocated at cap 2
		{0.0, 0.0, 0.0, 0.0}, // cell 1, vocab 4 -> cap 2 < 4, ptr set -> regrow
	}}
	student := DistillLogits{{
		{1.0, 0.0},
		{1.0, 0.0, 0.0, 0.0},
	}}
	mask := [][]float32{{1, 1}}

	loss, err := DistillationBatchLoss(teacher, student, mask, DistillConfig{Loss: DistillLossKL, Temperature: 1})
	if err != nil {
		t.Fatalf("DistillationBatchLoss(ascending per-cell vocab) error = %v", err)
	}
	if loss.Tokens != 2 {
		t.Fatalf("Tokens = %d, want 2 (both masked cells processed)", loss.Tokens)
	}
}

// --- validateDistillLogitShapes: per-cell vocab mismatch ---------------------

// TestDistillBranch_DistillationBatchLoss_PerCellVocabMismatch hits the
// non-empty per-cell vocab-mismatch arm of validateDistillLogitShapes
// (distill_loss.go line 379): a teacher cell of vocab 2 against a student cell
// of vocab 1. Sequence lengths match (one cell each) and the vocab is
// non-empty, so this is distinct from the empty-vocabulary guard one line up.
func TestDistillBranch_DistillationBatchLoss_PerCellVocabMismatch(t *testing.T) {
	_, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}}, // teacher cell vocab 2
		DistillLogits{{{0}}},    // student cell vocab 1
		nil,
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err == nil {
		t.Fatal("DistillationBatchLoss(per-cell vocab mismatch) error = nil, want vocabulary mismatch")
	}
}

// --- context cancellation seams ---------------------------------------------

// cancelOnNextDataset cancels its context on the Nth call to Next (1-based),
// returning a real sample each time so collection proceeds until the cancel
// lands mid-loop. It backs the distillCollectSamples cancellation seam, where
// the top-of-function ctx guard has already passed and only the in-loop guard
// can fire.
type cancelOnNextDataset struct {
	cancel   context.CancelFunc
	cancelAt int
	calls    int
}

func (d *cancelOnNextDataset) Next() (dataset.Sample, bool, error) {
	d.calls++
	if d.calls == d.cancelAt {
		d.cancel()
	}
	return dataset.Sample{Text: "x"}, true, nil
}

// TestDistillBranch_CollectSamples_CancelMidLoop covers the in-loop ctx.Err
// guard of distillCollectSamples (distill.go line 610). A pre-cancelled
// context cannot reach here — the top guard in RunKnowledgeDistillation and
// the distillBatches entry guard both catch it first — so the cancel must land
// while collection is already running. The dataset cancels on its first Next;
// the second loop iteration's guard then fires. MaxSamples > 1 keeps the loop
// going past the first sample so the in-loop guard is reached.
func TestDistillBranch_CollectSamples_CancelMidLoop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ds := &cancelOnNextDataset{cancel: cancel, cancelAt: 1}

	_, err := RunKnowledgeDistillation(ctx, DistillRunner{
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, ds, DistillConfig{MaxSamples: 4})
	if err == nil {
		t.Fatal("RunKnowledgeDistillation(cancel mid-collect) error = nil, want context cancelled")
	}
}

// TestDistillBranch_DistillBatches_CancelBeforeBuild covers the entry ctx.Err
// guard of distillBatches (distill.go line 407). The top guard in
// RunKnowledgeDistillation runs before any hook, so it cannot catch a cancel
// raised later; the StudentInfo hook (invoked after the top guard, before the
// epoch loop reaches distillBatches) cancels the context, so the next guard to
// run is the distillBatches entry. No MaxSamples, so distillBatches checks its
// guard before any collection.
func TestDistillBranch_DistillBatches_CancelBeforeBuild(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	_, err := RunKnowledgeDistillation(ctx, DistillRunner{
		StudentInfo: func(context.Context) ModelInfo {
			cancel()
			return ModelInfo{}
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
	}, oneSampleDataset(), DistillConfig{})
	if err == nil {
		t.Fatal("RunKnowledgeDistillation(cancel before distillBatches) error = nil, want context cancelled")
	}
}

// TestDistillBranch_RunEpoch_CancelBeforeBatchLoop covers the per-batch ctx.Err
// guard inside runDistillEpoch (distill.go line 353). BuildBatches returns a
// real batch *and* cancels the context: the distillBatches entry guard (407)
// has already passed by the time BuildBatches runs, so the batches are returned
// non-empty, and the first iteration of the step loop then sees the cancel.
func TestDistillBranch_RunEpoch_CancelBeforeBatchLoop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	_, err := RunKnowledgeDistillation(ctx, DistillRunner{
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			cancel()
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
	}, oneSampleDataset(), DistillConfig{})
	if err == nil {
		t.Fatal("RunKnowledgeDistillation(cancel before batch loop) error = nil, want context cancelled")
	}
}
