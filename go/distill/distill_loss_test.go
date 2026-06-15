// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestDistillationBatchLoss_SoftCrossEntropyUsesMask_Good(t *testing.T) {
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}, {0, 0}}},
		DistillLogits{{{0, 0}, {10, -10}}},
		[][]float32{{1, 0}},
		DistillConfig{Loss: DistillLossSoftCrossEntropy, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want mask to include one token", loss.Tokens)
	}
	if math.Abs(loss.SoftCrossEntropy-math.Log(2)) > 1e-6 {
		t.Fatalf("soft CE = %.9f, want ln(2)", loss.SoftCrossEntropy)
	}
	if math.Abs(loss.Value-loss.SoftCrossEntropy) > 1e-9 {
		t.Fatalf("loss value = %.9f, want soft CE %.9f", loss.Value, loss.SoftCrossEntropy)
	}
}

func TestDistillationBatchLoss_ValidationErrors_Bad(t *testing.T) {
	cases := []struct {
		name    string
		teacher DistillLogits
		student DistillLogits
		mask    [][]float32
		cfg     DistillConfig
		want    string
	}{
		{
			name:    "unsupported_loss",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{Loss: DistillLossKind("bad")},
			want:    "unsupported",
		},
		{
			name:    "empty_teacher",
			teacher: DistillLogits{},
			student: DistillLogits{},
			cfg:     DistillConfig{},
			want:    "empty",
		},
		{
			name:    "no_masked_tokens",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			mask:    [][]float32{{0}},
			cfg:     DistillConfig{},
			want:    "no masked",
		},
		{
			name:    "bad_temperature",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{Temperature: -1},
			want:    "temperature",
		},
		{
			name:    "nonfinite_logit",
			teacher: DistillLogits{{{float32(math.Inf(1))}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{},
			want:    "finite",
		},
		{
			name:    "batch_count_mismatch",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}, {{0}}},
			cfg:     DistillConfig{},
			want:    "batch",
		},
		{
			name:    "sequence_length_mismatch",
			teacher: DistillLogits{{{0}, {0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{},
			want:    "sequence",
		},
		{
			name:    "empty_vocabulary",
			teacher: DistillLogits{{{}}},
			student: DistillLogits{{{}}},
			cfg:     DistillConfig{},
			want:    "vocabulary",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DistillationBatchLoss(tc.teacher, tc.student, tc.mask, tc.cfg)
			if err == nil || !core.Contains(core.Lower(err.Error()), tc.want) {
				t.Fatalf("DistillationBatchLoss() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestDistillationBatchLoss_SoftCrossEntropyMaskShapes_Good(t *testing.T) {
	// SoftCrossEntropy loss-kind selects loss.Value = softCE (the
	// non-KL lossValue arm), and a mask shorter than the sequence
	// bounds the inner loop to the masked prefix.
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}, {0, 0}}},
		DistillLogits{{{1, -1}, {5, -5}}},
		[][]float32{{1}}, // mask covers only the first of two positions
		DistillConfig{Loss: DistillLossSoftCrossEntropy, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want short mask to bound to one token", loss.Tokens)
	}
	if loss.Kind != DistillLossSoftCrossEntropy || loss.Value != loss.SoftCrossEntropy {
		t.Fatalf("loss = %+v, want value == softCE for the soft-CE kind", loss)
	}
}

func TestDistillationBatchLoss_NilAndShortMaskRowsSkipped_Good(t *testing.T) {
	// A mask with fewer rows than the batch (second row absent) and a
	// nil mask row exercise the i>=len(maskRows) continue and the
	// maskRow==nil continue without producing a no-masked-tokens error.
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}, {{0, 0}}, {{0, 0}}},
		DistillLogits{{{2, -2}}, {{2, -2}}, {{2, -2}}},
		[][]float32{{1}, nil}, // row 0 masked, row 1 nil, row 2 absent
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want only the first masked row counted", loss.Tokens)
	}
}

func TestDistillationBatchLoss_StudentNonFinite_Ugly(t *testing.T) {
	// The teacher side is finite; the student side carries an Inf,
	// which must be rejected by the student log-softmax finite guard.
	_, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}},
		DistillLogits{{{float32(math.Inf(1)), 0}}},
		[][]float32{{1}},
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
		t.Fatalf("error = %v, want non-finite student logit rejection", err)
	}
}

func TestDistillationBatchLoss_VocabGrowthAcrossCells_Good(t *testing.T) {
	// validateDistillLogitShapes enforces per-cell teacher==student vocab
	// but NOT cross-cell uniformity, so a single row whose first cell has
	// vocab 2 and second cell has vocab 4 is valid. The second cell's
	// larger vocab trips the within-call scratch grow (the `else` arm that
	// re-makes the pooled buffers in place). Exercised on both the
	// mask-absent and mask-present inner loops.
	smallThenLarge := DistillLogits{{{0, 0}, {0, 0, 0, 0}}}
	studentSame := DistillLogits{{{1, -1}, {2, -2, 0, 0}}}

	t.Run("mask_absent", func(t *testing.T) {
		loss, err := DistillationBatchLoss(smallThenLarge, studentSame, nil, DistillConfig{Loss: DistillLossKL, Temperature: 1})
		if err != nil {
			t.Fatalf("DistillationBatchLoss() error = %v", err)
		}
		if loss.Tokens != 2 {
			t.Fatalf("tokens = %d, want both cells counted with no mask", loss.Tokens)
		}
	})

	t.Run("mask_present", func(t *testing.T) {
		loss, err := DistillationBatchLoss(smallThenLarge, studentSame, [][]float32{{1, 1}}, DistillConfig{Loss: DistillLossKL, Temperature: 1})
		if err != nil {
			t.Fatalf("DistillationBatchLoss() error = %v", err)
		}
		if loss.Tokens != 2 {
			t.Fatalf("tokens = %d, want both masked cells counted", loss.Tokens)
		}
	})
}

func TestDistillationBatchLoss_MaskAbsentStudentNonFinite_Ugly(t *testing.T) {
	// The mask-absent inner loop has its own student log-softmax call; a
	// non-finite student logit must be rejected there too (distinct from
	// the mask-present finite guard already covered).
	_, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}},
		DistillLogits{{{float32(math.Inf(1)), 0}}},
		nil, // no mask -> mask-absent branch
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
		t.Fatalf("error = %v, want non-finite student rejection on the mask-absent path", err)
	}
}

func TestCloneDistillLogits_EmptyAndRowsWithoutCells_Good(t *testing.T) {
	// Empty input clones to nil; a batch with rows but zero cells (no
	// vocab in any position) clones to a non-nil, equal-shaped result
	// with no flat-cell backing allocated.
	if got := cloneDistillLogits(nil); got != nil {
		t.Fatalf("cloneDistillLogits(nil) = %v, want nil", got)
	}
	rowsNoCells := DistillLogits{{}, {}}
	got := cloneDistillLogits(rowsNoCells)
	if got == nil || len(got) != 2 {
		t.Fatalf("cloneDistillLogits(rows-no-cells) = %v, want 2 empty rows", got)
	}
	if len(got[0]) != 0 || len(got[1]) != 0 {
		t.Fatalf("cloned rows = %v, want both empty", got)
	}
}
