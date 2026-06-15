// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestDistillLoss_DistillationBatchLoss_Good exercises the happy paths of
// DistillationBatchLoss: a masked soft cross-entropy, a short-mask bound,
// nil/short mask rows skipped, and cross-cell vocab growth. Each named
// case calls DistillationBatchLoss and asserts on the returned DistillLoss.
func TestDistillLoss_DistillationBatchLoss_Good(t *testing.T) {
	t.Run("soft_cross_entropy_uses_mask", func(t *testing.T) {
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
	})

	t.Run("soft_cross_entropy_mask_shapes", func(t *testing.T) {
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
	})

	t.Run("nil_and_short_mask_rows_skipped", func(t *testing.T) {
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
	})

	t.Run("vocab_growth_across_cells", func(t *testing.T) {
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
	})
}

// TestDistillLoss_DistillationBatchLoss_Bad asserts DistillationBatchLoss
// rejects every malformed input shape — unsupported loss kind, empty
// teacher, all-zero mask, bad temperature, non-finite logit, and batch /
// sequence / vocabulary shape mismatches — with a descriptive error.
func TestDistillLoss_DistillationBatchLoss_Bad(t *testing.T) {
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

// TestDistillLoss_DistillationBatchLoss_Ugly drives the degenerate
// non-finite arms of DistillationBatchLoss: a student Inf on the
// mask-present path, a student Inf on the mask-absent path, and a teacher
// Inf under an active mask. Each must be rejected by a log-softmax finite
// guard rather than silently producing a NaN loss.
func TestDistillLoss_DistillationBatchLoss_Ugly(t *testing.T) {
	t.Run("student_non_finite_mask_present", func(t *testing.T) {
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
	})

	t.Run("student_non_finite_mask_absent", func(t *testing.T) {
		// The mask-absent inner loop has its own student log-softmax call; a
		// non-finite student logit must be rejected there too (distinct from
		// the mask-present finite guard).
		_, err := DistillationBatchLoss(
			DistillLogits{{{0, 0}}},
			DistillLogits{{{float32(math.Inf(1)), 0}}},
			nil, // no mask -> mask-absent branch
			DistillConfig{Loss: DistillLossKL, Temperature: 1},
		)
		if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
			t.Fatalf("error = %v, want non-finite student rejection on the mask-absent path", err)
		}
	})

	t.Run("teacher_non_finite_mask_present", func(t *testing.T) {
		// The masked inner loop validates the teacher side too: a non-finite
		// teacher logit under an active mask must be rejected by the teacher
		// log-softmax finite guard (the mask-present counterpart of the
		// student-side rejection).
		_, err := DistillationBatchLoss(
			DistillLogits{{{float32(math.Inf(1)), 0}}},
			DistillLogits{{{0, 0}}},
			[][]float32{{1}},
			DistillConfig{Loss: DistillLossKL, Temperature: 1},
		)
		if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
			t.Fatalf("error = %v, want non-finite teacher rejection on the masked path", err)
		}
	})
}

// TestDistillLoss_DistillBatchCacheKey_Good shows DistillBatchCacheKey is
// deterministic: hashing the same SFTBatch contents twice yields the same
// non-empty key, so a teacher-logit cache lookup is reproducible.
func TestDistillLoss_DistillBatchCacheKey_Good(t *testing.T) {
	batch := SFTBatch{
		Batch:   Batch{Tokens: [][]int{{1, 2}}, LossMask: [][]float32{{1, 1}}},
		Targets: [][]int{{2, 3}},
	}
	key := DistillBatchCacheKey(batch)
	if key == "" {
		t.Fatal("DistillBatchCacheKey() = empty, want a stable hash")
	}
	if again := DistillBatchCacheKey(batch); again != key {
		t.Fatalf("DistillBatchCacheKey() = %q then %q, want identical keys", key, again)
	}
}

// TestDistillLoss_DistillBatchCacheKey_Bad shows DistillBatchCacheKey
// discriminates: batches whose token contents differ must hash to
// different keys, otherwise a cache would alias distinct teacher logits.
func TestDistillLoss_DistillBatchCacheKey_Bad(t *testing.T) {
	a := SFTBatch{
		Batch:   Batch{Tokens: [][]int{{1, 2}}, LossMask: [][]float32{{1, 1}}},
		Targets: [][]int{{2, 3}},
	}
	b := SFTBatch{
		Batch:   Batch{Tokens: [][]int{{9, 9}}, LossMask: [][]float32{{1, 1}}},
		Targets: [][]int{{2, 3}},
	}
	if DistillBatchCacheKey(a) == DistillBatchCacheKey(b) {
		t.Fatal("DistillBatchCacheKey() collided on distinct token contents")
	}
}

// TestDistillLoss_DistillBatchCacheKey_Ugly feeds DistillBatchCacheKey the
// degenerate empty batch (no tokens, targets, or mask). It must still
// return a stable non-empty key rather than panicking on the nil slices.
func TestDistillLoss_DistillBatchCacheKey_Ugly(t *testing.T) {
	empty := SFTBatch{}
	key := DistillBatchCacheKey(empty)
	if key == "" {
		t.Fatal("DistillBatchCacheKey(empty) = empty, want a stable hash for the zero batch")
	}
	if again := DistillBatchCacheKey(empty); again != key {
		t.Fatalf("DistillBatchCacheKey(empty) = %q then %q, want deterministic", key, again)
	}
}

// TestDistillLoss_cloneDistillLogits_Good covers cloneDistillLogits on the
// degenerate shapes: a nil input clones to nil, and a batch with rows but
// zero cells clones to a non-nil, equal-shaped result with no flat-cell
// backing allocated.
func TestDistillLoss_cloneDistillLogits_Good(t *testing.T) {
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
