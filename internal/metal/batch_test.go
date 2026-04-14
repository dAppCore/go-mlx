//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"
	"testing"
)

func TestBatch_BuildBatchMask_Shape_Good(t *testing.T) {
	// 2 prompts, max length 4, prompt lengths [3, 2].
	mask := buildBatchMask(2, 4, []int32{3, 2})
	if err := Eval(mask); err != nil {
		t.Fatalf("Eval mask: %v", err)
	}

	shape := mask.Shape()
	want := []int32{2, 1, 4, 4}
	if len(shape) != 4 {
		t.Fatalf("mask ndim = %d, want 4", len(shape))
	}
	for i, s := range shape {
		if s != want[i] {
			t.Errorf("mask shape[%d] = %d, want %d", i, s, want[i])
		}
	}
}

func TestBatch_BuildBatchMask_Values_Good(t *testing.T) {
	// Single prompt of length 3, padded to 4.
	// Expected mask [1, 1, 4, 4]:
	//   row 0: [0, -inf, -inf, -inf]  (can only attend to pos 0)
	//   row 1: [0, 0, -inf, -inf]     (attend to pos 0,1)
	//   row 2: [0, 0, 0, -inf]        (attend to pos 0,1,2)
	//   row 3: [0, 0, 0, -inf]        (row 3 is padding — causal says j<=3 but j<3 caps it)
	mask := buildBatchMask(1, 4, []int32{3})
	if err := Eval(mask); err != nil {
		t.Fatalf("Eval mask: %v", err)
	}

	// Flatten to get values.
	flat := Reshape(mask, 16)
	if err := Eval(flat); err != nil {
		t.Fatalf("Eval flat: %v", err)
	}
	vals := flat.Floats()

	negInf := float32(math.Inf(-1))
	expected := []float32{
		// row 0: attend j=0 only
		0, negInf, negInf, negInf,
		// row 1: attend j=0,1
		0, 0, negInf, negInf,
		// row 2: attend j=0,1,2
		0, 0, 0, negInf,
		// row 3: padding row — causal allows j<=3 but padding caps at j<3
		0, 0, 0, negInf,
	}

	for i, v := range vals {
		e := expected[i]
		if math.IsInf(float64(e), -1) {
			if !math.IsInf(float64(v), -1) {
				t.Errorf("vals[%d] = %f, want -inf", i, v)
			}
		} else if v != e {
			t.Errorf("vals[%d] = %f, want %f", i, v, e)
		}
	}
}

func TestBatch_BuildBatchMask_MultipleBatches_Good(t *testing.T) {
	// 2 prompts: lengths [2, 1], max length 2.
	mask := buildBatchMask(2, 2, []int32{2, 1})
	if err := Eval(mask); err != nil {
		t.Fatalf("Eval mask: %v", err)
	}

	flat := Reshape(mask, 8)
	if err := Eval(flat); err != nil {
		t.Fatalf("Eval flat: %v", err)
	}
	vals := flat.Floats()

	negInf := float32(math.Inf(-1))
	expected := []float32{
		// batch 0 (len=2): full causal, no padding
		0, negInf,
		0, 0,
		// batch 1 (len=1): only first position is real
		0, negInf,
		0, negInf, // row 1: causal allows j<=1 but padding caps at j<1
	}

	for i, v := range vals {
		e := expected[i]
		if math.IsInf(float64(e), -1) {
			if !math.IsInf(float64(v), -1) {
				t.Errorf("batch vals[%d] = %f, want -inf", i, v)
			}
		} else if v != e {
			t.Errorf("batch vals[%d] = %f, want %f", i, v, e)
		}
	}
}
