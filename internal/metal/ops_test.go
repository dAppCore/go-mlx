//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

const tol = 1e-5

func approx(a, b float64) bool { return math.Abs(a-b) < tol }

func floatSliceApprox(t *testing.T, got []float32, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if !approx(float64(got[i]), float64(want[i])) {
			t.Errorf("[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// --- Element-wise arithmetic ---

func TestOps_Add_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{4, 5, 6}, 3)
	c := Add(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{5, 7, 9})
}

func TestOps_AddScalar_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	c := AddScalar(a, 10.0)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{11, 12, 13})
}

func TestOps_Mul_Good(t *testing.T) {
	a := FromValues([]float32{2, 3, 4}, 3)
	b := FromValues([]float32{5, 6, 7}, 3)
	c := Mul(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{10, 18, 28})
}

func TestOps_MulScalar_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	c := MulScalar(a, 3.0)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{3, 6, 9})
}

func TestOps_Divide_Good(t *testing.T) {
	a := FromValues([]float32{10, 20, 30}, 3)
	b := FromValues([]float32{2, 5, 10}, 3)
	c := Divide(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{5, 4, 3})
}

func TestOps_Subtract_Good(t *testing.T) {
	a := FromValues([]float32{10, 20, 30}, 3)
	b := FromValues([]float32{1, 2, 3}, 3)
	c := Subtract(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{9, 18, 27})
}

func TestOps_Negative_Good(t *testing.T) {
	a := FromValues([]float32{1, -2, 3}, 3)
	c := Negative(a)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{-1, 2, -3})
}

// --- Math functions ---

func TestOps_Exp_Good(t *testing.T) {
	a := FromValues([]float32{0, 1, 2}, 3)
	c := Exp(a)
	Materialize(c)
	got := c.Floats()
	for i, x := range []float32{0, 1, 2} {
		want := float32(math.Exp(float64(x)))
		if !approx(float64(got[i]), float64(want)) {
			t.Errorf("Exp(%f) = %f, want %f", x, got[i], want)
		}
	}
}

func TestOps_Sigmoid_Good(t *testing.T) {
	a := FromValues([]float32{0, 100, -100}, 3)
	c := Sigmoid(a)
	Materialize(c)
	got := c.Floats()
	// sigmoid(0)=0.5, sigmoid(large)≈1, sigmoid(-large)≈0
	if !approx(float64(got[0]), 0.5) {
		t.Errorf("sigmoid(0) = %f, want 0.5", got[0])
	}
	if got[1] < 0.999 {
		t.Errorf("sigmoid(100) = %f, want ≈1.0", got[1])
	}
	if got[2] > 0.001 {
		t.Errorf("sigmoid(-100) = %f, want ≈0.0", got[2])
	}
}

func TestOps_SiLU_Good(t *testing.T) {
	// SiLU(x) = x * sigmoid(x)
	a := FromValues([]float32{0, 1, -1}, 3)
	c := SiLU(a)
	Materialize(c)
	got := c.Floats()
	// SiLU(0) = 0*0.5 = 0
	if !approx(float64(got[0]), 0.0) {
		t.Errorf("SiLU(0) = %f, want 0.0", got[0])
	}
	// SiLU(1) = 1 * sigmoid(1) = 1/(1+exp(-1)) ≈ 0.731059
	want := 1.0 / (1.0 + math.Exp(-1.0))
	if math.Abs(float64(got[1])-want) > 1e-4 {
		t.Errorf("SiLU(1) = %f, want %f", got[1], want)
	}
}

func TestOps_Tanh_Good(t *testing.T) {
	a := FromValues([]float32{0, 1, -1}, 3)
	c := Tanh(a)
	Materialize(c)
	got := c.Floats()
	for i, x := range []float32{0, 1, -1} {
		want := float32(math.Tanh(float64(x)))
		if !approx(float64(got[i]), float64(want)) {
			t.Errorf("Tanh(%f) = %f, want %f", x, got[i], want)
		}
	}
}

func TestOps_Sqrt_Good(t *testing.T) {
	a := FromValues([]float32{1, 4, 9, 16}, 4)
	c := Sqrt(a)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3, 4})
}

func TestOps_Rsqrt_Good(t *testing.T) {
	a := FromValues([]float32{1, 4, 16}, 3)
	c := Rsqrt(a)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1.0, 0.5, 0.25})
}

func TestOps_Reciprocal_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 4, 5}, 4)
	c := Reciprocal(a)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1.0, 0.5, 0.25, 0.2})
}

func TestOps_Square_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, -4}, 4)
	c := Square(a)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 4, 9, 16})
}

func TestOps_Power_Good(t *testing.T) {
	a := FromValues([]float32{2, 3, 4}, 3)
	b := FromValues([]float32{3, 2, 0.5}, 3)
	c := Power(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{8, 9, 2})
}

func TestOps_Maximum_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3}, 3)
	b := FromValues([]float32{4, 2, 6}, 3)
	c := Maximum(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{4, 5, 6})
}

func TestOps_Minimum_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3}, 3)
	b := FromValues([]float32{4, 2, 6}, 3)
	c := Minimum(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3})
}

// --- Matrix operations ---

func TestOps_Matmul_Good(t *testing.T) {
	// [1 2] @ [5 6]T = [1*5+2*7, 1*6+2*8] = [19, 22]
	// [3 4]   [7 8]    [3*5+4*7, 3*6+4*8]   [43, 50]
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	c := Matmul(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{19, 22, 43, 50})
}

func TestOps_Matmul_VectorMatrix_Good(t *testing.T) {
	// [1 2 3] @ [[1],[2],[3]] = [14]
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	b := FromValues([]float32{1, 2, 3}, 3, 1)
	c := Matmul(a, b)
	Materialize(c)

	if c.Size() != 1 {
		t.Fatalf("size = %d, want 1", c.Size())
	}
	if !approx(float64(c.Floats()[0]), 14.0) {
		t.Errorf("result = %f, want 14.0", c.Floats()[0])
	}
}

// --- Reductions ---

func TestOps_Softmax_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	c := Softmax(a)
	Materialize(c)

	got := c.Floats()
	// softmax values should sum to 1
	sum := float64(0)
	for _, v := range got {
		sum += float64(v)
	}
	if !approx(sum, 1.0) {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	// values should be monotonically increasing
	if got[0] >= got[1] || got[1] >= got[2] {
		t.Errorf("softmax not monotonic: %v", got)
	}
}

func TestOps_Argmax_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3, 2}, 1, 4)
	c := Argmax(a, -1, false)
	Materialize(c)

	if c.Int() != 1 {
		t.Errorf("argmax = %d, want 1", c.Int())
	}
}

func TestOps_TopK_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3, 7, 2}, 1, 5)
	c := TopK(a, 2)
	Materialize(c)

	got := c.Floats()
	if len(got) != 2 {
		t.Fatalf("topk returned %d elements, want 2", len(got))
	}
	// Top-2 from {1,5,3,7,2} should contain 7 and 5 (order not guaranteed)
	has7, has5 := false, false
	for _, v := range got {
		if v == 7 {
			has7 = true
		}
		if v == 5 {
			has5 = true
		}
	}
	if !has7 || !has5 {
		t.Errorf("topk = %v, want set {7, 5}", got)
	}
}

func TestOps_Sum_Good(t *testing.T) {
	// 2x3 matrix, sum along axis 1
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	c := Sum(a, 1, false)
	Materialize(c)
	// row 0: 1+2+3=6, row 1: 4+5+6=15
	floatSliceApprox(t, c.Floats(), []float32{6, 15})
}

func TestOps_Sum_KeepDims_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	c := Sum(a, 1, true)
	Materialize(c)

	if c.NumDims() != 2 {
		t.Errorf("ndim = %d, want 2 (keepDims)", c.NumDims())
	}
	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 1 {
		t.Errorf("shape = %v, want [2 1]", shape)
	}
}

func TestOps_Mean_Good(t *testing.T) {
	a := FromValues([]float32{2, 4, 6, 8}, 2, 2)
	c := Mean(a, 1, false)
	Materialize(c)
	// row 0: (2+4)/2=3, row 1: (6+8)/2=7
	floatSliceApprox(t, c.Floats(), []float32{3, 7})
}

func TestOps_LogSumExp_Axis_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	c := LogSumExp(a, -1, false)
	Materialize(c)

	// log(exp(1) + exp(2) + exp(3)) ≈ 3.4076
	want := math.Log(math.Exp(1) + math.Exp(2) + math.Exp(3))
	if !approx(c.Float(), want) {
		t.Errorf("LogSumExp = %f, want %f", c.Float(), want)
	}
}

// --- Shape operations ---

func TestOps_Reshape_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 6)
	c := Reshape(a, 2, 3)
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}
	// Data preserved
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3, 4, 5, 6})
}

func TestOps_Transpose_Good(t *testing.T) {
	// [[1 2 3], [4 5 6]] transposed -> shape [3 2]
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	c := Transpose(a)
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("shape = %v, want [3 2]", shape)
	}

	// Verify values via Reshape (forces contiguous copy)
	flat := Reshape(c, 6)
	Materialize(flat)
	floatSliceApprox(t, flat.Floats(), []float32{1, 4, 2, 5, 3, 6})
}

func TestOps_Transpose_WithAxes_Good(t *testing.T) {
	// 3D: (2,3,4) with axes (0,2,1) -> (2,4,3)
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	a := FromValues(data, 2, 3, 4)
	c := Transpose(a, 0, 2, 1)
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 4 || shape[2] != 3 {
		t.Errorf("shape = %v, want [2 4 3]", shape)
	}
}

func TestOps_ExpandDims_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	c := ExpandDims(a, 0)
	Materialize(c)

	shape := c.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 3 {
		t.Errorf("shape = %v, want [1 3]", shape)
	}
}

func TestOps_Squeeze_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	c := Squeeze(a, 0)
	Materialize(c)

	shape := c.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Errorf("shape = %v, want [3]", shape)
	}
}

func TestOps_Concatenate_Good(t *testing.T) {
	a := FromValues([]float32{1, 2}, 2)
	b := FromValues([]float32{3, 4, 5}, 3)
	c := Concatenate([]*Array{a, b}, 0)
	Materialize(c)

	if c.Size() != 5 {
		t.Fatalf("size = %d, want 5", c.Size())
	}
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3, 4, 5})
}

func TestOps_BroadcastTo_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	c := BroadcastTo(a, []int32{4, 3})
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 4 || shape[1] != 3 {
		t.Errorf("shape = %v, want [4 3]", shape)
	}
	if c.Size() != 12 {
		t.Errorf("size = %d, want 12", c.Size())
	}

	// Verify via Reshape (forces contiguous copy for broadcast views)
	flat := Reshape(c, 12)
	Materialize(flat)
	got := flat.Floats()
	want := []float32{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}
	floatSliceApprox(t, got, want)
}

func TestOps_AsType_Good(t *testing.T) {
	a := FromValues([]float32{1.5, 2.7, 3.9}, 3)
	c := AsType(a, DTypeInt32)
	Materialize(c)

	if c.Dtype() != DTypeInt32 {
		t.Errorf("dtype = %v, want int32", c.Dtype())
	}
	got := c.DataInt32()
	// Truncation to int
	want := []int32{1, 2, 3}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// --- Indexing ---

func TestOps_Take_Good(t *testing.T) {
	a := FromValues([]float32{10, 20, 30, 40, 50}, 5)
	indices := FromValues([]int32{0, 2, 4}, 3)
	c := Take(a, indices, 0)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{10, 30, 50})
}

func TestOps_Where_Good(t *testing.T) {
	cond := FromValues([]bool{true, false, true}, 3)
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{4, 5, 6}, 3)
	c := Where(cond, a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 5, 3})
}

func TestOps_TakeAlongAxis_Good(t *testing.T) {
	// 2x3 matrix, pick one element per row along axis 1
	a := FromValues([]float32{10, 20, 30, 40, 50, 60}, 2, 3)
	indices := FromValues([]int32{2, 0}, 2, 1) // row 0 pick col 2, row 1 pick col 0
	c := TakeAlongAxis(a, indices, 1)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{30, 40})
}

// --- Slicing ---

func TestOps_Slice_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	// Extract first row: [0:1, 0:3]
	c := Slice(a, []int32{0, 0}, []int32{1, 3})
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3})
}

func TestOps_SliceAxis_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	// Slice columns 1:3 from all rows
	c := SliceAxis(a, 1, 1, 3)
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 2 {
		t.Errorf("shape = %v, want [2 2]", shape)
	}
	// Reshape to force contiguous layout for value check
	flat := Reshape(c, 4)
	Materialize(flat)
	floatSliceApprox(t, flat.Floats(), []float32{2, 3, 5, 6})
}

func TestOps_SliceUpdateInplace_Good(t *testing.T) {
	a := Zeros([]int32{2, 3}, DTypeFloat32)
	update := FromValues([]float32{7, 8, 9}, 1, 3)
	// Put [7 8 9] in second row
	c := SliceUpdateInplace(a, update, []int32{1, 0}, []int32{2, 3})
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{0, 0, 0, 7, 8, 9})
}

// --- Broadcasting arithmetic ---

func TestOps_Add_Broadcasting_Good(t *testing.T) {
	// [2,3] + [1,3] should broadcast
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromValues([]float32{10, 20, 30}, 1, 3)
	c := Add(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{11, 22, 33, 14, 25, 36})
}

// --- Random ---

// --- Cumulative and sorting ops ---

func TestOps_CumSum_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	c := CumSum(a, -1, false, true)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 3, 6, 10})
}

func TestOps_CumSum_Exclusive_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	c := CumSum(a, -1, false, false) // exclusive
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{0, 1, 3, 6})
}

func TestOps_CumSum_Reverse_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	c := CumSum(a, -1, true, true) // reverse
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{10, 9, 7, 4})
}

func TestOps_Sort_Good(t *testing.T) {
	a := FromValues([]float32{3, 1, 4, 1, 5}, 1, 5)
	c := Sort(a, -1)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 1, 3, 4, 5})
}

func TestOps_Argsort_Good(t *testing.T) {
	a := FromValues([]float32{3, 1, 4, 1, 5}, 1, 5)
	c := Argsort(a, -1)
	Materialize(c)
	// indices of sorted order: [1, 3, 0, 2, 4]
	got := c.Ints()
	want := []int{1, 3, 0, 2, 4}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Argsort[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestOps_Greater_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3}, 3)
	b := FromValues([]float32{2, 2, 3}, 3)
	c := Greater(a, b)
	// Greater returns bool dtype — cast to int32 for data extraction
	c = AsType(c, DTypeInt32)
	Materialize(c)
	// 1>2=false, 5>2=true, 3>3=false
	got := c.DataInt32()
	want := []int32{0, 1, 0}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Greater[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestOps_MaxAxis_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
	c := MaxAxis(a, -1, false) // max per row
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{5, 6})
}

func TestOps_MaxAxis_KeepDims_Good(t *testing.T) {
	a := FromValues([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
	c := MaxAxis(a, -1, true)
	Materialize(c)

	shape := c.Shape()
	if shape[0] != 2 || shape[1] != 1 {
		t.Errorf("shape = %v, want [2 1]", shape)
	}
}

// --- Random ---

func TestOps_RandomCategorical_Good(t *testing.T) {
	// Heavily weighted towards index 2
	logprobs := FromValues([]float32{-100, -100, 0}, 1, 3)
	sample := RandomCategorical(logprobs)
	Materialize(sample)

	idx := sample.Int()
	if idx != 2 {
		t.Errorf("categorical sample = %d, want 2 (dominant logprob)", idx)
	}
}

func TestOps_RandomUniform_Good(t *testing.T) {
	a := RandomUniform(0, 1, []int32{100}, DTypeFloat32)
	Materialize(a)

	if a.Size() != 100 {
		t.Fatalf("size = %d, want 100", a.Size())
	}
	for i, v := range a.Floats() {
		if v < 0 || v >= 1 {
			t.Errorf("[%d] = %f, out of [0, 1) range", i, v)
		}
	}
}

// --- Any / AnyAxis ---

func TestOps_Any_AllFalse_Good(t *testing.T) {
	a := FromValues([]bool{false, false, false}, 3)
	c := Any(a, false)
	Materialize(c)
	if c.Bool() {
		t.Error("Any of all-false should be false")
	}
}

func TestOps_Any_SomeTrue_Good(t *testing.T) {
	a := FromValues([]bool{false, true, false}, 3)
	c := Any(a, false)
	Materialize(c)
	if !c.Bool() {
		t.Error("Any of [false, true, false] should be true")
	}
}

func TestOps_AnyAxis_PerRow_Good(t *testing.T) {
	// 2x3 bool matrix
	// row 0: [false, false, false] -> false
	// row 1: [false, true, false] -> true
	a := FromValues([]bool{false, false, false, false, true, false}, 2, 3)
	c := AnyAxis(a, 1, false)
	c = AsType(c, DTypeInt32)
	Materialize(c)
	got := c.DataInt32()
	want := []int32{0, 1}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("AnyAxis[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestOps_Any_KeepDims_Good(t *testing.T) {
	a := FromValues([]bool{true, false}, 1, 2)
	c := Any(a, true)
	Materialize(c)
	if c.NumDims() != 2 {
		t.Errorf("ndim = %d, want 2 (keepDims)", c.NumDims())
	}
}

func TestOps_Any_EmptyLike_Bad(t *testing.T) {
	// Single false element
	a := FromValues([]bool{false}, 1)
	c := Any(a, false)
	Materialize(c)
	if c.Bool() {
		t.Error("Any of single false should be false")
	}
}

// --- Arange ---

func TestOps_Arange_Int_Good(t *testing.T) {
	a := Arange(0, 5, 1, DTypeInt32)
	Materialize(a)

	if a.Size() != 5 {
		t.Fatalf("size = %d, want 5", a.Size())
	}
	got := a.DataInt32()
	want := []int32{0, 1, 2, 3, 4}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Arange[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestOps_Arange_Float_Good(t *testing.T) {
	a := Arange(0, 3, 0.5, DTypeFloat32)
	Materialize(a)

	if a.Size() != 6 {
		t.Fatalf("size = %d, want 6", a.Size())
	}
	floatSliceApprox(t, a.Floats(), []float32{0, 0.5, 1.0, 1.5, 2.0, 2.5})
}

func TestOps_Arange_Negative_Good(t *testing.T) {
	a := Arange(5, 0, -1, DTypeFloat32)
	Materialize(a)

	if a.Size() != 5 {
		t.Fatalf("size = %d, want 5", a.Size())
	}
	floatSliceApprox(t, a.Floats(), []float32{5, 4, 3, 2, 1})
}

func TestOps_Arange_EmptyRange_Bad(t *testing.T) {
	// start >= stop with positive step produces empty array
	a := Arange(5, 5, 1, DTypeFloat32)
	Materialize(a)

	if a.Size() != 0 {
		t.Errorf("size = %d, want 0 for empty range", a.Size())
	}
}

func TestOps_Arange_Float64_Ugly(t *testing.T) {
	// float64 is not supported on Metal GPU — Arange with DTypeFloat64
	// is expected to fail on Apple Silicon. Verify it fails gracefully.
	a := Arange(0, 3, 0.5, DTypeFloat64)
	if a.Valid() {
		// If it somehow succeeded (e.g. CPU fallback), verify correctness.
		Materialize(a)
		if a.Dtype() != DTypeFloat64 {
			t.Errorf("dtype = %v, want float64", a.Dtype())
		}
		if a.Size() != 6 {
			t.Fatalf("size = %d, want 6", a.Size())
		}
	} else {
		t.Log("float64 arange correctly unsupported on Metal GPU")
	}
	// Clear the global error state so subsequent tests are not affected.
	_ = lastError()
}

// --- IsNaN ---

func TestOps_IsNaN_NoNaN_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	c := IsNaN(a)
	c = AsType(c, DTypeInt32)
	Materialize(c)
	got := c.DataInt32()
	for i, v := range got {
		if v != 0 {
			t.Errorf("IsNaN[%d] = %d, want 0 (no NaN)", i, v)
		}
	}
}

func TestOps_IsNaN_WithNaN_Good(t *testing.T) {
	nan := float32(math.NaN())
	a := FromValues([]float32{1, nan, 3}, 3)
	c := IsNaN(a)
	c = AsType(c, DTypeInt32)
	Materialize(c)
	got := c.DataInt32()
	want := []int32{0, 1, 0}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("IsNaN[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestOps_IsNaN_AllNaN_Ugly(t *testing.T) {
	nan := float32(math.NaN())
	a := FromValues([]float32{nan, nan, nan}, 3)
	c := IsNaN(a)
	anyNaN := Any(c, false)
	Materialize(anyNaN)
	if !anyNaN.Bool() {
		t.Error("expected Any(IsNaN(all-NaN)) to be true")
	}
}
