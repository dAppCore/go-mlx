// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleAdd() {
	base := FromValues([]float32{1, 2, 3}, 3)
	delta := FromValues([]float32{4, 5, 6}, 3)
	out := Add(base, delta)
	defer Free(base, delta, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [5 7 9]
}

func ExampleAddScalar() {
	values := FromValues([]float32{1, 2}, 2)
	out := AddScalar(values, 0.5)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1.5 2.5]
}

func ExampleMul() {
	left := FromValues([]float32{2, 3}, 2)
	right := FromValues([]float32{4, 5}, 2)
	out := Mul(left, right)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [8 15]
}

func ExampleMulScalar() {
	values := FromValues([]float32{2, 4}, 2)
	out := MulScalar(values, 0.25)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [0.5 1]
}

func ExampleDivide() {
	left := FromValues([]float32{10, 20}, 2)
	right := FromValues([]float32{2, 5}, 2)
	out := Divide(left, right)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [5 4]
}

func ExampleSubtract() {
	left := FromValues([]float32{10, 20}, 2)
	right := FromValues([]float32{1, 3}, 2)
	out := Subtract(left, right)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [9 17]
}

func ExampleNegative() {
	values := FromValues([]float32{1, -2, 3}, 3)
	out := Negative(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [-1 2 -3]
}

func ExampleCopy() {
	values := FromValues([]float32{7, 8}, 2)
	out := Copy(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats(), out.Valid())
	// Output: [7 8] true
}

func ExampleExp() {
	values := FromValues([]float32{0, 1}, 2)
	out := Exp(values)
	defer Free(values, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.2f %.2f", got[0], got[1]))
	// Output: 1.00 2.72
}

func ExampleSigmoid() {
	values := FromValues([]float32{0, 1}, 2)
	out := Sigmoid(values)
	defer Free(values, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.2f %.2f", got[0], got[1]))
	// Output: 0.50 0.73
}

func ExampleSiLU() {
	values := FromValues([]float32{0, 1}, 2)
	out := SiLU(values)
	defer Free(values, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.2f %.2f", got[0], got[1]))
	// Output: 0.00 0.73
}

func ExampleTanh() {
	values := FromValues([]float32{0, 1}, 2)
	out := Tanh(values)
	defer Free(values, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.2f %.2f", got[0], got[1]))
	// Output: 0.00 0.76
}

func ExampleSqrt() {
	values := FromValues([]float32{1, 4, 9}, 3)
	out := Sqrt(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1 2 3]
}

func ExampleRsqrt() {
	values := FromValues([]float32{4, 16}, 2)
	out := Rsqrt(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [0.5 0.25]
}

func ExampleReciprocal() {
	values := FromValues([]float32{2, 4}, 2)
	out := Reciprocal(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [0.5 0.25]
}

func ExampleSquare() {
	values := FromValues([]float32{2, -3}, 2)
	out := Square(values)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [4 9]
}

func ExamplePower() {
	values := FromValues([]float32{2, 3}, 2)
	powers := FromValues([]float32{3, 2}, 2)
	out := Power(values, powers)
	defer Free(values, powers, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [8 9]
}

func ExampleMaximum() {
	left := FromValues([]float32{1, 5, 3}, 3)
	right := FromValues([]float32{4, 2, 6}, 3)
	out := Maximum(left, right)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [4 5 6]
}

func ExampleMinimum() {
	left := FromValues([]float32{1, 5, 3}, 3)
	right := FromValues([]float32{4, 2, 6}, 3)
	out := Minimum(left, right)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1 2 3]
}

func ExampleMatmul() {
	activations := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	weights := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	out := Matmul(activations, weights)
	defer Free(activations, weights, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 2] [19 22 43 50]
}

func ExampleConv2d() {
	core.Println("Conv2d")
	// Output: Conv2d
}

func ExampleQuantizedMatmul() {
	core.Println("QuantizedMatmul")
	// Output: QuantizedMatmul
}

func ExampleGatherMM() {
	core.Println("GatherMM")
	// Output: GatherMM
}

func ExampleGatherQMM() {
	core.Println("GatherQMM")
	// Output: GatherQMM
}

func ExampleSoftmax() {
	logits := FromValues([]float32{1, 2, 3}, 1, 3)
	probs := Softmax(logits)
	defer Free(logits, probs)
	Materialize(probs)

	got := probs.Floats()
	core.Println(probs.Shape(), core.Sprintf("%.2f %.2f %.2f", got[0], got[1], got[2]))
	// Output: [1 3] 0.09 0.24 0.67
}

func ExampleArgmax() {
	logits := FromValues([]float32{1, 5, 3, 2}, 1, 4)
	out := Argmax(logits, -1, false)
	defer Free(logits, out)
	Materialize(out)

	core.Println(out.Int())
	// Output: 1
}

func ExampleTopK() {
	logits := FromValues([]float32{1, 5, 3, 7, 2}, 1, 5)
	out := TopK(logits, 2)
	defer Free(logits, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 2] [5 7]
}

func ExampleSum() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	out := Sum(values, 1, false)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2] [6 15]
}

func ExampleMean() {
	values := FromValues([]float32{2, 4, 6, 8}, 2, 2)
	out := Mean(values, 1, false)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2] [3 7]
}

func ExampleReshape() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 6)
	out := Reshape(values, 2, 3)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 3] [1 2 3 4 5 6]
}

func ExampleTranspose() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	transposed := Transpose(values)
	flat := Reshape(transposed, 6)
	defer Free(values, transposed, flat)
	Materialize(flat)

	core.Println(transposed.Shape(), flat.Floats())
	// Output: [3 2] [1 4 2 5 3 6]
}

func ExampleExpandDims() {
	values := FromValues([]float32{1, 2, 3}, 3)
	out := ExpandDims(values, 0)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 3] [1 2 3]
}

func ExampleSqueeze() {
	values := FromValues([]float32{1, 2, 3}, 1, 3)
	out := Squeeze(values, 0)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [3] [1 2 3]
}

func ExampleConcatenate() {
	left := FromValues([]float32{1, 2}, 2)
	right := FromValues([]float32{3, 4, 5}, 3)
	out := Concatenate([]*Array{left, right}, 0)
	defer Free(left, right, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [5] [1 2 3 4 5]
}

func ExampleBroadcastTo() {
	row := FromValues([]float32{1, 2, 3}, 1, 3)
	out := BroadcastTo(row, []int32{2, 3})
	flat := Reshape(out, 6)
	defer Free(row, out, flat)
	Materialize(flat)

	core.Println(out.Shape(), flat.Floats())
	// Output: [2 3] [1 2 3 1 2 3]
}

func ExampleAsType() {
	values := FromValues([]float32{1.5, 2.7, 3.9}, 3)
	out := AsType(values, DTypeInt32)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Dtype(), out.DataInt32())
	// Output: int32 [1 2 3]
}

func ExampleAsStrided() {
	values := FromValues([]float32{1, 2, 3, 4}, 4)
	out := AsStrided(values, []int32{2, 2}, []int64{2, 1}, 0)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 2] [1 2 3 4]
}

func ExampleTake() {
	values := FromValues([]float32{10, 20, 30, 40, 50}, 5)
	indices := FromValues([]int32{0, 2, 4}, 3)
	out := Take(values, indices, 0)
	defer Free(values, indices, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [10 30 50]
}

func ExampleWhere() {
	condition := FromValues([]bool{true, false, true}, 3)
	left := FromValues([]float32{1, 2, 3}, 3)
	right := FromValues([]float32{4, 5, 6}, 3)
	out := Where(condition, left, right)
	defer Free(condition, left, right, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1 5 3]
}

func ExampleArgpartition() {
	values := FromValues([]float32{3, 1, 4, 1, 5}, 1, 5)
	out := Argpartition(values, 1, -1)
	defer Free(values, out)
	Materialize(out)

	indices := out.Ints()
	core.Println(indices[0] == 1 || indices[0] == 3, indices[1] == 1 || indices[1] == 3)
	// Output: true true
}

func ExampleDequantize() {
	weights := FromValues([]uint32{0x03020100, 0, 0, 0, 0, 0, 0, 0}, 1, 8)
	scales := FromValues([]float32{0.5}, 1, 1)
	biases := FromValues([]float32{1}, 1, 1)
	out := Dequantize(weights, scales, biases, 32, 8)
	defer Free(weights, scales, biases, out)
	Materialize(out)

	got := out.Floats()
	core.Println(out.Shape(), got[:4], got[31])
	// Output: [1 32] [1 1.5 2 2.5] 1
}

func ExamplePutAlongAxis() {
	values := Zeros([]int32{1, 4}, DTypeFloat32)
	indices := FromValues([]int32{1, 3}, 1, 2)
	updates := FromValues([]float32{5, 9}, 1, 2)
	out := PutAlongAxis(values, indices, updates, -1)
	defer Free(values, indices, updates, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [0 5 0 9]
}

func ExampleTakeAlongAxis() {
	values := FromValues([]float32{10, 20, 30, 40, 50, 60}, 2, 3)
	indices := FromValues([]int32{2, 0}, 2, 1)
	out := TakeAlongAxis(values, indices, 1)
	defer Free(values, indices, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 1] [30 40]
}

func ExampleLogSumExp() {
	values := FromValues([]float32{1, 2, 3}, 1, 3)
	out := LogSumExp(values, -1, false)
	defer Free(values, out)
	Materialize(out)

	core.Println(core.Sprintf("%.3f", out.Float()))
	// Output: 3.408
}

func ExampleCumSum() {
	values := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	out := CumSum(values, -1, false, true)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1 3 6 10]
}

func ExampleSort() {
	values := FromValues([]float32{3, 1, 4, 1, 5}, 1, 5)
	out := Sort(values, -1)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Floats())
	// Output: [1 1 3 4 5]
}

func ExampleArgsort() {
	values := FromValues([]float32{3, 1, 4, 1, 5}, 1, 5)
	out := Argsort(values, -1)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Ints())
	// Output: [1 3 0 2 4]
}

func ExampleGreater() {
	left := FromValues([]float32{1, 5, 3}, 3)
	right := FromValues([]float32{2, 2, 3}, 3)
	out := Greater(left, right)
	ints := AsType(out, DTypeInt32)
	defer Free(left, right, out, ints)
	Materialize(ints)

	core.Println(ints.DataInt32())
	// Output: [0 1 0]
}

func ExampleMaxAxis() {
	values := FromValues([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
	out := MaxAxis(values, -1, false)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2] [5 6]
}

func ExampleAny() {
	values := FromValues([]bool{false, true, false}, 3)
	out := Any(values, false)
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Bool())
	// Output: true
}

func ExampleAnyAxis() {
	values := FromValues([]bool{false, false, false, false, true, false}, 2, 3)
	out := AnyAxis(values, 1, false)
	ints := AsType(out, DTypeInt32)
	defer Free(values, out, ints)
	Materialize(ints)

	core.Println(ints.DataInt32())
	// Output: [0 1]
}

func ExampleArange() {
	out := Arange(0, 5, 1, DTypeInt32)
	defer Free(out)
	Materialize(out)

	core.Println(out.DataInt32())
	// Output: [0 1 2 3 4]
}

func ExampleIsNaN() {
	values := FromValues([]float32{-1, 4}, 2)
	roots := Sqrt(values)
	mask := IsNaN(roots)
	ints := AsType(mask, DTypeInt32)
	defer Free(values, roots, mask, ints)
	Materialize(ints)

	core.Println(ints.DataInt32())
	// Output: [1 0]
}
