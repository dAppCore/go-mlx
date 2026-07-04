// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleVJP() {
	x := FromValue(float32(3))
	cotangent := FromValue(float32(1))
	defer Free(x, cotangent)

	outputs, grads, err := VJP(func(inputs []*Array) []*Array {
		return []*Array{Mul(inputs[0], inputs[0])}
	}, []*Array{x}, []*Array{cotangent})
	if err != nil {
		core.Println(err)
		return
	}
	defer Free(outputs...)
	defer Free(grads...)
	Materialize(outputs[0], grads[0])

	core.Println(core.Sprintf("out=%.0f grad=%.0f", outputs[0].Float(), grads[0].Float()))
	// Output: out=9 grad=6
}

func ExampleJVP() {
	x := FromValue(float32(3))
	tangent := FromValue(float32(1))
	defer Free(x, tangent)

	outputs, tangents, err := JVP(func(inputs []*Array) []*Array {
		return []*Array{Mul(inputs[0], inputs[0])}
	}, []*Array{x}, []*Array{tangent})
	if err != nil {
		core.Println(err)
		return
	}
	defer Free(outputs...)
	defer Free(tangents...)
	Materialize(outputs[0], tangents[0])

	core.Println(core.Sprintf("out=%.0f tangent=%.0f", outputs[0].Float(), tangents[0].Float()))
	// Output: out=9 tangent=6
}

func ExampleValueAndGrad() {
	grad := ValueAndGrad(func(inputs []*Array) []*Array {
		return []*Array{inputs[0]}
	}, 0)
	defer grad.Free()

	core.Println(grad != nil, grad.cls.ctx != nil)
	// Output: true true
}

func ExampleGradFn_Apply() {
	grad := ValueAndGrad(func(inputs []*Array) []*Array {
		x := inputs[0]
		return []*Array{Add(Mul(x, x), MulScalar(x, 2))}
	}, 0)
	defer grad.Free()

	x := FromValue(float32(3))
	defer Free(x)
	values, grads, err := grad.Apply(x)
	if err != nil {
		core.Println(err)
		return
	}
	defer Free(values...)
	defer Free(grads...)
	Materialize(values[0], grads[0])

	core.Println(core.Sprintf("value=%.0f grad=%.0f", values[0].Float(), grads[0].Float()))
	// Output: value=15 grad=8
}

func ExampleGradFn_Free() {
	grad := ValueAndGrad(func(inputs []*Array) []*Array {
		return []*Array{inputs[0]}
	})
	before := grad.cls.ctx != nil
	grad.Free()

	core.Println(before, grad.cls.ctx == nil)
	// Output: true true
}

func ExampleCheckpoint() {
	checkpointed := Checkpoint(func(inputs []*Array) []*Array {
		return []*Array{Mul(inputs[0], inputs[0])}
	})
	x := FromValue(float32(5))
	defer Free(x)
	out := checkpointed([]*Array{x})
	defer Free(out...)
	Materialize(out[0])

	core.Println(core.Sprintf("value=%.0f", out[0].Float()))
	// Output: value=25
}

func ExampleCrossEntropyLoss() {
	logits := FromValues([]float32{0, 2}, 1, 1, 2)
	targets := FromValues([]int32{1}, 1, 1)
	defer Free(logits, targets)

	loss := CrossEntropyLoss(logits, targets)
	defer Free(loss)
	Materialize(loss)

	core.Println(core.Sprintf("loss=%.3f dims=%d", loss.Float(), loss.NumDims()))
	// Output: loss=0.127 dims=0
}

func ExampleMaskedCrossEntropyLoss() {
	logits := FromValues([]float32{0, 2, 3, 1}, 1, 2, 2)
	targets := FromValues([]int32{1, 0}, 1, 2)
	mask := FromValues([]float32{1, 0}, 1, 2)
	defer Free(logits, targets, mask)

	loss := MaskedCrossEntropyLoss(logits, targets, mask)
	defer Free(loss)
	Materialize(loss)

	core.Println(core.Sprintf("loss=%.3f dims=%d", loss.Float(), loss.NumDims()))
	// Output: loss=0.127 dims=0
}

func ExampleMSELoss() {
	predictions := FromValues([]float32{1, 2, 3}, 3)
	targets := FromValues([]float32{1.5, 2.5, 3.5}, 3)
	defer Free(predictions, targets)

	loss := MSELoss(predictions, targets)
	defer Free(loss)
	Materialize(loss)

	core.Println(core.Sprintf("loss=%.2f", loss.Float()))
	// Output: loss=0.25
}

func ExampleLog() {
	values := FromValues([]float32{1}, 1)
	defer Free(values)
	logValues := Log(values)
	defer Free(logValues)
	Materialize(logValues)

	core.Println(logValues.Floats())
	// Output: [0]
}

func ExampleSumAll() {
	values := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer Free(values)
	sum := SumAll(values)
	defer Free(sum)
	Materialize(sum)

	core.Println(core.Sprintf("sum=%.0f dims=%d", sum.Float(), sum.NumDims()))
	// Output: sum=10 dims=0
}

func ExampleMeanAll() {
	values := FromValues([]float32{2, 4, 6, 8}, 2, 2)
	defer Free(values)
	mean := MeanAll(values)
	defer Free(mean)
	Materialize(mean)

	core.Println(core.Sprintf("mean=%.0f dims=%d", mean.Float(), mean.NumDims()))
	// Output: mean=5 dims=0
}

func ExampleOnesLike() {
	values := FromValues([]float32{2, 4, 6}, 3)
	defer Free(values)
	ones := OnesLike(values)
	defer Free(ones)
	Materialize(ones)

	core.Println(ones.Shape(), ones.Floats())
	// Output: [3] [1 1 1]
}
