//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestVJP_SimpleSquare(t *testing.T) {
	// f(x) = x^2, df/dx = 2x
	// At x=3: f(3)=9, df/dx=6
	fn := func(inputs []*Array) []*Array {
		x := inputs[0]
		return []*Array{Mul(x, x)}
	}

	x := FromValue(float32(3.0))
	cotangent := FromValue(float32(1.0)) // upstream grad = 1

	outputs, grads, err := VJP(fn, []*Array{x}, []*Array{cotangent})
	if err != nil {
		t.Fatalf("VJP failed: %v", err)
	}

	Materialize(outputs[0], grads[0])

	got := outputs[0].Float()
	if math.Abs(got-9.0) > 1e-5 {
		t.Errorf("output = %f, want 9.0", got)
	}

	grad := grads[0].Float()
	if math.Abs(grad-6.0) > 1e-5 {
		t.Errorf("grad = %f, want 6.0", grad)
	}
}

func TestVJP_Addition(t *testing.T) {
	// f(x, y) = x + y, df/dx = 1, df/dy = 1
	fn := func(inputs []*Array) []*Array {
		return []*Array{Add(inputs[0], inputs[1])}
	}

	x := FromValue(float32(2.0))
	y := FromValue(float32(5.0))
	cotangent := FromValue(float32(1.0))

	_, grads, err := VJP(fn, []*Array{x, y}, []*Array{cotangent})
	if err != nil {
		t.Fatalf("VJP failed: %v", err)
	}

	Materialize(grads...)

	if math.Abs(grads[0].Float()-1.0) > 1e-5 {
		t.Errorf("dx = %f, want 1.0", grads[0].Float())
	}
	if math.Abs(grads[1].Float()-1.0) > 1e-5 {
		t.Errorf("dy = %f, want 1.0", grads[1].Float())
	}
}

func TestVJP_MatmulGrad(t *testing.T) {
	// f(W) = sum(W @ x) — gradient of sum(matmul) w.r.t. W
	// For W=[2,2], x=[2,1]: dL/dW = ones @ x^T
	x := FromValues([]float32{1.0, 2.0}, 2, 1)
	w := FromValues([]float32{1.0, 0.0, 0.0, 1.0}, 2, 2) // identity

	fn := func(inputs []*Array) []*Array {
		result := Matmul(inputs[0], x)
		return []*Array{SumAll(result)}
	}

	cotangent := FromValue(float32(1.0))

	outputs, grads, err := VJP(fn, []*Array{w}, []*Array{cotangent})
	if err != nil {
		t.Fatalf("VJP failed: %v", err)
	}

	Materialize(outputs[0], grads[0])

	// W @ x with W=I, x=[1,2]^T gives [1,2]^T, sum=3
	got := outputs[0].Float()
	if math.Abs(got-3.0) > 1e-5 {
		t.Errorf("output = %f, want 3.0", got)
	}

	// Gradient of sum(W@x) w.r.t. W is outer product: ones @ x^T
	// = [[1,2],[1,2]]
	gradFloats := grads[0].Floats()
	expected := []float32{1.0, 2.0, 1.0, 2.0}
	for i, exp := range expected {
		if math.Abs(float64(gradFloats[i]-exp)) > 1e-5 {
			t.Errorf("grad[%d] = %f, want %f", i, gradFloats[i], exp)
		}
	}
}

func TestJVP_SimpleSquare(t *testing.T) {
	// f(x) = x^2, JVP with tangent v: df = 2x * v
	// At x=3, v=1: df = 6
	fn := func(inputs []*Array) []*Array {
		x := inputs[0]
		return []*Array{Mul(x, x)}
	}

	x := FromValue(float32(3.0))
	tangent := FromValue(float32(1.0))

	outputs, jvps, err := JVP(fn, []*Array{x}, []*Array{tangent})
	if err != nil {
		t.Fatalf("JVP failed: %v", err)
	}

	Materialize(outputs[0], jvps[0])

	got := outputs[0].Float()
	if math.Abs(got-9.0) > 1e-5 {
		t.Errorf("output = %f, want 9.0", got)
	}

	jvp := jvps[0].Float()
	if math.Abs(jvp-6.0) > 1e-5 {
		t.Errorf("jvp = %f, want 6.0", jvp)
	}
}

func TestValueAndGrad_Quadratic(t *testing.T) {
	// f(x) = x^2 + 2x + 1 = (x+1)^2
	// f'(x) = 2x + 2
	// At x=3: f(3) = 16, f'(3) = 8
	fn := func(inputs []*Array) []*Array {
		x := inputs[0]
		x2 := Mul(x, x)
		two_x := MulScalar(x, 2.0)
		one := FromValue(float32(1.0))
		return []*Array{Add(Add(x2, two_x), one)}
	}

	grad := ValueAndGrad(fn, 0)
	defer grad.Free()

	x := FromValue(float32(3.0))
	values, grads, err := grad.Apply(x)
	if err != nil {
		t.Fatalf("ValueAndGrad failed: %v", err)
	}

	Materialize(values[0], grads[0])

	val := values[0].Float()
	if math.Abs(val-16.0) > 1e-5 {
		t.Errorf("value = %f, want 16.0", val)
	}

	g := grads[0].Float()
	if math.Abs(g-8.0) > 1e-5 {
		t.Errorf("grad = %f, want 8.0", g)
	}
}

func TestValueAndGrad_MultiArg(t *testing.T) {
	// f(x, y) = x*y, df/dx = y, df/dy = x
	// At x=3, y=4: f=12, dx=4, dy=3
	fn := func(inputs []*Array) []*Array {
		return []*Array{Mul(inputs[0], inputs[1])}
	}

	// Differentiate w.r.t. both arguments
	grad := ValueAndGrad(fn, 0, 1)
	defer grad.Free()

	x := FromValue(float32(3.0))
	y := FromValue(float32(4.0))
	values, grads, err := grad.Apply(x, y)
	if err != nil {
		t.Fatalf("ValueAndGrad failed: %v", err)
	}

	Materialize(values[0], grads[0], grads[1])

	val := values[0].Float()
	if math.Abs(val-12.0) > 1e-5 {
		t.Errorf("value = %f, want 12.0", val)
	}

	dx := grads[0].Float()
	if math.Abs(dx-4.0) > 1e-5 {
		t.Errorf("dx = %f, want 4.0 (y)", dx)
	}

	dy := grads[1].Float()
	if math.Abs(dy-3.0) > 1e-5 {
		t.Errorf("dy = %f, want 3.0 (x)", dy)
	}
}

func TestValueAndGrad_Reusable(t *testing.T) {
	// Verify GradFn can be called multiple times
	fn := func(inputs []*Array) []*Array {
		x := inputs[0]
		return []*Array{Mul(x, x)} // x^2, grad = 2x
	}

	grad := ValueAndGrad(fn)
	defer grad.Free()

	for _, tc := range []struct {
		x    float32
		want float64 // expected gradient
	}{
		{2.0, 4.0},
		{5.0, 10.0},
		{-3.0, -6.0},
		{0.0, 0.0},
	} {
		x := FromValue(tc.x)
		_, grads, err := grad.Apply(x)
		if err != nil {
			t.Fatalf("Apply failed for x=%f: %v", tc.x, err)
		}
		Materialize(grads[0])

		g := grads[0].Float()
		if math.Abs(g-tc.want) > 1e-5 {
			t.Errorf("x=%f: grad = %f, want %f", tc.x, g, tc.want)
		}
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Simple 3-class classification
	// logits = [1.0, 2.0, 3.0], target = 2 (class index)
	// Manual: logsumexp([1,2,3]) = 3 + log(exp(-2)+exp(-1)+1)
	//       = 3 + log(0.1353 + 0.3679 + 1.0) = 3 + log(1.5032) = 3.4076
	// loss = 3.4076 - 3.0 = 0.4076
	logits := FromValues([]float32{1.0, 2.0, 3.0}, 1, 3) // [1, 3]
	targets := FromValues([]int32{2}, 1)                   // [1]

	loss := CrossEntropyLoss(logits, targets)
	Materialize(loss)

	got := loss.Float()
	expected := 0.4076
	if math.Abs(got-expected) > 0.01 {
		t.Errorf("CrossEntropyLoss = %f, want ~%f", got, expected)
	}
}

func TestMSELoss(t *testing.T) {
	pred := FromValues([]float32{1.0, 2.0, 3.0}, 3)
	target := FromValues([]float32{1.5, 2.5, 3.5}, 3)

	loss := MSELoss(pred, target)
	Materialize(loss)

	// MSE = mean((0.5)^2, (0.5)^2, (0.5)^2) = mean(0.25, 0.25, 0.25) = 0.25
	got := loss.Float()
	if math.Abs(got-0.25) > 1e-5 {
		t.Errorf("MSELoss = %f, want 0.25", got)
	}
}

func TestLogSumExp(t *testing.T) {
	// logsumexp([1, 2, 3]) along axis -1
	a := FromValues([]float32{1.0, 2.0, 3.0}, 1, 3)
	result := LogSumExp(a, -1, false)
	Materialize(result)

	// = 3 + log(exp(-2) + exp(-1) + 1) = 3 + log(1.5032) ≈ 3.4076
	got := result.Float()
	expected := 3.4076
	if math.Abs(got-expected) > 0.01 {
		t.Errorf("LogSumExp = %f, want ~%f", got, expected)
	}
}

func TestOnesLike(t *testing.T) {
	a := FromValues([]float32{1.0, 2.0, 3.0}, 3)
	ones := OnesLike(a)
	Materialize(ones)

	floats := ones.Floats()
	for i, f := range floats {
		if f != 1.0 {
			t.Errorf("OnesLike[%d] = %f, want 1.0", i, f)
		}
	}
}

func TestCheckpoint(t *testing.T) {
	// Checkpoint should produce the same result as the original function
	fn := func(inputs []*Array) []*Array {
		x := inputs[0]
		return []*Array{Mul(x, x)}
	}

	cpFn := Checkpoint(fn)

	x := FromValue(float32(5.0))
	result := cpFn([]*Array{x})
	Materialize(result[0])

	got := result[0].Float()
	if math.Abs(got-25.0) > 1e-5 {
		t.Errorf("Checkpoint result = %f, want 25.0", got)
	}
}

func TestSumAll(t *testing.T) {
	a := FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
	result := SumAll(a)
	Materialize(result)

	got := result.Float()
	if math.Abs(got-10.0) > 1e-5 {
		t.Errorf("SumAll = %f, want 10.0", got)
	}
}

func TestMeanAll(t *testing.T) {
	a := FromValues([]float32{2.0, 4.0, 6.0, 8.0}, 2, 2)
	result := MeanAll(a)
	Materialize(result)

	got := result.Float()
	if math.Abs(got-5.0) > 1e-5 {
		t.Errorf("MeanAll = %f, want 5.0", got)
	}
}
