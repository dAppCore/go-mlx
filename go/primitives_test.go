// SPDX-Licence-Identifier: EUPL-1.2

// Tests for primitives.go (the former training.go) — the root facade over
// the metal tensor + autodiff primitives. These run real Metal math on
// tiny synthetic arrays (no model weights, AX-11), asserting the wrappers
// forward shape + values correctly. The model-bound facade entries
// (ConcreteAdapter, TrainingModel) are exercised by the live example
// suite — a TrainableModel is required and cannot be faked here.

package mlx

import (
	"testing"
)

func TestPrimitives_MatMul_Good(t *testing.T) {
	// [2x2] identity · [2x2] = the same matrix.
	a := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	defer Free(a, b)
	out := MatMul(a, b)
	defer Free(out)
	Materialize(out)
	got := out.Floats()
	want := []float32{5, 6, 7, 8}
	if len(got) != len(want) {
		t.Fatalf("MatMul len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatMul[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}
}

func TestPrimitives_Add_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{10, 20, 30}, 3)
	defer Free(a, b)
	out := Add(a, b)
	defer Free(out)
	Materialize(out)
	got := out.Floats()
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Add[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}
}

func TestPrimitives_Mul_Good(t *testing.T) {
	a := FromValues([]float32{2, 3, 4}, 3)
	b := FromValues([]float32{5, 6, 7}, 3)
	defer Free(a, b)
	out := Mul(a, b)
	defer Free(out)
	Materialize(out)
	got := out.Floats()
	want := []float32{10, 18, 28}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Mul[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}
}

func TestPrimitives_Softmax_Good(t *testing.T) {
	// Equal logits → uniform distribution that sums to 1.
	a := FromValues([]float32{1, 1, 1, 1}, 1, 4)
	defer Free(a)
	out := Softmax(a)
	defer Free(out)
	Materialize(out)
	got := out.Floats()
	if len(got) != 4 {
		t.Fatalf("Softmax len = %d, want 4", len(got))
	}
	var sum float32
	for _, v := range got {
		sum += v
		if v < 0.24 || v > 0.26 {
			t.Fatalf("Softmax element = %v, want ~0.25 (%v)", v, got)
		}
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("Softmax sum = %v, want ~1.0", sum)
	}
}

func TestPrimitives_Slice_Good(t *testing.T) {
	// Take rows [1,2) of a 3x2 matrix along axis 0 → the middle row.
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	defer Free(a)
	out := Slice(a, 1, 2, 0)
	defer Free(out)
	Materialize(out)
	got := out.Floats()
	want := []float32{3, 4}
	if len(got) != len(want) {
		t.Fatalf("Slice len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Slice[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}
}

func TestPrimitives_Slice_Ugly(t *testing.T) {
	// Slice accepts mixed int-compatible kinds for axis/start/end; an
	// out-of-int32-range argument is the documented panic path.
	a := FromValues([]float32{1, 2}, 2)
	defer Free(a)
	defer func() {
		if recover() == nil {
			t.Fatal("Slice(out-of-range start) did not panic")
		}
	}()
	out := Slice(a, int64(1)<<40, 1, 0)
	Free(out)
}

func TestPrimitives_Reshape_Good(t *testing.T) {
	// 6 elements reshaped 1x6 → 2x3, values preserved row-major.
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 6)
	defer Free(a)
	out := Reshape(a, 2, 3)
	defer Free(out)
	Materialize(out)
	if shape := out.Shape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("Reshape shape = %v, want [2 3]", shape)
	}
	got := out.Floats()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Reshape[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPrimitives_Reshape_Bad(t *testing.T) {
	// A shape whose product mismatches the element count does not panic:
	// the metal layer records the failure on the array, so the result is
	// an invalid (empty) array rather than a thrown error.
	a := FromValues([]float32{1, 2, 3, 4}, 4)
	defer Free(a)
	out := Reshape(a, 3, 3)
	defer Free(out)
	if out.Valid() {
		t.Fatalf("Reshape(incompatible) Valid = true, want false (shape=%v)", out.Shape())
	}
}

func TestPrimitives_VJP_Good(t *testing.T) {
	// f(x) = x*x; VJP with cotangent 1 gives df/dx = 2x. At x=3 → 6.
	primals := []*Array{FromValues([]float32{3}, 1)}
	cotangents := []*Array{FromValues([]float32{1}, 1)}
	defer Free(primals[0], cotangents[0])
	outputs, vjps, err := VJP(func(in []*Array) []*Array {
		return []*Array{Mul(in[0], in[0])}
	}, primals, cotangents)
	if err != nil {
		t.Fatalf("VJP error = %v", err)
	}
	defer Free(append(outputs, vjps...)...)
	if len(vjps) != 1 {
		t.Fatalf("VJP returned %d gradients, want 1", len(vjps))
	}
	Materialize(vjps[0])
	if got := vjps[0].Floats(); len(got) != 1 || got[0] != 6 {
		t.Fatalf("VJP gradient = %v, want [6] (d/dx x^2 at 3)", got)
	}
}

func TestPrimitives_JVP_Good(t *testing.T) {
	// f(x) = x*x; JVP with tangent 1 gives df = 2x. At x=4 → 8.
	primals := []*Array{FromValues([]float32{4}, 1)}
	tangents := []*Array{FromValues([]float32{1}, 1)}
	defer Free(primals[0], tangents[0])
	outputs, jvps, err := JVP(func(in []*Array) []*Array {
		return []*Array{Mul(in[0], in[0])}
	}, primals, tangents)
	if err != nil {
		t.Fatalf("JVP error = %v", err)
	}
	defer Free(append(outputs, jvps...)...)
	if len(jvps) != 1 {
		t.Fatalf("JVP returned %d tangents, want 1", len(jvps))
	}
	Materialize(jvps[0])
	if got := jvps[0].Floats(); len(got) != 1 || got[0] != 8 {
		t.Fatalf("JVP tangent = %v, want [8] (d/dx x^2 at 4)", got)
	}
}
