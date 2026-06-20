// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// ops_cover_test.go covers the tensor-op wrappers the core arithmetic suite
// leaves dark: the convolutions, axis padding, the equality and floor-division
// element ops, and the rank-3 reshape fast path. Synthetic arrays, value-checked
// against hand-computed expectations.

// FloorDivide is element-wise floor(a/b).
func TestOps_FloorDivide_Good(t *testing.T) {
	a := FromValues([]float32{7, 8, 9, 10}, 4)
	b := FromValues([]float32{2, 2, 4, 3}, 4)
	c := FloorDivide(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{3, 4, 2, 3})
}

// Equal returns 1 where elements match, 0 otherwise.
func TestOps_Equal_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 4)
	b := FromValues([]float32{1, 9, 3, 9}, 4)
	c := Equal(a, b)
	Materialize(c)
	floatSliceApprox(t, c.Floats(), []float32{1, 0, 1, 0})
}

// Reshape3 reshapes to a rank-3 shape; the element data is preserved in order.
func TestOps_Reshape3_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 12)
	c := Reshape3(a, 2, 3, 2)
	Materialize(c)
	if c.NumDims() != 3 || c.Dim(0) != 2 || c.Dim(1) != 3 || c.Dim(2) != 2 {
		t.Fatalf("Reshape3 shape = %v, want [2 3 2]", c.Shape())
	}
	floatSliceApprox(t, c.Floats(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
}

// PadAxis zero-pads one axis by low/high. Padding axis 0 of [1,2,3] with (2,1)
// yields [0,0,1,2,3,0].
func TestOps_PadAxis_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	c := PadAxis(a, 0, 2, 1)
	Materialize(c)
	if c.Dim(0) != 6 {
		t.Fatalf("PadAxis len = %d, want 6 (3 + 2 + 1)", c.Dim(0))
	}
	floatSliceApprox(t, c.Floats(), []float32{0, 0, 1, 2, 3, 0})
}

// Conv1d in NLC layout: a single 1-channel signal convolved with a length-2
// averaging-style kernel of unit weights produces sliding sums.
func TestOps_Conv1d_Good(t *testing.T) {
	// input [N=1, L=4, C=1] = 1,2,3,4
	in := FromValues([]float32{1, 2, 3, 4}, 1, 4, 1)
	// weight [out=1, k=2, in=1] = 1,1 → out[i] = x[i] + x[i+1]
	w := FromValues([]float32{1, 1}, 1, 2, 1)
	out := Conv1d(in, w, 1, 0, 1, 1)
	Materialize(out)
	// Valid conv over length 4 with kernel 2 → length 3: 1+2, 2+3, 3+4.
	if out.Dim(1) != 3 {
		t.Fatalf("Conv1d output length = %d, want 3", out.Dim(1))
	}
	floatSliceApprox(t, out.Floats(), []float32{3, 5, 7})
}

// Conv2d in NHWC layout: a 1-channel 2x2 image convolved with a 2x2 unit kernel
// produces a single summed output pixel.
func TestOps_Conv2d_Good(t *testing.T) {
	// input [N=1, H=2, W=2, C=1] = 1,2,3,4
	in := FromValues([]float32{1, 2, 3, 4}, 1, 2, 2, 1)
	// weight [out=1, kh=2, kw=2, in=1] = all ones → sum of the window.
	w := FromValues([]float32{1, 1, 1, 1}, 1, 2, 2, 1)
	out := Conv2d(in, w, 1, 1, 0, 0, 1, 1, 1)
	Materialize(out)
	// 2x2 image, 2x2 kernel, no pad → 1x1 output = 1+2+3+4 = 10.
	if out.Dim(1) != 1 || out.Dim(2) != 1 {
		t.Fatalf("Conv2d spatial output = %dx%d, want 1x1", out.Dim(1), out.Dim(2))
	}
	floatSliceApprox(t, out.Floats(), []float32{10})
}

// PadAxis on a non-float dtype exercises the AsType cast of the pad value.
func TestOps_PadAxis_NonFloatDtype_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	i32 := AsType(a, DTypeInt32)
	defer Free(a, i32)
	c := PadAxis(i32, 0, 1, 1)
	Materialize(c)
	if c.Dim(0) != 5 {
		t.Fatalf("int32 PadAxis len = %d, want 5", c.Dim(0))
	}
}
