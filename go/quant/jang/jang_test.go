// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Tests for the Metal-side JANG/JANGTQ dequant + projection wrappers.
//
// The three entry points (DequantizePackedTensor, ProjectPackedTensor,
// ProjectPackedTensorFused) all validate against the driver-neutral
// infjang lib first, then dispatch to a Metal kernel. The happy paths run
// real (tiny) GPU ops the same way the sibling autoround_dequant_test.go
// does — the JANG dequant kernel shares the exact affine LSB-first formula
// as the CPU infjang reference, so we use infjang.DequantizePackedTensor as
// the bit-exact oracle and assert the Metal result matches within tolerance.
//
// Run: MLX_METALLIB_PATH=<repo>/dist/lib/mlx.metallib \
//      go test -tags metal_runtime -cover ./quant/jang
package jang

import (
	"math"
	"testing"

	core "dappco.re/go"
	infjang "dappco.re/go/inference/quant/jang"
)

// jangTestInfo builds a single-bit-width JANGTQ Info that drives a routed
// expert descriptor at the requested bits + group size. Group bits flow
// through RoutedExpertBits so the inferTensorRole "experts." route picks it.
func jangTestInfo(bits, groupSize int) *infjang.Info {
	return &infjang.Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        groupSize,
		BitsDefault:      bits,
		RoutedExpertBits: bits,
	}
}

// jangFixture builds a descriptor + packed payload at the requested bit
// width, walking values across the full 0..maxValue range so every lane is
// exercised, with distinct per-group scale + bias so a mis-indexed group
// surfaces as a wrong magnitude rather than a silent identity.
func jangFixture(t *testing.T, bits int, shape []uint64, groupSize int) (infjang.PackedTensorDescriptor, []byte, []float32, []float32) {
	t.Helper()
	desc, err := infjang.NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", shape, jangTestInfo(bits, groupSize))
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor(%d-bit): %v", bits, err)
	}
	maxValue := uint8((1 << bits) - 1)
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i) & maxValue
	}
	packed, err := infjang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("PackQuantizedValues(%d-bit): %v", bits, err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.25 + float32(i)*0.0625
		biases[i] = -1 - float32(i)*0.5
	}
	return desc, packed, scales, biases
}

func assertClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("value[%d] = %v, want %v (delta=%v, tol=%v)", i, got[i], want[i], got[i]-want[i], tol)
		}
	}
}

// --- DequantizePackedTensor ---

// TestJang_DequantizePackedTensor_Good runs the real Metal dequant on a
// small 2-bit routed-expert tensor and asserts it matches the CPU infjang
// reference bit-for-bit (same affine LSB-first formula on both engines).
func TestJang_DequantizePackedTensor_Good(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{4, 16}, 64)

	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor() error = %v", err)
	}
	want, err := infjang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("infjang.DequantizePackedTensor() oracle error = %v", err)
	}
	if len(got) != int(desc.Elements) {
		t.Fatalf("output length = %d, want %d", len(got), desc.Elements)
	}
	assertClose(t, got, want, 1e-4)
}

// TestJang_DequantizePackedTensor_Bad feeds a packed slice whose length does
// not match the descriptor — the infjang validation gate must reject it
// before any Metal work happens.
func TestJang_DequantizePackedTensor_Bad(t *testing.T) {
	desc, _, scales, biases := jangFixture(t, 2, []uint64{4, 16}, 64)

	_, err := DequantizePackedTensor(desc, []byte{0, 1, 2}, scales, biases)
	if err == nil {
		t.Fatal("DequantizePackedTensor() error = nil, want packed length rejection")
	}
	if !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want packed length diagnostic", err)
	}
}

// TestJang_DequantizePackedTensor_Ugly exercises the short-tail edge: an
// element count that is NOT a multiple of groupSize, so the final group runs
// short. The Metal kernel must still match the CPU oracle on the ragged tail.
func TestJang_DequantizePackedTensor_Ugly(t *testing.T) {
	// 130 elements with groupSize 64 → 3 groups, last group has 2 elements.
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{130}, 64)

	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor() short-tail error = %v", err)
	}
	want, err := infjang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("infjang oracle error = %v", err)
	}
	assertClose(t, got, want, 1e-4)
}

// --- ProjectPackedTensor ---

// projectOracle computes input @ dequant(weight).T (+ bias) on the CPU using
// the infjang dequant as the weight source, mirroring what the Metal
// JANGPackedLinear kernel does, so we have an engine-independent reference.
func projectOracle(t *testing.T, desc infjang.PackedTensorDescriptor, packed []byte, scales, biases, input []float32, inShape []int32, bias []float32) []float32 {
	t.Helper()
	weight, err := infjang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("infjang.DequantizePackedTensor() oracle error = %v", err)
	}
	outDim := int(desc.Shape[0])
	inDim := int(desc.Shape[1])
	rows := len(input) / inDim
	out := make([]float32, rows*outDim)
	for r := 0; r < rows; r++ {
		for o := 0; o < outDim; o++ {
			var sum float32
			for k := 0; k < inDim; k++ {
				sum += input[r*inDim+k] * weight[o*inDim+k]
			}
			if len(bias) > 0 {
				sum += bias[o]
			}
			out[r*outDim+o] = sum
		}
	}
	_ = inShape
	return out
}

// TestJang_ProjectPackedTensor_Good runs the real Metal packed linear on a
// small [3,4] weight against a 2-row input and asserts it matches the CPU
// composed dequant+matmul reference, including the projection bias.
func TestJang_ProjectPackedTensor_Good(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	input := []float32{1, 2, 3, 4, -1, 0.5, 2, -0.5}
	inShape := []int32{2, 4}
	bias := []float32{0.25, -1, 2}

	res, err := ProjectPackedTensor(desc, packed, scales, biases, input, inShape, bias)
	if err != nil {
		t.Fatalf("ProjectPackedTensor() error = %v", err)
	}
	if len(res.Shape) != 2 || res.Shape[0] != 2 || res.Shape[1] != 3 {
		t.Fatalf("result shape = %+v, want [2 3]", res.Shape)
	}
	want := projectOracle(t, desc, packed, scales, biases, input, inShape, bias)
	assertClose(t, res.Values, want, 1e-3)
}

// TestJang_ProjectPackedTensor_Bad feeds an input whose flat length does not
// match the declared input shape — the wrapper's own shape check (not the
// Metal kernel) must reject it.
func TestJang_ProjectPackedTensor_Bad(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	// inShape says 2x4 = 8 elements but only 4 supplied.
	_, err := ProjectPackedTensor(desc, packed, scales, biases, []float32{1, 2, 3, 4}, []int32{2, 4}, nil)
	if err == nil {
		t.Fatal("ProjectPackedTensor() error = nil, want input length rejection")
	}
	if !core.Contains(err.Error(), "input length") {
		t.Fatalf("error = %v, want input length diagnostic", err)
	}
}

// TestJang_ProjectPackedTensor_Ugly covers two boundary rejections: a bias
// whose length disagrees with the output dimension, and an input whose last
// dimension disagrees with the weight's input dimension.
func TestJang_ProjectPackedTensor_Ugly(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)

	// Bias length 2, output dim is 3.
	_, err := ProjectPackedTensor(desc, packed, scales, biases, []float32{1, 2, 3, 4}, []int32{1, 4}, []float32{0, 0})
	if err == nil || !core.Contains(err.Error(), "bias length") {
		t.Fatalf("bias-length error = %v, want bias length diagnostic", err)
	}

	// Input last dim 3, weight input dim is 4.
	_, err = ProjectPackedTensor(desc, packed, scales, biases, []float32{1, 2, 3}, []int32{1, 3}, nil)
	if err == nil || !core.Contains(err.Error(), "last dimension") {
		t.Fatalf("last-dim error = %v, want last dimension diagnostic", err)
	}

	// A 3-D weight descriptor passes validation but is not a [out, in]
	// projection matrix — the wrapper's own rank check must reject it before
	// the Metal kernel.
	desc3D, packed3D, scales3D, biases3D := jangFixture(t, 2, []uint64{2, 2, 4}, 64)
	_, err = ProjectPackedTensor(desc3D, packed3D, scales3D, biases3D, []float32{1, 2, 3, 4}, []int32{1, 4}, nil)
	if err == nil || !core.Contains(err.Error(), "[out, in]") {
		t.Fatalf("3-D weight error = %v, want [out, in] rank diagnostic", err)
	}
}

// --- ProjectPackedTensorFused ---

// TestJang_ProjectPackedTensorFused_Good runs the fused Metal kernel and
// asserts it matches the non-fused ProjectPackedTensor result on the same
// inputs — the fused path must not change the maths, only skip materialising
// the dense weight.
func TestJang_ProjectPackedTensorFused_Good(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	input := []float32{1, 2, 3, 4, -1, 0.5, 2, -0.5}
	inShape := []int32{2, 4}
	bias := []float32{0.25, -1, 2}

	fused, err := ProjectPackedTensorFused(desc, packed, scales, biases, input, inShape, bias)
	if err != nil {
		t.Fatalf("ProjectPackedTensorFused() error = %v", err)
	}
	composed, err := ProjectPackedTensor(desc, packed, scales, biases, input, inShape, bias)
	if err != nil {
		t.Fatalf("ProjectPackedTensor() reference error = %v", err)
	}
	if len(fused.Shape) != len(composed.Shape) {
		t.Fatalf("fused shape = %+v, composed shape = %+v", fused.Shape, composed.Shape)
	}
	assertClose(t, fused.Values, composed.Values, 1e-3)
}

// TestJang_ProjectPackedTensorFused_Bad confirms the fused path runs the same
// validation gate — a packed-length mismatch is rejected before the kernel.
func TestJang_ProjectPackedTensorFused_Bad(t *testing.T) {
	desc, _, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	_, err := ProjectPackedTensorFused(desc, []byte{0}, scales, biases, []float32{1, 2, 3, 4}, []int32{1, 4}, nil)
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want packed length diagnostic", err)
	}
}

// TestJang_ProjectPackedTensorFused_Ugly runs the fused path with no
// projection bias (the bias-less kernel branch) and checks it still matches
// the non-fused result.
func TestJang_ProjectPackedTensorFused_Ugly(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	input := []float32{1, 2, 3, 4}
	inShape := []int32{1, 4}

	fused, err := ProjectPackedTensorFused(desc, packed, scales, biases, input, inShape, nil)
	if err != nil {
		t.Fatalf("ProjectPackedTensorFused() no-bias error = %v", err)
	}
	composed, err := ProjectPackedTensor(desc, packed, scales, biases, input, inShape, nil)
	if err != nil {
		t.Fatalf("ProjectPackedTensor() reference error = %v", err)
	}
	assertClose(t, fused.Values, composed.Values, 1e-3)
}

// --- MetalShape (pure Go) ---

func TestJang_MetalShape_Good(t *testing.T) {
	got, err := MetalShape([]uint64{2048, 4096})
	if err != nil {
		t.Fatalf("MetalShape() error = %v", err)
	}
	if len(got) != 2 || got[0] != 2048 || got[1] != 4096 {
		t.Fatalf("MetalShape() = %+v, want [2048 4096]", got)
	}
}

// TestJang_MetalShape_Bad covers a zero dimension and an overflow dimension
// (one that does not fit in a positive int32).
func TestJang_MetalShape_Bad(t *testing.T) {
	if _, err := MetalShape([]uint64{2048, 0}); err == nil || !core.Contains(err.Error(), "invalid") {
		t.Fatalf("zero-dim error = %v, want invalid shape", err)
	}
	// 1<<31 exceeds the max positive int32, so the bound check must reject it.
	if _, err := MetalShape([]uint64{1 << 31}); err == nil || !core.Contains(err.Error(), "invalid") {
		t.Fatalf("overflow-dim error = %v, want invalid shape", err)
	}
}

// TestJang_MetalShape_Ugly covers the empty-shape edge — a tensor with no
// dimensions is not a valid Metal shape.
func TestJang_MetalShape_Ugly(t *testing.T) {
	if _, err := MetalShape(nil); err == nil || !core.Contains(err.Error(), "required") {
		t.Fatalf("nil-shape error = %v, want required diagnostic", err)
	}
	if _, err := MetalShape([]uint64{}); err == nil {
		t.Fatal("empty-shape error = nil, want required diagnostic")
	}
}

// --- ShapeElements (pure Go) ---

func TestJang_ShapeElements_Good(t *testing.T) {
	got, err := ShapeElements([]int32{4, 28, 2048, 64})
	if err != nil {
		t.Fatalf("ShapeElements() error = %v", err)
	}
	if want := 4 * 28 * 2048 * 64; got != want {
		t.Fatalf("ShapeElements() = %d, want %d", got, want)
	}
}

// TestJang_ShapeElements_Bad covers a non-positive dimension.
func TestJang_ShapeElements_Bad(t *testing.T) {
	if _, err := ShapeElements([]int32{4, 0, 8}); err == nil || !core.Contains(err.Error(), "invalid") {
		t.Fatalf("zero-dim error = %v, want invalid shape", err)
	}
	if _, err := ShapeElements([]int32{-1, 8}); err == nil || !core.Contains(err.Error(), "invalid") {
		t.Fatalf("negative-dim error = %v, want invalid shape", err)
	}
}

// TestJang_ShapeElements_Ugly covers the empty shape and the overflow guard
// (a product that would exceed maxInt).
func TestJang_ShapeElements_Ugly(t *testing.T) {
	if _, err := ShapeElements(nil); err == nil || !core.Contains(err.Error(), "required") {
		t.Fatalf("nil-shape error = %v, want required diagnostic", err)
	}
	// Two dimensions whose product overflows a 64-bit int.
	big := int32(1 << 30)
	if _, err := ShapeElements([]int32{big, big, big}); err == nil || !core.Contains(err.Error(), "too large") {
		t.Fatalf("overflow error = %v, want too large diagnostic", err)
	}
}

// --- Int32SliceToInts (pure Go) ---

func TestJang_Int32SliceToInts_Good(t *testing.T) {
	got := Int32SliceToInts([]int32{4, 28, 2048, 64})
	want := []int{4, 28, 2048, 64}
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Int32SliceToInts()[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestJang_Int32SliceToInts_Bad covers negative values — the conversion is a
// straight widening cast, so negatives must be preserved, not clamped.
func TestJang_Int32SliceToInts_Bad(t *testing.T) {
	got := Int32SliceToInts([]int32{-1, -2048})
	if len(got) != 2 || got[0] != -1 || got[1] != -2048 {
		t.Fatalf("Int32SliceToInts() = %+v, want [-1 -2048] preserved", got)
	}
}

// TestJang_Int32SliceToInts_Ugly covers the empty and nil inputs — both yield
// a non-nil zero-length slice (the make-len-0 result), never a panic.
func TestJang_Int32SliceToInts_Ugly(t *testing.T) {
	if got := Int32SliceToInts(nil); len(got) != 0 {
		t.Fatalf("nil input = %+v, want empty", got)
	}
	if got := Int32SliceToInts([]int32{}); got == nil || len(got) != 0 {
		t.Fatalf("empty input = %+v, want non-nil empty", got)
	}
}
