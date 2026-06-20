// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Coverage-completion tests for the internal error returns of the Metal-side
// JANG wrappers. The happy/bad/ugly trios in jang_test.go drive the public
// validation gate (infjang.ValidatePackedTensor) and the real GPU paths; the
// blocks exercised here are the *post-validation* internal failures that the
// public-shape fixtures never reach:
//
//   - DequantizePackedTensor / projectPackedTensor — MetalShape(desc.Shape)
//     failing AFTER ValidatePackedTensor has passed.
//   - DequantizePackedTensor / projectPackedTensor — the Metal kernel
//     (DequantizeJANGPacked / JANGPackedLinear[Fused]) failing on a
//     shape/packed-length disagreement.
//   - projectPackedTensor — ShapeElements(inputShape) failing on a malformed
//     caller-supplied input shape.
//
// Reachability hinges on a real, non-synthetic gap: validateDescriptor (which
// ValidatePackedTensor calls) checks the derived counts — Elements, Bits,
// GroupSize, PackedBytes, ScaleCount, BiasCount — but NEVER the Shape slice.
// PackedTensorDescriptor carries json tags, so a deserialised / untrusted
// descriptor can legitimately arrive with a Shape that disagrees with its own
// counts (or a zero/overflow dimension). These internal guards are exactly the
// defence for that case, so reproducing it with a descriptor whose counts stay
// valid while Shape is rewritten is the descriptor a caller can really hand in,
// not an artificial one.
//
//	Run: MLX_METALLIB_PATH=<repo>/dist/lib/mlx.metallib \
//	     go test -tags metal_runtime -cover ./quant/jang
package jang

import (
	"testing"

	core "dappco.re/go"
)

// TestJang_DequantizePackedTensor_MetalShapeReject builds a valid 2-bit
// fixture, passes its validation gate, then rewrites only desc.Shape to carry
// a zero dimension. ValidatePackedTensor still passes (it ignores Shape), so
// execution reaches MetalShape, which rejects the zero dimension — covering the
// DequantizePackedTensor "MetalShape errored after validation" return.
func TestJang_DequantizePackedTensor_MetalShapeReject(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{4, 16}, 64)
	// All count fields stay consistent with packed/scales/biases; only the
	// Shape becomes invalid (zero dimension). MetalShape rejects it.
	desc.Shape = []uint64{4, 0}

	_, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err == nil || !core.Contains(err.Error(), "metal dequant shape is invalid") {
		t.Fatalf("error = %v, want metal dequant shape invalid diagnostic", err)
	}
}

// TestJang_DequantizePackedTensor_KernelReject rewrites desc.Shape so every
// dimension is MetalShape-valid but the element product disagrees with the
// packed payload the descriptor's counts describe. MetalShape therefore
// succeeds, and the disagreement surfaces inside metal.DequantizeJANGPacked
// (its recomputed expected packed length no longer matches the supplied packed
// array) — covering the DequantizePackedTensor "metal kernel errored" return.
func TestJang_DequantizePackedTensor_KernelReject(t *testing.T) {
	// Fixture: 4*16 = 64 elements, 2-bit → 16 packed bytes, 1 group of 64.
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{4, 16}, 64)
	// 4*32 = 128 elements: every dim positive + in-range so MetalShape passes,
	// but the kernel now expects (128*2+7)/8 = 32 packed bytes, not the 16 it
	// receives, so validateJANGPackedDequantInputs rejects before any GPU work.
	desc.Shape = []uint64{4, 32}

	_, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err == nil {
		t.Fatal("DequantizePackedTensor() error = nil, want metal kernel rejection")
	}
	// Assert on the Metal-layer wording ("mlx: JANG dequant ...", emitted by
	// metal.DequantizeJANGPacked's own input check) so the test proves it
	// reached the metal kernel return rather than the earlier infjang gate
	// (which phrases the same failure as "jang: packed tensor ... packed
	// length").
	if !core.Contains(err.Error(), "mlx: JANG dequant") {
		t.Fatalf("error = %v, want mlx JANG dequant diagnostic", err)
	}
}

// TestJang_ProjectPackedTensor_MetalShapeReject mirrors the dequant case for
// the projection wrapper: a validated descriptor whose Shape is rewritten with
// an overflow dimension reaches MetalShape (after ValidatePackedTensor passes)
// and is rejected there — covering projectPackedTensor's "MetalShape errored
// after validation" return.
func TestJang_ProjectPackedTensor_MetalShapeReject(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	// 1<<31 exceeds the max positive int32, so MetalShape's per-dimension
	// bound check rejects it while the count fields stay valid.
	desc.Shape = []uint64{3, 1 << 31}

	_, err := ProjectPackedTensor(desc, packed, scales, biases, []float32{1, 2, 3, 4}, []int32{1, 4}, nil)
	if err == nil || !core.Contains(err.Error(), "metal dequant shape is invalid") {
		t.Fatalf("error = %v, want metal dequant shape invalid diagnostic", err)
	}
}

// TestJang_ProjectPackedTensor_InputShapeReject drives projectPackedTensor past
// validation and the [out, in] rank check with a well-formed descriptor, then
// hands it a malformed caller input shape (a zero dimension). ShapeElements
// rejects that input shape — covering projectPackedTensor's "ShapeElements
// errored" return, the one internal branch the public ugly trio never reaches
// (its fixtures all pass well-formed input shapes).
func TestJang_ProjectPackedTensor_InputShapeReject(t *testing.T) {
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	// Descriptor + weight are valid [3,4]; the *input* shape carries a zero
	// dimension, which ShapeElements rejects ("input shape is invalid").
	_, err := ProjectPackedTensor(desc, packed, scales, biases, []float32{1, 2, 3, 4}, []int32{0, 4}, nil)
	if err == nil || !core.Contains(err.Error(), "input shape is invalid") {
		t.Fatalf("error = %v, want input shape invalid diagnostic", err)
	}
}

// TestJang_ProjectPackedTensor_KernelReject drives projectPackedTensor through
// every wrapper-side check with a descriptor whose rewritten Shape keeps the
// [out, in] rank and the input-dimension agreement the wrapper enforces, yet
// describes more weight elements than the packed payload holds. All the
// wrapper guards pass on weightShape[1] (the input last-dim) and weightShape[0]
// (the bias dim), so execution reaches metal.JANGPackedLinear, whose internal
// DequantizeJANGPacked rejects the packed/shape disagreement — covering
// projectPackedTensor's "metal kernel errored" return (jang.go:96-98), the
// last internal branch the public happy path leaves uncovered.
func TestJang_ProjectPackedTensor_KernelReject(t *testing.T) {
	// Fixture: 3*4 = 12 elements, 2-bit → 3 packed bytes, 1 group of 64.
	desc, packed, scales, biases := jangFixture(t, 2, []uint64{3, 4}, 64)
	// 3*8 = 24 elements: weightShape stays [out=3, in=8] so the rank check and
	// the input last-dim check (input len 8, last dim 8) both pass, but the
	// kernel now expects (24*2+7)/8 = 6 packed bytes, not the 3 supplied.
	desc.Shape = []uint64{3, 8}
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	_, err := ProjectPackedTensor(desc, packed, scales, biases, input, []int32{1, 8}, nil)
	if err == nil {
		t.Fatal("ProjectPackedTensor() error = nil, want metal kernel rejection")
	}
	if !core.Contains(err.Error(), "mlx: JANG dequant") {
		t.Fatalf("error = %v, want mlx JANG dequant diagnostic", err)
	}

	// The fused entry point shares the same wrapper, so it routes the same
	// disagreement into metal.JANGPackedLinearFused's own validation gate —
	// keeping the kernel-error return covered for both projection paths.
	_, errFused := ProjectPackedTensorFused(desc, packed, scales, biases, input, []int32{1, 8}, nil)
	if errFused == nil || !core.Contains(errFused.Error(), "mlx: JANG dequant") {
		t.Fatalf("fused error = %v, want mlx JANG dequant diagnostic", errFused)
	}
}
