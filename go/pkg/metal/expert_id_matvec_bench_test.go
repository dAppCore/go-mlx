// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// MoE expert-ID matvec bench coverage (W9-K, Wave 9).
//
// The quantizedExpertID*MatVec helpers in expert_id_matvec.go are the
// per-token, per-layer dispatch surface for fused-gather MoE decode on
// Gemma 4 26B A4B and minimax_m2. The Metal kernel itself is fully
// fused (single dispatch per call, gather-based via expert_ids[]
// indirection); IDEAS.md §5 prescribed shape is already in place.
//
// These benches measure the Go-side dispatch overhead — validation,
// kernel cache lookup, MetalKernelConfig setup, and the kernel.Apply
// cgo crossing — at realistic Gemma 4 MoE dimensions.
//
// Coverage:
//   - QuantizedExpertIDMatVec (the bare matvec)
//   - QuantizedExpertIDGELUSplitGateUpMatVec (gemma4 fused split path)
//   - QuantizedExpertIDWeightedMatVecSum (gemma4 down projection)
//
// Shapes:
//   - Tiny: matches the correctness tests (cheapest dispatch, surfaces
//     Go-side overhead).
//   - Gemma4-26B-ish: experts=128, top-2, hidden=2048, moeDim=2048,
//     groupSize=64, bits=4 — the actual MoE decode shape (GPU work
//     dominates, but lets us watch the dispatch path under load).

import (
	"testing"
)

// --- Synthetic q4 fixture builders (no *testing.T dependency) ---

// buildQ4ExpertIDFixture constructs a quantized expert-ID matvec fixture
// with the shapes (experts, outDim, inDim, groupSize) under affine q4
// packing. The packed weight is [experts, outDim, inDim/8] uint32 with
// each uint32 carrying 8 nibbles; scales/biases are
// [experts, outDim, inDim/groupSize] float32.
func buildQ4ExpertIDFixture(experts, outDim, inDim, groupSize, routes int) (input, weight, scales, biases, ids *Array) {
	if inDim%8 != 0 || inDim%groupSize != 0 {
		panic("buildQ4ExpertIDFixture: inDim must be a multiple of 8 and groupSize")
	}
	groups := inDim / groupSize
	packedIn := inDim / 8

	// Pack quantized values 4 bits each into uint32 nibbles. We synthesise
	// deterministic-ish bits via i*7 so the kernels see varied data; the
	// actual numerical accuracy is not validated by benches.
	packed := make([]uint32, experts*outDim*packedIn)
	for i := range packed {
		// 8 nibbles per uint32; each nibble is (i*7+offset) & 0xF.
		var v uint32
		for off := range 8 {
			v |= (uint32(i*7+off) & 0xF) << uint(off*4)
		}
		packed[i] = v
	}

	scalesVals := make([]float32, experts*outDim*groups)
	biasVals := make([]float32, experts*outDim*groups)
	for i := range scalesVals {
		scalesVals[i] = 0.025 * float32((i%9)+1)
		biasVals[i] = -0.45 + 0.05*float32(i%17)
	}

	inputVals := make([]float32, routes*inDim)
	for i := range inputVals {
		inputVals[i] = -1.5 + 0.0625*float32((i*5)%71)
	}

	idVals := make([]int32, routes)
	for i := range idVals {
		idVals[i] = int32(i % experts)
	}

	input = FromValues(inputVals, routes, inDim)
	weight = FromValues(packed, experts, outDim, packedIn)
	scales = FromValues(scalesVals, experts, outDim, groups)
	biases = FromValues(biasVals, experts, outDim, groups)
	ids = FromValues(idVals, routes)
	return input, weight, scales, biases, ids
}

// --- QuantizedExpertIDMatVec (bare matvec) ---

// Tiny shape — surfaces Go-side dispatch overhead.
func BenchmarkExpertIDMatVec_Q4_Tiny(b *testing.B) {
	input, weight, scales, biases, ids := buildQ4ExpertIDFixture(4, 8, 32, 16, 2)
	defer Free(input, weight, scales, biases, ids)
	Materialize(input, weight, scales, biases, ids)

	// Warm the kernel cache so we benchmark steady-state dispatch.
	warm, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 16, 4)
	if err != nil {
		b.Fatalf("warmup QuantizedExpertIDMatVec: %v", err)
	}
	Materialize(warm)
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		out, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 16, 4)
		if err != nil {
			b.Fatalf("QuantizedExpertIDMatVec: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}

// Gemma4 26B A4B realistic — experts=128, top-2, hidden=2048, moeDim=2048.
// inDim=2048 (router input width), outDim=2048 (moeDim output width).
func BenchmarkExpertIDMatVec_Q4_Gemma4_26B(b *testing.B) {
	input, weight, scales, biases, ids := buildQ4ExpertIDFixture(128, 2048, 2048, 64, 2)
	defer Free(input, weight, scales, biases, ids)
	Materialize(input, weight, scales, biases, ids)

	warm, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 64, 4)
	if err != nil {
		b.Fatalf("warmup QuantizedExpertIDMatVec: %v", err)
	}
	Materialize(warm)
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		out, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 64, 4)
		if err != nil {
			b.Fatalf("QuantizedExpertIDMatVec: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}

// --- QuantizedExpertIDGELUSplitGateUpMatVec (Gemma4 fused split gate/up) ---

// Tiny shape — surfaces Go-side dispatch overhead under the split-gate
// fused-activation path used by current Gemma4 26B q4 safetensors.
func BenchmarkExpertIDGELUSplitGateUpMatVec_Q4_Tiny(b *testing.B) {
	input, gateW, gateS, gateB, ids := buildQ4ExpertIDFixture(4, 8, 32, 16, 2)
	_, upW, upS, upB, _ := buildQ4ExpertIDFixture(4, 8, 32, 16, 2)
	defer Free(input, gateW, gateS, gateB, upW, upS, upB, ids)
	Materialize(input, gateW, gateS, gateB, upW, upS, upB, ids)

	warm, err := QuantizedExpertIDGELUSplitGateUpMatVec(input, gateW, gateS, gateB, upW, upS, upB, ids, 16, 4)
	if err != nil {
		b.Fatalf("warmup QuantizedExpertIDGELUSplitGateUpMatVec: %v", err)
	}
	Materialize(warm)
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		out, err := QuantizedExpertIDGELUSplitGateUpMatVec(input, gateW, gateS, gateB, upW, upS, upB, ids, 16, 4)
		if err != nil {
			b.Fatalf("QuantizedExpertIDGELUSplitGateUpMatVec: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}

// --- QuantizedExpertIDWeightedMatVecSum (Gemma4 down projection) ---

// Tiny shape — surfaces Go-side dispatch overhead.
func BenchmarkExpertIDWeightedMatVecSum_Q4_Tiny(b *testing.B) {
	input, weight, scales, biases, ids := buildQ4ExpertIDFixture(4, 8, 32, 16, 2)
	routeWeights := FromValues([]float32{0.65, 0.35}, 2)
	defer Free(input, weight, scales, biases, ids, routeWeights)
	Materialize(input, weight, scales, biases, ids, routeWeights)

	warm, err := QuantizedExpertIDWeightedMatVecSum(input, routeWeights, weight, scales, biases, ids, 16, 4)
	if err != nil {
		b.Fatalf("warmup QuantizedExpertIDWeightedMatVecSum: %v", err)
	}
	Materialize(warm)
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		out, err := QuantizedExpertIDWeightedMatVecSum(input, routeWeights, weight, scales, biases, ids, 16, 4)
		if err != nil {
			b.Fatalf("QuantizedExpertIDWeightedMatVecSum: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}
