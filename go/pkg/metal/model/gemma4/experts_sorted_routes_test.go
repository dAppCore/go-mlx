// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestExperts_gemma4SwitchLinearForwardSortedRoutes_Good drives the standalone
// sorted-route SwitchLinear matmul helper directly with a deterministic q4
// affine expert fixture and per-token expert ids. The helper has no production
// caller in the package — it is exercised in isolation here. The standard q4
// path routes through GatherQMM, which needs the affine_gather_qmv kernels; the
// test is gated on metal.GateGatherQMMReferenceTests, the same canonical signal
// the GatherQMM reference equivalence test uses for "this metallib ships the
// gather-QMM kernels". When the gate is off the helper cannot run correctly, so
// the test skips rather than asserting on a degenerate fallback shape.
func TestExperts_gemma4SwitchLinearForwardSortedRoutes_Good(t *testing.T) {
	requireMetalRuntime(t)
	if !metal.RuntimeGateEnabled(metal.GateGatherQMMReferenceTests) {
		t.Skip("enable metal.GateGatherQMMReferenceTests via SetRuntimeGate when the local metallib provides GatherQMM kernels")
	}

	const (
		experts   = 2
		seqLen    = 4
		topK      = 1
		outDim    = 8
		inDim     = 8
		groupSize = 4
		bits      = 4
	)
	linear := quantizedSwitchLinearExpertIDTest(t, experts, outDim, inDim, groupSize, bits, 7)
	defer metal.FreeSwitchLinear(linear)

	values := make([]float32, seqLen*inDim)
	for i := range values {
		values[i] = float32((i%9)-4) * 0.1
	}
	indices := make([]int32, seqLen*topK)
	for i := range indices {
		indices[i] = int32(i % experts)
	}
	input := metal.FromValues(values, 1, seqLen, inDim)
	expertIndices := metal.FromValues(indices, 1, seqLen, topK)
	defer metal.Free(input, expertIndices)

	out := gemma4SwitchLinearForwardSortedRoutes(linear, input, expertIndices)
	if out == nil {
		t.Fatal("gemma4SwitchLinearForwardSortedRoutes returned nil")
	}
	defer metal.Free(out)

	metal.Materialize(out)
	if err := metal.LastError(); err != nil {
		t.Skipf("GatherQMM kernel unavailable in local metallib: %v", err)
	}
	if !out.Valid() {
		t.Fatal("sorted-route output is not valid")
	}
	if shape := out.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != outDim {
		t.Fatalf("sorted-route output shape = %v, want [1 %d %d]", shape, seqLen, outDim)
	}
}
