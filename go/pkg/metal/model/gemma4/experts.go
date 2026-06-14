// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"sync"

	"dappco.re/go/mlx/pkg/metal"
)

// splitLastDimSliceScratch is a pooled []int32 backing buffer reused for the
// starts/ends slice pair that splitLastDimArray hands to metal.Slice. Those
// slices unavoidably escape: metal.Slice takes &starts[0] across the cgo
// boundary, so the compiler heap-allocates them regardless of a stack array
// (escape analysis confirms moved-to-heap on the var startsBuf/endsBuf form).
// splitLastDimArray fires per MoE block per layer per token on the fused
// gate_up decode path (the production tensor is rank-5), so reusing one
// 2*MaxTensorRank buffer removes the two per-call []int32 heap allocs.
var splitLastDimSliceScratch = sync.Pool{
	New: func() any {
		buf := make([]int32, 2*metal.MaxTensorRank)
		return &buf
	},
}

func (e *Gemma4Experts) forward(x, topKIndices, topKWeights *metal.Array, tracePrefix string) *metal.Array {
	trace := func(phase string, arrays ...*metal.Array) {
		if tracePrefix == "" {
			return
		}
		metal.TraceNativeMaterialize(tracePrefix+"."+phase, arrays...)
	}
	expanded1 := metal.ExpandDims(x, 2)
	expanded := metal.ExpandDims(expanded1, 2)
	metal.Free(expanded1)

	var gate, up *metal.Array
	if e.GateUpProj != nil && gemma4UseFusedExpertGateUp(x) {
		gateUp := e.GateUpProj.Forward(expanded, topKIndices)
		trace("gate_up", gateUp)
		var ok bool
		gate, up, ok = splitLastDimArray(gateUp)
		metal.Free(gateUp)
		if !ok {
			gate, up = nil, nil
		}
	}
	if gate == nil || up == nil {
		metal.Free(gate, up)
		up = e.UpProj.Forward(expanded, topKIndices)
		trace("up", up)
		gate = e.GateProj.Forward(expanded, topKIndices)
		trace("gate", gate)
	}
	metal.Free(expanded)
	activated := metal.GeluGateMul(gate, up)
	trace("activation", activated)
	metal.Free(gate, up)
	down := e.DownProj.Forward(activated, topKIndices)
	trace("down", down)
	metal.Free(activated)
	downSqueezed := metal.Squeeze(down, 3)
	metal.Free(down)

	weightsExpanded := metal.ExpandDims(topKWeights, 3)
	weighted := metal.Mul(weightsExpanded, downSqueezed)
	trace("weighted", weighted)
	metal.Free(weightsExpanded, downSqueezed)
	result := metal.Sum(weighted, -2, false)
	trace("sum", result)
	metal.Free(weighted)
	return result
}

func gemma4SwitchLinearForwardSortedRoutes(linear *metal.SwitchLinear, input, expertIndices *metal.Array) *metal.Array {
	var out *metal.Array
	if metal.RequiresDenseQuantizedMatmulFallback(linear.QuantizationMode) {
		denseWeight := metal.DequantizeMode(linear.Weight, linear.Scales, linear.Biases, linear.GroupSize, linear.Bits, linear.QuantizationMode)
		weightTranspose := metal.Transpose(denseWeight, 0, 2, 1)
		out = metal.GatherMM(input, weightTranspose, nil, expertIndices, true)
		metal.Free(denseWeight, weightTranspose)
	} else {
		out = metal.GatherQMM(input, linear.Weight, linear.Scales, linear.Biases, nil, expertIndices, true, linear.GroupSize, linear.Bits, linear.QuantizationMode, true)
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		bias := metal.Take(linear.Bias, expertIndices, 0)
		biasExpanded := metal.ExpandDims(bias, bias.NumDims()-1)
		oldOut := out
		out = metal.Add(out, biasExpanded)
		metal.Free(oldOut, bias, biasExpanded)
	}
	return out
}

func gemma4UseFusedExpertGateUp(x *metal.Array) bool {
	if x == nil || !x.Valid() {
		return false
	}
	// Branch on the row dim only — Shape() would heap-allocate a fresh
	// []int32 per MoE block per layer per token. Dim() is one C call.
	return x.NumDims() >= 2 && x.Dim(1) == 1
}

func splitLastDimArray(a *metal.Array) (*metal.Array, *metal.Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	// Stack-allocated shape scratch — called per MoE block on the
	// fused-gate-up split path. Avoids per-call []int32 heap alloc.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := a.ShapeInto(shapeBuf[:0])
	if len(shape) == 0 {
		return nil, nil, false
	}
	axis := len(shape) - 1
	mid := shape[axis] / 2
	if mid <= 0 || shape[axis]%2 != 0 {
		return nil, nil, false
	}
	// Pooled starts/ends backing buffer — these slices escape via metal.Slice's
	// cgo pointer, so a stack array does not help (see splitLastDimSliceScratch).
	// Reusing one buffer across both Slice calls is safe: mlx_slice_inline copies
	// starts/ends C-side before returning, exactly as the in-place starts[axis]=mid
	// mutation between the two calls already relies on.
	scratchPtr := splitLastDimSliceScratch.Get().(*[]int32)
	scratch := *scratchPtr
	starts := scratch[:len(shape)]
	ends := scratch[metal.MaxTensorRank : metal.MaxTensorRank+len(shape)]
	for i := range starts {
		starts[i] = 0
	}
	copy(ends, shape)
	ends[axis] = mid
	left := metal.Slice(a, starts, ends)
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := metal.Slice(a, starts, ends)
	splitLastDimSliceScratch.Put(scratchPtr)
	return left, right, true
}
