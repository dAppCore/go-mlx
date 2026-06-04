// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func (e *Gemma4Experts) forward(x, topKIndices, topKWeights *metal.Array, tracePrefix string) *metal.Array {
	trace := func(phase string, arrays ...*metal.Array) {
		if tracePrefix == "" {
			return
		}
		metal.TraceNativeMaterialize(tracePrefix+"."+phase, arrays...)
	}
	if result, ok := e.forwardExpertIDMatVec(x, topKIndices, topKWeights, trace); ok {
		return result
	}
	if result, ok := e.forwardSortedExpertPrefill(x, topKIndices, topKWeights, trace); ok {
		return result
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

func (e *Gemma4Experts) forwardSortedExpertPrefill(x, topKIndices, topKWeights *metal.Array, trace func(string, ...*metal.Array)) (*metal.Array, bool) {
	if !metal.SortedExpertPrefillEnabled() {
		return nil, false
	}
	if !gemma4SortedExpertPrefillCompatible(e) {
		return nil, false
	}
	if x == nil || topKIndices == nil || topKWeights == nil || !x.Valid() || !topKIndices.Valid() || !topKWeights.Valid() {
		return nil, false
	}
	// Stack-allocated shape scratch — sorted-expert prefill is called
	// per MoE block (× NumHiddenLayers) per prefill batch. Avoids 2-3
	// per-call []int32 heap allocs from x/topKIndices/DownProj.Weight Shape().
	var xShapeBuf, indicesShapeBuf, weightShapeBuf [metal.MaxTensorRank]int32
	xShape := x.ShapeInto(xShapeBuf[:0])
	indicesShape := topKIndices.ShapeInto(indicesShapeBuf[:0])
	if len(xShape) != 3 || len(indicesShape) != 3 || indicesShape[0] != xShape[0] || indicesShape[1] != xShape[1] {
		return nil, false
	}
	if xShape[1] <= 1 {
		return nil, false
	}
	batch := int(xShape[0])
	seqLen := int(xShape[1])
	hidden := int(xShape[2])
	topK := int(indicesShape[2])
	routes := topKIndices.Size()
	if batch <= 0 || seqLen <= 1 || hidden <= 0 || topK <= 0 || routes != batch*seqLen*topK || topKWeights.Size() != routes {
		return nil, false
	}
	numExperts := int(e.DownProj.Weight.ShapeInto(weightShapeBuf[:0])[0])
	if routes < 16 || numExperts <= 0 || routes/numExperts < 4 {
		return nil, false
	}

	flatIndices := metal.Reshape(topKIndices, int32(routes))
	sortOrder := metal.Argsort(flatIndices, -1)
	sortedIndices := metal.Take(flatIndices, sortOrder, 0)
	routePositions := metal.Arange(0, float64(routes), 1, metal.DTypeInt32)
	sortedRoutePositions := metal.Take(routePositions, sortOrder, 0)
	topKDivisor := metal.FromValue(topK)
	sortedTokenPositions := metal.FloorDivide(sortedRoutePositions, topKDivisor)
	flatX := metal.Reshape(x, int32(batch*seqLen), int32(hidden))
	sortedInputFlat := metal.Take(flatX, sortedTokenPositions, 0)
	sortedInput := metal.Reshape(sortedInputFlat, int32(routes), 1, int32(hidden))
	metal.Free(routePositions, sortedRoutePositions, topKDivisor, sortedTokenPositions, flatX, sortedInputFlat)
	defer metal.Free(flatIndices, sortOrder, sortedIndices, sortedInput)

	gate := gemma4SwitchLinearForwardSortedRoutes(e.GateProj, sortedInput, sortedIndices)
	trace("sorted_gate", gate)
	up := gemma4SwitchLinearForwardSortedRoutes(e.UpProj, sortedInput, sortedIndices)
	trace("sorted_up", up)
	activated := metal.GeluGateMul(gate, up)
	trace("sorted_activation", activated)
	metal.Free(gate, up)
	down := gemma4SwitchLinearForwardSortedRoutes(e.DownProj, activated, sortedIndices)
	trace("sorted_down", down)
	metal.Free(activated)

	flatWeights := metal.Reshape(topKWeights, int32(routes))
	sortedWeights := metal.Take(flatWeights, sortOrder, 0)
	weightsExpanded1 := metal.ExpandDims(sortedWeights, 1)
	weightsExpanded := metal.ExpandDims(weightsExpanded1, 2)
	weightedSorted := metal.Mul(weightsExpanded, down)
	trace("sorted_weighted", weightedSorted)
	metal.Free(flatWeights, sortedWeights, weightsExpanded1, weightsExpanded, down)

	inverseOrder := metal.Argsort(sortOrder, -1)
	weightedOriginal := metal.Take(weightedSorted, inverseOrder, 0)
	weightedSqueezed := metal.Squeeze(weightedOriginal, 1)
	grouped := metal.Reshape(weightedSqueezed, int32(batch), int32(seqLen), int32(topK), int32(hidden))
	result := metal.Sum(grouped, -2, false)
	trace("sorted_sum", result)
	metal.Free(weightedSorted, inverseOrder, weightedOriginal, weightedSqueezed, grouped)
	return result, true
}

func gemma4SortedExpertPrefillCompatible(e *Gemma4Experts) bool {
	return e != nil &&
		gemma4ExpertIDMatVecSwitchCompatible(e.GateProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.UpProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.DownProj)
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

func (e *Gemma4Experts) forwardExpertIDMatVec(x, topKIndices, topKWeights *metal.Array, trace func(string, ...*metal.Array)) (*metal.Array, bool) {
	if !metal.ExpertIDMatVecEnabled() {
		return nil, false
	}
	if e == nil || e.DownProj == nil {
		return nil, false
	}
	hasFusedGateUp := gemma4ExpertIDMatVecSwitchCompatible(e.GateUpProj)
	hasSplitGateUp := gemma4ExpertIDMatVecSwitchCompatible(e.GateProj) && gemma4ExpertIDMatVecSwitchCompatible(e.UpProj)
	if (!hasFusedGateUp && !hasSplitGateUp) || !gemma4ExpertIDMatVecSwitchCompatible(e.DownProj) {
		return nil, false
	}
	if x == nil || topKIndices == nil || topKWeights == nil || !x.Valid() || !topKIndices.Valid() || !topKWeights.Valid() {
		return nil, false
	}
	// Stack-allocated shape scratch — per-token decode MoE hot path.
	// Called once per MoE block × NumHiddenLayers per generated token.
	var xShapeBuf, indicesShapeBuf [metal.MaxTensorRank]int32
	xShape := x.ShapeInto(xShapeBuf[:0])
	indicesShape := topKIndices.ShapeInto(indicesShapeBuf[:0])
	if len(xShape) != 3 || xShape[0] != 1 || xShape[1] != 1 || len(indicesShape) != 3 || indicesShape[0] != 1 || indicesShape[1] != 1 {
		return nil, false
	}
	hidden := int(xShape[2])
	routes := int(indicesShape[2])
	if hidden <= 0 || routes <= 0 || topKWeights.Size() != routes {
		return nil, false
	}

	xFlat := metal.Reshape(x, 1, int32(hidden))
	idsFlat := metal.Reshape(topKIndices, int32(routes))
	defer metal.Free(xFlat, idsFlat)

	var activated *metal.Array
	if hasFusedGateUp && metal.ExpertIDFusedActivationEnabled() {
		var err error
		activated, err = metal.QuantizedExpertIDGELUGateUpMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id fused activation matvec failed; falling back", "error", err)
			return nil, false
		}
		trace("activation_id_matvec", activated)
	} else if hasFusedGateUp {
		gateUp, err := metal.QuantizedExpertIDMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec gate/up failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_up_id_matvec", gateUp)
		gate, up, ok := splitLastDimArray(gateUp)
		metal.Free(gateUp)
		if !ok {
			metal.Free(gate, up)
			return nil, false
		}
		activated = metal.GeluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		metal.Free(gate, up)
	} else if metal.ExpertIDFusedActivationEnabled() {
		var err error
		activated, err = metal.QuantizedExpertIDGELUSplitGateUpMatVec(
			xFlat,
			e.GateProj.Weight, e.GateProj.Scales, e.GateProj.Biases,
			e.UpProj.Weight, e.UpProj.Scales, e.UpProj.Biases,
			idsFlat,
			e.GateProj.GroupSize,
			e.GateProj.Bits,
		)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id split gate/up fused activation matvec failed; falling back", "error", err)
			return nil, false
		}
		trace("activation_split_id_matvec", activated)
	} else {
		up, err := metal.QuantizedExpertIDMatVec(xFlat, e.UpProj.Weight, e.UpProj.Scales, e.UpProj.Biases, idsFlat, e.UpProj.GroupSize, e.UpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec up failed; falling back", "error", err)
			return nil, false
		}
		trace("up_id_matvec", up)
		gate, err := metal.QuantizedExpertIDMatVec(xFlat, e.GateProj.Weight, e.GateProj.Scales, e.GateProj.Biases, idsFlat, e.GateProj.GroupSize, e.GateProj.Bits)
		if err != nil {
			metal.Free(up)
			core.Error("mlx: Gemma 4 expert id matvec gate failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_id_matvec", gate)
		activated = metal.GeluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		metal.Free(gate, up)
	}

	weightsFlat := metal.Reshape(topKWeights, int32(routes))
	down, err := metal.QuantizedExpertIDWeightedMatVecSum(activated, weightsFlat, e.DownProj.Weight, e.DownProj.Scales, e.DownProj.Biases, idsFlat, e.DownProj.GroupSize, e.DownProj.Bits)
	metal.Free(weightsFlat)
	metal.Free(activated)
	if err != nil {
		core.Error("mlx: Gemma 4 expert id weighted matvec down failed; falling back", "error", err)
		return nil, false
	}
	trace("down_weighted_sum_id_matvec", down)
	result := metal.Reshape(down, 1, 1, int32(hidden))
	metal.Free(down)
	return result, true
}

func gemma4ExpertIDMatVecSwitchCompatible(linear *metal.SwitchLinear) bool {
	return linear != nil &&
		linear.Weight != nil && linear.Weight.Valid() &&
		linear.Scales != nil && linear.Scales.Valid() &&
		linear.Biases != nil && linear.Biases.Valid() &&
		linear.GroupSize > 0 &&
		metal.IsAffineQuantizationMode(linear.QuantizationMode) &&
		(linear.Bits == 2 || linear.Bits == 4 || linear.Bits == 8)
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
	var startsBuf, endsBuf [metal.MaxTensorRank]int32
	starts := startsBuf[:len(shape)]
	ends := endsBuf[:len(shape)]
	copy(ends, shape)
	ends[axis] = mid
	left := metal.Slice(a, starts, ends)
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := metal.Slice(a, starts, ends)
	return left, right, true
}
