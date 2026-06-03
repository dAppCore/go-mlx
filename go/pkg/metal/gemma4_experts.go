// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func (e *Gemma4Experts) forward(x, topKIndices, topKWeights *Array, tracePrefix string) *Array {
	trace := func(phase string, arrays ...*Array) {
		if tracePrefix == "" {
			return
		}
		traceNativeMaterialize(tracePrefix+"."+phase, arrays...)
	}
	if result, ok := e.forwardExpertIDMatVec(x, topKIndices, topKWeights, trace); ok {
		return result
	}
	if result, ok := e.forwardSortedExpertPrefill(x, topKIndices, topKWeights, trace); ok {
		return result
	}
	expanded1 := ExpandDims(x, 2)
	expanded := ExpandDims(expanded1, 2)
	Free(expanded1)

	var gate, up *Array
	if e.GateUpProj != nil && gemma4UseFusedExpertGateUp(x) {
		gateUp := e.GateUpProj.Forward(expanded, topKIndices)
		trace("gate_up", gateUp)
		var ok bool
		gate, up, ok = splitLastDimArray(gateUp)
		Free(gateUp)
		if !ok {
			gate, up = nil, nil
		}
	}
	if gate == nil || up == nil {
		Free(gate, up)
		up = e.UpProj.Forward(expanded, topKIndices)
		trace("up", up)
		gate = e.GateProj.Forward(expanded, topKIndices)
		trace("gate", gate)
	}
	Free(expanded)
	activated := geluGateMul(gate, up)
	trace("activation", activated)
	Free(gate, up)
	down := e.DownProj.Forward(activated, topKIndices)
	trace("down", down)
	Free(activated)
	downSqueezed := Squeeze(down, 3)
	Free(down)

	weightsExpanded := ExpandDims(topKWeights, 3)
	weighted := Mul(weightsExpanded, downSqueezed)
	trace("weighted", weighted)
	Free(weightsExpanded, downSqueezed)
	result := Sum(weighted, -2, false)
	trace("sum", result)
	Free(weighted)
	return result
}

func (e *Gemma4Experts) forwardSortedExpertPrefill(x, topKIndices, topKWeights *Array, trace func(string, ...*Array)) (*Array, bool) {
	if !sortedExpertPrefillEnabled() {
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
	var xShapeBuf, indicesShapeBuf, weightShapeBuf [maxTensorRank]int32
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

	flatIndices := Reshape(topKIndices, int32(routes))
	sortOrder := Argsort(flatIndices, -1)
	sortedIndices := Take(flatIndices, sortOrder, 0)
	routePositions := Arange(0, float64(routes), 1, DTypeInt32)
	sortedRoutePositions := Take(routePositions, sortOrder, 0)
	topKDivisor := FromValue(topK)
	sortedTokenPositions := floorDivide(sortedRoutePositions, topKDivisor)
	flatX := Reshape(x, int32(batch*seqLen), int32(hidden))
	sortedInputFlat := Take(flatX, sortedTokenPositions, 0)
	sortedInput := Reshape(sortedInputFlat, int32(routes), 1, int32(hidden))
	Free(routePositions, sortedRoutePositions, topKDivisor, sortedTokenPositions, flatX, sortedInputFlat)
	defer Free(flatIndices, sortOrder, sortedIndices, sortedInput)

	gate := gemma4SwitchLinearForwardSortedRoutes(e.GateProj, sortedInput, sortedIndices)
	trace("sorted_gate", gate)
	up := gemma4SwitchLinearForwardSortedRoutes(e.UpProj, sortedInput, sortedIndices)
	trace("sorted_up", up)
	activated := geluGateMul(gate, up)
	trace("sorted_activation", activated)
	Free(gate, up)
	down := gemma4SwitchLinearForwardSortedRoutes(e.DownProj, activated, sortedIndices)
	trace("sorted_down", down)
	Free(activated)

	flatWeights := Reshape(topKWeights, int32(routes))
	sortedWeights := Take(flatWeights, sortOrder, 0)
	weightsExpanded1 := ExpandDims(sortedWeights, 1)
	weightsExpanded := ExpandDims(weightsExpanded1, 2)
	weightedSorted := Mul(weightsExpanded, down)
	trace("sorted_weighted", weightedSorted)
	Free(flatWeights, sortedWeights, weightsExpanded1, weightsExpanded, down)

	inverseOrder := Argsort(sortOrder, -1)
	weightedOriginal := Take(weightedSorted, inverseOrder, 0)
	weightedSqueezed := Squeeze(weightedOriginal, 1)
	grouped := Reshape(weightedSqueezed, int32(batch), int32(seqLen), int32(topK), int32(hidden))
	result := Sum(grouped, -2, false)
	trace("sorted_sum", result)
	Free(weightedSorted, inverseOrder, weightedOriginal, weightedSqueezed, grouped)
	return result, true
}

func gemma4SortedExpertPrefillCompatible(e *Gemma4Experts) bool {
	return e != nil &&
		gemma4ExpertIDMatVecSwitchCompatible(e.GateProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.UpProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.DownProj)
}

func gemma4SwitchLinearForwardSortedRoutes(linear *SwitchLinear, input, expertIndices *Array) *Array {
	var out *Array
	if requiresDenseQuantizedMatmulFallback(linear.QuantizationMode) {
		denseWeight := dequantizeMode(linear.Weight, linear.Scales, linear.Biases, linear.GroupSize, linear.Bits, linear.QuantizationMode)
		weightTranspose := Transpose(denseWeight, 0, 2, 1)
		out = GatherMM(input, weightTranspose, nil, expertIndices, true)
		Free(denseWeight, weightTranspose)
	} else {
		out = GatherQMM(input, linear.Weight, linear.Scales, linear.Biases, nil, expertIndices, true, linear.GroupSize, linear.Bits, linear.QuantizationMode, true)
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		bias := Take(linear.Bias, expertIndices, 0)
		biasExpanded := ExpandDims(bias, bias.NumDims()-1)
		oldOut := out
		out = Add(out, biasExpanded)
		Free(oldOut, bias, biasExpanded)
	}
	return out
}

func (e *Gemma4Experts) forwardExpertIDMatVec(x, topKIndices, topKWeights *Array, trace func(string, ...*Array)) (*Array, bool) {
	if !expertIDMatVecEnabled() {
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
	var xShapeBuf, indicesShapeBuf [maxTensorRank]int32
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

	xFlat := Reshape(x, 1, int32(hidden))
	idsFlat := Reshape(topKIndices, int32(routes))
	defer Free(xFlat, idsFlat)

	var activated *Array
	if hasFusedGateUp && expertIDFusedActivationEnabled() {
		var err error
		activated, err = quantizedExpertIDGELUGateUpMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id fused activation matvec failed; falling back", "error", err)
			return nil, false
		}
		trace("activation_id_matvec", activated)
	} else if hasFusedGateUp {
		gateUp, err := quantizedExpertIDMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec gate/up failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_up_id_matvec", gateUp)
		gate, up, ok := splitLastDimArray(gateUp)
		Free(gateUp)
		if !ok {
			Free(gate, up)
			return nil, false
		}
		activated = geluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		Free(gate, up)
	} else if expertIDFusedActivationEnabled() {
		var err error
		activated, err = quantizedExpertIDGELUSplitGateUpMatVec(
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
		up, err := quantizedExpertIDMatVec(xFlat, e.UpProj.Weight, e.UpProj.Scales, e.UpProj.Biases, idsFlat, e.UpProj.GroupSize, e.UpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec up failed; falling back", "error", err)
			return nil, false
		}
		trace("up_id_matvec", up)
		gate, err := quantizedExpertIDMatVec(xFlat, e.GateProj.Weight, e.GateProj.Scales, e.GateProj.Biases, idsFlat, e.GateProj.GroupSize, e.GateProj.Bits)
		if err != nil {
			Free(up)
			core.Error("mlx: Gemma 4 expert id matvec gate failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_id_matvec", gate)
		activated = geluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		Free(gate, up)
	}

	weightsFlat := Reshape(topKWeights, int32(routes))
	down, err := quantizedExpertIDWeightedMatVecSum(activated, weightsFlat, e.DownProj.Weight, e.DownProj.Scales, e.DownProj.Biases, idsFlat, e.DownProj.GroupSize, e.DownProj.Bits)
	Free(weightsFlat)
	Free(activated)
	if err != nil {
		core.Error("mlx: Gemma 4 expert id weighted matvec down failed; falling back", "error", err)
		return nil, false
	}
	trace("down_weighted_sum_id_matvec", down)
	result := Reshape(down, 1, 1, int32(hidden))
	Free(down)
	return result, true
}

func gemma4ExpertIDMatVecSwitchCompatible(linear *SwitchLinear) bool {
	return linear != nil &&
		linear.Weight != nil && linear.Weight.Valid() &&
		linear.Scales != nil && linear.Scales.Valid() &&
		linear.Biases != nil && linear.Biases.Valid() &&
		linear.GroupSize > 0 &&
		isAffineQuantizationMode(linear.QuantizationMode) &&
		(linear.Bits == 2 || linear.Bits == 4 || linear.Bits == 8)
}

func gemma4UseFusedExpertGateUp(x *Array) bool {
	if x == nil || !x.Valid() {
		return false
	}
	// Branch on the row dim only — Shape() would heap-allocate a fresh
	// []int32 per MoE block per layer per token. Dim() is one C call.
	return x.NumDims() >= 2 && x.Dim(1) == 1
}

func splitLastDimArray(a *Array) (*Array, *Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	// Stack-allocated shape scratch — called per MoE block on the
	// fused-gate-up split path. Avoids per-call []int32 heap alloc.
	var shapeBuf [maxTensorRank]int32
	shape := a.ShapeInto(shapeBuf[:0])
	if len(shape) == 0 {
		return nil, nil, false
	}
	axis := len(shape) - 1
	mid := shape[axis] / 2
	if mid <= 0 || shape[axis]%2 != 0 {
		return nil, nil, false
	}
	var startsBuf, endsBuf [maxTensorRank]int32
	starts := startsBuf[:len(shape)]
	ends := endsBuf[:len(shape)]
	copy(ends, shape)
	ends[axis] = mid
	left := Slice(a, starts, ends)
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := Slice(a, starts, ends)
	return left, right, true
}
