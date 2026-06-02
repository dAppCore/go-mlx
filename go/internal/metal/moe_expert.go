// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// MoESwiGLUExperts is the shared selected-expert SwiGLU dispatch used by
// Qwen/Mixtral-style sparse MoE layers once routing has chosen expert IDs.
type MoESwiGLUExperts struct {
	GateProj *SwitchLinear
	UpProj   *SwitchLinear
	DownProj *SwitchLinear
}

func (e *MoESwiGLUExperts) Forward(input, expertIDs, routeWeights *Array) (*Array, bool) {
	if !e.available(input, expertIDs, routeWeights) {
		return nil, false
	}

	expanded1 := ExpandDims(input, 2)
	expanded := ExpandDims(expanded1, 2)
	Free(expanded1)

	gate := e.GateProj.Forward(expanded, expertIDs)
	up := e.UpProj.Forward(expanded, expertIDs)
	Free(expanded)

	activated := siluGateMul(gate, up)
	Free(gate, up)

	down := e.DownProj.Forward(activated, expertIDs)
	Free(activated)

	downSqueezed := Squeeze(down, 3)
	Free(down)

	weightsExpanded := ExpandDims(routeWeights, 3)
	weighted := Mul(weightsExpanded, downSqueezed)
	Free(weightsExpanded, downSqueezed)

	result := Sum(weighted, -2, false)
	Free(weighted)
	return result, true
}

func newMoESwiGLUExpertsFromLinears(gate, up, down []*Linear) (*MoESwiGLUExperts, bool) {
	gateSwitch, ok := newMoESwitchLinearFromLinears(gate)
	if !ok {
		return nil, false
	}
	upSwitch, ok := newMoESwitchLinearFromLinears(up)
	if !ok {
		freeSwitchLinear(gateSwitch)
		return nil, false
	}
	downSwitch, ok := newMoESwitchLinearFromLinears(down)
	if !ok {
		freeSwitchLinear(gateSwitch)
		freeSwitchLinear(upSwitch)
		return nil, false
	}
	return &MoESwiGLUExperts{
		GateProj: gateSwitch,
		UpProj:   upSwitch,
		DownProj: downSwitch,
	}, true
}

func newMoESwitchLinearFromLinears(linears []*Linear) (*SwitchLinear, bool) {
	if len(linears) == 0 {
		return nil, false
	}

	weights := make([]*Array, 0, len(linears))
	scales := make([]*Array, 0, len(linears))
	qbiases := make([]*Array, 0, len(linears))
	biases := make([]*Array, 0, len(linears))
	first := linears[0]
	if first == nil || first.Weight == nil || !first.Weight.Valid() {
		return nil, false
	}
	hasQuant := first.Scales != nil && first.Scales.Valid()
	hasBias := first.Bias != nil && first.Bias.Valid()

	for _, linear := range linears {
		if !moeLinearStackCompatible(first, linear, hasQuant, hasBias) {
			return nil, false
		}
		weights = append(weights, ExpandDims(linear.Weight, 0))
		if hasQuant {
			scales = append(scales, ExpandDims(linear.Scales, 0))
			qbiases = append(qbiases, ExpandDims(linear.Biases, 0))
		}
		if hasBias {
			biases = append(biases, ExpandDims(linear.Bias, 0))
		}
	}
	defer Free(weights...)
	defer Free(scales...)
	defer Free(qbiases...)
	defer Free(biases...)

	weight := Concatenate(weights, 0)
	var bias *Array
	if hasBias {
		bias = Concatenate(biases, 0)
	}
	if !hasQuant {
		return NewSwitchLinear(weight, bias), true
	}
	scale := Concatenate(scales, 0)
	qbias := Concatenate(qbiases, 0)
	return newQuantizedSwitchLinearWithMode(weight, scale, qbias, bias, first.GroupSize, first.Bits, first.QuantizationMode), true
}

func moeLinearStackCompatible(first, linear *Linear, hasQuant, hasBias bool) bool {
	if linear == nil || linear.Weight == nil || !linear.Weight.Valid() {
		return false
	}
	if !sameMoEArrayShape(first.Weight, linear.Weight) {
		return false
	}
	if hasBias != (linear.Bias != nil && linear.Bias.Valid()) {
		return false
	}
	if hasBias && !sameMoEArrayShape(first.Bias, linear.Bias) {
		return false
	}
	if hasQuant != (linear.Scales != nil && linear.Scales.Valid()) {
		return false
	}
	if !hasQuant {
		return true
	}
	return linear.Biases != nil && linear.Biases.Valid() &&
		first.GroupSize == linear.GroupSize &&
		first.Bits == linear.Bits &&
		normalizeQuantizationMode(first.QuantizationMode) == normalizeQuantizationMode(linear.QuantizationMode) &&
		sameMoEArrayShape(first.Scales, linear.Scales) &&
		sameMoEArrayShape(first.Biases, linear.Biases)
}

func sameMoEArrayShape(a, b *Array) bool {
	if a == nil || b == nil || !a.Valid() || !b.Valid() {
		return false
	}
	var aBuf, bBuf [maxTensorRank]int32
	aShape := a.ShapeInto(aBuf[:0])
	bShape := b.ShapeInto(bBuf[:0])
	if len(aShape) != len(bShape) {
		return false
	}
	for i := range aShape {
		if aShape[i] != bShape[i] {
			return false
		}
	}
	return true
}

func freeMoESwiGLUExperts(e *MoESwiGLUExperts) {
	if e == nil {
		return
	}
	freeSwitchLinear(e.GateProj)
	freeSwitchLinear(e.UpProj)
	freeSwitchLinear(e.DownProj)
}

func moeSwiGLUTopK(topK int32) int {
	if topK <= 0 {
		return 0
	}
	return int(topK)
}

func (e *MoESwiGLUExperts) available(input, expertIDs, routeWeights *Array) bool {
	if e == nil || e.GateProj == nil || e.UpProj == nil || e.DownProj == nil {
		return false
	}
	if input == nil || expertIDs == nil || routeWeights == nil {
		return false
	}
	if !input.Valid() || !expertIDs.Valid() || !routeWeights.Valid() {
		return false
	}
	var inputShapeBuf, idsShapeBuf, weightsShapeBuf [maxTensorRank]int32
	inputShape := input.ShapeInto(inputShapeBuf[:0])
	idsShape := expertIDs.ShapeInto(idsShapeBuf[:0])
	weightsShape := routeWeights.ShapeInto(weightsShapeBuf[:0])
	if len(inputShape) != 3 || len(idsShape) != 3 || len(weightsShape) != 3 {
		return false
	}
	return inputShape[0] == idsShape[0] &&
		inputShape[1] == idsShape[1] &&
		idsShape[0] == weightsShape[0] &&
		idsShape[1] == weightsShape[1] &&
		idsShape[2] == weightsShape[2]
}

func moeSwiGLUForward(input *Array, router *Qwen3MoERouter, topK int32, experts *MoESwiGLUExperts) (*Array, bool) {
	expertIDs, routeWeights, ok, err := qwen3MoERouterSelectTopK(input, router, moeSwiGLUTopK(topK))
	if err != nil {
		core.Error("mlx: MoE router selected-expert dispatch failed; falling back", "error", err)
		return nil, false
	}
	if !ok {
		return nil, false
	}
	defer Free(expertIDs, routeWeights)
	return experts.Forward(input, expertIDs, routeWeights)
}
