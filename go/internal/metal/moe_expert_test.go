// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestMoESwiGLUExperts_Forward_Good(t *testing.T) {
	coverageTokens := "MoESwiGLUExperts Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	gateValues := []float32{
		0.7, -0.2,
		0.1, 0.5,
		-0.4, 0.3,
		0.8, -0.6,
	}
	upValues := []float32{
		0.2, 0.9,
		-0.3, 0.4,
		0.6, -0.1,
		0.5, 0.7,
	}
	downValues := []float32{
		0.5, -0.4,
		0.2, 0.3,
		-0.6, 0.1,
		0.7, 0.2,
	}
	experts := &MoESwiGLUExperts{
		GateProj: NewSwitchLinear(FromValues(gateValues, 2, 2, 2), nil),
		UpProj:   NewSwitchLinear(FromValues(upValues, 2, 2, 2), nil),
		DownProj: NewSwitchLinear(FromValues(downValues, 2, 2, 2), nil),
	}
	defer func() {
		freeSwitchLinear(experts.GateProj)
		freeSwitchLinear(experts.UpProj)
		freeSwitchLinear(experts.DownProj)
	}()

	inputValues := []float32{0.25, -0.75}
	expertIDsValues := []int32{1, 0}
	routeWeightValues := []float32{0.8, 0.2}
	input := FromValues(inputValues, 1, 1, 2)
	expertIDs := FromValues(expertIDsValues, 1, 1, 2)
	routeWeights := FromValues(routeWeightValues, 1, 1, 2)
	defer Free(input, expertIDs, routeWeights)

	got, ok := experts.Forward(input, expertIDs, routeWeights)
	if !ok {
		t.Fatal("MoESwiGLUExperts.Forward() ok = false, want true")
	}
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	want := moeSwiGLUExpertsCPUReference(inputValues, expertIDsValues, routeWeightValues, gateValues, upValues, downValues, 2, 2)
	floatSliceApprox(t, got.Floats(), want)
}

func TestMoESwiGLUExperts_Forward_Bad(t *testing.T) {
	coverageTokens := "MoESwiGLUExperts Forward Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	input := FromValues([]float32{0.25, -0.75}, 1, 1, 2)
	ids := FromValues([]int32{0}, 1, 1, 1)
	weights := FromValues([]float32{1}, 1, 1, 1)
	defer Free(input, ids, weights)

	if got, ok := (*MoESwiGLUExperts)(nil).Forward(input, ids, weights); ok || got != nil {
		t.Fatalf("nil experts Forward() = (%v, %v), want nil false", got, ok)
	}

	experts := &MoESwiGLUExperts{}
	if got, ok := experts.Forward(input, ids, weights); ok || got != nil {
		t.Fatalf("empty experts Forward() = (%v, %v), want nil false", got, ok)
	}
}

func moeSwiGLUExpertsCPUReference(input []float32, expertIDs []int32, routeWeights []float32, gateWeight, upWeight, downWeight []float32, outDim, inDim int) []float32 {
	result := make([]float32, outDim)
	for route, expertID := range expertIDs {
		expert := int(expertID)
		gate := moeSwitchLinearCPU(input, gateWeight, expert, outDim, inDim)
		up := moeSwitchLinearCPU(input, upWeight, expert, outDim, inDim)
		activated := make([]float32, outDim)
		for i := range activated {
			activated[i] = siluCPU(gate[i]) * up[i]
		}
		down := moeSwitchLinearCPU(activated, downWeight, expert, outDim, inDim)
		for i := range result {
			result[i] += routeWeights[route] * down[i]
		}
	}
	return result
}

func moeSwitchLinearCPU(input, weight []float32, expert, outDim, inDim int) []float32 {
	result := make([]float32, outDim)
	base := expert * outDim * inDim
	for out := 0; out < outDim; out++ {
		sum := float32(0)
		for in := 0; in < inDim; in++ {
			sum += input[in] * weight[base+out*inDim+in]
		}
		result[out] = sum
	}
	return result
}

func siluCPU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
