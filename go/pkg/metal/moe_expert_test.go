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

func TestMoETextRuntimeAvailable_Good(t *testing.T) {
	coverageTokens := "MoETextRuntimeAvailable Good"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	qRouter, qExperts, qCleanup := moeReadyRuntimeParts(t)
	defer qCleanup()
	qwen := &Qwen3MoEModel{
		Layers: []*Qwen3MoEDecoderLayer{{
			Dense: &Qwen3DecoderLayer{},
			MoE: &Qwen3MoEBlock{
				Router:        qRouter,
				Experts:       []*Qwen3MoEExpert{{}},
				SwitchExperts: qExperts,
			},
		}},
	}
	if !qwen3MoETextRuntimeAvailable(qwen) {
		t.Fatal("qwen3MoETextRuntimeAvailable() = false, want true")
	}

	mRouter, mExperts, mCleanup := moeReadyRuntimeParts(t)
	defer mCleanup()
	mixtral := &MixtralModel{
		Layers: []*MixtralDecoderLayer{{
			Dense: &Qwen3DecoderLayer{},
			MoE: &MixtralMoEBlock{
				Router:        mRouter,
				Experts:       []*MixtralExpert{{}},
				SwitchExperts: mExperts,
			},
		}},
	}
	if !mixtralTextRuntimeAvailable(mixtral) {
		t.Fatal("mixtralTextRuntimeAvailable() = false, want true")
	}

	kRouter, kExperts, kCleanup := moeReadyRuntimeParts(t)
	defer kCleanup()
	kimi := &KimiModel{
		Layers: []*KimiDecoderLayer{{
			Dense: &Qwen3DecoderLayer{},
			MoE: &KimiMoEBlock{
				Router:        kRouter,
				Experts:       []*KimiExpert{{}},
				SwitchExperts: kExperts,
			},
		}},
	}
	if !kimiTextRuntimeAvailable(kimi) {
		t.Fatal("kimiTextRuntimeAvailable() = false, want true")
	}

	gRouter, gExperts, gCleanup := moeReadyRuntimeParts(t)
	defer gCleanup()
	gptOss := &GptOssModel{
		Layers: []*GptOssDecoderLayer{{
			Dense: &Qwen3DecoderLayer{},
			MoE: &GptOssMoEBlock{
				Router:        gRouter,
				Experts:       []*GptOssExpert{{}},
				SwitchExperts: gExperts,
			},
		}},
	}
	if !gptOssTextRuntimeAvailable(gptOss) {
		t.Fatal("gptOssTextRuntimeAvailable() = false, want true")
	}
}

func TestMoETextRuntimeAvailable_Bad(t *testing.T) {
	coverageTokens := "MoETextRuntimeAvailable Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	if qwen3MoETextRuntimeAvailable(&Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{{Dense: &Qwen3DecoderLayer{}}}}) {
		t.Fatal("qwen3MoETextRuntimeAvailable(incomplete) = true, want false")
	}
	if mixtralTextRuntimeAvailable(&MixtralModel{Layers: []*MixtralDecoderLayer{{Dense: &Qwen3DecoderLayer{}}}}) {
		t.Fatal("mixtralTextRuntimeAvailable(incomplete) = true, want false")
	}
	if kimiTextRuntimeAvailable(&KimiModel{Layers: []*KimiDecoderLayer{{Dense: &Qwen3DecoderLayer{}}}}) {
		t.Fatal("kimiTextRuntimeAvailable(incomplete) = true, want false")
	}
	if gptOssTextRuntimeAvailable(&GptOssModel{Layers: []*GptOssDecoderLayer{{Dense: &Qwen3DecoderLayer{}}}}) {
		t.Fatal("gptOssTextRuntimeAvailable(incomplete) = true, want false")
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

func moeReadyRuntimeParts(t *testing.T) (*Qwen3MoERouter, *MoESwiGLUExperts, func()) {
	t.Helper()
	routerWeight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	experts := &MoESwiGLUExperts{
		GateProj: NewSwitchLinear(FromValues([]float32{1, 0, 0, 1}, 1, 2, 2), nil),
		UpProj:   NewSwitchLinear(FromValues([]float32{0.5, 0, 0, 0.5}, 1, 2, 2), nil),
		DownProj: NewSwitchLinear(FromValues([]float32{1, 0, 0, 1}, 1, 2, 2), nil),
	}
	cleanup := func() {
		Free(routerWeight)
		freeMoESwiGLUExperts(experts)
	}
	return &Qwen3MoERouter{Weight: routerWeight}, experts, cleanup
}

func moeSwitchLinearCPU(input, weight []float32, expert, outDim, inDim int) []float32 {
	result := make([]float32, outDim)
	base := expert * outDim * inDim
	for out := range outDim {
		sum := float32(0)
		for in := range inDim {
			sum += input[in] * weight[base+out*inDim+in]
		}
		result[out] = sum
	}
	return result
}

func siluCPU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
