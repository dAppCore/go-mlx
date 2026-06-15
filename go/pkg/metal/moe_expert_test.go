// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestMoESwiGLUExperts_Forward_Good(t *testing.T) {
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
		FreeSwitchLinear(experts.GateProj)
		FreeSwitchLinear(experts.UpProj)
		FreeSwitchLinear(experts.DownProj)
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

func TestMoETextLayersRuntimeAvailable_Good(t *testing.T) {
	layers := []*DenseDecoderLayer{{MLP: &SiLUMLP{}}, {MLP: &SiLUMLP{}}}
	if !MoETextLayersRuntimeAvailable(layers, func(layer *DenseDecoderLayer) MoETextLayerParts {
		return MoETextLayerParts{Dense: layer, OK: layer != nil}
	}) {
		t.Fatal("MoETextLayersRuntimeAvailable(dense layers) = false, want true")
	}
}

func TestMoETextLayersRuntimeAvailable_Bad(t *testing.T) {
	ready := &DenseDecoderLayer{MLP: &SiLUMLP{}}
	cases := []struct {
		name  string
		input []*DenseDecoderLayer
		parts func(*DenseDecoderLayer) MoETextLayerParts
	}{
		{name: "empty"},
		{name: "nil-parts", input: []*DenseDecoderLayer{ready}},
		{name: "nil-layer", input: []*DenseDecoderLayer{nil}, parts: func(layer *DenseDecoderLayer) MoETextLayerParts {
			return MoETextLayerParts{Dense: layer, OK: layer != nil}
		}},
		{name: "missing-mlp", input: []*DenseDecoderLayer{{}}, parts: func(layer *DenseDecoderLayer) MoETextLayerParts {
			return MoETextLayerParts{Dense: layer, OK: layer != nil}
		}},
		{name: "moe-missing-router", input: []*DenseDecoderLayer{ready}, parts: func(layer *DenseDecoderLayer) MoETextLayerParts {
			return MoETextLayerParts{Dense: layer, IsMoE: true, SwitchExperts: &MoESwiGLUExperts{}, OK: true}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if MoETextLayersRuntimeAvailable(tc.input, tc.parts) {
				t.Fatal("MoETextLayersRuntimeAvailable() = true, want false")
			}
		})
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

func TestMoEExpert_moeSwiGLUTopK_Good(t *testing.T) {
	// Pure-Go clamp: a positive top-K passes through unchanged.
	if got := moeSwiGLUTopK(4); got != 4 {
		t.Fatalf("moeSwiGLUTopK(4) = %d, want 4", got)
	}
}

func TestMoEExpert_moeSwiGLUTopK_Bad(t *testing.T) {
	// Zero and negative top-K both collapse to zero rather than driving a
	// negative dispatch downstream.
	for _, in := range []int{0, -1, -8} {
		if got := moeSwiGLUTopK(in); got != 0 {
			t.Fatalf("moeSwiGLUTopK(%d) = %d, want 0", in, got)
		}
	}
}

func TestMoEExpert_moeSwitchExpertsAvailable_Ugly(t *testing.T) {
	// A partially-built experts struct (any one projection nil) is unavailable;
	// only the fully-populated struct reports true. Uses unallocated SwitchLinear
	// placeholders — availability is a pointer-presence check, no Metal needed.
	if moeSwitchExpertsAvailable(nil) {
		t.Fatal("moeSwitchExpertsAvailable(nil) = true, want false")
	}
	g, u, d := &SwitchLinear{}, &SwitchLinear{}, &SwitchLinear{}
	cases := []*MoESwiGLUExperts{
		{},
		{GateProj: g},
		{GateProj: g, UpProj: u},
		{UpProj: u, DownProj: d},
	}
	for i, e := range cases {
		if moeSwitchExpertsAvailable(e) {
			t.Fatalf("case %d: moeSwitchExpertsAvailable(partial) = true, want false", i)
		}
	}
	if !moeSwitchExpertsAvailable(&MoESwiGLUExperts{GateProj: g, UpProj: u, DownProj: d}) {
		t.Fatal("moeSwitchExpertsAvailable(full) = false, want true")
	}
}

func TestMoEExpert_MoEDenseLayerTextReady_Good(t *testing.T) {
	// A dense (non-MoE) layer is text-ready as soon as its MLP slot is populated.
	dense := &DenseDecoderLayer{MLP: &SiLUMLP{}}
	if !MoEDenseLayerTextReady(dense, false, nil, nil) {
		t.Fatal("MoEDenseLayerTextReady(dense w/ MLP) = false, want true")
	}
}

func TestMoEExpert_MoEDenseLayerTextReady_Bad(t *testing.T) {
	// nil dense → not ready; dense-but-no-MLP → not ready.
	if MoEDenseLayerTextReady(nil, false, nil, nil) {
		t.Fatal("MoEDenseLayerTextReady(nil) = true, want false")
	}
	if MoEDenseLayerTextReady(&DenseDecoderLayer{}, false, nil, nil) {
		t.Fatal("MoEDenseLayerTextReady(no MLP) = true, want false")
	}
}

func TestMoEExpert_MoEDenseLayerTextReady_Ugly(t *testing.T) {
	// A layer flagged MoE needs BOTH a populated router and full experts — an
	// MLP alone is not enough, and a populated router with empty experts still
	// fails. Routers/experts here are presence-only (no weights → no Metal).
	dense := &DenseDecoderLayer{MLP: &SiLUMLP{}}
	if MoEDenseLayerTextReady(dense, true, nil, nil) {
		t.Fatal("MoE layer with nil router/experts = ready, want false")
	}
	if MoEDenseLayerTextReady(dense, true, nil, &MoESwiGLUExperts{}) {
		t.Fatal("MoE layer with nil router = ready, want false")
	}
}

func TestMoEExpert_NewMoESwiGLUExpertsFromLinears_Bad(t *testing.T) {
	// Empty / nil-first projection stacks are rejected before any Metal op:
	// these guard branches need no GPU.
	if got, ok := NewMoESwiGLUExpertsFromLinears(nil, nil, nil); ok || got != nil {
		t.Fatalf("NewMoESwiGLUExpertsFromLinears(nil...) = (%v,%v), want (nil,false)", got, ok)
	}
	if got, ok := newMoESwitchLinearFromLinears(nil); ok || got != nil {
		t.Fatalf("newMoESwitchLinearFromLinears(nil) = (%v,%v), want (nil,false)", got, ok)
	}
	if got, ok := newMoESwitchLinearFromLinears([]*Linear{nil}); ok || got != nil {
		t.Fatalf("newMoESwitchLinearFromLinears([nil]) = (%v,%v), want (nil,false)", got, ok)
	}
	if got, ok := newMoESwitchLinearFromLinears([]*Linear{{}}); ok || got != nil {
		t.Fatalf("newMoESwitchLinearFromLinears([{} invalid weight]) = (%v,%v), want (nil,false)", got, ok)
	}
}

func TestMoEExpert_sameMoEArrayShape_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	defer Free(a, b)
	if !sameMoEArrayShape(a, b) {
		t.Fatal("sameMoEArrayShape([2,2],[2,2]) = false, want true")
	}
}

func TestMoEExpert_sameMoEArrayShape_Bad(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	diffRank := FromValues([]float32{1, 2, 3, 4}, 4)
	diffDim := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	defer Free(a, diffRank, diffDim)

	if sameMoEArrayShape(a, nil) {
		t.Fatal("sameMoEArrayShape(a,nil) = true, want false")
	}
	if sameMoEArrayShape(a, diffRank) {
		t.Fatal("sameMoEArrayShape([2,2],[4]) = true, want false")
	}
	if sameMoEArrayShape(a, diffDim) {
		t.Fatal("sameMoEArrayShape([2,2],[2,3]) = true, want false")
	}
}

func TestMoEExpert_moeLinearStackCompatible_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	// first: a plain (unquantised, unbiased) [2,2] Linear. A second leaf with a
	// mismatched weight shape, or one that introduces a bias the first lacks,
	// must be rejected — these are the compatibility branches the batched
	// switch-expert builder relies on.
	first := &Linear{Weight: FromValues([]float32{1, 2, 3, 4}, 2, 2)}
	okPeer := &Linear{Weight: FromValues([]float32{5, 6, 7, 8}, 2, 2)}
	badShape := &Linear{Weight: FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)}
	biasedPeer := &Linear{
		Weight: FromValues([]float32{9, 8, 7, 6}, 2, 2),
		Bias:   FromValues([]float32{0.1, 0.2}, 2),
	}
	defer func() {
		Free(first.Weight, okPeer.Weight, badShape.Weight, biasedPeer.Weight, biasedPeer.Bias)
	}()

	if !moeLinearStackCompatible(first, okPeer, false, false) {
		t.Fatal("compatible same-shape unbiased peer reported incompatible")
	}
	if moeLinearStackCompatible(first, badShape, false, false) {
		t.Fatal("mismatched weight shape reported compatible")
	}
	if moeLinearStackCompatible(first, biasedPeer, false, false) {
		t.Fatal("peer carrying an unexpected bias reported compatible")
	}
	if moeLinearStackCompatible(first, nil, false, false) {
		t.Fatal("nil peer reported compatible")
	}
}

func TestMoEExpert_NewMoESwiGLUExpertsFromLinears_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Two experts, 2->2 plain Linears for each of gate/up/down. The builder
	// stacks them into batched SwitchLinears; FreeMoESwiGLUExperts releases them.
	mk := func() []*Linear {
		return []*Linear{
			{Weight: FromValues([]float32{1, 0, 0, 1}, 2, 2)},
			{Weight: FromValues([]float32{0, 1, 1, 0}, 2, 2)},
		}
	}
	gate, up, down := mk(), mk(), mk()
	defer func() {
		for _, set := range [][]*Linear{gate, up, down} {
			for _, l := range set {
				Free(l.Weight)
			}
		}
	}()

	experts, ok := NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok || experts == nil {
		t.Fatalf("NewMoESwiGLUExpertsFromLinears = (%v,%v), want a built struct", experts, ok)
	}
	if !moeSwitchExpertsAvailable(experts) {
		t.Fatal("built experts report unavailable")
	}
	FreeMoESwiGLUExperts(experts)
	FreeMoESwiGLUExperts(nil) // nil free must be a safe no-op
}

func TestMoEExpert_MoESwiGLUForward_Bad(t *testing.T) {
	requireMetalRuntime(t)

	// A nil router makes the top-K selection unavailable, so the whole MoE
	// forward reports (nil,false) without touching the experts.
	input := FromValues([]float32{0.25, -0.75}, 1, 1, 2)
	defer Free(input)
	if got, ok := MoESwiGLUForward(input, nil, 2, &MoESwiGLUExperts{}); ok || got != nil {
		t.Fatalf("MoESwiGLUForward(nil router) = (%v,%v), want (nil,false)", got, ok)
	}
}
