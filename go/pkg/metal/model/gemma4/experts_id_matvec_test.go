// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin Gemma4Experts.forward / forwardExpertIDMatVec — the expert-id
// matvec fast paths must match the gather-QMM reference across the fused
// gate_up, split gate/up, split fused-activation, and sorted-prefill variants.
// They moved here from package metal's expert_id_matvec_test.go with the
// Gemma4Experts type and its decode methods; the runtime gates are driven via
// the public metal.SetRuntimeGate seam.

func TestExpertIDMatVec_Gemma4SortedExpertPrefillMatchesGatherQMM_Good(t *testing.T) {
	requireMetalRuntime(t)
	if metal.RuntimeGateValue("GO_MLX_ENABLE_GATHER_QMM_REFERENCE_TESTS") != "1" {
		t.Skip("set GO_MLX_ENABLE_GATHER_QMM_REFERENCE_TESTS=1 when the local metallib provides GatherQMM reference kernels")
	}

	const (
		experts   = 2
		seqLen    = 16
		topK      = 1
		hidden    = 8
		moeDim    = 8
		groupSize = 4
		bits      = 4
	)
	layer := &Gemma4Experts{
		GateProj: quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 3),
		UpProj:   quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 5),
		DownProj: quantizedSwitchLinearExpertIDTest(t, experts, hidden, moeDim, groupSize, bits, 11),
	}
	defer func() {
		metal.FreeSwitchLinear(layer.GateProj)
		metal.FreeSwitchLinear(layer.UpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	values := make([]float32, seqLen*hidden)
	for i := range values {
		values[i] = float32((i%11)-5) * 0.125
	}
	indices := make([]int32, seqLen*topK)
	weights := make([]float32, seqLen*topK)
	for i := range indices {
		indices[i] = int32((i + 1) % experts)
		weights[i] = 0.5 + 0.025*float32(i%5)
	}
	x := metal.FromValues(values, 1, seqLen, hidden)
	topKIndices := metal.FromValues(indices, 1, seqLen, topK)
	topKWeights := metal.FromValues(weights, 1, seqLen, topK)
	defer metal.Free(x, topKIndices, topKWeights)

	restoreOff := metal.SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer metal.Free(want)

	restoreOn := metal.SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "1")
	got := layer.forward(x, topKIndices, topKWeights, "")
	restoreOn()
	defer metal.Free(got)

	metal.Materialize(want, got)
	if err := metal.LastError(); err != nil {
		t.Skipf("GatherQMM reference kernel unavailable: %v", err)
	}
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 6e-4)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 %d %d]", shape, seqLen, hidden)
	}
}

type gemma4ExpertIDQuantCPUFixture struct {
	quantized []uint8
	scales    []float32
	biases    []float32
	experts   int
	outDim    int
	inDim     int
	groupSize int
}

func gemma4ExpertIDQuantFixture(experts, outDim, inDim, groupSize, seed int) gemma4ExpertIDQuantCPUFixture {
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*seed + 5) & 15)
	}
	groups := inDim / groupSize
	scales := make([]float32, experts*outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.025 * float32((i%9)+1)
		biases[i] = -0.45 + 0.05*float32((i+seed)%17)
	}
	return gemma4ExpertIDQuantCPUFixture{
		quantized: quantized,
		scales:    scales,
		biases:    biases,
		experts:   experts,
		outDim:    outDim,
		inDim:     inDim,
		groupSize: groupSize,
	}
}

func gemma4ExpertIDQuantFixtureAsBF16(fixture gemma4ExpertIDQuantCPUFixture) gemma4ExpertIDQuantCPUFixture {
	for i, value := range fixture.scales {
		fixture.scales[i] = gemma4ExpertIDBF16Round(value)
	}
	for i, value := range fixture.biases {
		fixture.biases[i] = gemma4ExpertIDBF16Round(value)
	}
	return fixture
}

func gemma4ExpertIDCPUReference(input []float32, ids []int32, routeWeights []float32, hidden int, gateOrGateUp gemma4ExpertIDQuantCPUFixture, up *gemma4ExpertIDQuantCPUFixture, down gemma4ExpertIDQuantCPUFixture) []float32 {
	out := make([]float32, hidden)
	for route, expertID := range ids {
		expert := int(expertID)
		var gate, upValues []float32
		if up == nil {
			gateUp := gemma4ExpertIDQuantMatVecCPU(input, route, expert, gateOrGateUp)
			half := len(gateUp) / 2
			gate = gateUp[:half]
			upValues = gateUp[half:]
		} else {
			gate = gemma4ExpertIDQuantMatVecCPU(input, route, expert, gateOrGateUp)
			upValues = gemma4ExpertIDQuantMatVecCPU(input, route, expert, *up)
		}
		activated := make([]float32, len(gate))
		for i := range activated {
			activated[i] = gemma4ExpertIDGELUCPU(gate[i]) * upValues[i]
		}
		downValues := gemma4ExpertIDQuantMatVecCPU(activated, 0, expert, down)
		for i := range out {
			out[i] += routeWeights[route] * downValues[i]
		}
	}
	return out
}

func gemma4ExpertIDQuantMatVecCPU(input []float32, route int, expert int, fixture gemma4ExpertIDQuantCPUFixture) []float32 {
	out := make([]float32, fixture.outDim)
	groups := fixture.inDim / fixture.groupSize
	baseInput := 0
	if len(input) >= (route+1)*fixture.inDim {
		baseInput = route * fixture.inDim
	}
	for outCol := range fixture.outDim {
		var sum float32
		for inCol := range fixture.inDim {
			weightIndex := (expert*fixture.outDim+outCol)*fixture.inDim + inCol
			group := inCol / fixture.groupSize
			scaleIndex := (expert*fixture.outDim+outCol)*groups + group
			weight := float32(fixture.quantized[weightIndex])*fixture.scales[scaleIndex] + fixture.biases[scaleIndex]
			sum += input[baseInput+inCol] * weight
		}
		out[outCol] = sum
	}
	return out
}

func gemma4ExpertIDGELUCPU(x float32) float32 {
	cube := x * x * x
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*cube)))))
}

func gemma4ExpertIDBF16Round(x float32) float32 {
	bits := math.Float32bits(x)
	rounded := bits + 0x7fff + ((bits >> 16) & 1)
	return math.Float32frombits(rounded & 0xffff0000)
}
