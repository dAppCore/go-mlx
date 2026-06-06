// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func packMLXAffineQ4TestRows(t *testing.T, values []uint8) []uint32 {
	t.Helper()
	if len(values)%8 != 0 {
		t.Fatalf("q4 test rows must have a multiple of 8 values, got %d", len(values))
	}
	packed := make([]uint32, len(values)/8)
	for i, value := range values {
		if value > 15 {
			t.Fatalf("q4 value %d exceeds 15", value)
		}
		packed[i/8] |= uint32(value) << uint((i%8)*4)
	}
	return packed
}

func quantizedExpertIDMatVecCPUReference(input []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	out := make([]float32, len(ids)*outDim)
	for route, expertID := range ids {
		expert := int(expertID)
		for outCol := range outDim {
			var sum float32
			for inCol := range inDim {
				weightIndex := (expert*outDim+outCol)*inDim + inCol
				group := inCol / groupSize
				scaleIndex := (expert*outDim+outCol)*groups + group
				w := float32(quantized[weightIndex])*scales[scaleIndex] + biases[scaleIndex]
				sum += input[route*inDim+inCol] * w
			}
			out[route*outDim+outCol] = sum
		}
	}
	return out
}

func quantizedExpertIDGELUGateUpMatVecCPUReference(input []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	halfOut := outDim / 2
	out := make([]float32, len(ids)*halfOut)
	for route, expertID := range ids {
		expert := int(expertID)
		for outCol := range halfOut {
			var gateSum, upSum float32
			for inCol := range inDim {
				group := inCol / groupSize
				gateWeightIndex := (expert*outDim+outCol)*inDim + inCol
				upWeightIndex := (expert*outDim+outCol+halfOut)*inDim + inCol
				gateScaleIndex := (expert*outDim+outCol)*groups + group
				upScaleIndex := (expert*outDim+outCol+halfOut)*groups + group
				gateWeight := float32(quantized[gateWeightIndex])*scales[gateScaleIndex] + biases[gateScaleIndex]
				upWeight := float32(quantized[upWeightIndex])*scales[upScaleIndex] + biases[upScaleIndex]
				inputValue := input[route*inDim+inCol]
				gateSum += inputValue * gateWeight
				upSum += inputValue * upWeight
			}
			out[route*halfOut+outCol] = geluApproxFloat32(gateSum) * upSum
		}
	}
	return out
}

func geluApproxFloat32(x float32) float32 {
	cube := x * x * x
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*cube)))))
}

func quantizedExpertIDWeightedMatVecSumCPUReference(input, routeWeights []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	out := make([]float32, outDim)
	for route, expertID := range ids {
		expert := int(expertID)
		routeWeight := routeWeights[route]
		for outCol := range outDim {
			var sum float32
			for inCol := range inDim {
				weightIndex := (expert*outDim+outCol)*inDim + inCol
				group := inCol / groupSize
				scaleIndex := (expert*outDim+outCol)*groups + group
				w := float32(quantized[weightIndex])*scales[scaleIndex] + biases[scaleIndex]
				sum += input[route*inDim+inCol] * w
			}
			out[outCol] += routeWeight * sum
		}
	}
	return out
}

func quantizedSwitchLinearExpertIDTest(t *testing.T, experts, outDim, inDim, groupSize, bits, seed int) *SwitchLinear {
	t.Helper()
	if bits != 4 {
		t.Fatalf("test helper currently packs q4 only, got bits=%d", bits)
	}
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
	return NewQuantizedSwitchLinear(
		FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits)),
		FromValues(scales, experts, outDim, groups),
		FromValues(biases, experts, outDim, groups),
		nil,
		groupSize,
		bits,
	)
}

func quantizedSwitchLinearSidecarsAsType(linear *SwitchLinear, dtype DType) {
	if linear == nil || linear.Scales == nil || linear.Biases == nil {
		return
	}
	scales := AsType(linear.Scales, dtype)
	biases := AsType(linear.Biases, dtype)
	Free(linear.Scales, linear.Biases)
	linear.Scales = scales
	linear.Biases = biases
}
