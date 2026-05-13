// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	"dappco.re/go/mlx/internal/metal"
)

func skipIfNoUsableMetal(t *testing.T) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("usable Metal device unavailable")
	}
}

func float32SlicesRoughlyEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > epsilon {
			return false
		}
	}
	return true
}

func denseProjectionReference(input []float32, rows int, weight []float32, outDim, inDim int, bias []float32) []float32 {
	out := make([]float32, rows*outDim)
	for row := 0; row < rows; row++ {
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := float32(0)
			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += input[row*inDim+inIndex] * weight[outIndex*inDim+inIndex]
			}
			if len(bias) > 0 {
				sum += bias[outIndex]
			}
			out[row*outDim+outIndex] = sum
		}
	}
	return out
}
