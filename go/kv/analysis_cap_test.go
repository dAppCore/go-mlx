// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"math"
	"testing"
)

// referenceStridedDifferentiation computes 1 - mean pairwise cosine over the
// stride-sampled positions, the exact value the capped
// kvAnalysisPositionDifferentiation must produce above the position cap.
func referenceStridedDifferentiation(flat []float32, seqLen, headDim, stride int) (float64, int) {
	var normed [][]float64
	for src := 0; src < seqLen; src += stride {
		v := make([]float64, headDim)
		var sum float64
		for k := 0; k < headDim; k++ {
			v[k] = float64(flat[src*headDim+k])
			sum += v[k] * v[k]
		}
		if sum > 0 {
			inv := 1.0 / math.Sqrt(sum)
			for k := range v {
				v[k] *= inv
			}
		}
		normed = append(normed, v)
	}
	n := len(normed)
	var total float64
	pairs := 0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var dot float64
			for k := 0; k < headDim; k++ {
				dot += normed[i][k] * normed[j][k]
			}
			total += dot
			pairs++
		}
	}
	if pairs == 0 {
		return 0, 0
	}
	return 1.0 - total/float64(pairs), pairs
}

// TestPositionDifferentiation_CapMatchesStridedExact verifies the cap (a) leaves
// at/below-cap analysis byte-identical and (b) above the cap produces exactly the
// strided-position result (not garbage / not a panic). headDim>1 and headDim==1
// paths both covered.
func TestPositionDifferentiation_CapMatchesStridedExact(t *testing.T) {
	const cap = 4096 // mirrors maxExactPositions
	cases := []struct {
		name    string
		seqLen  int
		headDim int
	}{
		{"belowCap_headDim4_exact", 1000, 4},
		{"belowCap_headDim1_exact", 2000, 1},
		{"aboveCap_headDim4_sampled", 16384, 4},
		{"aboveCap_headDim1_sampled", 12000, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			flat := make([]float32, tc.seqLen*tc.headDim)
			for i := range flat {
				flat[i] = float32(math.Sin(float64(i)*0.017) + 0.3*math.Cos(float64(i)*0.005))
			}
			heads := []HeadSnapshot{{Key: flat, Value: flat}}

			got, gotLocked, gotPairs := kvAnalysisPositionDifferentiation(heads, tc.seqLen, tc.headDim, true, nil)

			stride := 1
			if tc.seqLen > cap {
				stride = (tc.seqLen + cap - 1) / cap
			}
			want, wantPairs := referenceStridedDifferentiation(flat, tc.seqLen, tc.headDim, stride)

			if math.Abs(got-want) > 1e-9 {
				t.Errorf("diff = %v, want strided-exact %v (stride %d)", got, want, stride)
			}
			if gotPairs != wantPairs {
				t.Errorf("pairs = %d, want %d", gotPairs, wantPairs)
			}
			if gotLocked < 0 || gotLocked > gotPairs {
				t.Errorf("locked %d out of range [0,%d]", gotLocked, gotPairs)
			}
		})
	}
}
