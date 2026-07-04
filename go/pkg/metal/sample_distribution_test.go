// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestSamplingDistribution_ValidAndTruncates pins the helper that speculative
// sampling weighs its accept coin against: the result must be a valid
// probability distribution (non-negative, sums to 1) and honour the truncation
// (top-k leaves at most k tokens with mass).
func TestSamplingDistribution_ValidAndTruncates(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 8)
	defer Free(logits)

	cases := []struct {
		name             string
		temp, topP, minP float32
		topK             int
	}{
		{"plain_temp1", 1, 0, 0, 0},
		{"temp2_flatter", 2, 0, 0, 0},
		{"topk3", 1, 0, 0, 3},
		{"topp_half", 1, 0.5, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dist := samplingDistribution(logits, tc.temp, tc.topP, tc.minP, tc.topK)
			defer Free(dist)
			Materialize(dist)
			p := dist.Floats()

			sum, nonzero := float64(0), 0
			for _, v := range p {
				if v < 0 {
					t.Fatalf("negative probability %f in %v", v, p)
				}
				sum += float64(v)
				if v > 1e-6 {
					nonzero++
				}
			}
			if sum < 0.999 || sum > 1.001 {
				t.Fatalf("distribution sum = %f, want 1.0 (%v)", sum, p)
			}
			if tc.topK > 0 && nonzero > tc.topK {
				t.Fatalf("top-k=%d left %d tokens with mass, want <= %d (%v)", tc.topK, nonzero, tc.topK, p)
			}
		})
	}
}

// TestSamplingDistribution_TemperatureFlattens checks temperature does what the
// accept-coin math assumes: higher temperature lowers the peak token's mass.
func TestSamplingDistribution_TemperatureFlattens(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 8)
	defer Free(logits)

	cold := samplingDistribution(logits, 0.5, 0, 0, 0)
	defer Free(cold)
	hot := samplingDistribution(logits, 2.0, 0, 0, 0)
	defer Free(hot)
	Materialize(cold, hot)

	// index 7 is the max logit → its probability must shrink as temp rises.
	if c, h := cold.Floats()[7], hot.Floats()[7]; h >= c {
		t.Fatalf("peak prob did not fall with temperature: temp0.5=%f temp2.0=%f", c, h)
	}
}
