// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Speculative-sampling acceptance (Leviathan/Chen): a drafted token x sampled
// from the drafter distribution q is accepted with probability min(1, p(x)/q(x))
// against the target distribution p; on rejection the replacement is drawn from
// the normalised residual (p-q)+. The result is sampled EXACTLY from p, so a
// served temperature-1 request is output-distribution-identical to plain
// sampling — only faster. Greedy is the temp=0 limit (p is a point mass at the
// argmax), which the codebase keeps as a separate fast path; these helpers carry
// the temperature>0 case. They operate on materialised probability vectors so
// the maths is pure and unit-testable off-GPU.

// speculativeAcceptToken reports whether drafted token x is accepted: u (uniform
// in [0,1)) must fall at or below p(x)/q(x). q(x) <= 0 can only happen if the
// drafter's truncated distribution gives x no mass (it cannot then have proposed
// x); treated as a reject for safety.
func speculativeAcceptToken(p, q []float32, x int32, u float32) bool {
	if x < 0 || int(x) >= len(p) || int(x) >= len(q) {
		return false
	}
	qx := q[x]
	if qx <= 0 {
		return false
	}
	// Strict: accept prob = min(1, p/q) for u in [0,1). With <=, a zero-target
	// token (p(x)=0) would be wrongly accepted at u=0; < rejects it.
	return float64(u) < float64(p[x])/float64(qx)
}

// speculativeResidualSample draws a replacement token from the normalised
// residual (p-q)+ using uniform r in [0,1). If the residual carries no mass
// (p <= q everywhere — e.g. identical distributions), it falls back to sampling
// the target distribution p directly.
func speculativeResidualSample(p, q []float32, r float32) int32 {
	n := min(len(p), len(q))
	var sum float64
	for i := 0; i < n; i++ {
		if d := p[i] - q[i]; d > 0 {
			sum += float64(d)
		}
	}
	if sum <= 0 {
		return sampleTokenFromProbs(p, r)
	}
	threshold := float64(r) * sum
	var acc float64
	for i := 0; i < n; i++ {
		if d := p[i] - q[i]; d > 0 {
			acc += float64(d)
			if acc >= threshold {
				return int32(i)
			}
		}
	}
	return int32(n - 1)
}

// sampleTokenFromProbs draws a token from probability vector p with uniform r.
func sampleTokenFromProbs(p []float32, r float32) int32 {
	var sum float64
	for _, v := range p {
		if v > 0 {
			sum += float64(v)
		}
	}
	if sum <= 0 {
		return 0
	}
	threshold := float64(r) * sum
	var acc float64
	for i, v := range p {
		if v > 0 {
			acc += float64(v)
			if acc >= threshold {
				return int32(i)
			}
		}
	}
	return int32(len(p) - 1)
}
