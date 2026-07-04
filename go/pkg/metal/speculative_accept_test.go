// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestSpeculativeAcceptToken(t *testing.T) {
	p := []float32{0.5, 0.3, 0.2}
	q := []float32{0.2, 0.3, 0.5}

	// p(0)/q(0) = 2.5 >= 1 → accept for any u.
	if !speculativeAcceptToken(p, q, 0, 0.999) {
		t.Fatal("x=0: p/q=2.5 must always accept")
	}
	// p(2)/q(2) = 0.4 → accept iff u <= 0.4.
	if !speculativeAcceptToken(p, q, 2, 0.30) {
		t.Fatal("x=2: p/q=0.4, u=0.30 must accept")
	}
	if speculativeAcceptToken(p, q, 2, 0.50) {
		t.Fatal("x=2: p/q=0.4, u=0.50 must reject")
	}
	// q(x)=0 → cannot accept.
	if speculativeAcceptToken([]float32{0.5, 0.5}, []float32{1, 0}, 1, 0.0) {
		t.Fatal("q(x)=0 must reject")
	}
}

// TestSpeculativeAccept_GreedyLimit proves greedy is the temp=0 special case:
// when p is a point mass at the argmax, accept reduces to "x == argmax".
func TestSpeculativeAccept_GreedyLimit(t *testing.T) {
	p := []float32{0, 1, 0} // argmax = index 1
	q := []float32{0.3, 0.4, 0.3}
	if !speculativeAcceptToken(p, q, 1, 0.999) {
		t.Fatal("greedy limit: x==argmax must always accept")
	}
	if speculativeAcceptToken(p, q, 0, 0.0) {
		t.Fatal("greedy limit: x!=argmax (p=0) must always reject")
	}
}

func TestSpeculativeResidualSample(t *testing.T) {
	// (p-q)+ has mass only at index 0 (0.6-0.3=0.3); 1 and 2 are <= q.
	p := []float32{0.6, 0.2, 0.2}
	q := []float32{0.3, 0.4, 0.3}
	for _, r := range []float32{0.0, 0.5, 0.999} {
		if got := speculativeResidualSample(p, q, r); got != 0 {
			t.Fatalf("residual mass only at index 0, got %d (r=%f)", got, r)
		}
	}
	// Identical distributions → empty residual → fall back to a valid p draw.
	if got := speculativeResidualSample(p, p, 0.5); got < 0 || int(got) >= len(p) {
		t.Fatalf("empty-residual fallback returned invalid index %d", got)
	}
}
