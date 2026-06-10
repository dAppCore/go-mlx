// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestSampledVerifyDecision(t *testing.T) {
	cfg := GenerateConfig{Temperature: 1}
	mid := func() float32 { return 0.5 }

	// Reject: drafted token 1, but the target overwhelmingly favours 0 → p(1)≈0,
	// so it rejects regardless of the coin; residual (p-q)+ → token 0.
	tgt := FromValues([]float32{10, 0}, 1, 2)
	drf := FromValues([]float32{0, 10}, 1, 2)
	defer Free(tgt, drf)
	acc, repl, all, err := sampledVerifyDecision([]*Array{tgt}, []*Array{drf}, []int32{1}, cfg, mid, nil)
	if err != nil {
		t.Fatalf("reject case err: %v", err)
	}
	if all || len(acc) != 0 || repl != 0 {
		t.Fatalf("reject case: acc=%v repl=%d all=%v, want acc=[] repl=0 all=false", acc, repl, all)
	}

	// Accept: drafted token 0, target and drafter agree → p(0)/q(0)≈1 → accept;
	// single-token block → allAccepted.
	tgt2 := FromValues([]float32{10, 0}, 1, 2)
	drf2 := FromValues([]float32{10, 0}, 1, 2)
	defer Free(tgt2, drf2)
	acc2, _, all2, err := sampledVerifyDecision([]*Array{tgt2}, []*Array{drf2}, []int32{0}, cfg, mid, nil)
	if err != nil {
		t.Fatalf("accept case err: %v", err)
	}
	if !all2 || len(acc2) != 1 || acc2[0] != 0 {
		t.Fatalf("accept case: acc=%v all=%v, want acc=[0] all=true", acc2, all2)
	}

	// Prefix: token 0 accepted (agree), token 1 rejected (target favours 0 at
	// position 2) → accepted=[0], replacement from residual = 0.
	tA := FromValues([]float32{10, 0}, 1, 2)
	tB := FromValues([]float32{10, 0}, 1, 2)
	dA := FromValues([]float32{10, 0}, 1, 2)
	dB := FromValues([]float32{0, 10}, 1, 2)
	defer Free(tA, tB, dA, dB)
	acc3, repl3, all3, err := sampledVerifyDecision([]*Array{tA, tB}, []*Array{dA, dB}, []int32{0, 1}, cfg, mid, nil)
	if err != nil {
		t.Fatalf("prefix case err: %v", err)
	}
	if all3 || len(acc3) != 1 || acc3[0] != 0 || repl3 != 0 {
		t.Fatalf("prefix case: acc=%v repl=%d all=%v, want acc=[0] repl=0 all=false", acc3, repl3, all3)
	}
}

// TestSampledVerifyDecision_NilLeadUnconditional proves a committed lead token
// (nil draft logit) is accepted regardless of the target distribution — the
// carryLead mechanism the sampling path relies on for its speedup.
func TestSampledVerifyDecision_NilLeadUnconditional(t *testing.T) {
	cfg := GenerateConfig{Temperature: 1}
	mid := func() float32 { return 0.5 }

	leadTgt := FromValues([]float32{0, 10}, 1, 2) // target favours token 1...
	tgt1 := FromValues([]float32{10, 0}, 1, 2)
	drf1 := FromValues([]float32{10, 0}, 1, 2)
	defer Free(leadTgt, tgt1, drf1)

	// ...but the lead is token 0 with a nil draft logit → unconditional accept;
	// then the real draft (token 0, agreed) accepts too.
	acc, _, all, err := sampledVerifyDecision(
		[]*Array{leadTgt, tgt1},
		[]*Array{nil, drf1},
		[]int32{0, 0},
		cfg, mid, nil)
	if err != nil {
		t.Fatalf("nil-lead case err: %v", err)
	}
	if !all || len(acc) != 2 || acc[0] != 0 || acc[1] != 0 {
		t.Fatalf("nil-lead: acc=%v all=%v, want acc=[0 0] all=true", acc, all)
	}
}
