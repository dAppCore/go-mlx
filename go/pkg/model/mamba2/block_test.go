// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

func mkBlockWeights(cfg BlockConfig, D int) *BlockWeights {
	return &BlockWeights{
		InProj:     syn(cfg.projDim()*D, 11),
		ConvWeight: syn(cfg.convDim()*cfg.ConvKernel, 12),
		ConvBias:   syn(cfg.convDim(), 13),
		ALog:       syn(cfg.NumHeads, 14),
		D:          syn(cfg.NumHeads, 15),
		DtBias:     syn(cfg.NumHeads, 16),
		Norm:       syn(cfg.dInner(), 17),
		OutProj:    syn(D*cfg.dInner(), 18),
	}
}

// TestBlockForwardShape checks the block produces [L,D] and advances both state slots.
func TestBlockForwardShape(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	out, nc, ns, err := BlockForwardF32(syn(L*D, 1), w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(nc) != (cfg.ConvKernel-1)*cfg.convDim() || len(ns) != cfg.NumHeads*cfg.HeadDim*cfg.StateDim {
		t.Fatalf("state shapes wrong: conv %d ssm %d", len(nc), len(ns))
	}
	t.Logf("mamba2 block: [%d,%d] in → out, conv-state %d + ssm-state %d advanced", L, D, len(nc), len(ns))
}

// TestBlockForwardCarry is the full-block decode invariant: running the block over a sequence in one
// pass is BIT-EXACT to running it as two chunks that carry BOTH the conv-state ring AND the SSM state
// across the boundary — so streaming Mamba-2 decode (state resident across calls) reproduces the
// one-pass prefill exactly.
func TestBlockForwardCarry(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, split, D = 7, 4, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)

	outFull, _, _, err := BlockForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, nc1, ns1, err := BlockForwardF32(x[:split*D], w, cfg, nil, nil, split, D)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, _, _, err := BlockForwardF32(x[split*D:], w, cfg, nc1, ns1, rem, D)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}
	for i := range o1 {
		if o1[i] != outFull[i] {
			t.Fatalf("chunk1 out[%d] = %v != full %v", i, o1[i], outFull[i])
		}
	}
	for i := range o2 {
		if o2[i] != outFull[split*D+i] {
			t.Fatalf("chunk2 out[%d] = %v != full %v", i, o2[i], outFull[split*D+i])
		}
	}
	t.Logf("mamba2 block decode invariant: split %d|%d, conv-state + SSM-state carry → output bit-exact to one-pass", split, rem)
}
