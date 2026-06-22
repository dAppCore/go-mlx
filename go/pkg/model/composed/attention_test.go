// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/mlx/pkg/model/qwen3"
)

func mkAttnMixer(cfg AttnConfig, D, seed int) Mixer {
	return NewAttnMixer(&AttnWeights{
		QProj: syn(cfg.Heads*cfg.HeadDim*D, seed+1),
		KProj: syn(cfg.KVHeads*cfg.HeadDim*D, seed+2),
		VProj: syn(cfg.KVHeads*cfg.HeadDim*D, seed+3),
		OProj: syn(D*cfg.Heads*cfg.HeadDim, seed+4),
		QNorm: syn(cfg.HeadDim, seed+5),
		KNorm: syn(cfg.HeadDim, seed+6),
	}, cfg)
}

// TestAttnMixerDecodeEqualsPrefill is the KV-cache correctness: stepping tokens one at a time through the
// attention mixer (growing the cache) produces outputs BIT-EXACT to a single prefill pass — causal
// attention over the cache reproduces full-sequence attention.
func TestAttnMixerDecodeEqualsPrefill(t *testing.T) {
	cfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	const D, L = 8, 6
	m := mkAttnMixer(cfg, D, 0)
	h := syn(L*D, 1)

	full, _, err := m.Forward(h, L, D, nil)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	var st any
	for t0 := 0; t0 < L; t0++ {
		o, next, err := m.Forward(h[t0*D:(t0+1)*D], 1, D, st)
		if err != nil {
			t.Fatalf("decode %d: %v", t0, err)
		}
		st = next
		for i := 0; i < D; i++ {
			if o[i] != full[t0*D+i] {
				t.Fatalf("token %d out[%d] = %v != prefill %v (KV cache diverged)", t0, i, o[i], full[t0*D+i])
			}
		}
	}
	t.Logf("attention mixer decode == prefill bit-exact over %d tokens (KV cache + partial rotary + GQA)", L)
}

// TestHybridDecodeEqualsPrefill is the orchestration's reason to exist: a ComposedModel that INTERLEAVES
// gated-delta and full-attention layers (the Qwen 3.6 schedule shape) decodes token-by-token BIT-EXACT to
// prefill — the session threads each layer's own state type (recurrent for gated-delta, KV for attention)
// through the same loop.
func TestHybridDecodeEqualsPrefill(t *testing.T) {
	const D, vocab, FF = 8, 32, 16
	gdCfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	atCfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	mk := func(li int, mx Mixer) Layer {
		return Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mx,
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	m := &ComposedModel{
		Embed: syn(vocab*D, 100), NormF: syn(D, 101), D: D, Vocab: vocab, Eps: 1e-5,
		Layers: []Layer{
			mk(0, mkGatedDeltaMixer(gdCfg, D, 20)), // linear_attention
			mk(1, mkAttnMixer(atCfg, D, 40)),       // full_attention
			mk(2, mkGatedDeltaMixer(gdCfg, D, 60)), // linear_attention
			mk(3, mkAttnMixer(atCfg, D, 80)),       // full_attention
		},
	}
	tokens := []int32{1, 5, 9, 2, 7, 3}

	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode %d: %v", t0, err)
		}
		for i := 0; i < D; i++ {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (hybrid decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("hybrid (gated-delta + full-attention interleaved) decode == prefill bit-exact over %d tokens", len(tokens))
}
