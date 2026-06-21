// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// mtp_attn_test.go validates SDPACausalBF16 against the REAL metal.ScaledDotProductAttention (the op
// metal's MTP verify calls). VERIFIED BYTE-IDENTICAL for the self-attention shape qL==kL (any GQA) —
// the path the K-row verify uses when the draft block has no carried cache prefix. For the cross-
// attention shape qL<kL (a draft block over a longer resident cache) there is a single token-robust
// 1-ULP divergence on the full-attention query (the probs·V / softmax accumulation edge metal takes
// for kL>qL); it cannot flip the verify argmax, so the MTP token stream is unaffected. Closing that
// 1-ULP to full hidden-state byte-identity is tracked as the audio-relK-style dispatch match.

func sdpaScale(D int) float32 { return float32(1.0 / math.Sqrt(float64(D))) }

// TestSDPACausalSelfAttention asserts SDPACausalBF16 == metal.ScaledDotProductAttention(causal) BYTE-
// IDENTICAL for qL==kL, across GQA factors and a single-query (decode-equivalent) case.
func TestSDPACausalSelfAttention(t *testing.T) {
	requireNativeRuntime(t)
	const D = 256
	cases := []struct{ H, Hkv, qL, kL int }{
		{4, 4, 5, 5}, // no GQA, qL==kL
		{8, 4, 6, 6}, // GQA 2:1, qL==kL
		{2, 1, 4, 4}, // GQA 2:1, qL==kL
		{2, 1, 1, 8}, // single query (kL>1) — decode-equivalent, also exact
	}
	for _, c := range cases {
		scale := sdpaScale(D)
		q := toBF16Bytes(syntheticFloat32(c.H*c.qL*D, 3))
		k := toBF16Bytes(syntheticFloat32(c.Hkv*c.kL*D, 5))
		v := toBF16Bytes(syntheticFloat32(c.Hkv*c.kL*D, 7))
		got, err := SDPACausalBF16(q, k, v, c.H, c.Hkv, c.qL, c.kL, D, scale)
		if err != nil {
			t.Fatalf("SDPACausalBF16(H%d Hkv%d qL%d kL%d): %v", c.H, c.Hkv, c.qL, c.kL, err)
		}
		r := mc.ScaledDotProductAttention(marr(q, 1, c.H, c.qL, D), marr(k, 1, c.Hkv, c.kL, D), marr(v, 1, c.Hkv, c.kL, D), scale, true)
		rb := mc.AsType(r, mc.DTypeBFloat16)
		mc.Materialize(rb)
		eqBytes(t, "SDPACausalBF16 vs metal SDPA (qL==kL)", got, append([]byte(nil), rb.RawBytes()...))
	}
}
