// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"

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

func TestSDPACausalBF16ScratchPoolKeepsShapesResident(t *testing.T) {
	small := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	putSDPACausalBF16Scratch(small)
	large := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	putSDPACausalBF16Scratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	defer putSDPACausalBF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("SDPA causal BF16 scratch pool evicted the small shape after using a larger shape")
	}

	gotLarge := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	defer putSDPACausalBF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("SDPA causal BF16 scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestSDPACausalBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	if _, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale); err != nil {
		t.Fatalf("SDPACausalBF16 warmup: %v", err)
	}

	var attnErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, attnErr = SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	})
	if attnErr != nil {
		t.Fatalf("SDPACausalBF16: %v", attnErr)
	}
	if allocs > 390 {
		t.Fatalf("SDPACausalBF16 allocations = %.0f, want <= 390", allocs)
	}
}

func TestSDPACausalBF16IntoReusesOutputBackingAndMatchesSDPACausalBF16(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	want, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16 reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, len(want))
	outPtr := unsafe.Pointer(&out[0])

	got, err := SDPACausalBF16Into(out, q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("SDPACausalBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "SDPACausalBF16Into", got, want)
}

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
