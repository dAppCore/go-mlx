// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestSDPA2PassMatchesReference validates the two-pass long-context SDPA path
// (sdpa_vector_2pass_1 → sdpa_vector_2pass_2) against a host float reference at a
// kvLen well past the single-pass degradation knee (2048). Pass 1 splits the cache
// across `blocks` threadgroups emitting per-block online-softmax partials; pass 2
// merges them. A pass proves the split-and-merge is token-identical to a straight
// softmax — the long-context KV lever ("improving the KV improves toks, more so as
// context grows"). It also cross-checks 2-pass against the proven single-pass SDPA
// at the same inputs: the two MLX kernels must agree.
func TestSDPA2PassMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	gqa := nHeads / nKV
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	qb := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	kb := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	vb := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	// --- host float reference: straight online softmax over bf16-rounded inputs ---
	rb := func(s []byte, i int) float32 { return bf16ToF32(s[i*2], s[i*2+1]) }
	ref := make([]byte, b*nHeads*headDim*2)
	for h := 0; h < nHeads; h++ {
		kvh := h / gqa
		m := float32(-3e38)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += rb(qb, h*headDim+d) * rb(kb, (kvh*kvLen+j)*headDim+d)
			}
			if dot*scale > m {
				m = dot * scale
			}
		}
		var denom float32
		acc := make([]float32, headDim)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += rb(qb, h*headDim+d) * rb(kb, (kvh*kvLen+j)*headDim+d)
			}
			p := float32(math.Exp(float64(dot*scale - m)))
			denom += p
			for d := 0; d < headDim; d++ {
				acc[d] += p * rb(vb, (kvh*kvLen+j)*headDim+d)
			}
		}
		for d := 0; d < headDim; d++ {
			o := f32ToBF16(acc[d] / denom)
			ref[(h*headDim+d)*2], ref[(h*headDim+d)*2+1] = byte(o), byte(o>>8)
		}
	}

	got, err := SDPA2Pass(qb, kb, vb, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass: %v", err)
	}
	if cos := cosineBF16(got, ref); cos < 0.999 {
		t.Fatalf("2-pass SDPA cosine=%.6f vs host reference — block split/merge broken", cos)
	} else {
		t.Logf("2-pass SDPA (kvLen=%d, blocks=%d): cosine=%.6f vs host reference — the cache reduction fans over %d threadgroups, token-identical", kvLen, sdpa2PassBlocks(kvLen), cos, sdpa2PassBlocks(kvLen))
	}

	// cross-check against the proven single-pass kernel at the same inputs.
	sp, err := SDPA(qb, kb, vb, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA (single-pass cross-check): %v", err)
	}
	if cos := cosineBF16(got, sp); cos < 0.999 {
		t.Fatalf("2-pass vs single-pass SDPA cosine=%.6f — the two MLX kernels disagree", cos)
	} else {
		t.Logf("2-pass vs single-pass SDPA: cosine=%.6f — agree", cos)
	}
}
