// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// mtp_attn.go is the multi-query causal attention the MTP batched verify needs — byte-identical to
// metal.ScaledDotProductAttention(q, k, v, scale, causal=true). The MTP verify runs K draft queries
// against the resident cache in one pass; gemma4's headDim is 256, which the fused steel attention
// does NOT support (it ships only bd128/64/80), so metal falls back to the f32-decomposed attention
// (instrumented: f32 QK^T → f32 softmax → f32 probs·V, output rounded to bf16 — bf16 intermediates
// diverge badly, f32 matches). Native composes the SAME with MatMulF32 (QK^T) + SoftmaxF32 (the GPU
// softmax that matches metal's) + MatMulF32 (probs·V) — the audio-attention pattern.

// sdpaCausalAttnInvalid is the masked-logit fill (underflows to 0 probability, like metal's -inf).
const sdpaCausalAttnInvalid = float32(-1e30)

// SDPACausalBF16 is causal scaled-dot-product attention on bf16 q/k/v in head-major [H, L, D] layout
// (within batch 1), returning bf16 [H, qL, D] — byte-identical to metal.ScaledDotProductAttention with
// causal=true. q has H heads, k/v have Hkv heads (GQA: head h reads kv head h/(H/Hkv)); query i (the
// last qL positions) attends keys [0 .. kL-qL+i]. Computed in f32 (widened weights), rounded to bf16.
func SDPACausalBF16(q, k, v []byte, H, Hkv, qL, kL, D int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(q) != H*qL*D*bf16Size || len(k) != Hkv*kL*D*bf16Size || len(v) != Hkv*kL*D*bf16Size {
		return nil, core.NewError("native.SDPACausalBF16: q/k/v sizes must match [H,qL,D]/[Hkv,kL,D] bf16")
	}
	if H%Hkv != 0 {
		return nil, core.NewError("native.SDPACausalBF16: H must be a multiple of Hkv")
	}
	qf, kf, vf := bf16ToF32Slice(q), bf16ToF32Slice(k), bf16ToF32Slice(v)
	gqa := H / Hkv
	out := make([]float32, H*qL*D)
	for h := 0; h < H; h++ {
		hk := h / gqa
		qh := qf[h*qL*D : (h+1)*qL*D]   // [qL, D]
		kh := kf[hk*kL*D : (hk+1)*kL*D] // [kL, D]
		vh := vf[hk*kL*D : (hk+1)*kL*D] // [kL, D]

		// scores = (qh · khᵀ)·scale, causal-masked: [qL, kL].
		scores, err := MatMulF32NT(qh, kh, qL, D, kL)
		if err != nil {
			return nil, err
		}
		for i := 0; i < qL; i++ {
			lim := kL - qL + i
			for j := 0; j < kL; j++ {
				if j <= lim {
					scores[i*kL+j] *= scale
				} else {
					scores[i*kL+j] = sdpaCausalAttnInvalid
				}
			}
		}
		probs, err := SoftmaxF32(scores, kL)
		if err != nil {
			return nil, err
		}
		// out_h = probs · vh : [qL, kL]·[kL, D] = [qL, D].
		oh, err := MatMulF32(probs, vh, qL, kL, D)
		if err != nil {
			return nil, err
		}
		copy(out[h*qL*D:(h+1)*qL*D], oh)
	}
	return f32ToBf16Slice(out), nil
}
