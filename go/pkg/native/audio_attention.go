// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// audio_attention.go ports the gemma4 Conformer chunked relative-position attention to the no-cgo
// path, BYTE-IDENTICAL to metal's Gemma4AudioAttention.Forward. The attention runs in float32 (metal
// .float()s q/k/v), so its matmuls go through MatMulF32 (the fused steel GEMM, byte-identical to
// metal.Matmul-f32) and its softmax through SoftmaxF32; the per-dim q-scale and tanh soft-cap use the
// byte-parity f32 Mul/Tanh; the blocked-context windowing, the Transformer-XL relShift, the validity
// mask and the masked select are host byte-copies/selects (no arithmetic, so byte-identical). The
// projections are bf16 (MatRowsBF16) widened to f32 (an exact AsType), and the result is rounded back
// to bf16 (f32ToBF16) before the bf16 output projection — exactly metal's dtype dance.

// AudioAttentionWeights holds the attention's weights: q/k/v/post projections (bf16, [H·D,hidden] /
// [hidden,H·D] for post), the relative-key projection (bf16, [H·D,hidden]), the per-dim q-scale
// (f32, [H·D] = q_scale·softplus(per_dim_scale), precomputed) and the sinusoid position table (f32,
// [P,hidden]). Projection clips (gradient clipping) are applied via the layer's ClipMin/ClipMax.
type AudioAttentionWeights struct {
	QProj, KProj, VProj, Post []byte
	RelativeKProj             []byte
	QScalePerDim              []float32 // [headDim] — broadcast over heads (metal's [1,1,1,headDim])
	PosEmbed                  []float32 // [P·hidden]
	PosCount                  int       // P
}

// audioContextSizeOf is chunk + past + future.
func audioContextSizeOf(cfg AudioConfig) int {
	return cfg.ChunkSize + cfg.PastHorizon + cfg.FutureHorizon
}

// audioBlockContextF32 pads the time axis of x [T, H, D] (fp32) by [past, future+chunk-1] (zeros) and
// unfolds overlapping windows strided by chunk → [nB, ctx, H, D] (fp32). Port of extractBlockContext.
func audioBlockContextF32(x []float32, T, H, D, nB, chunk, past, future int) []float32 {
	ctx := chunk + past + future
	out := make([]float32, nB*ctx*H*D)
	for b := 0; b < nB; b++ {
		for c := 0; c < ctx; c++ {
			// padded index = b*chunk + c; original time = padded - past.
			it := b*chunk + c - past
			if it < 0 || it >= T {
				continue // zero pad
			}
			copy(out[((b*ctx+c)*H)*D:((b*ctx+c)*H+H)*D], x[(it*H)*D:(it*H+H)*D])
		}
	}
	return out
}

// audioRelShiftF32 is the Transformer-XL relative shift: [H, nB, chunk, P] → [H, nB, chunk, ctx] by
// padding the position axis to ctx+1, folding chunk·(ctx+1), truncating to chunk·ctx, refolding. Port
// of relShift (B=1). Pure index remap (byte-copy / zero-pad), so byte-identical.
func audioRelShiftF32(x []float32, H, nB, chunk, P, ctx int) []float32 {
	padP := ctx + 1
	out := make([]float32, H*nB*chunk*ctx)
	for h := 0; h < H; h++ {
		for b := 0; b < nB; b++ {
			// folded[i*padP + p] = x[h,b,i,p] (p<P), else 0; then out[i,c] = folded[i*ctx + c].
			base := ((h*nB + b) * chunk)
			for i := 0; i < chunk; i++ {
				for c := 0; c < ctx; c++ {
					fi := i*ctx + c // index into the folded chunk·(ctx+1) stream
					row, col := fi/padP, fi%padP
					var v float32
					if col < P {
						v = x[((base+row)*P)+col]
					}
					out[((base+i)*ctx)+c] = v
				}
			}
		}
	}
	return out
}

// audioBlockedMask builds the [nB, chunk, ctx] validity mask: query q=blk·chunk+i may attend key
// kv=blk·chunk-past+j iff both in-sequence and kv∈[q-past, q+future]. Port of blockedMask.
func audioBlockedMask(seqLen, nB, chunk, ctx, past, future int) []bool {
	m := make([]bool, nB*chunk*ctx)
	for b := 0; b < nB; b++ {
		for i := 0; i < chunk; i++ {
			q := b*chunk + i
			for j := 0; j < ctx; j++ {
				kv := b*chunk - past + j
				if q < seqLen && kv >= 0 && kv < seqLen && kv >= q-past && kv <= q+future {
					m[(b*chunk+i)*ctx+j] = true
				}
			}
		}
	}
	return m
}

// AudioAttention runs the Conformer chunked relative-position attention on [T, hidden] bf16, returning
// [T, hidden] bf16 — byte-identical to metal's Gemma4AudioAttention.Forward (B=1).
func AudioAttention(x []byte, w *AudioAttentionWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	H, D := cfg.NumHeads, cfg.HeadDim
	hd := H * D
	T := len(x) / (cfg.Hidden * bf16Size)
	chunk := cfg.ChunkSize
	nB := (T + chunk - 1) / chunk
	ctx := audioContextSizeOf(cfg)
	past, future := cfg.PastHorizon, cfg.FutureHorizon

	// projections (bf16) widened to f32, reshaped [T, H, D].
	proj := func(weight []byte) ([]float32, error) {
		p, err := MatRowsBF16(weight, x, T, hd, cfg.Hidden)
		if err != nil {
			return nil, err
		}
		return bf16ToF32Slice(p), nil
	}
	qf, err := proj(w.QProj)
	if err != nil {
		return nil, err
	}
	kf, err := proj(w.KProj)
	if err != nil {
		return nil, err
	}
	vf, err := proj(w.VProj)
	if err != nil {
		return nil, err
	}

	// q *= QScalePerDim[d] (per-dim, broadcast over T and heads); k *= KScale.
	for i := 0; i < T*H; i++ {
		for d := 0; d < D; d++ {
			qf[i*D+d] *= w.QScalePerDim[d]
		}
	}
	for i := range kf {
		kf[i] *= cfg.KScale
	}

	// context windows for k,v: [nB, ctx, H, D].
	kc := audioBlockContextF32(kf, T, H, D, nB, chunk, past, future)
	vc := audioBlockContextF32(vf, T, H, D, nB, chunk, past, future)

	// relK = RelativeKProj.Forward(PosEmbed) = Matmul(PosEmbed, Transpose(weight)) → [P, H·D] (f32).
	// metal keeps PosEmbed in f32 and PROMOTES the bf16 weight to f32 (a clean widen — verified), then
	// runs the NT steel kernel (transpose_b). Native must use the SAME nt kernel: the nn kernel on a
	// materialised transpose picks a different accumulation order and diverges ~1 ULP at this shape.
	relK, err := MatMulF32NT(w.PosEmbed, bf16ToF32Slice(w.RelativeKProj), w.PosCount, cfg.Hidden, hd) // [P, H·D]
	if err != nil {
		return nil, err
	}

	// per query head h: matrix_ac[i,j] = Σ_d q[blk,i,h,d]·k_ctx[blk,j,h,d]; bd[i,p] = Σ_d q·relK[p,h,d];
	// logits = ac + relShift(bd); soft-cap; mask; softmax over ctx; out = Σ_j w[i,j]·v_ctx[blk,j,h,d].
	mask := audioBlockedMask(T, nB, chunk, ctx, past, future)
	outHeadMajor := make([]float32, H*nB*chunk*D)
	for h := 0; h < H; h++ {
		// gather this head's blocked q [nB·chunk, D], context k/v [nB,ctx,D].
		qh := make([]float32, nB*chunk*D)
		for b := 0; b < nB; b++ {
			for i := 0; i < chunk; i++ {
				t := b*chunk + i
				if t < T {
					copy(qh[(b*chunk+i)*D:(b*chunk+i)*D+D], qf[(t*H+h)*D:(t*H+h)*D+D])
				}
			}
		}
		// bd over all positions then per-block relShift: bd[nB·chunk, P] = qh @ relK_hᵀ.
		relKh := make([]float32, w.PosCount*D)
		for p := 0; p < w.PosCount; p++ {
			copy(relKh[p*D:p*D+D], relK[(p*H+h)*D:(p*H+h)*D+D])
		}
		bd, err := MatMulF32(qh, transposeF32(relKh, w.PosCount, D), nB*chunk, D, w.PosCount) // [nB·chunk, P]
		if err != nil {
			return nil, err
		}
		bdShift := audioRelShiftF32(bd, 1, nB, chunk, w.PosCount, ctx) // treat as [1,nB,chunk,P]→[1,nB,chunk,ctx]

		for b := 0; b < nB; b++ {
			kh := make([]float32, ctx*D)
			vh := make([]float32, ctx*D)
			for c := 0; c < ctx; c++ {
				copy(kh[c*D:c*D+D], kc[((b*ctx+c)*H+h)*D:((b*ctx+c)*H+h)*D+D])
				copy(vh[c*D:c*D+D], vc[((b*ctx+c)*H+h)*D:((b*ctx+c)*H+h)*D+D])
			}
			ac, err := MatMulF32(qh[b*chunk*D:(b+1)*chunk*D], transposeF32(kh, ctx, D), chunk, D, ctx) // [chunk, ctx]
			if err != nil {
				return nil, err
			}
			// soft-cap = LogitCap·tanh(logits/LogitCap), tanh via the GPU kernel (host math.Tanh is NOT
			// byte-identical to v_Tanhfloat32). MulScalar/Add are single f32 ops → byte-identical host-side.
			invCap := float32(1) / cfg.LogitCap
			scaled := make([]float32, chunk*ctx)
			for i := 0; i < chunk; i++ {
				for j := 0; j < ctx; j++ {
					scaled[i*ctx+j] = (ac[i*ctx+j] + bdShift[(b*chunk+i)*ctx+j]) * invCap
				}
			}
			capped, err := RunUnary("v_Tanhfloat32float32", scaled)
			if err != nil {
				return nil, err
			}
			masked := make([]float32, chunk*ctx)
			for i := 0; i < chunk; i++ {
				for j := 0; j < ctx; j++ {
					s := capped[i*ctx+j] * cfg.LogitCap
					if !mask[(b*chunk+i)*ctx+j] {
						s = cfg.InvalidLogit
					}
					masked[i*ctx+j] = s
				}
			}
			probs, err := SoftmaxF32(masked, ctx)
			if err != nil {
				return nil, err
			}
			o, err := MatMulF32(probs, vh, chunk, ctx, D) // [chunk, D]
			if err != nil {
				return nil, err
			}
			copy(outHeadMajor[((h*nB+b)*chunk)*D:((h*nB+b)*chunk+chunk)*D], o)
		}
	}

	// merge [H, nB, chunk, D] → [nB·chunk, H·D], trim to T, round to bf16, Post projection.
	merged := make([]float32, nB*chunk*hd)
	for h := 0; h < H; h++ {
		for b := 0; b < nB; b++ {
			for i := 0; i < chunk; i++ {
				copy(merged[((b*chunk+i)*hd)+h*D:((b*chunk+i)*hd)+h*D+D], outHeadMajor[((h*nB+b)*chunk+i)*D:((h*nB+b)*chunk+i)*D+D])
			}
		}
	}
	if len(merged) < T*hd {
		return nil, core.NewError("native.AudioAttention: internal merge size")
	}
	return MatRowsBF16(w.Post, f32ToBf16Slice(merged[:T*hd]), T, cfg.Hidden, hd)
}
