// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func (a *Gemma4Attention) applyRoPE(x *metal.Array, offset int) *metal.Array {
	if a.RopeFreqs != nil {
		return metal.RoPEWithFreqs(x, int(a.HeadDim), false, 0, 1.0, offset, a.RopeFreqs)
	}
	return metal.RoPE(x, int(a.RopeRotatedDim), false, a.RopeBase, 1.0, offset)
}

func attentionQueryForKV(query, key *metal.Array) (*metal.Array, *metal.Array) {
	if query == nil || key == nil || !query.Valid() || !key.Valid() {
		return query, nil
	}
	dtype := key.Dtype()
	if query.Dtype() == dtype {
		return query, nil
	}
	switch dtype {
	case metal.DTypeFloat16, metal.DTypeBFloat16:
		cast := metal.AsType(query, dtype)
		return cast, cast
	default:
		return query, nil
	}
}

func (a *Gemma4Attention) forward(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, prev sharedKV, cfg *Gemma4TextConfig, window int32, fixedMask *metal.Array, runtimeMasks *gemma4RuntimeMaskCache, materializePagedKVForReuse bool) (*metal.Array, sharedKV) {
	if metal.NativeGemma4FixedOwnerAttentionEnabled() && window == 0 && !prev.HasState() && L == 1 && mask == nil {
		if fixed, ok := c.(*metal.FixedKVCache); ok {
			if out, kv, ok, err := nativeGemma4FixedOwnerAttentionBlock(x, fixed, fixedMask, a, cfg); ok {
				return out, kv
			} else if err != nil {
				core.Error("mlx: native Gemma 4 fixed owner attention failed; falling back to Go graph", "error", err)
			}
		}
	}

	qProj := a.QProj.Forward(x)
	q := metal.AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, a.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * a.HeadDim), int64(a.HeadDim), int64(cfg.NumAttentionHeads * a.HeadDim), 1}, 0)
	metal.Free(qProj)
	oldQ := q
	q = metal.RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	metal.Free(oldQ)

	kv := prev
	offset := 0
	var out *metal.Array
	qRoPEApplied := false
	if !kv.HasState() {
		kProj := a.KProj.Forward(x)
		k := metal.AsStrided(kProj, []int32{B, a.NKVHeads, L, a.HeadDim},
			[]int64{int64(L * a.NKVHeads * a.HeadDim), int64(a.HeadDim), int64(a.NKVHeads * a.HeadDim), 1}, 0)
		metal.Free(kProj)

		var v *metal.Array
		if a.UseKEqV {
			// Gemma 4 K=V shares the projection source, not the final cache
			// tensors: K still takes KNorm+RoPE, while V takes value RMSNorm.
			v = k.Clone()
		} else {
			vProj := a.VProj.Forward(x)
			v = metal.AsStrided(vProj, []int32{B, a.NKVHeads, L, a.HeadDim},
				[]int64{int64(L * a.NKVHeads * a.HeadDim), int64(a.HeadDim), int64(a.NKVHeads * a.HeadDim), 1}, 0)
			metal.Free(vProj)
		}

		if c != nil {
			offset = c.Offset()
		}

		oldK := k
		k = metal.RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)
		metal.Free(oldK)
		kRoPE := a.applyRoPE(k, offset)
		metal.Free(k)
		k = kRoPE

		vNormed := metal.RMSNormNoScale(v, cfg.RMSNormEps)
		metal.Free(v)
		v = vNormed

		if c != nil {
			oldK, oldV := k, v
			if fixed, ok := c.(*metal.FixedKVCache); ok && L == 1 && mask == nil && fixed.maxSize > 0 {
				// Stack-allocated shape scratch — per-token per-layer hot path.
				// K/V are always rank-4 ([B,H,L,D]); avoids 2 × []int32 heap
				// allocs per layer per token (× NumHiddenLayers).
				var kShapeBuf, vShapeBuf [metal.MaxTensorRank]int32
				kShape := k.ShapeInto(kShapeBuf[:0])
				vShape := v.ShapeInto(vShapeBuf[:0])
				fixed.ensureShape(kShape[0], kShape[1], kShape[3], vShape[3], k.Dtype(), v.Dtype())
				state := fixed.BorrowedFixedState()
				if state.Keys != nil && state.Values != nil {
					qRoPE := a.applyRoPE(q, offset)
					metal.Free(q)
					q = qRoPE
					qRoPEApplied = true

					var nativeOut, nativeKeys, nativeValues *metal.Array
					var ok bool
					var err error
					var offsetArray *metal.Array
					if fixed.Offset()+int(L) <= fixed.maxSize {
						offsetArray = metal.FromValue(offset)
						nativeOut, nativeKeys, nativeValues, ok, err = metal.NativeFixedSingleTokenAttention(q, state.Keys, state.Values, k, v, offsetArray, nil, a.Scale)
					} else if metal.NativeFixedSlidingAttentionEnabled() && fixed.length >= fixed.maxSize {
						shiftIndices, lastIndex := fixed.slidingUpdateInputs()
						nativeOut, nativeKeys, nativeValues, ok, err = metal.NativeFixedSlidingSingleTokenAttention(q, state.Keys, state.Values, k, v, shiftIndices, lastIndex, a.Scale)
					}
					if err != nil {
						core.Error("mlx: native fixed owner attention failed; falling back to Go graph", "error", err)
						metal.Free(nativeOut, nativeKeys, nativeValues)
						nativeOut, nativeKeys, nativeValues = nil, nil, nil
						ok = false
					}
					if ok {
						if err := metal.ValidateGemma4LayerOutputShapes("mlx.nativeFixedSingleTokenAttention", q, nativeOut, nativeKeys, nativeValues, state.Keys, state.Values, true, true); err == nil {
							fixedState := fixed.ReplaceFixedFromNativeBorrowed(nativeKeys, nativeValues, int(L))
							if gemma4ValidKV(fixedState.Keys, fixedState.Values) {
								kv = sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}
								out = nativeOut
								fixed.RetireAfterNextEval(oldK, oldV, q, offsetArray)
								q = nil
								offsetArray = nil
							} else {
								core.Error("mlx: native fixed attention updated cache without valid K/V state; falling back to Go graph")
								metal.Free(nativeOut)
							}
						} else {
							core.Error("mlx: native fixed owner attention returned invalid K/V state; falling back to Go graph", "error", err)
							metal.Free(nativeOut, nativeKeys, nativeValues)
						}
					}
					metal.Free(offsetArray)
				}
			}
			if out == nil {
				if paged, ok := c.(*metal.PagedKVCache); ok && L == 1 && mask == nil {
					pages := paged.UpdateBorrowedPages(k, v, int(L))
					pagedKV := sharedKV{Pages: pages, Offset: offset}
					if pagedKV.HasPages() {
						metal.Free(oldK, oldV)
						kv = pagedKV
					} else {
						pages.Free()
						kv = sharedKV{Keys: oldK, Values: oldV, Offset: offset}
					}
				} else {
					k, v = c.Update(k, v, int(L))
					if gemma4ValidKV(k, v) {
						metal.Free(oldK, oldV)
						kv = sharedKV{Keys: k, Values: v, Offset: offset}
					} else {
						metal.Free(k, v)
						kv = sharedKV{Keys: oldK, Values: oldV, Offset: offset}
					}
				}
			}
		} else {
			kv = sharedKV{Keys: k, Values: v, Offset: offset}
		}
	} else {
		offset = kv.Offset
	}

	if out == nil {
		repeatFactor := cfg.NumAttentionHeads / a.NKVHeads
		if kv.HasPages() && L == 1 && mask == nil {
			qRoPE := a.applyRoPE(q, offset)
			metal.Free(q)
			q = qRoPE
			qRoPEApplied = true
			attentionQ := q
			var ownedAttentionQ *metal.Array
			if len(kv.Pages.Keys) > 0 {
				attentionQ, ownedAttentionQ = attentionQueryForKV(q, kv.Pages.Keys[0])
			} else if kv.Keys != nil {
				attentionQ, ownedAttentionQ = attentionQueryForKV(q, kv.Keys)
			}
			if gemma4ValidKV(kv.Keys, kv.Values) {
				out = metal.ScaledDotProductAttention(attentionQ, kv.Keys, kv.Values, a.Scale, false)
			}
			if out == nil && metal.NativePagedAttentionEnabled() && !materializePagedKVForReuse && len(kv.Pages.Keys) > 1 {
				var ok bool
				var err error
				out, ok, err = metal.NativePagedSingleTokenAttention(attentionQ, kv.Pages.Keys, kv.Pages.Values, a.Scale)
				if !ok || err != nil {
					if err != nil {
						core.Error("mlx: native paged attention failed; falling back to Go graph", "error", err)
					}
					out = nil
				}
			}
			if out == nil && metal.PagedDecodeFastConcatEnabled() && len(kv.Pages.Keys) > 1 {
				traceStart := time.Time{}
				if metal.NativePhaseTraceArmed() {
					traceStart = time.Now()
				}
				kBase, vBase := metal.ConcatenatePagedState(kv.Pages.Keys, kv.Pages.Values)
				tracePagedKVConcat("paged_kv.fast_concat."+gemma4AttentionWindowTraceName(window), traceStart, kv.Pages)
				concatQ := attentionQ
				var ownedConcatQ *metal.Array
				if ownedAttentionQ == nil {
					concatQ, ownedConcatQ = attentionQueryForKV(q, kBase)
				}
				out = metal.ScaledDotProductAttention(concatQ, kBase, vBase, a.Scale, false)
				metal.Free(ownedConcatQ)
				if window == 0 {
					kv.Keys = kBase
					kv.Values = vBase
				} else {
					metal.Free(kBase, vBase)
				}
			}
			if out == nil {
				kPages, vPages := kv.Pages.Keys, kv.Pages.Values
				var repeatedPages []*metal.Array
				if len(kPages) > 1 && metal.PagedStateNeedsMaterializedRepeat(kv.Pages, repeatFactor) {
					kPages, vPages, repeatedPages = metal.RepeatPagedState(kv.Pages, repeatFactor)
				}
				out = metal.ScaledDotProductAttentionPaged(attentionQ, kPages, vPages, a.Scale)
				metal.Free(repeatedPages...)
			}
			metal.Free(ownedAttentionQ)
		} else {
			kBase, vBase := kv.Keys, kv.Values
			var ownedContiguous []*metal.Array
			if (kBase == nil || vBase == nil) && kv.HasPages() {
				traceStart := time.Time{}
				if metal.NativePhaseTraceArmed() {
					traceStart = time.Now()
				}
				kBase, vBase = metal.ConcatenatePagedState(kv.Pages.Keys, kv.Pages.Values)
				tracePagedKVConcat("paged_kv.contiguous."+gemma4AttentionWindowTraceName(window), traceStart, kv.Pages)
				ownedContiguous = append(ownedContiguous, kBase, vBase)
			}
			if !gemma4ValidKV(kBase, vBase) {
				metal.Free(q)
				metal.Free(ownedContiguous...)
				panic("mlx: Gemma 4 attention missing valid K/V state")
			}
			if mask == nil && offset > 0 && L > 1 && window > 0 {
				localContextLen := gemma4SlidingCausalContextLen(L, int32(kBase.Dim(2)), window)
				tailK, tailV := metal.CacheTail(kBase, vBase, localContextLen)
				if tailK != kBase {
					ownedContiguous = append(ownedContiguous, tailK)
					kBase = tailK
				}
				if tailV != vBase {
					ownedContiguous = append(ownedContiguous, tailV)
					vBase = tailV
				}
			}
			var cachedMask *metal.Array
			cachedMaskOwned := false
			useCausalAttention := false
			if mask == nil && offset > 0 && L > 1 {
				keyLen := int32(kBase.Dim(2))
				if gemma4CanUseOffsetCausalAttention(L, keyLen, window) {
					useCausalAttention = true
				} else {
					keyStart := max(int32(offset)+L-keyLen, 0)
					if runtimeMasks != nil {
						cachedMask = runtimeMasks.CachedAttentionMask(B, L, keyLen, int32(offset), keyStart, window)
					} else {
						cachedMask = buildGemma4CachedAttentionMask(B, L, keyLen, int32(offset), keyStart, window)
						cachedMaskOwned = true
					}
					mask = cachedMask
				}
			} else if kv.Fixed && L == 1 && mask == nil {
				offsetArray := metal.FromValue(offset)
				cachedMask = metal.SingleTokenCausalMask(int(kBase.Dim(2)), offsetArray)
				metal.Free(offsetArray)
				cachedMaskOwned = true
				mask = cachedMask
			}
			if !qRoPEApplied {
				qRoPE := a.applyRoPE(q, offset)
				metal.Free(q)
				q = qRoPE
				qRoPEApplied = true
			}
			attentionQ, ownedAttentionQ := attentionQueryForKV(q, kBase)
			if mask != nil {
				out = metal.ScaledDotProductAttentionWithMask(attentionQ, kBase, vBase, mask, a.Scale)
			} else if useCausalAttention {
				out = metal.ScaledDotProductAttention(attentionQ, kBase, vBase, a.Scale, true)
			} else {
				out = metal.ScaledDotProductAttention(attentionQ, kBase, vBase, a.Scale, L > 1)
			}
			metal.Free(ownedAttentionQ)
			if cachedMaskOwned {
				metal.Free(cachedMask)
			}
			metal.Free(ownedContiguous...)
		}
	}
	if !qRoPEApplied {
		qRoPE := a.applyRoPE(q, offset)
		metal.Free(q)
		q = qRoPE
		qRoPEApplied = true
	}
	metal.Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := metal.Transpose4(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, cfg.NumAttentionHeads*a.HeadDim)
	metal.Free(transposed)
	result := a.forwardOProjection(reshaped)
	metal.Free(reshaped)
	return result, kv
}

func (a *Gemma4Attention) forwardOProjection(x *metal.Array) *metal.Array {
	if metal.NativeGemma4AttentionOMatVecEnabled() {
		out, ok, err := metal.QuantizedDenseMatVec(x, a.OProj)
		if err != nil {
			core.Error("mlx: native Gemma 4 attention output matvec failed; falling back to Go graph", "error", err)
			metal.Free(out)
		} else if ok {
			return out
		}
	}
	return a.OProj.Forward(x)
}
