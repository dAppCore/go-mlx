// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Compiled (metal.CompiledFunc) Gemma 4 decode-layer graph builders.

func compiledGemma4DecodeLayer(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, fixedMask *metal.Array) (*metal.Array, sharedKV, bool, error) {
	if !metal.CompiledGemma4LayerEnabled() {
		return nil, sharedKV{}, false, nil
	}
	if !gemma4CompiledDecodeLayerBoundaryAvailable(x, c, B, L, mask, perLayerInput, prev, layer, cfg) {
		return nil, sharedKV{}, false, nil
	}

	offset := 0
	var prevKeys, prevValues *metal.Array
	var pageState metal.PagedKVState
	var fixedState metal.FixedKVState
	ownsKV := !prev.HasState()
	fixedKV := prev.Fixed
	if ownsKV {
		switch cache := c.(type) {
		case *metal.PagedKVCache:
			offset = cache.Offset()
			pageState = cache.PageState()
			if len(pageState.Keys) != 1 || len(pageState.Values) != 1 {
				pageState.Free()
				return nil, sharedKV{}, false, nil
			}
			prevKeys = pageState.Keys[0]
			prevValues = pageState.Values[0]
			defer pageState.Free()
		case *metal.FixedKVCache:
			offset = cache.Offset()
			fixedState = cache.BorrowedFixedState()
			if fixedState.Keys == nil || fixedState.Values == nil {
				return nil, sharedKV{}, false, nil
			}
			prevKeys = fixedState.Keys
			prevValues = fixedState.Values
			fixedKV = true
		default:
			return nil, sharedKV{}, false, nil
		}
	} else {
		offset = prev.Offset
		switch {
		case prev.Keys != nil && prev.Values != nil:
			prevKeys, prevValues = prev.Keys, prev.Values
		case prev.HasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
			prevKeys, prevValues = prev.Pages.Keys[0], prev.Pages.Values[0]
		default:
			return nil, sharedKV{}, false, nil
		}
	}
	if prevKeys == nil || prevValues == nil || !prevKeys.Valid() || !prevValues.Valid() {
		return nil, sharedKV{}, false, nil
	}

	compiled := layer.compiledNativeSharedDecode
	failed := &layer.compiledNativeSharedFailed
	slot := &layer.compiledNativeSharedDecode
	useFixedMask := fixedKV && fixedMask != nil && fixedMask.Valid()
	if fixedKV {
		compiled = layer.compiledNativeFixedSharedDecode
		failed = &layer.compiledNativeFixedSharedFailed
		slot = &layer.compiledNativeFixedSharedDecode
		if useFixedMask {
			compiled = layer.compiledNativeFixedMaskedSharedDecode
			failed = &layer.compiledNativeFixedMaskedSharedFailed
			slot = &layer.compiledNativeFixedMaskedSharedDecode
		}
	}
	if *failed {
		return nil, sharedKV{}, false, nil
	}
	if ownsKV {
		if fixedKV {
			compiled = layer.compiledNativeFixedOwnerDecode
			failed = &layer.compiledNativeFixedOwnerFailed
			slot = &layer.compiledNativeFixedOwnerDecode
			if useFixedMask {
				compiled = layer.compiledNativeFixedMaskedOwnerDecode
				failed = &layer.compiledNativeFixedMaskedOwnerFailed
				slot = &layer.compiledNativeFixedMaskedOwnerDecode
			}
		} else {
			compiled = layer.compiledNativeOwnerDecode
			failed = &layer.compiledNativeOwnerFailed
			slot = &layer.compiledNativeOwnerDecode
		}
		if *failed {
			return nil, sharedKV{}, false, nil
		}
	}
	if compiled == nil || !compiled.Valid() {
		compiled = compileGemma4DecodeLayer(layer, cfg, ownsKV, fixedKV, useFixedMask)
		*slot = compiled
	}

	offsetArray := metal.FromValue(offset)
	defer metal.Free(offsetArray)
	inputs := []*metal.Array{x, prevKeys, prevValues, perLayerInput, offsetArray}
	if useFixedMask {
		inputs = append(inputs, fixedMask)
	}
	outs, callErr := metal.CallCompiledGemma4DecodeLayer(compiled, inputs...)
	if callErr != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		return nil, sharedKV{}, true, callErr
	}
	if err := metal.ValidateGemma4LayerOutputs("mlx.compiledGemma4DecodeLayer", outs, ownsKV); err != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		metal.Free(outs...)
		return nil, sharedKV{}, true, err
	}
	if err := metal.ValidateGemma4LayerOutputShapes("mlx.compiledGemma4DecodeLayer", x, outs[0], metal.OutputAt(outs, 1), metal.OutputAt(outs, 2), prevKeys, prevValues, ownsKV, fixedKV); err != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		metal.Free(outs...)
		return nil, sharedKV{}, true, err
	}
	if ownsKV {
		if fixedKV {
			fixed, _ := c.(*metal.FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(outs[1], outs[2], int(L))
			return outs[0], sharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
		}
		paged, _ := c.(*metal.PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(outs[1], outs[2], int(L))
		return outs[0], sharedKV{Pages: pages, Offset: offset}, true, nil
	}
	return outs[0], prev, true, nil
}

func compileGemma4DecodeLayer(layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV, fixedMask bool) *metal.CompiledFunc {
	return metal.CompileShapeless(func(inputs []*metal.Array) []*metal.Array {
		if len(inputs) < 5 {
			return nil
		}
		var mask *metal.Array
		if fixedMask {
			if len(inputs) < 6 {
				return nil
			}
			mask = inputs[5]
		}
		out, keys, values := gemma4DecodeLayerGraph(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], mask, layer, cfg, ownsKV, fixedKV)
		if ownsKV {
			return []*metal.Array{out, keys, values}
		}
		return []*metal.Array{out}
	}, true)
}

func gemma4DecodeLayerGraph(x, prevKeys, prevValues, perLayerInput, offset, fixedMask *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*metal.Array, *metal.Array, *metal.Array) {
	residual := x
	normed := metal.RMSNorm(x, layer.InputNormScaled, cfg.RMSNormEps)
	attnOut, keys, values := gemma4AttentionGraph(normed, prevKeys, prevValues, offset, fixedMask, layer.Attention, cfg, ownsKV, fixedKV)
	metal.Free(normed)
	attnNormed := metal.RMSNorm(attnOut, layer.PostAttnNormScaled, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(residual, attnNormed)
	metal.Free(attnNormed)

	ffResidual := gemma4DecodeFFNGraph(h, layer, cfg)

	hNext := metal.Add(h, ffResidual)
	metal.Free(h, ffResidual)

	gate := layer.PerLayerInputGate.Forward(hNext)
	multiplied := metal.GeluGateMul(gate, perLayerInput)
	metal.Free(gate)
	projected := layer.PerLayerProjection.Forward(multiplied)
	metal.Free(multiplied)
	projectedNormed := metal.RMSNorm(projected, layer.PostPerLayerInputNormScaled, cfg.RMSNormEps)
	metal.Free(projected)
	gated := metal.Add(hNext, projectedNormed)
	metal.Free(hNext, projectedNormed)
	hNext = gated

	scaled := metal.Mul(hNext, layer.LayerScalar)
	metal.Free(hNext)
	return scaled, keys, values
}

func gemma4DecodeFFNGraph(h *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) *metal.Array {
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil {
		h1In := metal.RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
		h1 := metal.Gemma4MLPGraph(h1In, layer.MLP)
		metal.Free(h1In)
		h1Normed := metal.RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
		metal.Free(h1)

		h2In := metal.RMSNorm(h, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := layer.Router.forward(h)
		h2 := layer.Experts.forward(h2In, topKIndices, topKWeights, "")
		metal.Free(h2In, topKIndices, topKWeights)
		h2Normed := metal.RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
		metal.Free(h2)

		combined := metal.Add(h1Normed, h2Normed)
		metal.Free(h1Normed, h2Normed)
		ffResidual := metal.RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
		metal.Free(combined)
		return ffResidual
	}

	ffIn := metal.RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
	ff := metal.Gemma4MLPGraph(ffIn, layer.MLP)
	metal.Free(ffIn)
	ffResidual := metal.RMSNorm(ff, layer.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(ff)
	return ffResidual
}

func gemma4AttentionGraph(x, prevKeys, prevValues, offset, fixedMask *metal.Array, attn *Gemma4Attention, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*metal.Array, *metal.Array, *metal.Array) {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	qProj := attn.QProj.Forward(x)
	qReshaped := metal.Reshape(qProj, B, L, cfg.NumAttentionHeads, attn.HeadDim)
	metal.Free(qProj)
	q := metal.Transpose(qReshaped, 0, 2, 1, 3)
	metal.Free(qReshaped)
	oldQ := q
	q = metal.RMSNorm(q, attn.QNormScaled, cfg.RMSNormEps)
	metal.Free(oldQ)

	var keys, values *metal.Array
	var out *metal.Array
	qHasRoPE := false
	if ownsKV {
		kProj := attn.KProj.Forward(x)
		kReshaped := metal.Reshape(kProj, B, L, attn.NKVHeads, attn.HeadDim)
		metal.Free(kProj)
		k := metal.Transpose(kReshaped, 0, 2, 1, 3)
		metal.Free(kReshaped)

		var v *metal.Array
		if attn.UseKEqV {
			v = k.Clone()
		} else {
			vProj := attn.VProj.Forward(x)
			vReshaped := metal.Reshape(vProj, B, L, attn.NKVHeads, attn.HeadDim)
			metal.Free(vProj)
			v = metal.Transpose(vReshaped, 0, 2, 1, 3)
			metal.Free(vReshaped)
		}

		oldK := k
		k = metal.RMSNorm(k, attn.KNormScaled, cfg.RMSNormEps)
		metal.Free(oldK)
		k = gemma4ApplyRoPEDynamic(attn, k, offset)

		vNormed := metal.RMSNormNoScale(v, cfg.RMSNormEps)
		metal.Free(v)
		v = vNormed

		if fixedKV {
			q = gemma4ApplyRoPEDynamic(attn, q, offset)
			qHasRoPE = true
			if nativeOut, nativeKeys, nativeValues, ok, err := metal.NativeFixedSingleTokenAttention(q, prevKeys, prevValues, k, v, offset, fixedMask, attn.Scale); ok {
				out = nativeOut
				keys = nativeKeys
				values = nativeValues
			} else {
				if err != nil {
					core.Error("mlx: native fixed single-token attention failed; falling back to Go graph", "error", err)
				}
				keys = metal.SingleTokenCacheUpdate(prevKeys, k, offset)
				values = metal.SingleTokenCacheUpdate(prevValues, v, offset)
			}
			metal.Free(k, v)
		} else {
			keys = metal.Concatenate2(prevKeys, k, 2)
			values = metal.Concatenate2(prevValues, v, 2)
			metal.Free(k, v)
		}
	} else {
		keys = prevKeys
		values = prevValues
	}

	if !qHasRoPE {
		q = gemma4ApplyRoPEDynamic(attn, q, offset)
	}
	if out == nil {
		if fixedKV {
			mask := fixedMask
			if mask == nil || !mask.Valid() {
				mask = metal.SingleTokenCausalMask(int(keys.Dim(2)), offset)
				defer metal.Free(mask)
			}
			out = metal.ScaledDotProductAttentionWithMask(q, keys, values, mask, attn.Scale)
		} else {
			out = metal.ScaledDotProductAttention(q, keys, values, attn.Scale, false)
		}
	}
	metal.Free(q)

	transposed := metal.Transpose(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, cfg.NumAttentionHeads*attn.HeadDim)
	metal.Free(transposed)
	result := attn.OProj.Forward(reshaped)
	metal.Free(reshaped)
	if !ownsKV {
		return result, nil, nil
	}
	return result, keys, values
}

func gemma4ApplyRoPEDynamic(attn *Gemma4Attention, x, offset *metal.Array) *metal.Array {
	old := x
	if attn.RopeFreqs != nil {
		x = metal.RoPEWithOffsetArray(x, int(attn.HeadDim), false, 0, 1.0, offset, attn.RopeFreqs)
	} else {
		x = metal.RoPEWithOffsetArray(x, int(attn.RopeRotatedDim), false, attn.RopeBase, 1.0, offset, nil)
	}
	metal.Free(old)
	return x
}
