// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func (l *Gemma4DecoderLayer) forward(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, cfg *Gemma4TextConfig, fixedMask *metal.Array, runtimeMasks *gemma4RuntimeMaskCache, materializePagedKVForReuse bool) (*metal.Array, sharedKV) {
	defer func() {
		if recovered := recover(); recovered != nil {
			panic(core.Sprintf("Gemma 4 layer %d %s: %v", l.LayerIdx, l.LayerType, recovered))
		}
	}()
	traceEnabled := metal.NativePhaseMaterializeTraceEnabled() && metal.NativePhaseTraceArmed()
	if out, kv, ok, err := compiledGemma4DecodeLayer(x, c, B, L, mask, perLayerInput, prev, l, cfg, fixedMask); ok {
		if err == nil {
			l.traceNativeMaterialize(traceEnabled, "compiled_layer", out)
			return out, kv
		}
		core.Error("mlx: compiled Gemma 4 decode layer failed; falling back to Go graph", "layer", l.LayerIdx, "type", l.LayerType, "error", err)
	}
	if out, kv, ok, err := nativeGemma4DecodeLayer(x, c, B, L, mask, perLayerInput, prev, l, cfg, fixedMask); ok {
		if err == nil {
			l.traceNativeMaterialize(traceEnabled, "native_layer", out)
			return out, kv
		}
		core.Error("mlx: native Gemma 4 decode layer failed; falling back to Go graph", "layer", l.LayerIdx, "type", l.LayerType, "error", err)
	}

	residual := x

	normed := metal.RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	window := int32(0)
	if l.IsSliding {
		window = cfg.SlidingWindow
	}
	var h *metal.Array
	var kv sharedKV
	if metal.NativeGemma4FixedOwnerAttentionResidualEnabled() && !l.IsSliding && !prev.HasState() && L == 1 && mask == nil {
		if fixed, ok := c.(*metal.FixedKVCache); ok {
			if nativeH, nativeKV, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(residual, normed, fixed, fixedMask, l.Attention, l.PostAttnNormScaled, cfg); ok {
				h = nativeH
				kv = nativeKV
				l.traceNativeMaterialize(traceEnabled, "attention_residual", h)
			} else if err != nil {
				core.Error("mlx: native Gemma 4 fixed owner attention residual failed; falling back to Go graph", "error", err)
			}
		}
	}
	if h == nil {
		attnOut, nativeKV := l.Attention.forward(normed, c, B, L, mask, prev, cfg, window, fixedMask, runtimeMasks, materializePagedKVForReuse)
		kv = nativeKV
		l.traceNativeMaterialize(traceEnabled, "attention", attnOut)
		if metal.NativeGemma4ResidualNormEnabled() {
			if nativeH, ok, err := metal.NativeResidualNormAdd(residual, attnOut, l.PostAttnNormScaled, cfg.RMSNormEps); ok {
				h = nativeH
			} else if err != nil {
				core.Error("mlx: native Gemma 4 attention residual failed; falling back to Go graph", "error", err)
			}
		}
		if h == nil {
			attnNormed := metal.RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
			h = metal.Add(residual, attnNormed)
			metal.Free(attnNormed)
		}
		metal.Free(attnOut)
		l.traceNativeMaterialize(traceEnabled, "attention_residual", h)
	}
	metal.Free(normed)

	residual = h
	var ffResidual *metal.Array
	var hNext *metal.Array
	if l.EnableMoE && l.Router != nil && l.Experts != nil {
		h1In := metal.RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		h1 := l.MLP.forward(h1In)
		l.traceNativeMaterialize(traceEnabled, "ffn_local_mlp", h1)
		metal.Free(h1In)

		h2In := metal.RMSNorm(h, l.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := l.Router.forward(h)
		l.traceNativeMaterialize(traceEnabled, "ffn_router", topKIndices, topKWeights)
		expertTracePrefix := ""
		if traceEnabled {
			expertTracePrefix = l.nativeTraceName("ffn_expert")
		}
		h2 := l.Experts.forward(h2In, topKIndices, topKWeights, expertTracePrefix)
		l.traceNativeMaterialize(traceEnabled, "ffn_experts", h2)
		metal.Free(h2In, topKIndices, topKWeights)

		if nativeOut, ok, err := metal.NativeGemma4FFNResidual(residual, h1, h2, l.PostFFNorm1Scaled, l.PostFFNorm2Scaled, l.PostFFNormScaled, cfg.RMSNormEps); ok {
			if err == nil {
				hNext = nativeOut
				l.traceNativeMaterialize(traceEnabled, "ffn_residual", hNext)
			} else {
				core.Error("mlx: native Gemma 4 FFN residual failed; falling back to Go graph", "error", err)
			}
		}
		if hNext == nil {
			h1Normed := metal.RMSNorm(h1, l.PostFFNorm1Scaled, cfg.RMSNormEps)
			l.traceNativeMaterialize(traceEnabled, "ffn_local_norm", h1Normed)
			h2Normed := metal.RMSNorm(h2, l.PostFFNorm2Scaled, cfg.RMSNormEps)
			l.traceNativeMaterialize(traceEnabled, "ffn_expert_norm", h2Normed)

			// Gemma 4 MoE layers normalise each branch independently, then apply
			// the standard post-feedforward norm to the combined branch output
			// before adding it back to the residual path.
			combined := metal.Add(h1Normed, h2Normed)
			metal.Free(h1Normed, h2Normed)
			ffResidual = metal.RMSNorm(combined, l.PostFFNormScaled, cfg.RMSNormEps)
			metal.Free(combined)
		}
		metal.Free(h1, h2)
	} else {
		ffIn := metal.RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		ff := l.MLP.forward(ffIn)
		metal.Free(ffIn)
		ffResidual = metal.RMSNorm(ff, l.PostFFNormScaled, cfg.RMSNormEps)
		metal.Free(ff)
	}
	if ffResidual != nil {
		l.traceNativeMaterialize(traceEnabled, "ffn", ffResidual)
	}

	if hNext == nil {
		hNext = metal.Add(residual, ffResidual)
		metal.Free(ffResidual)
	}
	metal.Free(h)

	if l.PerLayerInputGate != nil && l.PerLayerProjection != nil && l.PostPerLayerInputNormScaled != nil && perLayerInput != nil {
		gate := l.PerLayerInputGate.Forward(hNext)
		multiplied := metal.GeluGateMul(gate, perLayerInput)
		metal.Free(gate)
		projected := l.PerLayerProjection.Forward(multiplied)
		metal.Free(multiplied)
		projectedNormed := metal.RMSNorm(projected, l.PostPerLayerInputNormScaled, cfg.RMSNormEps)
		metal.Free(projected)
		gated := metal.Add(hNext, projectedNormed)
		metal.Free(hNext, projectedNormed)
		hNext = gated
	}

	if l.LayerScalar != nil && l.LayerScalar.Valid() {
		scaled := metal.Mul(hNext, l.LayerScalar)
		metal.Free(hNext)
		hNext = scaled
	}
	l.traceNativeMaterialize(traceEnabled, "output", hNext)

	return hNext, kv
}

func (l *Gemma4DecoderLayer) traceNativeMaterialize(enabled bool, phase string, arrays ...*metal.Array) {
	if !enabled {
		return
	}
	metal.TraceNativeMaterialize(l.nativeTraceName(phase), arrays...)
}

func gemma4AttentionWindowTraceName(window int32) string {
	if window > 0 {
		return "local"
	}
	return "global"
}

func tracePagedKVConcat(name string, start time.Time, state metal.PagedKVState) {
	if !metal.NativePhaseTraceArmed() || name == "" || start.IsZero() {
		return
	}
	duration := time.Since(start)
	if duration <= 0 {
		duration = time.Nanosecond
	}
	metal.AppendNativePhaseTraceEvent(metal.NativePhaseTrace{
		Name:     name,
		Duration: duration,
		Pages:    len(state.Keys),
		Tokens:   state.Length,
	})
}

func (l *Gemma4DecoderLayer) nativeTraceName(phase string) string {
	return core.Sprintf("gemma4.layer.%02d.%s", l.LayerIdx, phase)
}
