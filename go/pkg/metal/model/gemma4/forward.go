// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func (m *Gemma4Model) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

// ForwardMasked runs the forward pass with an explicit attention mask.
func (m *Gemma4Model) ForwardMasked(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	h, _, _ := m.forwardHidden(tokens, mask, caches)
	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out
}

// ForwardLastTokenLogits runs prefill while projecting only the final sequence
// position. Long local-context warmup needs KV cache updates for every token,
// but generation only consumes logits from the last token; avoiding full
// [sequence, vocab] logits keeps Gemma 4 prefill inside Apple memory limits.
func (m *Gemma4Model) ForwardLastTokenLogits(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	out, hidden := m.ForwardLastTokenLogitsAndHidden(tokens, mask, caches)
	metal.Free(hidden)
	return out
}

// ForwardLastTokenLogitsAndHidden runs prefill while returning both final
// position logits and the corresponding target hidden state before output
// normalisation. The hidden state is the seed consumed by attached MTP
// assistants.
func (m *Gemma4Model) ForwardLastTokenLogitsAndHidden(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) (*metal.Array, *metal.Array) {
	h, _, L := m.forwardHidden(tokens, mask, caches)
	h = gemma4LastSequenceHidden(h, L)
	h = gemma4ProjectionHidden(h)
	h = gemma4ContiguousHidden(h)
	if gemma4PreferNativeLastTokenOutputLogits(m.Output) {
		if out, ok, err := metal.NativeLastTokenOutputLogits(h, m.NormScaled, m.Output, m.Cfg.RMSNormEps, m.Cfg.FinalLogitSoftcapping); ok {
			if err == nil {
				return out, h
			}
			core.Error("mlx: native Gemma 4 last-token output failed; falling back to Go graph", "error", err)
		}
	}
	return m.forwardLastTokenOutputGraph(h), h
}

func gemma4PreferNativeLastTokenOutputLogits(output *metal.Linear) bool {
	if output == nil {
		return false
	}
	if output.Scales != nil {
		return false
	}
	return true
}

func (m *Gemma4Model) forwardLastTokenOutputGraph(h *metal.Array) *metal.Array {
	if m == nil || m.Cfg == nil {
		return nil
	}
	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out
}

// ForwardAllTokenLogitsAndHidden runs the forward pass returning logits AND the
// pre-output-norm hidden state at EVERY sequence position (not just the last).
// Batched MTP verification uses it to check a whole draft block in ONE target
// pass: logits[:,i,:] is the target's prediction after consuming the i-th input
// token, and hidden[:,i,:] seeds the next draft from the last accepted position.
// Returns ([1,L,vocab], [1,L,hidden]); the caches are advanced by L tokens.
func (m *Gemma4Model) ForwardAllTokenLogitsAndHidden(tokens *metal.Array, caches []metal.Cache) (*metal.Array, *metal.Array) {
	h, _, _ := m.forwardHidden(tokens, nil, caches)
	hidden := metal.Copy(h)
	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	metal.Free(h)
	out := m.Output.Forward(normed)
	metal.Free(normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out, hidden
}

// ForwardGreedyToken runs a forward pass and returns the metal.Greedy next token
// directly. Final logit softcapping is monotonic, so metal.Greedy selection can skip
// materialising a softcapped logits tensor.
func (m *Gemma4Model) ForwardGreedyToken(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	return m.forwardGreedyTokenWithSuppressionArray(tokens, mask, caches, nil, nil)
}

// ForwardGreedyTokenWithSuppression runs the same metal.Greedy decode path while
// masking chat-template and modality token IDs before argmax.
func (m *Gemma4Model) ForwardGreedyTokenWithSuppression(tokens *metal.Array, mask *metal.Array, caches []metal.Cache, suppressTokens []int32) *metal.Array {
	return m.forwardGreedyTokenWithSuppressionArray(tokens, mask, caches, suppressTokens, nil)
}

func (m *Gemma4Model) forwardGreedyTokenWithSuppressionArray(tokens *metal.Array, mask *metal.Array, caches []metal.Cache, suppressTokens []int32, suppress *metal.Array) *metal.Array {
	h, _, L := m.forwardHidden(tokens, mask, caches)
	h = gemma4LastSequenceHidden(h, L)
	h = gemma4ProjectionHidden(h)
	h = gemma4ContiguousHidden(h)
	if out, ok, err := metal.NativeLastTokenGreedyTokenWithArray(h, m.NormScaled, m.Output, m.Cfg.RMSNormEps, suppress, suppressTokens...); ok {
		if err == nil {
			metal.Free(h)
			return out
		}
		core.Error("mlx: native Gemma 4 metal.Greedy token failed; falling back to Go graph", "error", err)
	}
	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	logits := m.Output.Forward(normed)
	var out *metal.Array
	if len(suppressTokens) > 0 {
		var err error
		sampler := metal.NewSamplerWithSuppression(0, 0, 0, 0, suppressTokens)
		out, err = metal.SampleTokenWithSuppressionGuard(logits, sampler, suppressTokens)
		metal.CloseSampler(sampler)
		if err != nil {
			core.Error("mlx: Gemma 4 suppressed metal.Greedy fallback failed; falling back to unsuppressed argmax", "error", err)
			metal.Free(out)
			out = metal.Argmax(logits, -1, false)
		}
	} else {
		out = metal.Argmax(logits, -1, false)
	}
	metal.Free(h, normed, logits)
	return out
}

func gemma4LastSequenceHidden(h *metal.Array, seqLen int32) *metal.Array {
	if h == nil || !h.Valid() || seqLen <= 1 {
		return h
	}
	ndim := h.NumDims()
	var axis int
	switch {
	case ndim >= 3:
		axis = ndim - 2
	case ndim == 2:
		axis = 0
	default:
		return h
	}
	dim := h.Dim(axis)
	if dim <= 1 {
		return h
	}
	start := int32(dim - 1)
	if seqLen > 0 && seqLen <= int32(dim) {
		start = seqLen - 1
	}
	last := metal.SliceAxis(h, axis, start, start+1)
	metal.Free(h)
	return last
}

func gemma4ProjectionHidden(h *metal.Array) *metal.Array {
	if h == nil || !h.Valid() {
		return h
	}
	switch h.NumDims() {
	case 1:
		out := metal.Reshape(h, 1, 1, int32(h.Dim(0)))
		metal.Free(h)
		return out
	case 2:
		out := metal.Reshape(h, 1, int32(h.Dim(0)), int32(h.Dim(1)))
		metal.Free(h)
		return out
	default:
		return h
	}
}

func gemma4ContiguousHidden(h *metal.Array) *metal.Array {
	if h == nil || !h.Valid() || h.IsRowContiguous() {
		return h
	}
	out := metal.Contiguous(h)
	metal.Free(h)
	return out
}

// gemma4ForwardOverrides redirects forwardHidden for non-causal forwards.
// The block-diffusion canvas step supplies explicit bidirectional masks
// (replacing the causal builders entirely) and an embed hook that injects
// the self-conditioning signal after embedding scale.
type gemma4ForwardOverrides struct {
	fullMask    *metal.Array
	slidingMask *metal.Array
	embedHook   func(h *metal.Array) *metal.Array
}

func (m *Gemma4Model) forwardHidden(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) (*metal.Array, int32, int32) {
	return m.forwardHiddenOverride(tokens, mask, caches, nil)
}

func (m *Gemma4Model) forwardHiddenOverride(tokens *metal.Array, mask *metal.Array, caches []metal.Cache, ov *gemma4ForwardOverrides) (*metal.Array, int32, int32) {
	m.ensureCacheLayout()

	// Stack-allocated shape scratch — per-forward-pass hot path. Avoids
	// the per-call []int32 heap alloc from tokens.Shape().
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)
	scaledH := metal.MulScalar(h, m.Cfg.EmbeddingScale)
	metal.Free(h)
	h = scaledH
	if ov != nil && ov.embedHook != nil {
		h = ov.embedHook(h)
	}

	perLayerInputTensor := m.computePerLayerInputTensor(tokens, h, B, L)
	defer metal.Free(perLayerInputTensor)

	var ownedMasks []*metal.Array
	var runtimeMasks *gemma4RuntimeMaskCache
	if L > 1 {
		runtimeMasks = newGemma4RuntimeMaskCache()
		defer runtimeMasks.Free()
	}
	fixedMasks := newFixedGemma4AttentionMaskSet(B, L, mask)
	defer fixedMasks.Free()
	fullMask := mask
	slidingMask := mask
	if ov != nil && (ov.fullMask != nil || ov.slidingMask != nil) {
		// Explicit per-layer-type masks (the diffusion canvas step): the
		// causal builders are bypassed — the masks carry ALL semantics.
		fullMask = ov.fullMask
		slidingMask = ov.slidingMask
	} else if mask == nil {
		if L > 1 && m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
			slidingMask = buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
			ownedMasks = append(ownedMasks, slidingMask)
		}
	} else if m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
		windowMask := buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
		combined := gemma4CombineMasks(mask, windowMask)
		metal.Free(windowMask)
		slidingMask = combined
		ownedMasks = append(ownedMasks, combined)
	}
	defer metal.Free(ownedMasks...)

	var stackIntermediates [64]sharedKV
	var intermediates []sharedKV
	var stackSharedSources [64]bool
	var sharedSources []bool
	if len(m.Layers) <= len(stackIntermediates) {
		intermediates = stackIntermediates[:len(m.Layers)]
		sharedSources = stackSharedSources[:len(m.Layers)]
	} else {
		intermediates = make([]sharedKV, len(m.Layers))
		sharedSources = make([]bool, len(m.Layers))
	}
	for i, prevIdx := range m.PreviousKVs {
		if i >= len(sharedSources) {
			break
		}
		if prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(sharedSources)) {
			sharedSources[prevIdx] = true
		}
	}
	defer func() {
		for _, kv := range intermediates {
			kv.Free()
		}
	}()
	for i, layer := range m.Layers {
		var prev sharedKV
		if prevIdx := m.PreviousKVs[i]; prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(intermediates)) {
			prev = intermediates[prevIdx]
		}

		var cache metal.Cache
		if m.PreviousKVs[i] == int32(i) && i < len(m.CacheIndexByLayer) {
			if cacheIdx := m.CacheIndexByLayer[i]; cacheIdx >= 0 && int(cacheIdx) < len(caches) {
				cache = caches[cacheIdx]
			}
		}

		layerMask := fullMask
		if layer.IsSliding {
			layerMask = slidingMask
		}

		pli := m.perLayerInputForLayer(perLayerInputTensor, B, L, int32(i))

		fixedMask := fixedMasks.ForLayer(cache, prev)
		prevAvailable := prev.HasState()
		materializePagedKVForReuse := m.PreviousKVs[i] == int32(i) && sharedSources[i]
		nextH, kv := layer.forward(h, cache, B, L, layerMask, pli, prev, m.Cfg, fixedMask, runtimeMasks, materializePagedKVForReuse)
		metal.Free(pli)
		metal.Free(h)
		h = nextH
		if m.PreviousKVs[i] == int32(i) || !prevAvailable {
			if sharedSources[i] {
				intermediates[i] = moveSharedKV(&kv)
			}
			kv.Free()
		}
	}
	return h, B, L
}

func logitSoftcap(x *metal.Array, softcap float32) *metal.Array {
	scaled := metal.MulScalar(x, 1.0/softcap)
	capped := metal.Tanh(scaled)
	metal.Free(scaled)
	out := metal.MulScalar(capped, softcap)
	metal.Free(capped)
	return out
}
