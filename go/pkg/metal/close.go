// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// ModelCloser capability (go-mlx #45): each model releases its Metal arrays via
// the free helper defined alongside it, so Model.Close dispatches on the
// capability interface instead of a concrete type-switch. These wrappers travel
// with their workers when a model moves out of package metal.
// KimiModel's CloseModel travels with the model in package metal/model/kimi.
// MixtralModel's CloseModel travels with the model in package metal/model/mixtral.
// GptOssModel's CloseModel travels with the model in package metal/model/gptoss.
// Qwen3Model's CloseModel travels with the model in package metal/model/qwen3.
// Qwen3MoEModel's CloseModel travels with the model in package metal/model/qwen3.

// FreeLinear releases all weight arrays held by a Linear layer.
func FreeLinear(l *Linear) {
	if l == nil {
		return
	}
	Free(l.Weight, l.Scales, l.Biases, l.Bias, l.DenseFallbackT)
	if l.LoRA != nil {
		Free(l.LoRA.A, l.LoRA.B)
	}
}

// FreeSwitchLinear releases all weight arrays held by a SwitchLinear layer.
func FreeSwitchLinear(l *SwitchLinear) {
	if l == nil {
		return
	}
	Free(l.Weight, l.WeightT, l.Scales, l.Biases, l.Bias)
}

// FreeEmbedding releases all weight arrays held by an Embedding layer.
func FreeEmbedding(e *Embedding) {
	if e == nil {
		return
	}
	Free(e.Weight, e.Scales, e.Biases)
}

// FreeRMSNorm releases the weight array held by an RMSNormModule.
func FreeRMSNorm(r *RMSNormModule) {
	if r == nil {
		return
	}
	Free(r.Weight)
}

// FreeCaches releases all key/value arrays held by a slice of caches.
func FreeCaches(caches []Cache) {
	for _, c := range caches {
		if c == nil {
			continue
		}
		if s := c.State(); s != nil {
			Free(s...)
		}
	}
}

// closeGemma travels with the Gemma 3 model in package metal/model/gemma3.

// closeGemma4 releases all Metal arrays held by a Gemma4Model.

// closeQwen3 travels with the model in package metal/model/qwen3.
