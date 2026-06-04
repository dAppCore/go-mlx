// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import "dappco.re/go/mlx/pkg/metal"

func (m *GemmaModel) CloseModel() { closeGemma(m) }

// closeGemma releases all Metal arrays held by a GemmaModel.
func closeGemma(m *GemmaModel) {
	if m == nil {
		return
	}
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeRMSNorm(m.Norm)
	metal.Free(m.NormScaled)

	// Output may be tied to EmbedTokens — only free if it has its own weight.
	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		metal.FreeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		metal.FreeRMSNorm(layer.InputNorm)
		metal.FreeRMSNorm(layer.PostAttnNorm)
		metal.FreeRMSNorm(layer.PreFFNorm)
		metal.FreeRMSNorm(layer.PostFFNorm)
		metal.Free(layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled)

		attn := layer.Attention
		if attn != nil {
			metal.FreeLinear(attn.QProj)
			metal.FreeLinear(attn.KProj)
			metal.FreeLinear(attn.VProj)
			metal.FreeLinear(attn.OProj)
			metal.FreeRMSNorm(attn.QNorm)
			metal.FreeRMSNorm(attn.KNorm)
			metal.Free(attn.QNormScaled, attn.KNormScaled)
		}

		mlp := layer.MLP
		if mlp != nil {
			metal.FreeLinear(mlp.GateProj)
			metal.FreeLinear(mlp.UpProj)
			metal.FreeLinear(mlp.DownProj)
		}
	}
}
