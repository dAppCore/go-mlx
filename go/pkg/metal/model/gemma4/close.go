// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "dappco.re/go/mlx/pkg/metal"

func (m *Gemma4Model) CloseModel() { closeGemma4(m) }

func closeGemma4(m *Gemma4Model) {
	if m == nil {
		return
	}
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeEmbedding(m.EmbedTokensPerLayer)
	closeGemma4Vision(m.VisionTower, m.MultiModalProjector)
	metal.FreeRMSNorm(m.Norm)
	metal.FreeLinear(m.PerLayerModelProj)
	metal.FreeRMSNorm(m.PerLayerProjNorm)
	metal.Free(m.NormScaled, m.PerLayerProjNormScaled)
	if m.compiledPerLayerInputs != nil {
		m.compiledPerLayerInputs.Free()
	}

	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		metal.FreeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		if layer.compiledNativeOwnerDecode != nil {
			layer.compiledNativeOwnerDecode.Free()
		}
		if layer.compiledNativeSharedDecode != nil {
			layer.compiledNativeSharedDecode.Free()
		}
		if layer.compiledNativeFixedOwnerDecode != nil {
			layer.compiledNativeFixedOwnerDecode.Free()
		}
		if layer.compiledNativeFixedSharedDecode != nil {
			layer.compiledNativeFixedSharedDecode.Free()
		}
		if layer.compiledNativeFixedMaskedOwnerDecode != nil {
			layer.compiledNativeFixedMaskedOwnerDecode.Free()
		}
		if layer.compiledNativeFixedMaskedSharedDecode != nil {
			layer.compiledNativeFixedMaskedSharedDecode.Free()
		}
		metal.FreeRMSNorm(layer.InputNorm)
		metal.FreeRMSNorm(layer.PostAttnNorm)
		metal.FreeRMSNorm(layer.PreFFNorm)
		metal.FreeRMSNorm(layer.PostFFNorm)
		metal.FreeRMSNorm(layer.PreFFNorm2)
		metal.FreeRMSNorm(layer.PostFFNorm1)
		metal.FreeRMSNorm(layer.PostFFNorm2)
		metal.FreeRMSNorm(layer.PostPerLayerInputNorm)
		metal.Free(
			layer.InputNormScaled,
			layer.PostAttnNormScaled,
			layer.PreFFNormScaled,
			layer.PostFFNormScaled,
			layer.PreFFNorm2Scaled,
			layer.PostFFNorm1Scaled,
			layer.PostFFNorm2Scaled,
			layer.PostPerLayerInputNormScaled,
			layer.LayerScalar,
		)

		attn := layer.Attention
		if attn != nil {
			metal.FreeLinear(attn.QProj)
			metal.FreeLinear(attn.KProj)
			metal.FreeLinear(attn.VProj)
			metal.FreeLinear(attn.OProj)
			metal.FreeRMSNorm(attn.QNorm)
			metal.FreeRMSNorm(attn.KNorm)
			metal.Free(attn.QNormScaled, attn.KNormScaled, attn.RopeFreqs)
		}

		mlp := layer.MLP
		if mlp != nil {
			metal.FreeLinear(mlp.GateProj)
			metal.FreeLinear(mlp.UpProj)
			metal.FreeLinear(mlp.DownProj)
		}

		if layer.Router != nil {
			metal.FreeLinear(layer.Router.Proj)
			metal.Free(layer.Router.Scale, layer.Router.PerExpertScale, layer.Router.ScaleScaled)
		}

		if layer.Experts != nil {
			metal.FreeSwitchLinear(layer.Experts.GateUpProj)
			metal.FreeSwitchLinear(layer.Experts.GateProj)
			metal.FreeSwitchLinear(layer.Experts.UpProj)
			metal.FreeSwitchLinear(layer.Experts.DownProj)
		}

		metal.FreeLinear(layer.PerLayerInputGate)
		metal.FreeLinear(layer.PerLayerProjection)
	}
}
