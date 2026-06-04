// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import "dappco.re/go/mlx/pkg/metal"

// CloseModel releases all Metal arrays held by the model (metal.ModelCloser).
func (m *KimiModel) CloseModel() { closeKimi(m) }

func closeKimi(m *KimiModel) {
	if m == nil {
		return
	}
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeRMSNorm(m.Norm)

	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		metal.FreeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		if layer == nil || layer.Dense == nil {
			continue
		}
		if layer.Dense.Attention != nil {
			metal.FreeLinear(layer.Dense.Attention.QProj)
			metal.FreeLinear(layer.Dense.Attention.KProj)
			metal.FreeLinear(layer.Dense.Attention.VProj)
			metal.FreeLinear(layer.Dense.Attention.OProj)
		}
		metal.FreeRMSNorm(layer.Dense.InputNorm)
		metal.FreeRMSNorm(layer.Dense.PostAttnNorm)
		if layer.Dense.MLP != nil {
			metal.FreeLinear(layer.Dense.MLP.GateProj)
			metal.FreeLinear(layer.Dense.MLP.UpProj)
			metal.FreeLinear(layer.Dense.MLP.DownProj)
		}
		if layer.MoE != nil {
			if layer.MoE.Router != nil {
				metal.Free(layer.MoE.Router.Weight, layer.MoE.Router.Scales, layer.MoE.Router.Biases)
			}
			metal.FreeMoESwiGLUExperts(layer.MoE.SwitchExperts)
			for _, expert := range layer.MoE.Experts {
				metal.FreeLinear(expert.GateProj)
				metal.FreeLinear(expert.UpProj)
				metal.FreeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}
