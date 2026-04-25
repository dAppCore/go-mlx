//go:build darwin && arm64 && !nomlx

package metal

// freeLinear releases all weight arrays held by a Linear layer.
func freeLinear(l *Linear) {
	if l == nil {
		return
	}
	Free(l.Weight, l.Scales, l.Biases, l.Bias)
	if l.LoRA != nil {
		Free(l.LoRA.A, l.LoRA.B)
	}
}

// freeSwitchLinear releases all weight arrays held by a SwitchLinear layer.
func freeSwitchLinear(l *SwitchLinear) {
	if l == nil {
		return
	}
	Free(l.Weight, l.WeightT, l.Scales, l.Biases, l.Bias)
}

// freeEmbedding releases all weight arrays held by an Embedding layer.
func freeEmbedding(e *Embedding) {
	if e == nil {
		return
	}
	Free(e.Weight, e.Scales, e.Biases)
}

// freeRMSNorm releases the weight array held by an RMSNormModule.
func freeRMSNorm(r *RMSNormModule) {
	if r == nil {
		return
	}
	Free(r.Weight)
}

// freeCaches releases all key/value arrays held by a slice of caches.
func freeCaches(caches []Cache) {
	for _, c := range caches {
		if s := c.State(); s != nil {
			Free(s...)
		}
	}
}

// closeGemma releases all Metal arrays held by a GemmaModel.
func closeGemma(m *GemmaModel) {
	freeEmbedding(m.EmbedTokens)
	freeRMSNorm(m.Norm)
	Free(m.NormScaled)

	// Output may be tied to EmbedTokens — only free if it has its own weight.
	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		freeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		freeRMSNorm(layer.InputNorm)
		freeRMSNorm(layer.PostAttnNorm)
		freeRMSNorm(layer.PreFFNorm)
		freeRMSNorm(layer.PostFFNorm)
		Free(layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled)

		attn := layer.Attention
		if attn != nil {
			freeLinear(attn.QProj)
			freeLinear(attn.KProj)
			freeLinear(attn.VProj)
			freeLinear(attn.OProj)
			freeRMSNorm(attn.QNorm)
			freeRMSNorm(attn.KNorm)
			Free(attn.QNormScaled, attn.KNormScaled)
		}

		mlp := layer.MLP
		if mlp != nil {
			freeLinear(mlp.GateProj)
			freeLinear(mlp.UpProj)
			freeLinear(mlp.DownProj)
		}
	}
}

// closeGemma4 releases all Metal arrays held by a Gemma4Model.
func closeGemma4(m *Gemma4Model) {
	freeEmbedding(m.EmbedTokens)
	freeEmbedding(m.EmbedTokensPerLayer)
	closeGemma4Vision(m.VisionTower, m.MultiModalProjector)
	freeRMSNorm(m.Norm)
	freeLinear(m.PerLayerModelProj)
	freeRMSNorm(m.PerLayerProjNorm)
	Free(m.NormScaled, m.PerLayerProjNormScaled)

	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		freeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		freeRMSNorm(layer.InputNorm)
		freeRMSNorm(layer.PostAttnNorm)
		freeRMSNorm(layer.PreFFNorm)
		freeRMSNorm(layer.PostFFNorm)
		freeRMSNorm(layer.PreFFNorm2)
		freeRMSNorm(layer.PostFFNorm1)
		freeRMSNorm(layer.PostFFNorm2)
		freeRMSNorm(layer.PostPerLayerInputNorm)
		Free(
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
			freeLinear(attn.QProj)
			freeLinear(attn.KProj)
			freeLinear(attn.VProj)
			freeLinear(attn.OProj)
			freeRMSNorm(attn.QNorm)
			freeRMSNorm(attn.KNorm)
			Free(attn.QNormScaled, attn.KNormScaled, attn.RopeFreqs)
		}

		mlp := layer.MLP
		if mlp != nil {
			freeLinear(mlp.GateProj)
			freeLinear(mlp.UpProj)
			freeLinear(mlp.DownProj)
		}

		if layer.Router != nil {
			freeLinear(layer.Router.Proj)
			Free(layer.Router.Scale, layer.Router.PerExpertScale, layer.Router.ScaleScaled)
		}

		if layer.Experts != nil {
			freeSwitchLinear(layer.Experts.GateProj)
			freeSwitchLinear(layer.Experts.UpProj)
			freeSwitchLinear(layer.Experts.DownProj)
		}

		freeLinear(layer.PerLayerInputGate)
		freeLinear(layer.PerLayerProjection)
	}
}

// closeQwen3 releases all Metal arrays held by a Qwen3Model.
func closeQwen3(m *Qwen3Model) {
	freeEmbedding(m.EmbedTokens)
	freeRMSNorm(m.Norm)

	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		freeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		freeRMSNorm(layer.InputNorm)
		freeRMSNorm(layer.PostAttnNorm)

		attn := layer.Attention
		if attn != nil {
			freeLinear(attn.QProj)
			freeLinear(attn.KProj)
			freeLinear(attn.VProj)
			freeLinear(attn.OProj)
			freeRMSNorm(attn.QNorm)
			freeRMSNorm(attn.KNorm)
		}

		mlp := layer.MLP
		if mlp != nil {
			freeLinear(mlp.GateProj)
			freeLinear(mlp.UpProj)
			freeLinear(mlp.DownProj)
		}
	}
}
