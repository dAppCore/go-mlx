// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// TransformerConfig is the architecture-neutral transformer-config core — the
// fields every HuggingFace config.json carries regardless of family (Llama,
// Qwen, Gemma, Mixtral, …). Family configs EMBED it so the common shape is
// declared exactly once: DenseConfig here, the gemma4 package's
// Gemma4TextConfig, and (as they adopt it) every other model config. Go field
// promotion keeps every `cfg.HiddenSize`-style access — and stdlib JSON
// unmarshalling of the embedded tags — working unchanged.
//
//	type FamilyConfig struct {
//	    metal.TransformerConfig          // promotes ModelType/HiddenSize/…
//	    FamilySpecificField int32 `json:"family_specific"`
//	}
type TransformerConfig struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
}

// DenseConfig holds the Llama-family dense-transformer configuration shared by
// the pre-norm SwiGLU GQA models on the metal SDK (Qwen 2/3, Llama, Mistral,
// Hermes, Granite, Phi, GLM and the Mixtral/GPT-OSS/Kimi dense blocks). It
// embeds the neutral TransformerConfig core and adds the dense/MoE/RoPE fields.
// The sparse-MoE checkpoints reuse the same shape and populate the expert
// fields; the per-family loaders differ only in which checkpoint weight names
// they probe.
type DenseConfig struct {
	TransformerConfig // embedded core: ModelType/HiddenSize/NumHiddenLayers/… promoted

	MoEIntermediateSize int32    `json:"moe_intermediate_size"`
	NumExperts          int32    `json:"num_experts"`
	NumExpertsPerTok    int32    `json:"num_experts_per_tok"`
	DecoderSparseStep   int32    `json:"decoder_sparse_step"`
	RopeTheta           float32  `json:"rope_theta"`
	PartialRotaryFactor float32  `json:"partial_rotary_factor"`
	LayerTypes          []string `json:"layer_types"`

	Quantization *QuantizationConfig `json:"-"`
	Scale        float32             `json:"-"` // 1/sqrt(head_dim)
}

// DenseDecoderLayer is a single pre-norm transformer block: standard pre-norm
// residual (norm→attn→add, norm→mlp→add). The dense and MoE families compose it
// directly — MoE layers swap the MLP slot for an expert block.
type DenseDecoderLayer struct {
	InputNorm    *RMSNormModule // Pre-attention norm
	PostAttnNorm *RMSNormModule // Pre-MLP norm (confusingly named post_attention_layernorm)
	Attention    *GQAAttention
	MLP          *SiLUMLP
}

// GQAAttention implements grouped-query attention with optional Q/K RMS
// normalization (Qwen 3 has the Q/K norms; Qwen 2 and the other dense families
// leave them nil). The same algo serves every dense and MoE family on the SDK.
type GQAAttention struct {
	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear
	QNorm *RMSNormModule
	KNorm *RMSNormModule
}

// Forward runs one dense decoder layer. Exported so model packages can compose
// the shared dense transformer block without living inside package metal.
func (l *DenseDecoderLayer) Forward(x *Array, c Cache, B, L int32, mask *Array, cfg *DenseConfig) *Array {
	// Pre-attention norm → attention → residual add
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, mask, cfg)
	Free(normed)
	h := Add(x, attnOut)
	Free(attnOut)

	// Pre-MLP norm → MLP → residual add
	normed2 := l.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.Forward(normed2)
	Free(normed2)
	result := Add(h, mlpOut)
	Free(h, mlpOut)
	return result
}

func (l *DenseDecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *DenseConfig) *Array {
	return l.Forward(x, c, B, L, mask, cfg)
}

// Forward runs grouped-query attention for one decoder layer. Exported so
// models on the metal SDK (e.g. metal/model/mixtral) can drive the shared
// attention algo from outside package metal.
func (a *GQAAttention) Forward(x *Array, c Cache, B, L int32, mask *Array, cfg *DenseConfig) *Array {
	return a.forward(x, c, B, L, mask, cfg)
}

func (a *GQAAttention) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *DenseConfig) *Array {
	qProj := a.QProj.Forward(x)
	kProj := a.KProj.Forward(x)
	vProj := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim] via stride manipulation.
	// AsStrided creates a view (C refcount keeps source alive), so Free source after.
	q := AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	Free(qProj)
	k := AsStrided(kProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	Free(kProj)
	v := AsStrided(vProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	Free(vProj)

	// Q/K RMS normalization (Qwen 3 has this; Qwen 2 does not)
	if a.QNorm != nil && a.QNorm.Weight != nil {
		oldQ := q
		q = a.QNorm.Forward(q, cfg.RMSNormEps)
		Free(oldQ)
	}
	if a.KNorm != nil && a.KNorm.Weight != nil {
		oldK := k
		k = a.KNorm.Forward(k, cfg.RMSNormEps)
		Free(oldK)
	}

	// RoPE — single theta for all layers (no sliding window)
	oldQ := q
	q = RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	Free(oldQ)
	oldK := k
	k = RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	Free(oldK)

	// Scaled dot-product attention
	var out *Array
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if paged, ok := c.(*PagedKVCache); ok && L == 1 && mask == nil {
		oldK, oldV := k, v
		pages := paged.UpdatePages(k, v, int(L))
		Free(oldK, oldV)
		kPages, vPages := pages.Keys, pages.Values
		var repeatedPages []*Array
		if PagedStateNeedsMaterializedRepeat(pages, repeatFactor) {
			kPages, vPages, repeatedPages = RepeatPagedState(pages, repeatFactor)
		}
		out = ScaledDotProductAttentionPaged(q, kPages, vPages, cfg.Scale)
		Free(repeatedPages...)
		pages.Free()
	} else {
		// Update KV cache — returns Slice views into cache buffer; free our pre-update handles.
		oldK, oldV := k, v
		k, v = c.Update(k, v, int(L))
		Free(oldK, oldV)

		// GQA: repeat K/V heads to match Q heads
		kAttn, vAttn := k, v
		if repeatFactor > 1 {
			kAttn = RepeatKV(k, repeatFactor)
			vAttn = RepeatKV(v, repeatFactor)
			Free(k, v) // Free Slice views from cache.Update; RepeatKV holds copies
		}

		if mask != nil {
			out = ScaledDotProductAttentionWithMask(q, kAttn, vAttn, mask, cfg.Scale)
		} else {
			out = ScaledDotProductAttention(q, kAttn, vAttn, cfg.Scale, L > 1)
		}
		Free(kAttn, vAttn) // Always free — when repeatFactor==1 this frees the Slice views
	}
	Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := Transpose4(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	Free(transposed)
	result := a.OProj.Forward(reshaped)
	Free(reshaped)
	return result
}
