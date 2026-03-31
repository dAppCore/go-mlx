//go:build darwin && arm64

package metal

// Linear is a fully-connected layer: y = x @ W.T + bias.
// For quantized models, set Scales/Biases/GroupSize/Bits to use QuantizedMatmul.
// Set LoRA to inject a low-rank adapter (training only).
type Linear struct {
	Weight    *Array `weight:"weight"`
	Scales    *Array `weight:"scales"`
	Biases    *Array `weight:"biases"`
	Bias      *Array `weight:"bias"`
	GroupSize int
	Bits      int

	LoRA *LoRALinear // Optional LoRA adapter — if set, Forward routes through it
}

// NewLinear creates a dense Linear layer with optional bias.
//
//	projection := metal.NewLinear(weights["q_proj.weight"], nil) // attention query projection
func NewLinear(weight, bias *Array) *Linear {
	return &Linear{Weight: weight, Bias: bias}
}

// NewQuantizedLinear creates a quantized Linear layer.
//
//	projection := metal.NewQuantizedLinear(w, scales, biases, nil, 64, 4) // 4-bit, group=64
func NewQuantizedLinear(weight, scales, biases, bias *Array, groupSize, bits int) *Linear {
	return &Linear{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		Bias:      bias,
		GroupSize: groupSize,
		Bits:      bits,
	}
}

// Forward computes the linear transformation.
// If a LoRA adapter is attached, routes through it instead (base + low-rank delta).
// Uses QuantizedMatmul when quantization parameters are present.
//
//	y := projection.Forward(input) // input: [B, L, in_dim] → y: [B, L, out_dim]
func (linear *Linear) Forward(input *Array) *Array {
	if linear.LoRA != nil {
		return linear.LoRA.Forward(input)
	}
	return linear.baseForward(input)
}

// baseForward is the raw linear transformation without LoRA.
// Used internally by LoRALinear to avoid infinite recursion.
func (linear *Linear) baseForward(input *Array) *Array {
	var out *Array
	if linear.Scales != nil {
		out = QuantizedMatmul(input, linear.Weight, linear.Scales, linear.Biases, true, linear.GroupSize, linear.Bits)
	} else {
		weightTranspose := Transpose(linear.Weight)
		out = Matmul(input, weightTranspose)
		Free(weightTranspose)
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		oldOut := out
		out = Add(out, linear.Bias)
		Free(oldOut)
	}
	return out
}

// Embedding is a lookup table for token embeddings.
// For quantized models, set Scales/Biases/GroupSize/Bits to dequantize before lookup.
type Embedding struct {
	Weight    *Array `weight:"weight"`
	Scales    *Array `weight:"scales"`
	Biases    *Array `weight:"biases"`
	GroupSize int
	Bits      int
}

// Forward looks up embeddings for the given token indices.
//
//	y := emb.Forward(tokenIDs) // tokenIDs: [B, L] int32 → y: [B, L, hidden_dim]
func (embedding *Embedding) Forward(tokenIDs *Array) *Array {
	if embedding.Scales != nil {
		w := Dequantize(embedding.Weight, embedding.Scales, embedding.Biases, embedding.GroupSize, embedding.Bits)
		res := Take(w, tokenIDs, 0)
		Free(w)
		return res
	}
	return Take(embedding.Weight, tokenIDs, 0)
}

// AsLinear returns a Linear layer using the embedding weights (for tied output).
//
//	output := embedding.AsLinear() // share embed_tokens weights with lm_head (Gemma3)
func (embedding *Embedding) AsLinear() *Linear {
	return &Linear{
		Weight:    embedding.Weight,
		Scales:    embedding.Scales,
		Biases:    embedding.Biases,
		GroupSize: embedding.GroupSize,
		Bits:      embedding.Bits,
	}
}

// RMSNormModule is an RMS normalization layer wrapping the fused kernel.
type RMSNormModule struct {
	Weight *Array `weight:"weight"`
}

// Forward applies RMS normalization.
//
//	normed := norm.Forward(input, 1e-6) // input: [B, L, hidden] → normed: same shape
func (norm *RMSNormModule) Forward(input *Array, eps float32) *Array {
	return RMSNorm(input, norm.Weight, eps)
}

// RepeatKV repeats key/value heads for grouped-query attention (GQA).
// Input shape: [B, num_kv_heads, L, D] → output: [B, num_kv_heads*factor, L, D].
//
//	// Gemma3: 16 KV heads, 16 query groups → factor=1 (no-op)
//	// Qwen3:   8 KV heads, 32 query heads  → factor=4
//	kExpanded := metal.RepeatKV(k, int32(numQueryHeads/numKVHeads))
func RepeatKV(input *Array, factor int32) *Array {
	if factor <= 1 {
		return input
	}
	shape := input.Shape()
	B, H, L, D := shape[0], shape[1], shape[2], shape[3]

	// Expand: [B, H, 1, L, D] then broadcast to [B, H, factor, L, D]
	expanded := ExpandDims(input, 2)
	broadcasted := BroadcastTo(expanded, []int32{B, H, factor, L, D})
	Free(expanded)

	res := Reshape(broadcasted, B, H*factor, L, D)
	Free(broadcasted)
	return res
}
