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
func NewLinear(weight, bias *Array) *Linear {
	return &Linear{Weight: weight, Bias: bias}
}

// NewQuantizedLinear creates a quantized Linear layer.
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
func (l *Linear) Forward(x *Array) *Array {
	if l.LoRA != nil {
		return l.LoRA.Forward(x)
	}
	return l.baseForward(x)
}

// baseForward is the raw linear transformation without LoRA.
// Used internally by LoRALinear to avoid infinite recursion.
func (l *Linear) baseForward(x *Array) *Array {
	var out *Array
	if l.Scales != nil {
		out = QuantizedMatmul(x, l.Weight, l.Scales, l.Biases, true, l.GroupSize, l.Bits)
	} else {
		wT := Transpose(l.Weight)
		out = Matmul(x, wT)
		Free(wT)
	}
	if l.Bias != nil && l.Bias.Valid() {
		oldOut := out
		out = Add(out, l.Bias)
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
func (e *Embedding) Forward(indices *Array) *Array {
	if e.Scales != nil {
		w := Dequantize(e.Weight, e.Scales, e.Biases, e.GroupSize, e.Bits)
		res := Take(w, indices, 0)
		Free(w)
		return res
	}
	return Take(e.Weight, indices, 0)
}

// AsLinear returns a Linear layer using the embedding weights (for tied output).
func (e *Embedding) AsLinear() *Linear {
	return &Linear{
		Weight:    e.Weight,
		Scales:    e.Scales,
		Biases:    e.Biases,
		GroupSize: e.GroupSize,
		Bits:      e.Bits,
	}
}

// RMSNormModule is an RMS normalization layer wrapping the fused kernel.
type RMSNormModule struct {
	Weight *Array `weight:"weight"`
}

// Forward applies RMS normalization.
func (r *RMSNormModule) Forward(x *Array, eps float32) *Array {
	return RMSNorm(x, r.Weight, eps)
}

// RepeatKV repeats key/value heads for grouped-query attention.
// Input shape: [B, num_kv_heads, L, D]
// Output shape: [B, num_kv_heads * factor, L, D]
func RepeatKV(x *Array, factor int32) *Array {
	if factor <= 1 {
		return x
	}
	shape := x.Shape()
	B, H, L, D := shape[0], shape[1], shape[2], shape[3]

	// Expand: [B, H, 1, L, D] then broadcast to [B, H, factor, L, D]
	expanded := ExpandDims(x, 2)
	broadcasted := BroadcastTo(expanded, []int32{B, H, factor, L, D})
	Free(expanded)

	res := Reshape(broadcasted, B, H*factor, L, D)
	Free(broadcasted)
	return res
}
