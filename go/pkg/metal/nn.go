// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Linear is a fully-connected layer: y = x @ W.T + bias.
// For quantized models, set Scales/Biases/GroupSize/Bits to use QuantizedMatmul.
// Set LoRA to inject a low-rank adapter (training only).
type Linear struct {
	Weight           *Array `weight:"weight"`
	Scales           *Array `weight:"scales"`
	Biases           *Array `weight:"biases"`
	Bias             *Array `weight:"bias"`
	DenseFallbackT   *Array
	GroupSize        int
	Bits             int
	QuantizationMode string

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
	return NewQuantizedLinearWithMode(weight, scales, biases, bias, groupSize, bits, "affine")
}

// NewQuantizedLinearWithMode creates a quantized Linear layer for a specific
// MLX quantization mode.
func NewQuantizedLinearWithMode(weight, scales, biases, bias *Array, groupSize, bits int, mode string) *Linear {
	return &Linear{
		Weight:           weight,
		Scales:           scales,
		Biases:           biases,
		Bias:             bias,
		GroupSize:        groupSize,
		Bits:             bits,
		QuantizationMode: NormalizeQuantizationMode(mode),
	}
}

// SwitchLinear is an expert-indexed linear layer backed by gather_mm / gather_qmm.
type SwitchLinear struct {
	Weight           *Array `weight:"weight"`
	WeightT          *Array
	Scales           *Array `weight:"scales"`
	Biases           *Array `weight:"biases"`
	Bias             *Array `weight:"bias"`
	GroupSize        int
	Bits             int
	QuantizationMode string
}

// NewSwitchLinear creates a dense expert-indexed linear layer.
func NewSwitchLinear(weight, bias *Array) *SwitchLinear {
	layer := &SwitchLinear{
		Weight: weight,
		Bias:   bias,
	}
	if weight != nil && weight.Valid() {
		layer.WeightT = Transpose(weight, 0, 2, 1)
	}
	return layer
}

// NewQuantizedSwitchLinear creates a quantized expert-indexed linear layer.
func NewQuantizedSwitchLinear(weight, scales, biases, bias *Array, groupSize, bits int) *SwitchLinear {
	return NewQuantizedSwitchLinearWithMode(weight, scales, biases, bias, groupSize, bits, "affine")
}

// NewQuantizedSwitchLinearWithMode creates a quantized expert-indexed linear
// layer for a specific MLX quantization mode.
func NewQuantizedSwitchLinearWithMode(weight, scales, biases, bias *Array, groupSize, bits int, mode string) *SwitchLinear {
	return &SwitchLinear{
		Weight:           weight,
		Scales:           scales,
		Biases:           biases,
		Bias:             bias,
		GroupSize:        groupSize,
		Bits:             bits,
		QuantizationMode: NormalizeQuantizationMode(mode),
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
		if RequiresDenseQuantizedMatmulFallback(linear.QuantizationMode) {
			if linear.DenseFallbackT == nil || !linear.DenseFallbackT.Valid() {
				denseWeight := DequantizeMode(linear.Weight, linear.Scales, linear.Biases, linear.GroupSize, linear.Bits, linear.QuantizationMode)
				linear.DenseFallbackT = Transpose(denseWeight)
				Free(denseWeight)
			}
			out = Matmul(input, linear.DenseFallbackT)
		} else if IsAffineQuantizationMode(linear.QuantizationMode) && nativeLinearMatVecRuntimeEnabled() {
			if nativeOut, ok, err := QuantizedDenseMatVec(input, linear); ok {
				if err == nil {
					return nativeOut
				}
				core.Error("mlx: native linear matvec failed; falling back to quantized matmul", "error", err)
				Free(nativeOut)
			}
			out = quantizedMatmulMode(input, linear.Weight, linear.Scales, linear.Biases, true, linear.GroupSize, linear.Bits, linear.QuantizationMode)
		} else {
			out = quantizedMatmulMode(input, linear.Weight, linear.Scales, linear.Biases, true, linear.GroupSize, linear.Bits, linear.QuantizationMode)
		}
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

// Forward computes the expert-indexed linear transformation selected by expertIndices.
func (linear *SwitchLinear) Forward(input, expertIndices *Array) *Array {
	var out *Array
	if linear.Scales != nil {
		if RequiresDenseQuantizedMatmulFallback(linear.QuantizationMode) {
			if linear.WeightT == nil || !linear.WeightT.Valid() {
				denseWeight := DequantizeMode(linear.Weight, linear.Scales, linear.Biases, linear.GroupSize, linear.Bits, linear.QuantizationMode)
				linear.WeightT = Transpose(denseWeight, 0, 2, 1)
				Free(denseWeight)
			}
			out = GatherMM(input, linear.WeightT, nil, expertIndices, false)
		} else {
			out = GatherQMM(input, linear.Weight, linear.Scales, linear.Biases, nil, expertIndices, true, linear.GroupSize, linear.Bits, linear.QuantizationMode, false)
		}
	} else {
		if linear.WeightT == nil && linear.Weight != nil && linear.Weight.Valid() {
			linear.WeightT = Transpose(linear.Weight, 0, 2, 1)
		}
		out = GatherMM(input, linear.WeightT, nil, expertIndices, false)
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		bias := Take(linear.Bias, expertIndices, 0)
		biasExpanded := ExpandDims(bias, bias.NumDims()-1)
		oldOut := out
		out = Add(out, biasExpanded)
		Free(oldOut, bias, biasExpanded)
	}
	return out
}

// Embedding is a lookup table for token embeddings.
// For quantized models, set Scales/Biases/GroupSize/Bits to dequantize before lookup.
type Embedding struct {
	Weight           *Array `weight:"weight"`
	Scales           *Array `weight:"scales"`
	Biases           *Array `weight:"biases"`
	GroupSize        int
	Bits             int
	QuantizationMode string
}

// Forward looks up embeddings for the given token indices.
//
//	y := emb.Forward(tokenIDs) // tokenIDs: [B, L] int32 → y: [B, L, hidden_dim]
func (embedding *Embedding) Forward(tokenIDs *Array) *Array {
	if embedding.Scales != nil {
		// Gather packed rows before dequantising to avoid materialising the full
		// vocabulary table for a single decode token.
		rows := Take(embedding.Weight, tokenIDs, 0)
		scales := Take(embedding.Scales, tokenIDs, 0)
		var biases *Array
		if embedding.Biases != nil && embedding.Biases.Valid() {
			biases = Take(embedding.Biases, tokenIDs, 0)
		}
		res := DequantizeMode(rows, scales, biases, embedding.GroupSize, embedding.Bits, embedding.QuantizationMode)
		Free(rows, scales, biases)
		return res
	}
	return Take(embedding.Weight, tokenIDs, 0)
}

// AsLinear returns a Linear layer using the embedding weights (for tied output).
//
//	output := embedding.AsLinear() // share embed_tokens weights with lm_head (Gemma3)
func (embedding *Embedding) AsLinear() *Linear {
	return &Linear{
		Weight:           embedding.Weight,
		Scales:           embedding.Scales,
		Biases:           embedding.Biases,
		GroupSize:        embedding.GroupSize,
		Bits:             embedding.Bits,
		QuantizationMode: embedding.QuantizationMode,
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
//	kExpanded := metal.RepeatKV(k, int32(NumQueryHeads/numKVHeads))
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
