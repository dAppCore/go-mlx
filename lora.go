//go:build darwin && arm64

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"math"
	"sort"
	"unsafe"
)

// LoRALinear wraps a frozen Linear layer with low-rank trainable adapters.
//
// Forward: base(x) + scale * (x @ A^T) @ B^T
//
// A is [rank, in_features]  — initialised with Kaiming normal
// B is [out_features, rank] — initialised to zero
// Scale = alpha / rank
//
// Only A and B are trainable. The base weights stay frozen.
type LoRALinear struct {
	Base  *Linear // Frozen base weights (may be quantized)
	A     *Array  // [rank, in_features] — trainable
	B     *Array  // [out_features, rank] — trainable
	Scale float32 // alpha / rank
	Rank  int
	Alpha float32
}

// NewLoRALinear wraps an existing Linear layer with LoRA adapters.
// rank: decomposition rank (typically 4, 8, or 16)
// alpha: scaling factor (typically 2*rank)
func NewLoRALinear(base *Linear, rank int, alpha float32) *LoRALinear {
	// Determine dimensions from the base weight.
	// Weight shape is [out_features, in_features] for standard linear,
	// or quantized shape which we handle via the base layer.
	var inFeatures, outFeatures int32

	if base.Scales != nil {
		// Quantized: weight is packed. Compute dims from scales.
		// scales shape is [out_features, in_features / group_size]
		scaleShape := base.Scales.Shape()
		outFeatures = scaleShape[0]
		inFeatures = scaleShape[1] * int32(base.GroupSize)
	} else {
		wShape := base.Weight.Shape()
		outFeatures = wShape[0]
		inFeatures = wShape[1]
	}

	// A: Kaiming normal initialisation — N(0, 1/sqrt(in_features))
	stddev := float32(1.0 / math.Sqrt(float64(inFeatures)))
	a := RandomNormal(0, stddev, []int32{int32(rank), inFeatures}, DTypeFloat32)

	// B: zero initialisation — LoRA starts as identity (no change to base)
	b := Zeros([]int32{outFeatures, int32(rank)}, DTypeFloat32)

	Materialize(a, b)

	return &LoRALinear{
		Base:  base,
		A:     a,
		B:     b,
		Scale: alpha / float32(rank),
		Rank:  rank,
		Alpha: alpha,
	}
}

// Forward computes: base(x) + scale * (x @ A^T) @ B^T
// Calls baseForward on the underlying Linear to avoid infinite recursion
// when the Linear's Forward method dispatches through LoRA.
func (l *LoRALinear) Forward(x *Array) *Array {
	baseOut := l.Base.baseForward(x)

	// LoRA path: x @ A^T gives [B, L, rank], then @ B^T gives [B, L, out]
	loraOut := Matmul(x, Transpose(l.A))
	loraOut = Matmul(loraOut, Transpose(l.B))
	loraOut = MulScalar(loraOut, l.Scale)

	return Add(baseOut, loraOut)
}

// TrainableParams returns the LoRA A and B arrays for gradient computation.
func (l *LoRALinear) TrainableParams() []*Array {
	return []*Array{l.A, l.B}
}

// SetParams updates the LoRA A and B arrays (used by optimiser after gradient step).
func (l *LoRALinear) SetParams(a, b *Array) {
	l.A = a
	l.B = b
}

// ParamCount returns the number of trainable parameters.
func (l *LoRALinear) ParamCount() int {
	aShape := l.A.Shape()
	bShape := l.B.Shape()
	return int(aShape[0]*aShape[1] + bShape[0]*bShape[1])
}

// --- LoRA Application to Models ---

// LoRAConfig specifies which layers to apply LoRA to and with what parameters.
type LoRAConfig struct {
	Rank       int     // Decomposition rank (default 8)
	Alpha      float32 // Scaling factor (default 16)
	TargetKeys []string // Weight name suffixes to target (default: q_proj, v_proj)
}

// DefaultLoRAConfig returns the standard LoRA configuration for LLM fine-tuning.
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
}

// LoRAAdapter holds all LoRA layers applied to a model.
type LoRAAdapter struct {
	Layers map[string]*LoRALinear // keyed by weight path prefix
	Config LoRAConfig
}

// TotalParams returns the total number of trainable parameters across all LoRA layers.
func (a *LoRAAdapter) TotalParams() int {
	total := 0
	for _, l := range a.Layers {
		total += l.ParamCount()
	}
	return total
}

// SortedNames returns layer names in deterministic sorted order.
func (a *LoRAAdapter) SortedNames() []string {
	names := make([]string, 0, len(a.Layers))
	for name := range a.Layers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// AllTrainableParams returns all trainable arrays (A and B from every layer),
// in a deterministic order sorted by layer name.
func (a *LoRAAdapter) AllTrainableParams() []*Array {
	names := a.SortedNames()
	params := make([]*Array, 0, len(names)*2)
	for _, name := range names {
		l := a.Layers[name]
		params = append(params, l.A, l.B)
	}
	return params
}

// SetAllParams updates all LoRA A and B arrays from a flat slice,
// in the same deterministic order as AllTrainableParams.
func (a *LoRAAdapter) SetAllParams(params []*Array) {
	names := a.SortedNames()
	for i, name := range names {
		l := a.Layers[name]
		l.A = params[i*2]
		l.B = params[i*2+1]
	}
}

// Save writes the LoRA adapter weights to a safetensors file.
// Only saves the A and B matrices — not the frozen base weights.
func (a *LoRAAdapter) Save(path string) error {
	weights := make(map[string]*Array)
	for name, l := range a.Layers {
		weights[name+".lora_a"] = l.A
		weights[name+".lora_b"] = l.B
	}
	return SaveSafetensors(path, weights)
}

// --- Random Normal ---

// RandomNormal generates normal (Gaussian) random values with given mean and stddev.
func RandomNormal(mean, stddev float32, shape []int32, dtype DType) *Array {
	Init()
	out := New("RANDOM_NORMAL")
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	key := C.mlx_array_new()
	defer C.mlx_array_free(key)
	C.mlx_random_normal(
		&out.ctx,
		&cShape[0], C.size_t(len(cShape)),
		C.mlx_dtype(dtype),
		C.float(mean), C.float(stddev),
		key, // null key = use default RNG
		DefaultStream().ctx,
	)
	return out
}

// --- SaveSafetensors ---

// SaveSafetensors saves a map of named arrays to a .safetensors file.
func SaveSafetensors(path string, weights map[string]*Array) error {
	Init()

	// Build the map
	cMap := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(cMap)

	for name, arr := range weights {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(cMap, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	// Empty metadata
	cMeta := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(cMeta)

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.mlx_save_safetensors(cPath, cMap, cMeta)
	if rc != 0 {
		checkError()
		return fmt.Errorf("mlx: save safetensors failed: %s", path)
	}
	return nil
}
