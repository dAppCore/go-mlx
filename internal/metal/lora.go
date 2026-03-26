//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"log/slog"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"unsafe"

	"dappco.re/go/core"

	coreio "forge.lthn.ai/core/go-io"
	coreerr "forge.lthn.ai/core/go-log"
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
// dtype: optional training dtype (pass 0 or DTypeFloat32 for default)
func NewLoRALinear(base *Linear, rank int, alpha float32, dtype ...DType) *LoRALinear {
	dt := DTypeFloat32
	if len(dtype) > 0 && dtype[0] != 0 {
		dt = dtype[0]
	}
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
	a := RandomNormal(0, stddev, []int32{int32(rank), inFeatures}, dt)

	// B: zero initialisation — LoRA starts as identity (no change to base)
	b := Zeros([]int32{outFeatures, int32(rank)}, dt)

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
	ta := Transpose(l.A)
	loraOut := Matmul(x, ta)
	Free(ta)

	tb := Transpose(l.B)
	loraOut2 := Matmul(loraOut, tb)
	Free(loraOut, tb)

	loraOut3 := MulScalar(loraOut2, l.Scale)
	Free(loraOut2)

	res := Add(baseOut, loraOut3)
	Free(baseOut, loraOut3)
	return res
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
	Rank       int      // Decomposition rank (default 8)
	Alpha      float32  // Scaling factor (default 16)
	TargetKeys []string // Weight name suffixes to target (default: q_proj, v_proj)
	DType      DType    // Training dtype for A/B (default Float32; use BFloat16 for mixed precision)
}

// DefaultLoRAConfig returns the standard LoRA configuration for LLM fine-tuning.
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
		DType:      DTypeFloat32,
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
	return slices.Sorted(maps.Keys(a.Layers))
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
	out := newArray("RANDOM_NORMAL")
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

// --- Adapter Loading (Inference) ---

// adapterConfig holds the metadata from adapter_config.json produced by mlx-lm training.
type adapterConfig struct {
	Rank       int      `json:"rank"`
	Alpha      float32  `json:"alpha"`
	NumLayers  int      `json:"num_layers"`
	TargetKeys []string `json:"lora_layers"` // e.g. ["self_attn.q_proj", "self_attn.v_proj"]
}

// parseAdapterConfig reads and parses an adapter_config.json file.
func parseAdapterConfig(path string) (*adapterConfig, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, coreerr.E("lora.parseAdapterConfig", "read adapter_config.json", err)
	}
	var cfg adapterConfig
	if r := core.JSONUnmarshal([]byte(str), &cfg); !r.OK {
		return nil, coreerr.E("lora.parseAdapterConfig", "parse adapter_config.json", nil)
	}
	// Apply defaults matching mlx-lm conventions.
	if cfg.Rank == 0 {
		cfg.Rank = 8
	}
	if cfg.Alpha == 0 {
		cfg.Alpha = float32(cfg.Rank) * 2 // mlx-lm default: alpha = 2 * rank
	}
	return &cfg, nil
}

// loadAdapterWeights loads all safetensors files from an adapter directory into a flat weight map.
func loadAdapterWeights(dir string) (map[string]*Array, error) {
	matches := core.PathGlob(core.JoinPath(dir, "*.safetensors"))
	if len(matches) == 0 {
		return nil, coreerr.E("lora.loadAdapterWeights", "no .safetensors files found in "+dir, nil)
	}

	weights := make(map[string]*Array)
	for _, path := range matches {
		for name, arr := range LoadSafetensors(path) {
			weights[name] = arr
		}
		if err := lastError(); err != nil {
			return nil, coreerr.E("lora.loadAdapterWeights", "load adapter weights "+core.PathBase(path), err)
		}
	}
	return weights, nil
}

// resolveLinear returns the *Linear for a given projection path within a model.
// projPath is e.g. "self_attn.q_proj" and the function resolves layer index + field.
func resolveLinear(model InternalModel, layerIdx int, projPath string) *Linear {
	switch m := model.(type) {
	case *Qwen3Model:
		if layerIdx >= len(m.Layers) {
			return nil
		}
		layer := m.Layers[layerIdx]
		switch projPath {
		case "self_attn.q_proj":
			return layer.Attention.QProj
		case "self_attn.k_proj":
			return layer.Attention.KProj
		case "self_attn.v_proj":
			return layer.Attention.VProj
		case "self_attn.o_proj":
			return layer.Attention.OProj
		}
	case *GemmaModel:
		if layerIdx >= len(m.Layers) {
			return nil
		}
		layer := m.Layers[layerIdx]
		switch projPath {
		case "self_attn.q_proj":
			return layer.Attention.QProj
		case "self_attn.k_proj":
			return layer.Attention.KProj
		case "self_attn.v_proj":
			return layer.Attention.VProj
		case "self_attn.o_proj":
			return layer.Attention.OProj
		}
	}
	return nil
}

// parseLoRAWeightName extracts the layer index, projection path, and A/B suffix
// from an adapter weight name. Returns (-1, "", "") if the name is not a recognised
// LoRA weight.
//
// Examples:
//
//	"layers.0.self_attn.q_proj.lora_a" → (0, "self_attn.q_proj", "lora_a")
//	"model.layers.12.self_attn.v_proj.lora_b" → (12, "self_attn.v_proj", "lora_b")
func parseLoRAWeightName(name string) (layerIdx int, projPath, suffix string) {
	// Strip optional "model." prefix.
	name = core.TrimPrefix(name, "model.")

	// Must start with "layers.{N}."
	if !core.HasPrefix(name, "layers.") {
		return -1, "", ""
	}

	// Must end with ".lora_a" or ".lora_b".
	if core.HasSuffix(name, ".lora_a") {
		suffix = "lora_a"
	} else if core.HasSuffix(name, ".lora_b") {
		suffix = "lora_b"
	} else {
		return -1, "", ""
	}

	// Remove "layers." prefix and ".lora_X" suffix.
	inner := name[len("layers."):]
	inner = inner[:len(inner)-len("."+suffix)]

	// Split off the layer index.
	dotIdx := strings.Index(inner, ".")
	if dotIdx < 0 {
		return -1, "", ""
	}
	idxStr := inner[:dotIdx]
	projPath = inner[dotIdx+1:]

	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		return -1, "", ""
	}

	return idx, projPath, suffix
}

// applyLoadedLoRA loads a trained LoRA adapter from disk and injects it into the model
// for inference. The adapter weights are frozen (no gradients needed).
func applyLoadedLoRA(model InternalModel, adapterDir string) error {
	// Step 1: Read adapter configuration.
	cfg, err := parseAdapterConfig(core.JoinPath(adapterDir, "adapter_config.json"))
	if err != nil {
		return err
	}

	// Step 2: Load adapter safetensors.
	weights, err := loadAdapterWeights(adapterDir)
	if err != nil {
		return err
	}

	// Materialise all adapter weights onto Metal.
	var allArrays []*Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	Materialize(allArrays...)

	// Step 3: Group weights by (layerIdx, projPath) and inject LoRA.
	type loraKey struct {
		layerIdx int
		projPath string
	}
	type loraPair struct {
		a *Array
		b *Array
	}
	pairs := make(map[loraKey]*loraPair)

	for name, arr := range weights {
		layerIdx, projPath, suffix := parseLoRAWeightName(name)
		if layerIdx < 0 {
			slog.Warn("adapter: skipping unrecognised weight", "name", name)
			continue
		}
		key := loraKey{layerIdx, projPath}
		pair, ok := pairs[key]
		if !ok {
			pair = &loraPair{}
			pairs[key] = pair
		}
		switch suffix {
		case "lora_a":
			pair.a = arr
		case "lora_b":
			pair.b = arr
		}
	}

	scale := cfg.Alpha / float32(cfg.Rank)
	injected := 0

	for key, pair := range pairs {
		if pair.a == nil || pair.b == nil {
			slog.Warn("adapter: incomplete LoRA pair, skipping",
				"layer", key.layerIdx, "proj", key.projPath)
			continue
		}

		linear := resolveLinear(model, key.layerIdx, key.projPath)
		if linear == nil {
			slog.Warn("adapter: target layer not found, skipping",
				"layer", key.layerIdx, "proj", key.projPath)
			continue
		}

		lora := &LoRALinear{
			Base:  linear,
			A:     pair.a,
			B:     pair.b,
			Scale: scale,
			Rank:  cfg.Rank,
			Alpha: cfg.Alpha,
		}
		linear.LoRA = lora
		injected++
	}

	if injected == 0 {
		return coreerr.E("lora.applyLoadedLoRA", "no LoRA layers injected from "+adapterDir, nil)
	}

	slog.Info("adapter loaded",
		"path", adapterDir,
		"rank", cfg.Rank,
		"alpha", cfg.Alpha,
		"scale", scale,
		"layers_injected", injected,
	)
	return nil
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
		if err := lastError(); err != nil {
			return err
		}
		return coreerr.E("mlx.SaveSafetensors", "save safetensors failed: "+path, nil)
	}
	return nil
}
