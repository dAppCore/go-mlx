//go:build darwin && arm64

package mlx

import (
	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-mlx/internal/metal"
)

// Re-export training types from internal/metal for direct use by go-ml.

// Array is a Metal GPU tensor.
type Array = metal.Array

// LoRAAdapter holds all LoRA layers applied to a model.
type LoRAAdapter = metal.LoRAAdapter

// LoRAConfig specifies which layers to apply LoRA to and with what parameters.
type LoRAConfig = metal.LoRAConfig

// DefaultLoRAConfig returns the standard LoRA configuration for LLM fine-tuning.
var DefaultLoRAConfig = metal.DefaultLoRAConfig

// GradFn computes both loss values and gradients via reverse-mode autodiff.
type GradFn = metal.GradFn

// AdamW is the decoupled weight decay optimiser.
type AdamW = metal.AdamW

// Cache is a per-layer KV cache.
type Cache = metal.Cache

// DType represents a Metal array data type.
type DType = metal.DType

// InternalModel is the training-level model interface with Forward/NewCache.
type InternalModel = metal.InternalModel

// Common DType constants.
var (
	DTypeFloat32  = metal.DTypeFloat32
	DTypeBFloat16 = metal.DTypeBFloat16
)

// --- Training Primitives ---

// ValueAndGrad creates a GradFn that computes both the function value and
// gradients with respect to the arguments at the given indices.
//
//	lossFn := func(params []*Array) []*Array { return []*Array{loss} }
//	grad := mlx.ValueAndGrad(lossFn, 0, 1, 2)
//	values, grads, err := grad.Apply(params...)
func ValueAndGrad(fn func([]*Array) []*Array, argnums ...int) *GradFn {
	return metal.ValueAndGrad(fn, argnums...)
}

// NewAdamW creates an AdamW optimiser with default hyperparameters.
func NewAdamW(lr float64) *AdamW { return metal.NewAdamW(lr) }

// CrossEntropyLoss computes cross-entropy loss between logits and integer targets.
// Returns scalar loss averaged over all positions.
func CrossEntropyLoss(logits, targets *Array) *Array {
	return metal.CrossEntropyLoss(logits, targets)
}

// MaskedCrossEntropyLoss computes cross-entropy loss only on masked positions.
// mask: 1.0 = compute loss, 0.0 = ignore. Averaged over masked positions.
func MaskedCrossEntropyLoss(logits, targets, mask *Array) *Array {
	return metal.MaskedCrossEntropyLoss(logits, targets, mask)
}

// Checkpoint wraps a function for memory-efficient gradient recomputation.
// During backward, intermediates are recomputed rather than stored.
func Checkpoint(fn func([]*Array) []*Array) func([]*Array) []*Array {
	return metal.Checkpoint(fn)
}

// FromValues creates a Metal Array from a Go slice with the given shape.
func FromValues[S ~[]E, E metal.ArrayElement](s S, shape ...int) *metal.Array {
	return metal.FromValues(s, shape...)
}

// Materialize forces evaluation of lazy Metal arrays.
func Materialize(arrays ...*Array) { metal.Materialize(arrays...) }

// Free releases Metal arrays immediately without waiting for GC.
func Free(arrays ...*Array) { metal.Free(arrays...) }

// Zeros creates an array of zeros with the given shape and dtype.
func Zeros(shape []int32, dtype metal.DType) *Array { return metal.Zeros(shape, dtype) }

// --- metalAdapter training methods ---
// These make metalAdapter satisfy inference.TrainableModel.

func (a *metalAdapter) ApplyLoRA(cfg inference.LoRAConfig) inference.Adapter {
	mcfg := metal.LoRAConfig{
		Rank:       cfg.Rank,
		Alpha:      cfg.Alpha,
		TargetKeys: cfg.TargetKeys,
	}
	if mcfg.Rank == 0 {
		mcfg.Rank = 8
	}
	if mcfg.Alpha == 0 {
		mcfg.Alpha = 16
	}
	if len(mcfg.TargetKeys) == 0 {
		mcfg.TargetKeys = []string{"q_proj", "v_proj"}
	}
	if cfg.BFloat16 {
		mcfg.DType = metal.DTypeBFloat16
	}
	return a.m.ApplyLoRA(mcfg)
}

func (a *metalAdapter) Encode(text string) []int32 {
	return a.m.Encode(text)
}

func (a *metalAdapter) Decode(ids []int32) string {
	return a.m.Decode(ids)
}

func (a *metalAdapter) NumLayers() int {
	return a.m.NumLayers()
}

// InternalModel returns the underlying metal.InternalModel for direct training access.
// This gives go-ml access to Forward(), NewCache(), and concrete LoRA operations
// without wrapping everything in opaque types.
func (a *metalAdapter) InternalModel() metal.InternalModel {
	return a.m.Internal()
}

// ConcreteAdapter returns the concrete *LoRAAdapter from an inference.Adapter.
// Panics if the adapter is not from this backend.
func ConcreteAdapter(a inference.Adapter) *LoRAAdapter {
	return a.(*LoRAAdapter)
}

// TrainingModel returns the InternalModel from a TrainableModel loaded via the
// Metal backend. This gives go-ml direct access to Forward() and NewCache()
// for the training loop.
// Panics if the model is not from the Metal backend.
func TrainingModel(tm inference.TrainableModel) InternalModel {
	return tm.(*metalAdapter).InternalModel()
}
