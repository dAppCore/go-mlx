//go:build darwin && arm64 && !nomlx

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
)

// Array is a Metal GPU tensor.
type Array = metal.Array

// LoRAAdapter holds all LoRA layers applied to a model.
type LoRAAdapter = metal.LoRAAdapter

// LoRAConfig specifies which layers to apply LoRA to and with what parameters.
type LoRAConfig = metal.LoRAConfig

// Batch describes one RFC-style training batch.
type Batch = metal.Batch

// TrainConfig holds RFC-style training loop settings.
type TrainConfig = metal.TrainConfig

// DefaultLoRAConfig returns the standard LoRA configuration for LLM fine-tuning.
//
//	config := mlx.DefaultLoRAConfig() // rank=8, alpha=16, targets=[q_proj, v_proj]
var DefaultLoRAConfig = metal.DefaultLoRAConfig

// DefaultAdamWConfig returns the standard AdamW hyperparameters.
var DefaultAdamWConfig = metal.DefaultAdamWConfig

// GradFn computes both loss values and gradients via reverse-mode autodiff.
type GradFn = metal.GradFn

// AdamW is the decoupled weight decay optimiser.
type AdamW = metal.AdamW

// AdamWConfig configures AdamW construction.
type AdamWConfig = metal.AdamWConfig

// Cache is a per-layer KV cache.
type Cache = metal.Cache

// DType represents a Metal array data type.
type DType = metal.DType

// InternalModel is the training-level model interface with Forward/NewCache.
//
//	internalModel := mlx.TrainingModel(trainableModel)
//	logits := internalModel.Forward(tokens, caches)
type InternalModel = metal.InternalModel

var (
	DTypeFloat32  = metal.DTypeFloat32
	DTypeBFloat16 = metal.DTypeBFloat16
)

// ValueAndGrad creates a GradFn that computes both the function value and
// gradients with respect to the arguments at the given indices.
//
//	lossFunction := func(parameters []*Array) []*Array { return []*Array{loss} }
//	grad := mlx.ValueAndGrad(lossFunction, 0, 1, 2)
//	values, grads, err := grad.Apply(parameters...)
func ValueAndGrad(lossFunction func([]*Array) []*Array, argumentIndices ...int) *GradFn {
	return metal.ValueAndGrad(lossFunction, argumentIndices...)
}

// NewAdamW creates an AdamW optimiser with default hyperparameters.
//
//	optimizer := mlx.NewAdamW(1e-4)
//	optimizer := mlx.NewAdamW(&mlx.AdamWConfig{LearningRate: 1e-4, Beta1: 0.85})
func NewAdamW(config any) *AdamW { return metal.NewAdamW(config) }

// CrossEntropyLoss computes cross-entropy loss between logits and integer targets.
//
//	loss := mlx.CrossEntropyLoss(logits, targets) // logits: [B, L, V], targets: [B, L]
func CrossEntropyLoss(logits, targets *Array) *Array {
	return metal.CrossEntropyLoss(logits, targets)
}

// MaskedCrossEntropyLoss computes cross-entropy loss only on masked positions.
//
//	loss := mlx.MaskedCrossEntropyLoss(logits, targets, mask) // mask: 1.0 = include, 0.0 = ignore
func MaskedCrossEntropyLoss(logits, targets, mask *Array) *Array {
	return metal.MaskedCrossEntropyLoss(logits, targets, mask)
}

// Checkpoint wraps a function for memory-efficient gradient recomputation.
//
//	checkpointedBlock := mlx.Checkpoint(func(hidden []*Array) []*Array {
//	    return []*Array{decoder.Forward(hidden[0])}
//	})
func Checkpoint(forwardPass func([]*Array) []*Array) func([]*Array) []*Array {
	return metal.Checkpoint(forwardPass)
}

// FromValues creates a Metal Array from a Go slice with the given shape.
//
//	tokens := mlx.FromValues([]int32{1, 2, 3}, 1, 3) // [1, L] token tensor
func FromValues[S ~[]E, E metal.ArrayElement](values S, shape ...int) *Array {
	return metal.FromValues(values, shape...)
}

// Materialize forces evaluation of lazy Metal arrays.
//
//	mlx.Materialize(firstOutput, secondOutput, thirdOutput) // block until GPU eval completes
func Materialize(arrays ...*Array) { metal.Materialize(arrays...) }

// Free releases Metal arrays immediately without waiting for GC.
//
//	mlx.Free(embeddingOutput, hiddenState, previousLogits) // explicit release after each decode step
func Free(arrays ...*Array) { metal.Free(arrays...) }

// Zeros creates an array of zeros with the given shape and dtype.
//
//	zeroMatrix := mlx.Zeros([]int32{outFeatures, rank}, mlx.DTypeFloat32) // zero-init LoRA B matrix
func Zeros(shape []int32, dtype metal.DType) *Array { return metal.Zeros(shape, dtype) }

func (adapter *metalAdapter) ApplyLoRA(config inference.LoRAConfig) inference.Adapter {
	mcfg := metal.LoRAConfig{
		Rank:       config.Rank,
		Alpha:      config.Alpha,
		TargetKeys: config.TargetKeys,
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
	if config.BFloat16 {
		mcfg.DType = metal.DTypeBFloat16
	}
	return adapter.model.ApplyLoRA(mcfg)
}

func (adapter *metalAdapter) Encode(text string) []int32 {
	return adapter.model.Encode(text)
}

func (adapter *metalAdapter) Decode(tokenIDs []int32) string {
	return adapter.model.Decode(tokenIDs)
}

func (adapter *metalAdapter) NumLayers() int {
	return adapter.model.NumLayers()
}

func (adapter *metalAdapter) InternalModel() metal.InternalModel {
	return adapter.model.Internal()
}

// ConcreteAdapter returns the concrete *LoRAAdapter from an inference.Adapter.
// Panics if the adapter is not from the Metal backend.
//
//	loraAdapter := mlx.ConcreteAdapter(adapter)
//	trainableParameters := loraAdapter.AllTrainableParams()
func ConcreteAdapter(adapter inference.Adapter) *LoRAAdapter {
	return adapter.(*LoRAAdapter)
}

// TrainingModel returns the InternalModel from a Metal-loaded TrainableModel.
// Gives direct access to Forward() and NewCache() for the training loop.
// Panics if the model is not from the Metal backend.
//
//	internalModel := mlx.TrainingModel(trainableModel)
//	logits := internalModel.Forward(tokens, caches)
func TrainingModel(trainableModel inference.TrainableModel) InternalModel {
	return trainableModel.(*metalAdapter).InternalModel()
}
