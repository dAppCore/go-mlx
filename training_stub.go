//go:build !(darwin && arm64) || nomlx

package mlx

import (
	// Note: AX-6 - iter.Seq is the public Array.Iter contract; core has no iterator alias.
	"iter"

	"dappco.re/go/core"
	"dappco.re/go/inference"
)

func unsupportedBuildError() error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// Array is a stub tensor on unsupported builds.
type Array struct {
	shape []int32
	dtype DType
}

// DType is a stub array dtype on unsupported builds.
type DType uint8

const (
	dtypeUnknown DType = iota
	dtypeFloat32
	dtypeBFloat16
)

func (d DType) String() string {
	switch d {
	case dtypeFloat32:
		return "float32"
	case dtypeBFloat16:
		return "bfloat16"
	default:
		return "unknown"
	}
}

// LoRAAdapter holds stub adapter metadata on unsupported builds.
type LoRAAdapter struct {
	Config LoRAConfig
}

// LoRAConfig mirrors the supported-build LoRA config shape.
type LoRAConfig struct {
	Rank         int
	Alpha        float32
	Scale        float32
	TargetKeys   []string
	TargetLayers []string
	Lambda       float32
	DType        DType
}

// Batch describes one RFC-style training batch.
type Batch struct {
	Tokens [][]int
	Length []int
}

// TrainConfig holds RFC-style training loop settings.
type TrainConfig struct {
	Epochs         int
	BatchSize      int
	LearningRate   float64
	EvalInterval   int
	SaveInterval   int
	EvalLossThresh float64
}

// AdamW is a stub optimiser on unsupported builds.
type AdamW struct{}

// AdamWConfig mirrors the supported-build config shape.
type AdamWConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Eps          float64
	WeightDecay  float64
}

// GradFn is a stub autodiff handle on unsupported builds.
type GradFn struct{}

// Cache mirrors the supported-build cache interface.
type Cache interface {
	Update(k, v *Array, seqLen int) (*Array, *Array)
	Offset() int
	Len() int
	State() []*Array
	Reset()
	Detach()
}

// InternalModel mirrors the supported-build training interface.
type InternalModel interface {
	Forward(tokens *Array, caches []Cache) *Array
	ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array
	NewCache() []Cache
	NumLayers() int
	Tokenizer() *Tokenizer
	ModelType() string
	ApplyLoRA(cfg LoRAConfig) *LoRAAdapter
}

var (
	// DTypeFloat32 is the float32 array dtype.
	DTypeFloat32 = dtypeFloat32
	// DTypeBFloat16 is the bfloat16 array dtype.
	DTypeBFloat16 = dtypeBFloat16

	// DefaultLoRAConfig returns the standard LoRA configuration.
	DefaultLoRAConfig = func() LoRAConfig {
		return LoRAConfig{
			Rank:         8,
			Alpha:        16,
			Scale:        2,
			TargetKeys:   []string{"q_proj", "v_proj"},
			TargetLayers: []string{"q_proj", "v_proj"},
			DType:        DTypeFloat32,
		}
	}

	// DefaultAdamWConfig returns the standard AdamW hyperparameters.
	DefaultAdamWConfig = func() AdamWConfig {
		return AdamWConfig{
			LearningRate: 1e-5,
			Beta1:        0.9,
			Beta2:        0.999,
			Eps:          1e-8,
			WeightDecay:  0.01,
		}
	}
)

func cloneShape(shape []int32) []int32 {
	if len(shape) == 0 {
		return nil
	}
	return append([]int32(nil), shape...)
}

func newStubArray(shape []int32, dtype DType) *Array {
	return &Array{shape: cloneShape(shape), dtype: dtype}
}

// Set replaces the stub array metadata with another array's metadata.
func (a *Array) Set(other *Array) {
	if a == nil {
		return
	}
	if other == nil {
		a.shape = nil
		a.dtype = 0
		return
	}
	a.shape = cloneShape(other.shape)
	a.dtype = other.dtype
}

// Clone returns a shallow stub copy.
func (a *Array) Clone() *Array {
	if a == nil {
		return nil
	}
	return newStubArray(a.shape, a.dtype)
}

// Valid reports whether the stub array is non-nil.
func (a *Array) Valid() bool { return a != nil }

// String returns a short stub description.
func (a *Array) String() string { return "mlx.Array(unavailable)" }

// Shape returns the recorded stub shape.
func (a *Array) Shape() []int32 {
	if a == nil {
		return nil
	}
	return cloneShape(a.shape)
}

// NumDims returns the number of dimensions in the recorded shape.
func (a *Array) NumDims() int {
	if a == nil {
		return 0
	}
	return len(a.shape)
}

// Dim returns the size of dimension i or zero when unavailable.
func (a *Array) Dim(i int) int {
	if a == nil || i < 0 || i >= len(a.shape) {
		return 0
	}
	return int(a.shape[i])
}

// Dims returns the recorded dimensions as ints.
func (a *Array) Dims() []int {
	if a == nil {
		return nil
	}
	dims := make([]int, len(a.shape))
	for i, dim := range a.shape {
		dims[i] = int(dim)
	}
	return dims
}

// Dtype returns the recorded stub dtype.
func (a *Array) Dtype() DType {
	if a == nil {
		return 0
	}
	return a.dtype
}

// Int returns zero on unsupported builds.
func (a *Array) Int() int { return 0 }

// Float returns zero on unsupported builds.
func (a *Array) Float() float64 { return 0 }

// Bool returns false on unsupported builds.
func (a *Array) Bool() bool { return false }

// SetFloat64 is a no-op on unsupported builds.
func (a *Array) SetFloat64(_ float64) {}

// Ints returns nil on unsupported builds.
func (a *Array) Ints() []int { return nil }

// DataInt32 returns nil on unsupported builds.
func (a *Array) DataInt32() []int32 { return nil }

// Floats returns nil on unsupported builds.
func (a *Array) Floats() []float32 { return nil }

// Iter yields no values on unsupported builds.
func (a *Array) Iter() iter.Seq[float32] {
	return func(func(float32) bool) {}
}

// TotalParams reports zero on unsupported builds.
func (adapter *LoRAAdapter) TotalParams() int { return 0 }

// SortedNames reports no layer names on unsupported builds.
func (adapter *LoRAAdapter) SortedNames() []string { return nil }

// AllTrainableParams reports no trainable arrays on unsupported builds.
func (adapter *LoRAAdapter) AllTrainableParams() []*Array { return nil }

// SetAllParams is a no-op on unsupported builds.
func (adapter *LoRAAdapter) SetAllParams(_ []*Array) {}

// Step returns nil on unsupported builds.
func (adapter *LoRAAdapter) Step(_ Batch, _ [][]int, _ *AdamW) *Array { return nil }

// Save returns an availability error on unsupported builds.
func (adapter *LoRAAdapter) Save(_ string) error { return unsupportedBuildError() }

// Merge is a no-op on unsupported builds.
func (adapter *LoRAAdapter) Merge() {}

// Step returns the input parameters unchanged on unsupported builds.
func (optimizer *AdamW) Step(parameters []*Array, _ []*Array) []*Array { return parameters }

// Reset is a no-op on unsupported builds.
func (optimizer *AdamW) Reset() {}

// Apply returns an availability error on unsupported builds.
func (g *GradFn) Apply(_ ...*Array) (values []*Array, grads []*Array, err error) {
	return nil, nil, unsupportedBuildError()
}

// Free is a no-op on unsupported builds.
func (g *GradFn) Free() {}

// ValueAndGrad creates a stub GradFn.
func ValueAndGrad(_ func([]*Array) []*Array, _ ...int) *GradFn { return &GradFn{} }

// NewAdamW creates a stub AdamW.
func NewAdamW(_ any) *AdamW { return &AdamW{} }

// CrossEntropyLoss returns nil on unsupported builds.
func CrossEntropyLoss(_, _ *Array) *Array { return nil }

// MaskedCrossEntropyLoss returns nil on unsupported builds.
func MaskedCrossEntropyLoss(_, _, _ *Array) *Array { return nil }

// Checkpoint returns the original function on unsupported builds.
func Checkpoint(forwardPass func([]*Array) []*Array) func([]*Array) []*Array {
	return forwardPass
}

type stubArrayElement interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}

// FromValues records shape metadata only on unsupported builds.
func FromValues[S ~[]E, E stubArrayElement](_ S, shape ...int) *Array {
	out := make([]int32, len(shape))
	for i, dim := range shape {
		out[i] = int32(dim)
	}
	return newStubArray(out, DTypeFloat32)
}

// Materialize is a no-op on unsupported builds.
func Materialize(_ ...*Array) {}

// Free is a no-op on unsupported builds.
func Free(_ ...*Array) {}

// Zeros records shape metadata only on unsupported builds.
func Zeros(shape []int32, dtype DType) *Array { return newStubArray(shape, dtype) }

// MatMul returns a stub array using the left-hand shape when available.
func MatMul(a, _ *Array) *Array {
	if a == nil {
		return nil
	}
	return a.Clone()
}

// Add returns a stub array using the left-hand shape when available.
func Add(a, b *Array) *Array {
	if a != nil {
		return a.Clone()
	}
	if b != nil {
		return b.Clone()
	}
	return nil
}

// Mul returns a stub array using the left-hand shape when available.
func Mul(a, b *Array) *Array { return Add(a, b) }

// Softmax returns a stub clone on unsupported builds.
func Softmax(a *Array) *Array {
	if a == nil {
		return nil
	}
	return a.Clone()
}

// Slice records an updated size along the requested axis when possible.
func Slice(a *Array, start, end, axis any) *Array {
	if a == nil {
		return nil
	}
	out := a.Clone()
	axisInt := normalizeRootIntArg("axis", axis)
	startInt := normalizeRootInt32Arg("start", start)
	endInt := normalizeRootInt32Arg("end", end)
	if axisInt >= 0 && axisInt < len(out.shape) && endInt >= startInt {
		out.shape[axisInt] = endInt - startInt
	}
	return out
}

// Reshape records the requested shape.
func Reshape(a *Array, shape ...any) *Array {
	dtype := DTypeFloat32
	if a != nil {
		dtype = a.dtype
	}
	return newStubArray(normalizeRootShapeArgs(shape), dtype)
}

// VJP returns an availability error on unsupported builds.
func VJP(_ func([]*Array) []*Array, _ []*Array, _ []*Array) (outputs []*Array, vjps []*Array, err error) {
	return nil, nil, unsupportedBuildError()
}

// JVP returns an availability error on unsupported builds.
func JVP(_ func([]*Array) []*Array, _ []*Array, _ []*Array) (outputs []*Array, jvps []*Array, err error) {
	return nil, nil, unsupportedBuildError()
}

// ConcreteAdapter returns nil on unsupported builds.
func ConcreteAdapter(_ inference.Adapter) *LoRAAdapter { return nil }

// TrainingModel returns nil on unsupported builds.
func TrainingModel(_ inference.TrainableModel) InternalModel { return nil }
