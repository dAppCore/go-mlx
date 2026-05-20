// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "math"

// AdamW implements the AdamW optimiser (Adam with decoupled weight decay).
//
// Update rule per parameter:
//
//	m = beta1 * m + (1 - beta1) * grad
//	v = beta2 * v + (1 - beta2) * grad^2
//	m_hat = m / (1 - beta1^t)
//	v_hat = v / (1 - beta2^t)
//	param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
type AdamW struct {
	LR          float64 // Learning rate (default 1e-5)
	Beta1       float64 // First moment decay (default 0.9)
	Beta2       float64 // Second moment decay (default 0.999)
	Eps         float64 // Numerical stability (default 1e-8)
	WeightDecay float64 // Decoupled weight decay (default 0.01)
	PackedState bool    // Store moments in contiguous slabs when parameter layout permits.

	step int      // Number of updates performed
	m    []*Array // First moment estimates (positional, parallel to params)
	v    []*Array // Second moment estimates (positional, parallel to params)

	packed *adamWPackedState
}

// AdamWConfig configures AdamW optimiser construction.
type AdamWConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Eps          float64
	WeightDecay  float64
	PackedState  bool

	LearningRateSet bool
	Beta1Set        bool
	Beta2Set        bool
	EpsSet          bool
	WeightDecaySet  bool
	PackedStateSet  bool
}

// DefaultAdamWConfig returns the standard AdamW hyperparameters.
func DefaultAdamWConfig() AdamWConfig {
	return AdamWConfig{
		LearningRate: 1e-5,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.01,
		PackedState:  true,
	}
}

// NewAdamW creates an AdamW optimiser with default hyperparameters.
//
//	optimizer := metal.NewAdamW(1e-4)
//	optimizer := metal.NewAdamW(&AdamWConfig{LearningRate: 1e-4, Beta1: 0.85})
func NewAdamW(config any) *AdamW {
	cfg := DefaultAdamWConfig()
	switch v := config.(type) {
	case nil:
	case float64:
		cfg.LearningRate = v
	case float32:
		cfg.LearningRate = float64(v)
	case int:
		cfg.LearningRate = float64(v)
	case int32:
		cfg.LearningRate = float64(v)
	case int64:
		cfg.LearningRate = float64(v)
	case AdamWConfig:
		cfg = mergeAdamWConfig(cfg, v)
	case *AdamWConfig:
		if v != nil {
			cfg = mergeAdamWConfig(cfg, *v)
		}
	default:
		panic("metal.NewAdamW: unsupported config type")
	}
	return &AdamW{
		LR:          cfg.LearningRate,
		Beta1:       cfg.Beta1,
		Beta2:       cfg.Beta2,
		Eps:         cfg.Eps,
		WeightDecay: cfg.WeightDecay,
		PackedState: cfg.PackedState,
	}
}

func mergeAdamWConfig(defaults AdamWConfig, override AdamWConfig) AdamWConfig {
	cfg := defaults
	if override.LearningRate != 0 || override.LearningRateSet {
		cfg.LearningRate = override.LearningRate
	}
	if override.Beta1 != 0 || override.Beta1Set {
		cfg.Beta1 = override.Beta1
	}
	if override.Beta2 != 0 || override.Beta2Set {
		cfg.Beta2 = override.Beta2
	}
	if override.Eps != 0 || override.EpsSet {
		cfg.Eps = override.Eps
	}
	if override.WeightDecay != 0 || override.WeightDecaySet {
		cfg.WeightDecay = override.WeightDecay
	}
	if override.PackedState || override.PackedStateSet {
		cfg.PackedState = override.PackedState
	}
	return cfg
}

type adamWPackedParam struct {
	start int32
	end   int32
	shape []int32
}

type adamWPackedState struct {
	m      *Array
	v      *Array
	dtype  DType
	layout []adamWPackedParam
}

// Step performs one optimisation step: updates parameters using gradients.
// Parameters and gradients must be parallel slices of the same length.
// Returns the updated parameter arrays (parameters are replaced in-place).
//
//	parameters = optimizer.Step(parameters, gradients) // one Adam step per mini-batch
func (optimizer *AdamW) Step(parameters []*Array, gradients []*Array) []*Array {
	optimizer.step++
	packed := optimizer.ensurePackedState(parameters)

	// Bias correction factors: compensate for zero-initialised moments.
	biasCorrection1 := 1.0 - math.Pow(optimizer.Beta1, float64(optimizer.step))
	biasCorrection2 := 1.0 - math.Pow(optimizer.Beta2, float64(optimizer.step))

	updated := make([]*Array, len(parameters))

	// Grow moment slices if needed (first call or param count increased)
	for len(optimizer.m) < len(parameters) {
		optimizer.m = append(optimizer.m, nil)
		optimizer.v = append(optimizer.v, nil)
	}

	var nextM, nextV []*Array
	if packed {
		nextM = make([]*Array, len(parameters))
		nextV = make([]*Array, len(parameters))
	}

	for i, parameter := range parameters {
		gradient := gradients[i]

		// Initialise moments on first use
		if optimizer.m[i] == nil {
			shape := parameter.Shape()
			optimizer.m[i] = Zeros(shape, parameter.Dtype())
			optimizer.v[i] = Zeros(shape, parameter.Dtype())
		}
		oldM := optimizer.m[i]
		oldV := optimizer.v[i]

		// m = beta1 * m + (1 - beta1) * grad
		scaledM := MulScalar(oldM, float32(optimizer.Beta1))
		scaledGrad := MulScalar(gradient, float32(1.0-optimizer.Beta1))
		m := Add(scaledM, scaledGrad)
		Free(scaledM, scaledGrad)

		// v = beta2 * v + (1 - beta2) * grad^2
		gradSquared := Square(gradient)
		scaledV := MulScalar(oldV, float32(optimizer.Beta2))
		scaledGradSquared := MulScalar(gradSquared, float32(1.0-optimizer.Beta2))
		v := Add(scaledV, scaledGradSquared)
		Free(gradSquared, scaledV, scaledGradSquared)

		// Bias-corrected estimates
		mHat := MulScalar(m, float32(1.0/biasCorrection1))
		vHat := MulScalar(v, float32(1.0/biasCorrection2))

		// Weight decay: param = param * (1 - lr * weight_decay)
		decayed := MulScalar(parameter, float32(1.0-optimizer.LR*optimizer.WeightDecay))

		// Update: param = decayed - lr * m_hat / (sqrt(v_hat) + eps)
		sqrtVHat := Sqrt(vHat)
		denom := AddScalar(sqrtVHat, float32(optimizer.Eps))
		stepBase := Divide(mHat, denom)
		step := MulScalar(stepBase, float32(optimizer.LR))
		newParam := Subtract(decayed, step)
		Free(mHat, vHat, decayed, sqrtVHat, denom, stepBase, step)

		// Store updated moments
		if packed {
			nextM[i] = m
			nextV[i] = v
		} else {
			optimizer.m[i] = m
			optimizer.v[i] = v
			Free(oldM, oldV)
		}

		updated[i] = newParam
	}

	if packed {
		optimizer.replacePackedMoments(nextM, nextV)
	}

	return updated
}

// Reset clears the optimiser state (moments and step counter).
//
//	optimizer.Reset() // start a new training run from scratch
func (optimizer *AdamW) Reset() {
	Free(optimizer.m...)
	Free(optimizer.v...)
	if optimizer.packed != nil {
		Free(optimizer.packed.m, optimizer.packed.v)
		optimizer.packed = nil
	}
	optimizer.step = 0
	optimizer.m = nil
	optimizer.v = nil
}

func (optimizer *AdamW) ensurePackedState(parameters []*Array) bool {
	if optimizer == nil || !optimizer.PackedState {
		optimizer.releasePackedStateOnly()
		return false
	}
	layout, dtype, ok := adamWPackedLayout(parameters)
	if !ok {
		optimizer.releasePackedStateOnly()
		return false
	}
	if optimizer.packed != nil && adamWPackedLayoutEqual(optimizer.packed.layout, layout) && optimizer.packed.dtype == dtype {
		if len(optimizer.m) == len(layout) && len(optimizer.v) == len(layout) {
			return true
		}
		Free(optimizer.m...)
		Free(optimizer.v...)
		optimizer.m, optimizer.v = optimizer.packed.views()
		return true
	}

	Free(optimizer.m...)
	Free(optimizer.v...)
	if optimizer.packed != nil {
		Free(optimizer.packed.m, optimizer.packed.v)
	}
	total := int(layout[len(layout)-1].end)
	optimizer.packed = &adamWPackedState{
		m:      Zeros([]int32{int32(total)}, dtype),
		v:      Zeros([]int32{int32(total)}, dtype),
		dtype:  dtype,
		layout: cloneAdamWPackedLayout(layout),
	}
	optimizer.m, optimizer.v = optimizer.packed.views()
	return true
}

func (optimizer *AdamW) releasePackedStateOnly() {
	if optimizer == nil || optimizer.packed == nil {
		return
	}
	Free(optimizer.m...)
	Free(optimizer.v...)
	Free(optimizer.packed.m, optimizer.packed.v)
	optimizer.packed = nil
	optimizer.m = nil
	optimizer.v = nil
}

func (optimizer *AdamW) replacePackedMoments(nextM, nextV []*Array) {
	if optimizer == nil || optimizer.packed == nil || len(nextM) == 0 || len(nextM) != len(nextV) {
		return
	}
	mFlat := make([]*Array, len(nextM))
	vFlat := make([]*Array, len(nextV))
	for i := range nextM {
		mFlat[i] = Reshape(nextM[i], optimizer.packed.layout[i].end-optimizer.packed.layout[i].start)
		vFlat[i] = Reshape(nextV[i], optimizer.packed.layout[i].end-optimizer.packed.layout[i].start)
	}
	oldMViews, oldVViews := optimizer.m, optimizer.v
	oldMSlab, oldVSlab := optimizer.packed.m, optimizer.packed.v
	if len(mFlat) == 1 {
		optimizer.packed.m = mFlat[0].Clone()
		optimizer.packed.v = vFlat[0].Clone()
	} else {
		optimizer.packed.m = Concatenate(mFlat, 0)
		optimizer.packed.v = Concatenate(vFlat, 0)
	}
	optimizer.m, optimizer.v = optimizer.packed.views()
	Free(oldMViews...)
	Free(oldVViews...)
	Free(oldMSlab, oldVSlab)
	Free(mFlat...)
	Free(vFlat...)
	Free(nextM...)
	Free(nextV...)
}

func (state *adamWPackedState) views() ([]*Array, []*Array) {
	if state == nil || state.m == nil || state.v == nil {
		return nil, nil
	}
	momentsM := make([]*Array, len(state.layout))
	momentsV := make([]*Array, len(state.layout))
	for i, desc := range state.layout {
		momentsM[i] = adamWPackedView(state.m, desc)
		momentsV[i] = adamWPackedView(state.v, desc)
	}
	return momentsM, momentsV
}

func adamWPackedView(slab *Array, desc adamWPackedParam) *Array {
	flat := Slice(slab, []int32{desc.start}, []int32{desc.end})
	view := Reshape(flat, desc.shape...)
	Free(flat)
	return view
}

func adamWPackedLayout(parameters []*Array) ([]adamWPackedParam, DType, bool) {
	if len(parameters) == 0 {
		return nil, 0, false
	}
	layout := make([]adamWPackedParam, len(parameters))
	var dtype DType
	var offset int32
	for i, parameter := range parameters {
		if parameter == nil || !parameter.Valid() {
			return nil, 0, false
		}
		shape := parameter.Shape()
		if len(shape) == 0 {
			return nil, 0, false
		}
		size, ok := adamWShapeSize(shape)
		if !ok {
			return nil, 0, false
		}
		if i == 0 {
			dtype = parameter.Dtype()
		} else if parameter.Dtype() != dtype {
			return nil, 0, false
		}
		next := offset + int32(size)
		if next <= offset {
			return nil, 0, false
		}
		layout[i] = adamWPackedParam{
			start: offset,
			end:   next,
			shape: append([]int32(nil), shape...),
		}
		offset = next
	}
	return layout, dtype, true
}

func adamWShapeSize(shape []int32) (int, bool) {
	if len(shape) == 0 {
		return 0, false
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return 0, false
		}
		if total > int(^uint32(0)>>1)/int(dim) {
			return 0, false
		}
		total *= int(dim)
	}
	return total, true
}

func adamWPackedLayoutEqual(a, b []adamWPackedParam) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].start != b[i].start || a[i].end != b[i].end || len(a[i].shape) != len(b[i].shape) {
			return false
		}
		for j := range a[i].shape {
			if a[i].shape[j] != b[i].shape[j] {
				return false
			}
		}
	}
	return true
}

func cloneAdamWPackedLayout(src []adamWPackedParam) []adamWPackedParam {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]adamWPackedParam, len(src))
	for i, desc := range src {
		cloned[i] = adamWPackedParam{
			start: desc.start,
			end:   desc.end,
			shape: append([]int32(nil), desc.shape...),
		}
	}
	return cloned
}
