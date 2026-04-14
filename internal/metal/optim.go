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

	step int      // Number of updates performed
	m    []*Array // First moment estimates (positional, parallel to params)
	v    []*Array // Second moment estimates (positional, parallel to params)
}

// AdamWConfig configures AdamW optimiser construction.
type AdamWConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Eps          float64
	WeightDecay  float64
}

// DefaultAdamWConfig returns the standard AdamW hyperparameters.
func DefaultAdamWConfig() AdamWConfig {
	return AdamWConfig{
		LearningRate: 1e-5,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.01,
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
	}
}

func mergeAdamWConfig(defaults AdamWConfig, override AdamWConfig) AdamWConfig {
	cfg := defaults
	if override.LearningRate != 0 {
		cfg.LearningRate = override.LearningRate
	}
	if override.Beta1 != 0 {
		cfg.Beta1 = override.Beta1
	}
	if override.Beta2 != 0 {
		cfg.Beta2 = override.Beta2
	}
	if override.Eps != 0 {
		cfg.Eps = override.Eps
	}
	if override.WeightDecay != 0 {
		cfg.WeightDecay = override.WeightDecay
	}
	return cfg
}

// Step performs one optimisation step: updates parameters using gradients.
// Parameters and gradients must be parallel slices of the same length.
// Returns the updated parameter arrays (parameters are replaced in-place).
//
//	parameters = optimizer.Step(parameters, gradients) // one Adam step per mini-batch
func (optimizer *AdamW) Step(parameters []*Array, gradients []*Array) []*Array {
	optimizer.step++

	// Bias correction factors: compensate for zero-initialised moments.
	biasCorrection1 := 1.0 - math.Pow(optimizer.Beta1, float64(optimizer.step))
	biasCorrection2 := 1.0 - math.Pow(optimizer.Beta2, float64(optimizer.step))

	updated := make([]*Array, len(parameters))

	// Grow moment slices if needed (first call or param count increased)
	for len(optimizer.m) < len(parameters) {
		optimizer.m = append(optimizer.m, nil)
		optimizer.v = append(optimizer.v, nil)
	}

	for i, parameter := range parameters {
		gradient := gradients[i]

		// Initialise moments on first use
		if optimizer.m[i] == nil {
			shape := parameter.Shape()
			optimizer.m[i] = Zeros(shape, parameter.Dtype())
			optimizer.v[i] = Zeros(shape, parameter.Dtype())
		}

		// m = beta1 * m + (1 - beta1) * grad
		m := Add(
			MulScalar(optimizer.m[i], float32(optimizer.Beta1)),
			MulScalar(gradient, float32(1.0-optimizer.Beta1)),
		)

		// v = beta2 * v + (1 - beta2) * grad^2
		v := Add(
			MulScalar(optimizer.v[i], float32(optimizer.Beta2)),
			MulScalar(Square(gradient), float32(1.0-optimizer.Beta2)),
		)

		// Bias-corrected estimates
		mHat := MulScalar(m, float32(1.0/biasCorrection1))
		vHat := MulScalar(v, float32(1.0/biasCorrection2))

		// Weight decay: param = param * (1 - lr * weight_decay)
		decayed := MulScalar(parameter, float32(1.0-optimizer.LR*optimizer.WeightDecay))

		// Update: param = decayed - lr * m_hat / (sqrt(v_hat) + eps)
		denom := AddScalar(Sqrt(vHat), float32(optimizer.Eps))
		step := MulScalar(Divide(mHat, denom), float32(optimizer.LR))
		newParam := Subtract(decayed, step)

		// Store updated moments
		optimizer.m[i] = m
		optimizer.v[i] = v

		updated[i] = newParam
	}

	return updated
}

// Reset clears the optimiser state (moments and step counter).
//
//	optimizer.Reset() // start a new training run from scratch
func (optimizer *AdamW) Reset() {
	optimizer.step = 0
	optimizer.m = nil
	optimizer.v = nil
}
