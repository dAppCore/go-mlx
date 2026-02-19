//go:build darwin && arm64

package mlx

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

// NewAdamW creates an AdamW optimiser with default hyperparameters.
func NewAdamW(lr float64) *AdamW {
	return &AdamW{
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.01,
	}
}

// Step performs one optimisation step: updates params using gradients.
// params and grads must be parallel slices of the same length.
// Returns the updated parameter arrays (params are replaced in-place).
func (o *AdamW) Step(params []*Array, grads []*Array) []*Array {
	o.step++

	// Bias correction factors
	bc1 := 1.0 - math.Pow(o.Beta1, float64(o.step))
	bc2 := 1.0 - math.Pow(o.Beta2, float64(o.step))

	updated := make([]*Array, len(params))

	// Grow moment slices if needed (first call or param count increased)
	for len(o.m) < len(params) {
		o.m = append(o.m, nil)
		o.v = append(o.v, nil)
	}

	for i, param := range params {
		grad := grads[i]

		// Initialise moments on first use
		if o.m[i] == nil {
			shape := param.Shape()
			o.m[i] = Zeros(shape, param.Dtype())
			o.v[i] = Zeros(shape, param.Dtype())
		}

		// m = beta1 * m + (1 - beta1) * grad
		m := Add(
			MulScalar(o.m[i], float32(o.Beta1)),
			MulScalar(grad, float32(1.0-o.Beta1)),
		)

		// v = beta2 * v + (1 - beta2) * grad^2
		v := Add(
			MulScalar(o.v[i], float32(o.Beta2)),
			MulScalar(Square(grad), float32(1.0-o.Beta2)),
		)

		// Bias-corrected estimates
		mHat := MulScalar(m, float32(1.0/bc1))
		vHat := MulScalar(v, float32(1.0/bc2))

		// Weight decay: param = param * (1 - lr * weight_decay)
		decayed := MulScalar(param, float32(1.0-o.LR*o.WeightDecay))

		// Update: param = decayed - lr * m_hat / (sqrt(v_hat) + eps)
		denom := AddScalar(Sqrt(vHat), float32(o.Eps))
		step := MulScalar(Divide(mHat, denom), float32(o.LR))
		newParam := Subtract(decayed, step)

		// Store updated moments
		o.m[i] = m
		o.v[i] = v

		updated[i] = newParam
	}

	return updated
}

// Reset clears the optimiser state (moments and step counter).
func (o *AdamW) Reset() {
	o.step = 0
	o.m = nil
	o.v = nil
}
