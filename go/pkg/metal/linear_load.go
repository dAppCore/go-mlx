// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// LoadLinear builds a Linear from a checkpoint's weights by prefix, applying the
// model's quantization config when the weight is packed (scales present). It is
// the generic, model-agnostic projection loader: a config-composed model and a
// mixer builder's ctx.Linear resolve their projections through it, so the
// quant-loader registry (the NewQuantizedLinearWithMode dispatch) stays the
// single quant boundary and the caller never handles a quant format. Returns nil
// when the weight is absent — the caller decides whether that is fatal.
//
//	inProj := metal.LoadLinear(weights, prefix+".in_proj", cfg.Quantization)
func LoadLinear(weights map[string]*Array, prefix string, q *QuantizationConfig) *Linear {
	weight := weights[prefix+".weight"]
	if weight == nil {
		return nil
	}
	bias := weights[prefix+".bias"]
	if scales := weights[prefix+".scales"]; scales != nil && q != nil {
		return NewQuantizedLinearWithMode(weight, scales, weights[prefix+".biases"], bias, q.GroupSize, q.Bits, q.Mode)
	}
	return NewLinear(weight, bias)
}
