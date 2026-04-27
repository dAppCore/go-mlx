// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

func mergedLoRAWeight(layer *LoRALinear) *Array {
	delta := Matmul(layer.B, layer.A)
	scaled := MulScalar(delta, layer.Scale)
	Free(delta)
	return scaled
}

func mergeLinearLoRA(linear *Linear) {
	if linear == nil || linear.LoRA == nil || linear.LoRA.A == nil || linear.LoRA.B == nil {
		return
	}

	oldWeight := linear.Weight
	oldScales := linear.Scales
	oldBiases := linear.Biases
	delta := mergedLoRAWeight(linear.LoRA)

	baseWeight := oldWeight
	if oldScales != nil {
		baseWeight = Dequantize(oldWeight, oldScales, oldBiases, linear.GroupSize, linear.Bits)
	}

	merged := Add(baseWeight, delta)
	Materialize(merged)
	Detach(merged)

	if baseWeight != oldWeight {
		Free(baseWeight)
	}
	Free(delta)

	linear.Weight = merged
	linear.Scales = nil
	linear.Biases = nil
	linear.GroupSize = 0
	linear.Bits = 0

	Free(oldWeight, oldScales, oldBiases, linear.LoRA.A, linear.LoRA.B)
	linear.LoRA.A = nil
	linear.LoRA.B = nil
	linear.LoRA = nil
}

// Merge folds all active LoRA deltas into their base linear weights in place.
// After Merge, generation uses the merged dense weights directly and the adapter
// no longer holds trainable matrices.
func (adapter *LoRAAdapter) Merge() {
	if adapter == nil {
		return
	}
	for _, layer := range adapter.Layers {
		if layer == nil || layer.Base == nil {
			continue
		}
		mergeLinearLoRA(layer.Base)
	}
	adapter.Layers = map[string]*LoRALinear{}
}
