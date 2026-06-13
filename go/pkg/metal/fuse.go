// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"slices"

	core "dappco.re/go"
)

// FuseLoRAIntoWeights folds a trained LoRA adapter's deltas into a dense base
// weights map, returning a NEW map where each targeted base weight W is replaced
// by W + (B·A)·scale (dense) and every other tensor is carried through unchanged.
// It also returns the sorted names of the layers it fused, for diagnostics.
//
// This is the disk-fuse core (the P-phase on-disk fuse): it works at the
// weights-map level so a fused checkpoint serialises straight back out with
// SaveSafetensors, without walking a live model. scale is the LoRA alpha/rank.
//
// The base must be DENSE (unquantized): a fused layer becomes a dense matrix, and
// a single config-level quantization block cannot describe a checkpoint that
// mixes fused-dense layers with quantized neighbours. A quantized base is refused
// here — fusing one means dequantize-merge-then-requantize the whole model, a
// separate path. A targeted layer missing its base weight, or a shape that does
// not match the delta, is a loud error rather than a silent skip.
//
//	scale := alpha / float32(rank)
//	merged, fused, err := metal.FuseLoRAIntoWeights(base, adapter, scale)
func FuseLoRAIntoWeights(base, adapter map[string]*Array, scale float32) (map[string]*Array, []string, error) {
	const op = "metal.FuseLoRAIntoWeights"

	merged := make(map[string]*Array, len(base))
	for name, arr := range base {
		merged[name] = arr
	}

	var fused []string
	for key, a := range adapter {
		if !core.HasSuffix(key, ".lora_a") {
			continue
		}
		layer := key[:len(key)-len(".lora_a")]
		b := adapter[layer+".lora_b"]
		if a == nil || b == nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: lora_a/lora_b pair incomplete", layer), nil)
		}

		baseName := layer + ".weight"
		baseW := ResolveWeight(base, baseName)
		if baseW == nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: no base weight %q for the adapter delta", layer, baseName), nil)
		}
		// A dense base only — refuse a quantized base rather than mis-merging a
		// packed weight against a dense delta.
		if ResolveWeight(base, layer+".scales") != nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: base is quantized; fuse requires a dense (bf16/fp32) base", layer), nil)
		}

		// delta = (B·A)·scale, shape [out, in] to match the [out, in] base weight.
		ba := Matmul(b, a)
		delta := MulScalar(ba, scale)
		Free(ba)

		if !slices.Equal(baseW.Shape(), delta.Shape()) {
			Free(delta)
			return nil, nil, core.E(op, core.Sprintf("layer %q: base weight shape %v != lora delta shape %v", layer, baseW.Shape(), delta.Shape()), nil)
		}

		mergedW := Add(baseW, delta)
		Materialize(mergedW)
		Detach(mergedW)
		Free(delta)

		merged[baseName] = mergedW
		fused = append(fused, layer)
	}

	slices.Sort(fused)
	return merged, fused, nil
}
