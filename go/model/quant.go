// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// QuantFormat names a model's quantisation scheme, loader-neutral — the same
// value means the same thing whether the model later runs on Metal, ROCm or the
// pure-Go CPU floor.
type QuantFormat string

const (
	// QuantNone is full precision (bf16/fp16) — no quantisation block.
	QuantNone QuantFormat = ""
	// QuantAffine is MLX group-affine: uint32-packed weights with per-group
	// scales + biases (the mlx-community / HF MetalConfig pack).
	QuantAffine QuantFormat = "affine"
)

// QuantSpec is the neutral, loader-independent description of a model's
// quantisation, read once from the model's own bytes before any backend touches
// it. Each backend's kernel factory reacts to this spec; it is never inferred
// from the filename. Exclude lists modules kept at full precision
// (HF's modules_to_not_convert) — the canonical way mixed-precision is declared.
type QuantSpec struct {
	Format    QuantFormat
	Bits      int // 0 for full precision
	GroupSize int
	Exclude   []string
}

// ResolveQuant detects a model's quantisation from its own bytes. The declared
// config.json quantization block supplies the group size; the packed tensor
// geometry confirms the bit-width (deriveAffineBits); the two are cross-checked
// and a mismatch fails loud. The model is the truth — the filename is never
// consulted.
//
//	spec, err := model.ResolveQuant(modelDir) // {affine, 4, 64} for gemma-4-e2b-it-4bit
func ResolveQuant(modelDir string) (QuantSpec, error) {
	config, err := readModelConfig(modelDir)
	if err != nil {
		return QuantSpec{}, err
	}
	declaredBits := config.quantBits()
	if declaredBits == 0 {
		// No quantization block — the model ships full precision.
		return QuantSpec{Format: QuantNone}, nil
	}
	group := config.quantGroup()
	if group <= 0 {
		return QuantSpec{}, core.E("model.ResolveQuant",
			"config declares quantization bits but no group_size", nil)
	}
	index, err := safetensorsIndex(modelDir)
	if err != nil {
		return QuantSpec{}, err
	}
	derivedBits, ok := deriveQuantFromIndex(index, group)
	if !ok {
		return QuantSpec{}, core.E("model.ResolveQuant",
			"could not derive quant bits from packed tensor geometry", nil)
	}
	if derivedBits != declaredBits {
		return QuantSpec{}, core.E("model.ResolveQuant",
			core.Sprintf("config declares %d-bit but tensors are %d-bit", declaredBits, derivedBits), nil)
	}
	return QuantSpec{Format: QuantAffine, Bits: derivedBits, GroupSize: group}, nil
}

// deriveAffineBits returns the bit-width of an MLX affine-quantised linear from
// its packed-weight and scales last-dimensions and the quantisation group size.
//
// MLX packs an affine-quantised weight of logical shape [out, in] as a uint32
// tensor [out, in*bits/32], with a per-group scales (and biases) companion of
// shape [out, in/group]. The two last-dims therefore pin the bit-width exactly:
//
//	bits = 32 * weightLast / (scalesLast * group)
//
// It returns ok=false for any shape that doesn't reduce to a clean 1..8-bit
// pack: the engine reads bits from the bytes the model actually ships, and must
// fail loud on a layout it can't read rather than return a plausible guess.
func deriveAffineBits(weightLast, scalesLast, group int) (int, bool) {
	if weightLast <= 0 || scalesLast <= 0 || group <= 0 {
		return 0, false
	}
	numerator := 32 * weightLast
	denominator := scalesLast * group
	if numerator%denominator != 0 {
		return 0, false
	}
	bits := numerator / denominator
	if bits < 1 || bits > 8 {
		return 0, false
	}
	return bits, true
}

// deriveQuantFromIndex finds an affine-quantised linear in the tensor index (a
// tensor with a ".scales" companion) and derives its bit-width from the packed
// weight/scales geometry. The first clean match wins — uniform packs (every
// linear the same bits) are the common case; per-tensor mixed-bit (MoQ)
// detection is a follow-up that walks the whole set.
func deriveQuantFromIndex(index safetensors.Index, group int) (int, bool) {
	for name, scales := range index.Tensors {
		if !core.HasSuffix(name, ".scales") {
			continue
		}
		weightName := core.TrimSuffix(name, ".scales") + ".weight"
		weight, ok := index.Tensors[weightName]
		if !ok || len(weight.Shape) == 0 || len(scales.Shape) == 0 {
			continue
		}
		weightLast := int(weight.Shape[len(weight.Shape)-1])
		scalesLast := int(scales.Shape[len(scales.Shape)-1])
		if bits, ok := deriveAffineBits(weightLast, scalesLast, group); ok {
			return bits, true
		}
	}
	return 0, false
}

// safetensorsIndex reads the tensor header index across every .safetensors
// shard in a model directory (single-file or sharded alike).
func safetensorsIndex(modelDir string) (safetensors.Index, error) {
	listed := core.ReadDir(core.DirFS(modelDir), ".")
	if !listed.OK {
		return safetensors.Index{}, listed.Value.(error)
	}
	var paths []string
	for _, entry := range listed.Value.([]core.FsDirEntry) {
		name := entry.Name()
		if core.HasSuffix(name, ".safetensors") {
			paths = append(paths, core.PathJoin(modelDir, name))
		}
	}
	if len(paths) == 0 {
		return safetensors.Index{}, core.E("model.safetensorsIndex", "no .safetensors files in "+modelDir, nil)
	}
	return safetensors.IndexFiles(paths)
}
