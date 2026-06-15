// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Runnable examples for the public jang API. The pure-Go shape helpers print
// their real computed output; the Metal dequant example shows the call shape
// and prints the descriptor-derived output length (deterministic — no GPU
// float formatting in the asserted Output block).
package jang_test

import (
	core "dappco.re/go"
	infjang "dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/quant/jang"
)

// ExampleMetalShape converts a uint64 model-weight shape into the int32 form
// the Metal kernels expect, with the per-dimension bound check applied.
func ExampleMetalShape() {
	shape, err := jang.MetalShape([]uint64{4096, 14336})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(shape[0], shape[1])
	// Output: 4096 14336
}

// ExampleShapeElements returns the overflow-checked element product of a
// tensor shape — the count of values a dequant must produce.
func ExampleShapeElements() {
	n, err := jang.ShapeElements([]int32{2, 4, 8})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(n)
	// Output: 64
}

// ExampleInt32SliceToInts widens an int32 shape to the []int form Metal's
// FromValues variadic shape argument consumes.
func ExampleInt32SliceToInts() {
	out := jang.Int32SliceToInts([]int32{1, 2, 4})
	core.Println(out[0], out[1], out[2])
	// Output: 1 2 4
}

// ExampleDequantizePackedTensor shows the descriptor → pack → Metal dequant
// call shape. The asserted output is the deterministic element count derived
// from the descriptor (the dequantised float values themselves come from the
// GPU and are covered by the round-trip test, not printed here).
func ExampleDequantizePackedTensor() {
	info := &infjang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", GroupSize: 64, BitsDefault: 2, RoutedExpertBits: 2}
	desc, err := infjang.NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{2, 16}, info)
	if err != nil {
		core.Println(err.Error())
		return
	}
	packed := make([]byte, desc.PackedBytes)
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 1
	}
	out, err := jang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(len(out))
	// Output: 32
}
