// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleSlice() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	out := Slice(values, []int32{0, 0}, []int32{1, 3})
	defer Free(values, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 3] [1 2 3]
}

func ExampleSliceAxis() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	out := SliceAxis(values, 1, 1, 3)
	flat := Reshape(out, 4)
	defer Free(values, out, flat)
	Materialize(flat)

	core.Println(out.Shape(), flat.Floats())
	// Output: [2 2] [2 3 5 6]
}

func ExampleSliceUpdateInplace() {
	cache := Zeros([]int32{2, 3}, DTypeFloat32)
	update := FromValues([]float32{7, 8, 9}, 1, 3)
	out := SliceUpdateInplace(cache, update, []int32{1, 0}, []int32{2, 3})
	defer Free(cache, update, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 3] [0 0 0 7 8 9]
}
