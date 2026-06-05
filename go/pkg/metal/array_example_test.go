// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleFromValue() {
	value := FromValue(float32(3.5))
	defer Free(value)
	Materialize(value)

	core.Println(core.Sprintf("%.1f", value.Float()), value.Dtype(), value.NumDims(), value.Size())
	// Output: 3.5 float32 0 1
}

func ExampleFromValues() {
	values := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer Free(values)
	Materialize(values)

	core.Println(values.Shape(), values.Floats())
	// Output: [2 2] [1 2 3 4]
}

func ExampleZeros() {
	values := Zeros([]int32{2, 3}, DTypeFloat32)
	defer Free(values)
	Materialize(values)

	core.Println(values.Shape(), values.Floats())
	// Output: [2 3] [0 0 0 0 0 0]
}

func ExampleArray_Set() {
	values := FromValue(float32(1))
	other := FromValue(float32(2))
	defer Free(values, other)

	values.Set(other)
	Materialize(values)

	core.Println(core.Sprintf("%.0f", values.Float()))
	// Output: 2
}

func ExampleArray_Clone() {
	values := FromValue(float32(7))
	clone := values.Clone()
	defer Free(values, clone)
	Materialize(clone)

	core.Println(core.Sprintf("%.0f", clone.Float()), clone.Valid())
	// Output: 7 true
}

func ExampleArray_Valid() {
	values := FromValue(float32(1))
	core.Println(values.Valid())
	Free(values)
	core.Println(values.Valid())
	// Output:
	// true
	// false
}

func ExampleArray_String() {
	values := FromValue(float32(42))
	defer Free(values)
	Materialize(values)
	text := values.String()

	core.Println(core.Contains(text, "42"), core.Contains(text, "float32"))
	// Output: true true
}

func ExampleArray_Shape() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(values)

	core.Println(values.Shape())
	// Output: [1 2 3]
}

func ExampleArray_Size() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	defer Free(values)

	core.Println(values.Size())
	// Output: 6
}

func ExampleArray_NumBytes() {
	values := FromValues([]float32{1, 2, 3, 4}, 4)
	defer Free(values)

	core.Println(values.NumBytes())
	// Output: 16
}

func ExampleArray_NumDims() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(values)

	core.Println(values.NumDims())
	// Output: 3
}

func ExampleArray_Dim() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(values)

	core.Println(values.Dim(0), values.Dim(1), values.Dim(2))
	// Output: 1 2 3
}

func ExampleArray_Dims() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(values)

	core.Println(values.Dims())
	// Output: [1 2 3]
}

func ExampleArray_Dtype() {
	values := FromValues([]int32{10, 20}, 2)
	defer Free(values)

	core.Println(values.Dtype())
	// Output: int32
}

func ExampleArray_Int() {
	value := FromValue(42)
	defer Free(value)
	Materialize(value)

	core.Println(value.Int())
	// Output: 42
}

func ExampleArray_Float() {
	value := FromValue(float32(1.5))
	defer Free(value)
	Materialize(value)

	core.Println(core.Sprintf("%.1f", value.Float()))
	// Output: 1.5
}

func ExampleArray_Bool() {
	value := FromValue(true)
	defer Free(value)
	Materialize(value)

	core.Println(value.Bool())
	// Output: true
}

func ExampleArray_SetFloat64() {
	value := FromValue(float64(1))
	defer Free(value)
	value.SetFloat64(2.5)
	Materialize(value)

	core.Println(core.Sprintf("%.1f", value.Float()))
	// Output: 2.5
}

func ExampleArray_ShapeRaw() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	defer Free(values)
	raw := values.ShapeRaw()

	core.Println(shapeRawDim(raw, 0), shapeRawDim(raw, 1))
	// Output: 2 3
}

func ExampleArray_IsRowContiguous() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	transposed := Transpose(values)
	defer Free(values, transposed)
	Materialize(transposed)

	core.Println(values.IsRowContiguous(), transposed.IsRowContiguous())
	// Output: true false
}

func ExampleContiguous() {
	values := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	transposed := Transpose(values)
	contiguous := Contiguous(transposed)
	defer Free(values, transposed, contiguous)
	Materialize(contiguous)

	core.Println(contiguous.IsRowContiguous(), contiguous.Shape(), contiguous.Floats())
	// Output: true [3 2] [1 4 2 5 3 6]
}

func ExampleArray_Bytes() {
	values := FromValues([]uint8{1, 2, 3, 4}, 4)
	defer Free(values)
	Materialize(values)

	core.Println(values.Bytes())
	// Output: [1 2 3 4]
}

func ExampleArray_Ints() {
	values := FromValues([]int32{10, 20, 30}, 3)
	defer Free(values)
	Materialize(values)

	core.Println(values.Ints())
	// Output: [10 20 30]
}

func ExampleArray_DataInt32() {
	values := FromValues([]int32{10, 20, 30}, 3)
	defer Free(values)
	Materialize(values)

	core.Println(values.DataInt32())
	// Output: [10 20 30]
}

func ExampleArray_Floats() {
	values := FromValues([]float32{1, 2, 3}, 3)
	defer Free(values)
	Materialize(values)

	core.Println(values.Floats())
	// Output: [1 2 3]
}

func ExampleFree() {
	values := FromValues([]float32{1, 2, 3, 4}, 4)
	Materialize(values)

	core.Println(Free(values))
	// Output: 16
}

func ExampleArray_Iter() {
	values := FromValues([]float32{1, 2, 3}, 3)
	defer Free(values)
	Materialize(values)
	var sum float32
	for value := range values.Iter() {
		sum += value
	}

	core.Println(core.Sprintf("%.0f", sum))
	// Output: 6
}
