// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"encoding/binary"
	"fmt"
	"math"

	"dappco.re/go/mlx/safetensors"
)

// ExampleDTypeByteSize shows how to resolve the on-disk byte width of a
// supported dense safetensors dtype. The dtype string is matched
// case-insensitively, so a header carrying "bf16" resolves the same as
// the canonical "BF16".
func ExampleDTypeByteSize() {
	for _, dtype := range []string{"F16", "BF16", "F32", "F64", "bf16"} {
		size, err := safetensors.DTypeByteSize(dtype)
		if err != nil {
			fmt.Printf("%s: %v\n", dtype, err)
			continue
		}
		fmt.Printf("%s -> %d bytes/element\n", dtype, size)
	}
	// Output:
	// F16 -> 2 bytes/element
	// BF16 -> 2 bytes/element
	// F32 -> 4 bytes/element
	// F64 -> 8 bytes/element
	// bf16 -> 2 bytes/element
}

// ExampleDecodeFloatData decodes a little-endian F32 payload (the raw
// bytes as they appear in a safetensors tensor blob) into a []float32.
// The element count is supplied by the caller from the tensor's shape;
// DecodeFloatData validates that the payload length matches.
func ExampleDecodeFloatData() {
	// Three F32 values: 1.0, 2.5, -3.0 laid out little-endian.
	raw := make([]byte, 3*4)
	binary.LittleEndian.PutUint32(raw[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(raw[4:], math.Float32bits(2.5))
	binary.LittleEndian.PutUint32(raw[8:], math.Float32bits(-3.0))

	values, err := safetensors.DecodeFloatData("F32", raw, 3)
	if err != nil {
		fmt.Println("decode:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2.5 -3]
}

// ExampleFloat16ToFloat32 converts a single IEEE-754 half-precision bit
// pattern to float32. 0x3c00 is the half-precision encoding of 1.0 and
// 0xc000 is -2.0.
func ExampleFloat16ToFloat32() {
	fmt.Println(safetensors.Float16ToFloat32(0x3c00))
	fmt.Println(safetensors.Float16ToFloat32(0xc000))
	// Output:
	// 1
	// -2
}

// ExampleRefFromHeader builds a TensorRef from a parsed header entry. The
// returned ref carries the absolute byte range of the tensor payload in
// the source file (DataStart is dataStart + the entry's begin offset) and
// the element count derived from the shape.
func ExampleRefFromHeader() {
	entry := safetensors.HeaderEntry{
		DType:       "F32",
		Shape:       []int64{2, 3},
		DataOffsets: []int64{0, 24},
	}
	ref, err := safetensors.RefFromHeader("model.safetensors", "weight", entry, 128)
	if err != nil {
		fmt.Println("ref:", err)
		return
	}
	fmt.Printf("name=%s dtype=%s elements=%d start=%d len=%d\n",
		ref.Name, ref.DType, ref.Elements, ref.DataStart, ref.ByteLen)
	// Output:
	// name=weight dtype=F32 elements=6 start=128 len=24
}
