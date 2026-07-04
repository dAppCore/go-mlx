// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestPinnedArray_FromPinnedRawBytes_Good(t *testing.T) {
	requireMetalRuntime(t)

	raw := pinnedArrayFloat32Bytes([]float32{1, 2, 3, 4})
	array, err := fromPinnedRawBytes(raw, []int{1, 1, 2, 2}, DTypeFloat32)
	if err != nil {
		t.Fatalf("fromPinnedRawBytes() error = %v", err)
	}
	defer Free(array)

	if got := array.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("pinned array floats = %v, want [1 2 3 4]", got)
	}
}

func TestPinnedArray_FromPinnedRawBytes_Bad(t *testing.T) {
	requireMetalRuntime(t)

	_, err := fromPinnedRawBytes([]byte{1, 2}, []int{1, 1, 1, 1}, DTypeFloat32)
	if err == nil {
		t.Fatal("fromPinnedRawBytes() error = nil, want byte length validation error")
	}
}

func TestPinnedArray_FromPinnedRawBytesStrided_Good(t *testing.T) {
	requireMetalRuntime(t)

	raw := pinnedArrayFloat32Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	array, err := fromPinnedRawBytesStrided(
		raw,
		[]int{1, 1, 4, 2},
		[]int{1, 1, 2, 2},
		[]int64{8, 8, 2, 1},
		2,
		DTypeFloat32,
	)
	if err != nil {
		t.Fatalf("fromPinnedRawBytesStrided() error = %v", err)
	}
	defer Free(array)

	if got := array.Floats(); !reflect.DeepEqual(got, []float32{3, 4, 5, 6}) {
		t.Fatalf("strided pinned array floats = %v, want [3 4 5 6]", got)
	}
}

func TestPinnedArray_FromPinnedRawBytesStrided_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	raw := pinnedArrayFloat32Bytes([]float32{1, 2, 3, 4})
	_, err := fromPinnedRawBytesStrided(
		raw,
		[]int{1, 1, 2, 2},
		[]int{1, 1, 3, 2},
		[]int64{4, 4, 2, 1},
		0,
		DTypeFloat32,
	)
	if err == nil {
		t.Fatal("fromPinnedRawBytesStrided() error = nil, want bounds validation error")
	}
}

func pinnedArrayFloat32Bytes(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return raw
}
