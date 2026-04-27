// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"reflect"
	"testing"
)

func TestReshape_AcceptsShapeSlices_Good(t *testing.T) {
	arr := FromValues([]float32{1, 2, 3, 4}, 4)
	reshapedInts := Reshape(arr, []int{2, 2})
	reshapedInt32s := Reshape(arr, []int32{1, 4})
	defer Free(arr, reshapedInts, reshapedInt32s)

	if got, want := reshapedInts.Shape(), []int32{2, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Reshape([]int) shape = %v, want %v", got, want)
	}
	if got, want := reshapedInt32s.Shape(), []int32{1, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Reshape([]int32) shape = %v, want %v", got, want)
	}
}

func TestSlice_AcceptsPlainInts_Good(t *testing.T) {
	arr := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	sliced := Slice(arr, 0, 1, 1)
	defer Free(arr, sliced)

	if got, want := sliced.Shape(), []int32{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Slice(int, int, int) shape = %v, want %v", got, want)
	}
}

func TestWithReturnLogits_Alias_Good(t *testing.T) {
	cfg := applyGenerateOptions([]GenerateOption{WithReturnLogits()})
	if !cfg.ReturnLogits {
		t.Fatal("WithReturnLogits() did not enable ReturnLogits")
	}
}
