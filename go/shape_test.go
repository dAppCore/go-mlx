// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"testing"

	core "dappco.re/go"
)

func TestAPIShape_NormalizeRootInt32Arg_Good(t *testing.T) {
	cases := []struct {
		name  string
		value any
		want  int32
	}{
		{name: "int", value: int(7), want: 7},
		{name: "int8", value: int8(-8), want: -8},
		{name: "int16", value: int16(16), want: 16},
		{name: "int32", value: int32(32), want: 32},
		{name: "int64", value: int64(64), want: 64},
		{name: "uint", value: uint(7), want: 7},
		{name: "uint8", value: uint8(8), want: 8},
		{name: "uint16", value: uint16(16), want: 16},
		{name: "uint32", value: uint32(32), want: 32},
		{name: "uint64", value: uint64(64), want: 64},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeRootInt32Arg("shape", tc.value); got != tc.want {
				t.Fatalf("normalizeRootInt32Arg() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestAPIShape_NormalizeRootShapeArgs_Good(t *testing.T) {
	cases := []struct {
		name  string
		shape []any
		want  []int32
	}{
		{name: "plain_args", shape: []any{int(2), int64(3), uint8(4)}, want: []int32{2, 3, 4}},
		{name: "int_slice", shape: []any{[]int{2, 3}}, want: []int32{2, 3}},
		{name: "int32_slice", shape: []any{[]int32{4, 5}}, want: []int32{4, 5}},
		{name: "int64_slice", shape: []any{[]int64{6, 7}}, want: []int32{6, 7}},
		{name: "uint_slice", shape: []any{[]uint{8, 9}}, want: []int32{8, 9}},
		{name: "uint32_slice", shape: []any{[]uint32{10, 11}}, want: []int32{10, 11}},
		{name: "uint64_slice", shape: []any{[]uint64{12, 13}}, want: []int32{12, 13}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeRootShapeArgs(tc.shape); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("normalizeRootShapeArgs() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAPIShape_NormalizeRootIntArg_Bad(t *testing.T) {
	if got := normalizeRootIntArg("axis", int16(3)); got != 3 {
		t.Fatalf("normalizeRootIntArg() = %d, want 3", got)
	}
	assertRootShapePanic(t, func() { normalizeRootInt32Arg("shape", int64(rootMaxInt32)+1) }, "out of int32 range")
	assertRootShapePanic(t, func() { normalizeRootInt32Arg("shape", uint64(rootMaxInt32)+1) }, "out of int32 range")
	assertRootShapePanic(t, func() { normalizeRootInt32Arg("shape", "2") }, "int-compatible")
}

func assertRootShapePanic(t *testing.T, fn func(), want string) {
	t.Helper()
	defer func() {
		recovered := recover()
		if recovered == nil {
			t.Fatal("expected panic")
		}
		text, ok := recovered.(string)
		if !ok || !core.Contains(text, want) {
			t.Fatalf("panic = %#v, want substring %q", recovered, want)
		}
	}()
	fn()
}
// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"reflect"
	"testing"
)

func TestReshape_AcceptsShapeSlices_Good(t *testing.T) {
	coverageTokens := "AcceptsShapeSlices"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "AcceptsPlainInts"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	arr := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	sliced := Slice(arr, 0, 1, 1)
	defer Free(arr, sliced)

	if got, want := sliced.Shape(), []int32{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Slice(int, int, int) shape = %v, want %v", got, want)
	}
}

func TestWithReturnLogits_Alias_Good(t *testing.T) {
	coverageTokens := "Alias"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cfg := applyGenerateOptions([]GenerateOption{WithReturnLogits()})
	if !cfg.ReturnLogits {
		t.Fatal("WithReturnLogits() did not enable ReturnLogits")
	}
}
