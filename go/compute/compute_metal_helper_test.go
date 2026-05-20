// SPDX-Licence-Identifier: EUPL-1.2

package compute

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestComputeDarwinHelpers_Scalars_Good(t *testing.T) {
	if got := minInt(2, 9); got != 2 {
		t.Fatalf("minInt() = %d, want 2", got)
	}
	if got := maxInt(2, 9); got != 9 {
		t.Fatalf("maxInt() = %d, want 9", got)
	}
	if x, y := threadGroup(99, 3); x != 16 || y != 3 {
		t.Fatalf("threadGroup(99,3) = (%d,%d), want (16,3)", x, y)
	}
	if x, y := threadGroup(0, -4); x != 1 || y != 1 {
		t.Fatalf("threadGroup(0,-4) = (%d,%d), want (1,1)", x, y)
	}

	if got := quantizeUnitScalar(0.5); got != 128 {
		t.Fatalf("quantizeUnitScalar(0.5) = %d, want 128", got)
	}
	if got := quantizeUnitScalar(-1); got != 0 {
		t.Fatalf("quantizeUnitScalar(-1) = %d, want 0", got)
	}
	if got := quantizeUnitScalar(2); got != 256 {
		t.Fatalf("quantizeUnitScalar(2) = %d, want 256", got)
	}
}

func TestComputeDarwinHelpers_RequireBuffer_Bad(t *testing.T) {
	_, err := requireBuffer(nil, KernelNearestScale, "src")
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("requireBuffer(nil) error = %v, want missing buffer", err)
	}

	_, err = requireBuffer(map[string]Buffer{}, KernelNearestScale, "src")
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("requireBuffer(missing) error = %v, want missing buffer", err)
	}

	want := &bufferbase{size: 4}
	got, err := requireBuffer(map[string]Buffer{"src": want}, KernelNearestScale, "src")
	if err != nil {
		t.Fatalf("requireBuffer(existing): %v", err)
	}
	if got != want {
		t.Fatalf("requireBuffer(existing) = %p, want %p", got, want)
	}
}

func TestComputeDarwinHelpers_UnitScalar_Ugly(t *testing.T) {
	cases := []struct {
		name string
		args KernelArgs
		want int
	}{
		{name: "nil_scalars", args: KernelArgs{}, want: 64},
		{name: "missing_scalar", args: KernelArgs{Scalars: map[string]float64{}}, want: 64},
		{name: "explicit_scalar", args: KernelArgs{Scalars: map[string]float64{"strength": 0.25}}, want: 64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := unitScalar(tc.args, KernelScanlineFilter, "strength", 0.25)
			if err != nil {
				t.Fatalf("unitScalar(): %v", err)
			}
			if got != tc.want {
				t.Fatalf("unitScalar() = %d, want %d", got, tc.want)
			}
		})
	}

	badCases := []struct {
		name  string
		value float64
	}{
		{name: "nan", value: math.NaN()},
		{name: "inf", value: math.Inf(1)},
		{name: "negative", value: -0.1},
		{name: "too_large", value: 1.1},
	}
	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := unitScalar(KernelArgs{Scalars: map[string]float64{"strength": tc.value}}, KernelScanlineFilter, "strength", 0.25)
			if !core.Is(err, ErrComputeInvalidScalar) {
				t.Fatalf("unitScalar(%v) error = %v, want invalid scalar", tc.value, err)
			}
		})
	}
}

func TestComputeDarwinHelpers_ValidateFilterBuffers_Bad(t *testing.T) {
	src := &pixelbuffer{desc: PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}}
	dst := &pixelbuffer{desc: PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}}
	if err := validateFilterBuffers(src, dst, KernelScanlineFilter); err != nil {
		t.Fatalf("validateFilterBuffers(valid): %v", err)
	}
	if !sameDimensions(src.desc, dst.desc) {
		t.Fatal("sameDimensions(valid) = false, want true")
	}

	cases := []struct {
		name string
		dst  *pixelbuffer
	}{
		{name: "dimensions", dst: &pixelbuffer{desc: PixelBufferDesc{Width: 3, Height: 2, Stride: 12, Format: PixelRGBA8}}},
		{name: "format", dst: &pixelbuffer{desc: PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8}}},
		{name: "stride", dst: &pixelbuffer{desc: PixelBufferDesc{Width: 2, Height: 2, Stride: 16, Format: PixelRGBA8}}},
		{name: "unsupported", dst: &pixelbuffer{desc: PixelBufferDesc{Width: 2, Height: 2, Stride: 4, Format: PixelRGB565}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testSrc := src
			if tc.name == "unsupported" {
				testSrc = &pixelbuffer{desc: tc.dst.desc}
			}
			err := validateFilterBuffers(testSrc, tc.dst, KernelScanlineFilter)
			if !core.Is(err, ErrComputeInvalidKernelArgs) {
				t.Fatalf("validateFilterBuffers(%s) error = %v, want invalid kernel args", tc.name, err)
			}
		})
	}
}
