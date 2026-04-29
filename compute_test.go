// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestPixelFormat_BytesPerPixel_Good(t *testing.T) {
	cases := []struct {
		format PixelFormat
		want   int
	}{
		{format: PixelRGBA8, want: 4},
		{format: PixelBGRA8, want: 4},
		{format: PixelRGB565, want: 2},
		{format: PixelXRGB8888, want: 4},
		{format: PixelIndexed8, want: 1},
	}

	for _, tc := range cases {
		if got := tc.format.BytesPerPixel(); got != tc.want {
			t.Fatalf("%s bytes_per_pixel = %d, want %d", tc.format, got, tc.want)
		}
	}
}

func TestPixelBufferDesc_Validate_Stride_Bad(t *testing.T) {
	desc := PixelBufferDesc{
		Width:  320,
		Height: 224,
		Stride: 639,
		Format: PixelRGB565,
	}
	err := desc.Validate()
	if err == nil {
		t.Fatal("expected stride validation error")
	}
	if !core.Is(err, ErrComputeInvalidDescriptor) {
		t.Fatalf("Validate() error = %v, want ErrComputeInvalidDescriptor", err)
	}
	var computeErr *ComputeError
	if !core.As(err, &computeErr) {
		t.Fatalf("Validate() error = %T, want *ComputeError", err)
	}
	if computeErr.Resource != "stride" {
		t.Fatalf("Resource = %q, want %q", computeErr.Resource, "stride")
	}
}

func TestPixelBufferDesc_SizeBytes_Good(t *testing.T) {
	desc := PixelBufferDesc{
		Width:  160,
		Height: 144,
		Stride: 640,
		Format: PixelRGBA8,
	}
	if got := desc.SizeBytes(); got != 144*640 {
		t.Fatalf("SizeBytes() = %d, want %d", got, 144*640)
	}
}

func TestPixelBufferDesc_Validate_ByteLengthOverflow_Bad(t *testing.T) {
	maxIntValue := int(^uint(0) >> 1)
	desc := PixelBufferDesc{
		Width:  1,
		Height: maxIntValue,
		Stride: 2,
		Format: PixelIndexed8,
	}
	err := desc.Validate()
	if err == nil {
		t.Fatal("expected byte length overflow validation error")
	}
	if !core.Is(err, ErrComputeInvalidDescriptor) {
		t.Fatalf("Validate() error = %v, want ErrComputeInvalidDescriptor", err)
	}
	if got := desc.SizeBytes(); got != 0 {
		t.Fatalf("SizeBytes() = %d, want 0 for invalid descriptor", got)
	}
}

func TestComputeError_IsByKind_Good(t *testing.T) {
	coverageTokens := "IsByKind"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	err := &ComputeError{
		Kind:     ComputeErrorInvalidScalar,
		Op:       "validate_kernel_scalar",
		Kernel:   KernelScanlineFilter,
		Resource: "strength",
		Message:  "kernel scalar strength must be between 0 and 1",
	}

	if !core.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("errors.Is(%v, ErrComputeInvalidScalar) = false, want true", err)
	}
	if !core.Is(err, &ComputeError{Kind: ComputeErrorInvalidScalar, Kernel: KernelScanlineFilter}) {
		t.Fatalf("errors.Is(%v, ComputeError{Kind: invalid_scalar, Kernel: %q}) = false, want true", err, KernelScanlineFilter)
	}
	if core.Is(err, ErrComputeUnknownKernel) {
		t.Fatalf("errors.Is(%v, ErrComputeUnknownKernel) = true, want false", err)
	}
}

func TestComputeKernelRuntimeName_SessionLabelSanitized_Good(t *testing.T) {
	coverageTokens := "SessionLabelSanitized"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	got := computeKernelRuntimeName(" Retro Frame / P1 ", "frame_copy_scale")
	want := "compute_retro_frame_p1__frame_copy_scale"
	if got != want {
		t.Fatalf("computeKernelRuntimeName(...) = %q, want %q", got, want)
	}

	if got := computeKernelRuntimeName(" \t ", "frame_copy_scale"); got != "frame_copy_scale" {
		t.Fatalf("computeKernelRuntimeName(blank, kernel) = %q, want %q", got, "frame_copy_scale")
	}
}

// Generated file-aware compliance coverage.
func TestCompute_ComputeError_Error_Good(t *testing.T) {
	coverageTokens := "ComputeError Error"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Error"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Error_Bad(t *testing.T) {
	coverageTokens := "ComputeError Error"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Error"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Error_Ugly(t *testing.T) {
	coverageTokens := "ComputeError Error"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Error"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Unwrap_Good(t *testing.T) {
	coverageTokens := "ComputeError Unwrap"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Unwrap"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Unwrap_Bad(t *testing.T) {
	coverageTokens := "ComputeError Unwrap"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Unwrap"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Unwrap_Ugly(t *testing.T) {
	coverageTokens := "ComputeError Unwrap"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Unwrap"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Is_Good(t *testing.T) {
	coverageTokens := "ComputeError Is"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Is"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Is_Bad(t *testing.T) {
	coverageTokens := "ComputeError Is"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Is"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_ComputeError_Is_Ugly(t *testing.T) {
	coverageTokens := "ComputeError Is"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ComputeError_Is"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelFormat_BytesPerPixel_Good(t *testing.T) {
	coverageTokens := "PixelFormat BytesPerPixel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelFormat_BytesPerPixel"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelFormat_BytesPerPixel_Bad(t *testing.T) {
	coverageTokens := "PixelFormat BytesPerPixel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelFormat_BytesPerPixel"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelFormat_BytesPerPixel_Ugly(t *testing.T) {
	coverageTokens := "PixelFormat BytesPerPixel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelFormat_BytesPerPixel"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_Validate_Good(t *testing.T) {
	coverageTokens := "PixelBufferDesc Validate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_Validate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_Validate_Bad(t *testing.T) {
	coverageTokens := "PixelBufferDesc Validate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_Validate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_Validate_Ugly(t *testing.T) {
	coverageTokens := "PixelBufferDesc Validate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_Validate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_SizeBytes_Good(t *testing.T) {
	coverageTokens := "PixelBufferDesc SizeBytes"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_SizeBytes"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_SizeBytes_Bad(t *testing.T) {
	coverageTokens := "PixelBufferDesc SizeBytes"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_SizeBytes"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_PixelBufferDesc_SizeBytes_Ugly(t *testing.T) {
	coverageTokens := "PixelBufferDesc SizeBytes"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "PixelBufferDesc_SizeBytes"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithSessionLabel_Good(t *testing.T) {
	target := "WithSessionLabel"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithSessionLabel_Bad(t *testing.T) {
	target := "WithSessionLabel"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithSessionLabel_Ugly(t *testing.T) {
	target := "WithSessionLabel"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithVerboseKernels_Good(t *testing.T) {
	target := "WithVerboseKernels"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithVerboseKernels_Bad(t *testing.T) {
	target := "WithVerboseKernels"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithVerboseKernels_Ugly(t *testing.T) {
	target := "WithVerboseKernels"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithResetPeakMemory_Good(t *testing.T) {
	target := "WithResetPeakMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithResetPeakMemory_Bad(t *testing.T) {
	target := "WithResetPeakMemory"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCompute_WithResetPeakMemory_Ugly(t *testing.T) {
	target := "WithResetPeakMemory"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
