package mlx

import (
	"errors"
	"testing"
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

func TestPixelBufferDesc_Validate_BadStride(t *testing.T) {
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
	if !errors.Is(err, ErrComputeInvalidDescriptor) {
		t.Fatalf("Validate() error = %v, want ErrComputeInvalidDescriptor", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
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

func TestComputeError_IsByKind_Good(t *testing.T) {
	err := &ComputeError{
		Kind:     ComputeErrorInvalidScalar,
		Op:       "validate_kernel_scalar",
		Kernel:   KernelScanlineFilter,
		Resource: "strength",
		Message:  "kernel scalar strength must be between 0 and 1",
	}

	if !errors.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("errors.Is(%v, ErrComputeInvalidScalar) = false, want true", err)
	}
	if !errors.Is(err, &ComputeError{Kind: ComputeErrorInvalidScalar, Kernel: KernelScanlineFilter}) {
		t.Fatalf("errors.Is(%v, ComputeError{Kind: invalid_scalar, Kernel: %q}) = false, want true", err, KernelScanlineFilter)
	}
	if errors.Is(err, ErrComputeUnknownKernel) {
		t.Fatalf("errors.Is(%v, ErrComputeUnknownKernel) = true, want false", err)
	}
}
