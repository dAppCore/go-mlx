// SPDX-Licence-Identifier: EUPL-1.2

package compute

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func TestCompute_PixelFormat_BytesPerPixel_Good(t *testing.T) {
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

func TestCompute_PixelBufferDesc_Validate_Bad(t *testing.T) {
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

func TestCompute_PixelBufferDesc_SizeBytes_Good(t *testing.T) {
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

func TestCompute_PixelBufferDesc_SizeBytes_Bad(t *testing.T) {
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

func TestCompute_PixelBufferDesc_Validate_Ugly(t *testing.T) {
	cases := []struct {
		name     string
		desc     PixelBufferDesc
		wantKind *ComputeError
		resource string
	}{
		{
			name:     "width",
			desc:     PixelBufferDesc{Height: 1, Stride: 4, Format: PixelRGBA8},
			wantKind: ErrComputeInvalidDescriptor,
			resource: "width",
		},
		{
			name:     "height",
			desc:     PixelBufferDesc{Width: 1, Stride: 4, Format: PixelRGBA8},
			wantKind: ErrComputeInvalidDescriptor,
			resource: "height",
		},
		{
			name:     "stride",
			desc:     PixelBufferDesc{Width: 1, Height: 1, Format: PixelRGBA8},
			wantKind: ErrComputeInvalidDescriptor,
			resource: "stride",
		},
		{
			name:     "format",
			desc:     PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelFormat("rgba16")},
			wantKind: ErrComputeUnsupportedPixelFormat,
			resource: "format",
		},
		{
			name:     "row_overflow",
			desc:     PixelBufferDesc{Width: int(^uint(0) >> 1), Height: 1, Stride: int(^uint(0) >> 1), Format: PixelRGBA8},
			wantKind: ErrComputeInvalidDescriptor,
			resource: "width",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.desc.Validate()
			if err == nil {
				t.Fatal("expected descriptor validation error")
			}
			if !core.Is(err, tc.wantKind) {
				t.Fatalf("Validate() error = %v, want %v", err, tc.wantKind)
			}
			var computeErr *ComputeError
			if !core.As(err, &computeErr) {
				t.Fatalf("Validate() error = %T, want *ComputeError", err)
			}
			if computeErr.Resource != tc.resource {
				t.Fatalf("Resource = %q, want %q", computeErr.Resource, tc.resource)
			}
		})
	}
}

func TestCompute_ComputeError_Error_Good(t *testing.T) {
	cases := []struct {
		name string
		err  *ComputeError
		want string
	}{
		{name: "nil", err: nil, want: "<nil>"},
		{name: "unavailable", err: ErrComputeUnavailable, want: "mlx: Metal compute is unavailable"},
		{name: "closed", err: ErrComputeClosed, want: "mlx: compute session is closed"},
		{name: "invalid_state", err: ErrComputeInvalidState, want: "mlx: invalid compute state"},
		{name: "invalid_descriptor", err: ErrComputeInvalidDescriptor, want: "mlx: invalid compute descriptor"},
		{name: "unsupported_pixel_format", err: ErrComputeUnsupportedPixelFormat, want: "mlx: unsupported pixel format"},
		{name: "invalid_buffer", err: ErrComputeInvalidBuffer, want: "mlx: invalid compute buffer"},
		{name: "buffer_size_mismatch", err: ErrComputeBufferSizeMismatch, want: "mlx: buffer size mismatch"},
		{name: "invalid_allocation", err: ErrComputeInvalidAllocation, want: "mlx: invalid compute allocation"},
		{name: "missing_kernel_buffer", err: ErrComputeMissingKernelBuffer, want: "mlx: missing kernel buffer"},
		{name: "invalid_kernel_args", err: ErrComputeInvalidKernelArgs, want: "mlx: invalid kernel arguments"},
		{name: "invalid_scalar", err: ErrComputeInvalidScalar, want: "mlx: invalid kernel scalar"},
		{name: "unknown_kernel", err: ErrComputeUnknownKernel, want: "mlx: unknown compute kernel"},
		{name: "internal", err: ErrComputeInternal, want: "mlx: internal compute error"},
		{name: "unknown", err: &ComputeError{}, want: "mlx: compute error"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.want {
				t.Fatalf("Error() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCompute_ComputeError_Error_Bad(t *testing.T) {
	cause := core.NewError("metal blew up")
	err := computeWrap(ComputeErrorInternal, "dispatch_kernel", KernelNearestScale, "dst", "dispatch failed", cause)
	if !core.Is(err, cause) {
		t.Fatalf("wrapped error does not expose cause")
	}
	if got := err.Error(); got != "mlx: dispatch failed: metal blew up" {
		t.Fatalf("Error() = %q, want wrapped detail", got)
	}
	if core.Is(err, &ComputeError{Kind: ComputeErrorInternal, Op: "other"}) {
		t.Fatalf("errors.Is matched mismatched op")
	}
	if core.Is(err, &ComputeError{Kind: ComputeErrorInternal, Kernel: KernelBilinearScale}) {
		t.Fatalf("errors.Is matched mismatched kernel")
	}
	if core.Is(err, &ComputeError{Kind: ComputeErrorInternal, Resource: "src"}) {
		t.Fatalf("errors.Is matched mismatched resource")
	}
}

func TestCompute_newSessionConfig_Options_Good(t *testing.T) {
	cfg := newSessionConfig([]SessionOption{
		WithSessionLabel("Render Pass"),
		nil,
		WithVerboseKernels(true),
		WithResetPeakMemory(false),
	})

	if cfg.label != "Render Pass" {
		t.Fatalf("label = %q, want %q", cfg.label, "Render Pass")
	}
	if !cfg.verboseKernels {
		t.Fatal("verboseKernels = false, want true")
	}
	if cfg.resetPeakMemory {
		t.Fatal("resetPeakMemory = true, want false")
	}

	defaults := newSessionConfig(nil)
	if !defaults.resetPeakMemory {
		t.Fatal("default resetPeakMemory = false, want true")
	}
}

func TestCompute_sanitizeComputeLabel_Good(t *testing.T) {
	cases := []struct {
		label string
		want  string
	}{
		{label: "__Hello--World__", want: "hello_world"},
		{label: "Ångström βeta 42", want: "ångström_βeta_42"},
		{label: "///", want: ""},
	}

	for _, tc := range cases {
		if got := sanitizeComputeLabel(tc.label); got != tc.want {
			t.Fatalf("sanitizeComputeLabel(%q) = %q, want %q", tc.label, got, tc.want)
		}
	}
}

func TestCompute_ComputeError_Is_Good(t *testing.T) {
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

func TestCompute_computeKernelRuntimeName_Good(t *testing.T) {
	got := computeKernelRuntimeName(" Retro Frame / P1 ", "frame_copy_scale")
	want := "compute_retro_frame_p1__frame_copy_scale"
	if got != want {
		t.Fatalf("computeKernelRuntimeName(...) = %q, want %q", got, want)
	}

	if got := computeKernelRuntimeName(" \t ", "frame_copy_scale"); got != "frame_copy_scale" {
		t.Fatalf("computeKernelRuntimeName(blank, kernel) = %q, want %q", got, "frame_copy_scale")
	}
}

func TestCompute_DefaultCompute_TinyKernelPipeline_Good(t *testing.T) {
	session := newTinyComputeSession(t)
	defer session.Close()

	if !DefaultCompute().Available() {
		t.Fatal("DefaultCompute().Available() = false after session creation")
	}
	if DefaultCompute().DeviceInfo().Architecture == "" {
		t.Fatal("DeviceInfo().Architecture is empty on available compute backend")
	}

	rgbaSrc := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{10, 20, 30, 40})
	bgraDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelBGRA8}, []byte{0, 0, 0, 0})
	if err := session.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame() error = %v", err)
	}
	if err := session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc},
		Outputs: map[string]Buffer{"dst": bgraDst},
	}); err != nil {
		t.Fatalf("Run(%s) error = %v", KernelRGBA8ToBGRA8, err)
	}
	frame, err := session.FinishFrame()
	if err != nil {
		t.Fatalf("FinishFrame() error = %v", err)
	}
	if frame.Passes != 1 || frame.LastKernel != KernelRGBA8ToBGRA8 {
		t.Fatalf("frame metrics = %+v, want one swizzle pass", frame)
	}
	assertBufferBytes(t, bgraDst, []byte{30, 20, 10, 40})

	roundTrip := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	runPixelKernel(t, session, KernelBGRA8ToRGBA8, map[string]Buffer{"src": bgraDst}, map[string]Buffer{"dst": roundTrip}, nil)
	assertBufferBytes(t, roundTrip, []byte{10, 20, 30, 40})

	nearestDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}, make([]byte, 16))
	runPixelKernel(t, session, KernelNearestScale, map[string]Buffer{"src": rgbaSrc}, map[string]Buffer{"dst": nearestDst}, nil)
	assertBufferBytes(t, nearestDst, []byte{
		10, 20, 30, 40, 10, 20, 30, 40,
		10, 20, 30, 40, 10, 20, 30, 40,
	})

	integerDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}, make([]byte, 16))
	runPixelKernel(t, session, KernelIntegerScale, map[string]Buffer{"src": rgbaSrc}, map[string]Buffer{"dst": integerDst}, nil)
	assertBufferBytes(t, integerDst, []byte{
		10, 20, 30, 40, 10, 20, 30, 40,
		10, 20, 30, 40, 10, 20, 30, 40,
	})

	bilinearDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	runPixelKernel(t, session, KernelBilinearScale, map[string]Buffer{"src": rgbaSrc}, map[string]Buffer{"dst": bilinearDst}, nil)
	assertBufferBytes(t, bilinearDst, []byte{10, 20, 30, 40})

	rgb565Src := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565}, []byte{0x00, 0xf8})
	rgb565Dst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	runPixelKernel(t, session, KernelRGB565ToRGBA8, map[string]Buffer{"src": rgb565Src}, map[string]Buffer{"dst": rgb565Dst}, nil)
	assertBufferBytes(t, rgb565Dst, []byte{255, 0, 0, 255})

	xrgbSrc := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelXRGB8888}, []byte{3, 2, 1, 0})
	xrgbDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	runPixelKernel(t, session, KernelXRGB8888ToRGBA8, map[string]Buffer{"src": xrgbSrc}, map[string]Buffer{"dst": xrgbDst}, nil)
	assertBufferBytes(t, xrgbDst, []byte{1, 2, 3, 255})

	indexedSrc := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 1, Format: PixelIndexed8}, []byte{2})
	palette := make([]byte, 256*4)
	copy(palette[8:12], []byte{9, 8, 7, 6})
	paletteBuffer := newByteBufferWithData(t, session, palette)
	paletteDst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	runPixelKernel(t, session, KernelPaletteExpandRGBA, map[string]Buffer{"src": indexedSrc, "palette": paletteBuffer}, map[string]Buffer{"dst": paletteDst}, nil)
	assertBufferBytes(t, paletteDst, []byte{9, 8, 7, 6})

	for _, kernel := range []string{KernelScanlineFilter, KernelCRTFilter, KernelSoftenFilter, KernelSharpenFilter} {
		dst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
		runPixelKernel(t, session, kernel, map[string]Buffer{"src": rgbaSrc}, map[string]Buffer{"dst": dst}, map[string]float64{"strength": 0.25, "scanline_strength": 0.25, "mask_strength": 0.25})
		if got, err := dst.Read(); err != nil || len(got) != 4 {
			t.Fatalf("%s Read() = %v/%v, want four bytes", kernel, got, err)
		}
	}

	metrics := session.Metrics()
	if metrics.Passes < 10 || metrics.LastKernel == "" {
		t.Fatalf("session metrics = %+v, want accumulated passes", metrics)
	}
	if err := session.Sync(); err != nil {
		t.Fatalf("Sync() error = %v", err)
	}
}

func TestCompute_Run_TinyErrorPaths_Bad(t *testing.T) {
	session := newTinyComputeSession(t)
	defer session.Close()

	if _, err := session.NewByteBuffer(0); !core.Is(err, ErrComputeInvalidAllocation) {
		t.Fatalf("NewByteBuffer(0) error = %v, want invalid allocation", err)
	}
	src := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{1, 2, 3, 4})
	dst := newPixelBufferWithData(t, session, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}, []byte{0, 0, 0, 0})
	bytes := newByteBufferWithData(t, session, []byte{1, 2, 3, 4})

	if err := src.Upload([]byte{1}); !core.Is(err, ErrComputeBufferSizeMismatch) {
		t.Fatalf("PixelBuffer.Upload(short) error = %v, want size mismatch", err)
	}
	if err := bytes.Upload([]byte{1}); !core.Is(err, ErrComputeBufferSizeMismatch) {
		t.Fatalf("ByteBuffer.Upload(short) error = %v, want size mismatch", err)
	}
	if err := session.Run("missing_kernel", KernelArgs{}); !core.Is(err, ErrComputeUnknownKernel) {
		t.Fatalf("Run(unknown) error = %v, want unknown kernel", err)
	}
	if err := session.Run(KernelNearestScale, KernelArgs{}); !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(missing buffers) error = %v, want missing buffer", err)
	}
	if err := session.Run(KernelNearestScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": bytes},
		Outputs: map[string]Buffer{"dst": dst},
	}); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(byte src) error = %v, want invalid buffer", err)
	}
	if err := session.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 2},
	}); !core.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("Run(invalid scalar) error = %v, want invalid scalar", err)
	}
	if err := session.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame() error = %v", err)
	}
	if err := session.BeginFrame(); !core.Is(err, ErrComputeInvalidState) {
		t.Fatalf("BeginFrame(active) error = %v, want invalid state", err)
	}
	if _, err := session.FinishFrame(); err != nil {
		t.Fatalf("FinishFrame() error = %v", err)
	}
	if _, err := session.FinishFrame(); !core.Is(err, ErrComputeInvalidState) {
		t.Fatalf("FinishFrame(inactive) error = %v, want invalid state", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := session.Run(KernelNearestScale, KernelArgs{}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("Run(closed) error = %v, want closed", err)
	}
	if err := session.Sync(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("Sync(closed) error = %v, want closed", err)
	}
	if _, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("NewPixelBuffer(closed) error = %v, want closed", err)
	}
	if _, err := session.NewByteBuffer(4); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("NewByteBuffer(closed) error = %v, want closed", err)
	}
	if _, err := src.Read(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("Read(closed) error = %v, want closed", err)
	}
}

func TestCompute_NewSession_UnavailableAndValidationPaths_Bad(t *testing.T) {
	_ = DefaultCompute().DeviceInfo()
	if _, err := NewSession(WithResetPeakMemory(false)); !DefaultCompute().Available() && !core.Is(err, ErrComputeUnavailable) {
		t.Fatalf("NewSession(unavailable) error = %v, want unavailable", err)
	}

	closed := &computesession{closed: true, kernels: map[string]*metal.MetalKernel{}, buffers: map[*bufferbase]struct{}{}}
	if err := closed.Close(); err != nil {
		t.Fatalf("Close(closed) error = %v", err)
	}
	if err := closed.BeginFrame(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("BeginFrame(closed) error = %v, want closed", err)
	}
	if _, err := closed.FinishFrame(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("FinishFrame(closed) error = %v, want closed", err)
	}
	if err := closed.Run(KernelNearestScale, KernelArgs{}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("Run(closed) error = %v, want closed", err)
	}
	if err := closed.Sync(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("Sync(closed) error = %v, want closed", err)
	}
	if _, err := closed.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("NewPixelBuffer(closed) error = %v, want closed", err)
	}
	if _, err := closed.NewByteBuffer(4); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("NewByteBuffer(closed) error = %v, want closed", err)
	}

	open := &computesession{kernels: map[string]*metal.MetalKernel{}, buffers: map[*bufferbase]struct{}{}}
	if _, err := open.NewPixelBuffer(PixelBufferDesc{}); !core.Is(err, ErrComputeInvalidDescriptor) {
		t.Fatalf("NewPixelBuffer(invalid desc) error = %v, want invalid descriptor", err)
	}
	if _, err := open.NewByteBuffer(0); !core.Is(err, ErrComputeInvalidAllocation) {
		t.Fatalf("NewByteBuffer(0) error = %v, want invalid allocation", err)
	}
	if _, err := open.NewByteBuffer(int(^uint32(0))); !core.Is(err, ErrComputeInvalidAllocation) {
		t.Fatalf("NewByteBuffer(large) error = %v, want invalid allocation", err)
	}
	if err := open.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame() error = %v", err)
	}
	if err := open.BeginFrame(); !core.Is(err, ErrComputeInvalidState) {
		t.Fatalf("BeginFrame(active) error = %v, want invalid state", err)
	}

	noFrame := &computesession{kernels: map[string]*metal.MetalKernel{}, buffers: map[*bufferbase]struct{}{}}
	if _, err := noFrame.FinishFrame(); !core.Is(err, ErrComputeInvalidState) {
		t.Fatalf("FinishFrame(inactive) error = %v, want invalid state", err)
	}
	if err := noFrame.Run("unknown_kernel", KernelArgs{}); !core.Is(err, ErrComputeUnknownKernel) {
		t.Fatalf("Run(unknown) error = %v, want unknown kernel", err)
	}
	if err := noFrame.Run(KernelNearestScale, KernelArgs{}); !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(missing buffers) error = %v, want missing buffer", err)
	}
	if err := noFrame.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame(noFrame) error = %v", err)
	}
	if got := noFrame.FrameMetrics(); got.Frame != 1 {
		t.Fatalf("FrameMetrics(active frame) = %+v, want frame 1", got)
	}
	_ = noFrame.Metrics()

	foreign := &computesession{kernels: map[string]*metal.MetalKernel{}, buffers: map[*bufferbase]struct{}{}}
	src := fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	dst := fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelBGRA8})
	other := fakeOpenPixelBuffer(foreign, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	bytes := fakeOpenByteBuffer(noFrame, 4)
	if err := noFrame.Run(KernelNearestScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": bytes},
		Outputs: map[string]Buffer{"dst": dst},
	}); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(byte src) error = %v, want invalid buffer", err)
	}
	if err := noFrame.Run(KernelNearestScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": other},
		Outputs: map[string]Buffer{"dst": dst},
	}); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(foreign src) error = %v, want invalid buffer", err)
	}
	if err := noFrame.Run(KernelNearestScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(format mismatch) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelIntegerScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 3, Height: 2, Stride: 12, Format: PixelRGBA8})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(integer mismatch) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(filter format mismatch) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})},
		Scalars: map[string]float64{"strength": 2},
	}); !core.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("Run(invalid scalar) error = %v, want invalid scalar", err)
	}

	if err := noFrame.Run(KernelBilinearScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(bilinear unsupported format) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgb565 bad source) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": dst},
		Outputs: map[string]Buffer{"dst": dst},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(swizzle bad source) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelXRGB8888ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(xrgb bad source) error = %v, want invalid args", err)
	}
	if err := noFrame.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs: map[string]Buffer{
			"src":     fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 1, Format: PixelIndexed8}),
			"palette": fakeOpenByteBuffer(noFrame, 4),
		},
		Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})},
	}); !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(short palette) error = %v, want invalid args", err)
	}
	for _, kernel := range []string{KernelCRTFilter, KernelSoftenFilter, KernelSharpenFilter} {
		if err := noFrame.Run(kernel, KernelArgs{
			Inputs:  map[string]Buffer{"src": src},
			Outputs: map[string]Buffer{"dst": fakeOpenPixelBuffer(noFrame, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})},
			Scalars: map[string]float64{"strength": 2, "mask_strength": 2},
		}); !core.Is(err, ErrComputeInvalidScalar) {
			t.Fatalf("Run(%s invalid scalar) error = %v, want invalid scalar", kernel, err)
		}
	}

	(&bufferbase{}).bufferHandle()
	if src.Size() != 4 || src.Descriptor().Format != PixelRGBA8 {
		t.Fatalf("fake pixel buffer = size %d desc %+v, want RGBA8 size 4", src.Size(), src.Descriptor())
	}
	closedPixel := fakeOpenPixelBuffer(closed, PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err := closedPixel.Upload([]byte{1, 2, 3, 4}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("closed PixelBuffer.Upload() error = %v, want closed", err)
	}
	if _, err := closedPixel.Read(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("closed PixelBuffer.Read() error = %v, want closed", err)
	}
	closedBytes := fakeOpenByteBuffer(closed, 4)
	if closedBytes.Size() != 4 {
		t.Fatalf("closed byte buffer size = %d, want 4", closedBytes.Size())
	}
	if err := closedBytes.Upload([]byte{1, 2, 3, 4}); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("closed ByteBuffer.Upload() error = %v, want closed", err)
	}
	if _, err := closedBytes.Read(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("closed ByteBuffer.Read() error = %v, want closed", err)
	}
	base := &bufferbase{session: noFrame}
	first := &metal.Array{}
	second := &metal.Array{}
	base.replaceLocked(first)
	base.replaceLocked(second)
	if len(noFrame.retired) == 0 {
		t.Fatal("replaceLocked did not retire previous array")
	}
}

func newTinyComputeSession(t *testing.T) Session {
	t.Helper()
	if !DefaultCompute().Available() {
		t.Skip("Metal compute is unavailable")
	}
	session, err := NewSession(WithSessionLabel("tiny coverage"), WithResetPeakMemory(false))
	if err != nil {
		if core.Is(err, ErrComputeUnavailable) {
			t.Skipf("Metal compute is unavailable: %v", err)
		}
		t.Fatalf("NewSession() error = %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func fakeOpenPixelBuffer(session *computesession, desc PixelBufferDesc) PixelBuffer {
	return &pixelbuffer{
		bufferbase: bufferbase{session: session, array: &metal.Array{}, size: desc.SizeBytes()},
		desc:       desc,
	}
}

func fakeOpenByteBuffer(session *computesession, size int) ByteBuffer {
	return &bytebuffer{bufferbase: bufferbase{session: session, array: &metal.Array{}, size: size}}
}

func newPixelBufferWithData(t *testing.T, session Session, desc PixelBufferDesc, data []byte) PixelBuffer {
	t.Helper()
	buffer, err := session.NewPixelBuffer(desc)
	if err != nil {
		t.Fatalf("NewPixelBuffer(%+v) error = %v", desc, err)
	}
	if err := buffer.Upload(data); err != nil {
		t.Fatalf("PixelBuffer.Upload(%+v) error = %v", desc, err)
	}
	return buffer
}

func newByteBufferWithData(t *testing.T, session Session, data []byte) ByteBuffer {
	t.Helper()
	buffer, err := session.NewByteBuffer(len(data))
	if err != nil {
		t.Fatalf("NewByteBuffer(%d) error = %v", len(data), err)
	}
	if err := buffer.Upload(data); err != nil {
		t.Fatalf("ByteBuffer.Upload(%d) error = %v", len(data), err)
	}
	return buffer
}

func runPixelKernel(t *testing.T, session Session, kernel string, inputs map[string]Buffer, outputs map[string]Buffer, scalars map[string]float64) {
	t.Helper()
	if err := session.Run(kernel, KernelArgs{Inputs: inputs, Outputs: outputs, Scalars: scalars}); err != nil {
		t.Fatalf("Run(%s) error = %v", kernel, err)
	}
}

func assertBufferBytes(t *testing.T, buffer interface{ Read() ([]byte, error) }, want []byte) {
	t.Helper()
	got, err := buffer.Read()
	if err != nil {
		t.Fatalf("Read() error = %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("Read() = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("Read() = %v, want %v", got, want)
		}
	}
}

// --- v0.9.0 audit triplets for compute.go public symbols ---

func TestCompute_ComputeError_Error_Ugly(t *testing.T) {
	// Ugly: a custom Message overrides the kind default, and a wrapped cause is
	// appended even when the kind is the zero value.
	err := &ComputeError{Message: "custom failure"}
	if got := err.Error(); got != "mlx: custom failure" {
		t.Fatalf("Error() = %q, want %q", got, "mlx: custom failure")
	}
	wrapped := &ComputeError{Err: core.NewError("root cause")}
	if got := wrapped.Error(); got != "mlx: compute error: root cause" {
		t.Fatalf("Error() = %q, want wrapped zero-kind detail", got)
	}
}

func TestCompute_ComputeError_Unwrap_Good(t *testing.T) {
	cause := core.NewError("backing store gone")
	err := computeWrap(ComputeErrorInternal, "read_buffer", "", "", "readback failed", cause).(*ComputeError)
	if got := err.Unwrap(); got != cause {
		t.Fatalf("Unwrap() = %v, want the wrapped cause", got)
	}
}

func TestCompute_ComputeError_Unwrap_Bad(t *testing.T) {
	// Bad: an error constructed without a cause unwraps to nil, so a chain walk
	// terminates rather than matching a foreign sentinel.
	err := computeErr(ComputeErrorClosed, "require_buffer", "", "", "closed")
	if got := err.(*ComputeError).Unwrap(); got != nil {
		t.Fatalf("Unwrap() = %v, want nil for an unwrapped error", got)
	}
}

func TestCompute_ComputeError_Unwrap_Ugly(t *testing.T) {
	// Ugly: a directly nested ComputeError unwraps one level at a time.
	inner := &ComputeError{Kind: ComputeErrorInvalidScalar}
	outer := &ComputeError{Kind: ComputeErrorInternal, Err: inner}
	if got := outer.Unwrap(); got != inner {
		t.Fatalf("Unwrap() = %v, want the inner ComputeError", got)
	}
	if got := inner.Unwrap(); got != nil {
		t.Fatalf("inner Unwrap() = %v, want nil", got)
	}
}

func TestCompute_ComputeError_Is_Bad(t *testing.T) {
	// Bad: a non-ComputeError target never matches, and a kind mismatch is rejected.
	err := &ComputeError{Kind: ComputeErrorClosed}
	if err.Is(core.NewError("plain")) {
		t.Fatal("Is(plain error) = true, want false")
	}
	if err.Is(&ComputeError{Kind: ComputeErrorUnavailable}) {
		t.Fatal("Is(mismatched kind) = true, want false")
	}
}

func TestCompute_ComputeError_Is_Ugly(t *testing.T) {
	// Ugly: an empty template matches by kind only; every populated field on the
	// template must also match for Is to report true.
	err := &ComputeError{Kind: ComputeErrorInvalidKernelArgs, Op: "dispatch", Kernel: KernelCRTFilter, Resource: "dst"}
	if !err.Is(&ComputeError{}) {
		t.Fatal("Is(empty template) = false, want true")
	}
	if !err.Is(&ComputeError{Kind: ComputeErrorInvalidKernelArgs, Op: "dispatch", Kernel: KernelCRTFilter, Resource: "dst"}) {
		t.Fatal("Is(fully matching template) = false, want true")
	}
	if err.Is(&ComputeError{Resource: "src"}) {
		t.Fatal("Is(mismatched resource) = true, want false")
	}
}

func TestCompute_PixelFormat_BytesPerPixel_Bad(t *testing.T) {
	// Bad: an unrecognised format reports 0 bytes-per-pixel, which the validator
	// treats as an unsupported layout.
	if got := PixelFormat("rgba16").BytesPerPixel(); got != 0 {
		t.Fatalf("BytesPerPixel() = %d, want 0 for an unknown format", got)
	}
	if got := PixelFormat("").BytesPerPixel(); got != 0 {
		t.Fatalf("BytesPerPixel(empty) = %d, want 0", got)
	}
}

func TestCompute_PixelFormat_BytesPerPixel_Ugly(t *testing.T) {
	// Ugly: case sensitivity is significant — an upper-cased spelling of a known
	// format is not recognised and reports 0.
	if got := PixelFormat("RGBA8").BytesPerPixel(); got != 0 {
		t.Fatalf("BytesPerPixel(RGBA8 upper) = %d, want 0", got)
	}
	if got := PixelRGB565.BytesPerPixel(); got != 2 {
		t.Fatalf("BytesPerPixel(rgb565) = %d, want 2", got)
	}
}

func TestCompute_PixelBufferDesc_Validate_Good(t *testing.T) {
	desc := PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}
	if err := desc.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil for a well-formed descriptor", err)
	}
}

func TestCompute_PixelBufferDesc_SizeBytes_Ugly(t *testing.T) {
	// Ugly: a one-row indexed buffer is the minimum valid case; SizeBytes is
	// exactly Height*Stride for a descriptor that passes validation.
	desc := PixelBufferDesc{Width: 1, Height: 1, Stride: 1, Format: PixelIndexed8}
	if got := desc.SizeBytes(); got != 1 {
		t.Fatalf("SizeBytes() = %d, want 1", got)
	}
	tall := PixelBufferDesc{Width: 1, Height: 3, Stride: 4, Format: PixelRGBA8}
	if got := tall.SizeBytes(); got != 12 {
		t.Fatalf("SizeBytes() = %d, want 12", got)
	}
}

func TestCompute_WithSessionLabel_Good(t *testing.T) {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("Render Pass")})
	if cfg.label != "Render Pass" {
		t.Fatalf("WithSessionLabel set label = %q, want %q", cfg.label, "Render Pass")
	}
}

func TestCompute_WithSessionLabel_Bad(t *testing.T) {
	// Bad: an empty label leaves the config label empty, so no kernel prefix is
	// later derived from it.
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("")})
	if cfg.label != "" {
		t.Fatalf("WithSessionLabel(\"\") label = %q, want empty", cfg.label)
	}
}

func TestCompute_WithSessionLabel_Ugly(t *testing.T) {
	// Ugly: the last WithSessionLabel applied wins when several are supplied.
	cfg := newSessionConfig([]SessionOption{
		WithSessionLabel("first"),
		WithSessionLabel("second"),
	})
	if cfg.label != "second" {
		t.Fatalf("WithSessionLabel last-wins label = %q, want %q", cfg.label, "second")
	}
}

func TestCompute_WithVerboseKernels_Good(t *testing.T) {
	cfg := newSessionConfig([]SessionOption{WithVerboseKernels(true)})
	if !cfg.verboseKernels {
		t.Fatal("WithVerboseKernels(true) verboseKernels = false, want true")
	}
}

func TestCompute_WithVerboseKernels_Bad(t *testing.T) {
	// Bad: WithVerboseKernels(false) leaves verbose logging off, matching the
	// zero-value default.
	cfg := newSessionConfig([]SessionOption{WithVerboseKernels(false)})
	if cfg.verboseKernels {
		t.Fatal("WithVerboseKernels(false) verboseKernels = true, want false")
	}
}

func TestCompute_WithVerboseKernels_Ugly(t *testing.T) {
	// Ugly: a later WithVerboseKernels overrides an earlier one.
	cfg := newSessionConfig([]SessionOption{
		WithVerboseKernels(true),
		WithVerboseKernels(false),
	})
	if cfg.verboseKernels {
		t.Fatal("WithVerboseKernels last-wins verboseKernels = true, want false")
	}
}

func TestCompute_WithResetPeakMemory_Good(t *testing.T) {
	cfg := newSessionConfig([]SessionOption{WithResetPeakMemory(false)})
	if cfg.resetPeakMemory {
		t.Fatal("WithResetPeakMemory(false) resetPeakMemory = true, want false")
	}
}

func TestCompute_WithResetPeakMemory_Bad(t *testing.T) {
	// Bad: WithResetPeakMemory(true) restores the reset behaviour even though the
	// default already enables it, so the option is observable on its own.
	cfg := newSessionConfig([]SessionOption{WithResetPeakMemory(false), WithResetPeakMemory(true)})
	if !cfg.resetPeakMemory {
		t.Fatal("WithResetPeakMemory(true) after false resetPeakMemory = false, want true")
	}
}

func TestCompute_WithResetPeakMemory_Ugly(t *testing.T) {
	// Ugly: the no-options default leaves resetPeakMemory enabled; a single
	// WithResetPeakMemory(false) is the only way to disable it.
	defaults := newSessionConfig(nil)
	if !defaults.resetPeakMemory {
		t.Fatal("default resetPeakMemory = false, want true")
	}
	opted := newSessionConfig([]SessionOption{WithResetPeakMemory(false)})
	if opted.resetPeakMemory {
		t.Fatal("WithResetPeakMemory(false) resetPeakMemory = true, want false")
	}
}
