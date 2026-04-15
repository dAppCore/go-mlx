package mlx

import (
	"strings"
	"time"
	"unicode"
)

// ComputeErrorKind classifies non-LLM compute failures for frame-oriented callers.
type ComputeErrorKind string

const (
	ComputeErrorUnavailable            ComputeErrorKind = "unavailable"
	ComputeErrorClosed                 ComputeErrorKind = "closed"
	ComputeErrorInvalidState           ComputeErrorKind = "invalid_state"
	ComputeErrorInvalidDescriptor      ComputeErrorKind = "invalid_descriptor"
	ComputeErrorUnsupportedPixelFormat ComputeErrorKind = "unsupported_pixel_format"
	ComputeErrorInvalidBuffer          ComputeErrorKind = "invalid_buffer"
	ComputeErrorBufferSizeMismatch     ComputeErrorKind = "buffer_size_mismatch"
	ComputeErrorInvalidAllocation      ComputeErrorKind = "invalid_allocation"
	ComputeErrorMissingKernelBuffer    ComputeErrorKind = "missing_kernel_buffer"
	ComputeErrorInvalidKernelArgs      ComputeErrorKind = "invalid_kernel_args"
	ComputeErrorInvalidScalar          ComputeErrorKind = "invalid_scalar"
	ComputeErrorUnknownKernel          ComputeErrorKind = "unknown_kernel"
	ComputeErrorInternal               ComputeErrorKind = "internal"
)

var (
	ErrComputeUnavailable            = &ComputeError{Kind: ComputeErrorUnavailable}
	ErrComputeClosed                 = &ComputeError{Kind: ComputeErrorClosed}
	ErrComputeInvalidState           = &ComputeError{Kind: ComputeErrorInvalidState}
	ErrComputeInvalidDescriptor      = &ComputeError{Kind: ComputeErrorInvalidDescriptor}
	ErrComputeUnsupportedPixelFormat = &ComputeError{Kind: ComputeErrorUnsupportedPixelFormat}
	ErrComputeInvalidBuffer          = &ComputeError{Kind: ComputeErrorInvalidBuffer}
	ErrComputeBufferSizeMismatch     = &ComputeError{Kind: ComputeErrorBufferSizeMismatch}
	ErrComputeInvalidAllocation      = &ComputeError{Kind: ComputeErrorInvalidAllocation}
	ErrComputeMissingKernelBuffer    = &ComputeError{Kind: ComputeErrorMissingKernelBuffer}
	ErrComputeInvalidKernelArgs      = &ComputeError{Kind: ComputeErrorInvalidKernelArgs}
	ErrComputeInvalidScalar          = &ComputeError{Kind: ComputeErrorInvalidScalar}
	ErrComputeUnknownKernel          = &ComputeError{Kind: ComputeErrorUnknownKernel}
	ErrComputeInternal               = &ComputeError{Kind: ComputeErrorInternal}
)

// ComputeError is the structured error returned by the non-LLM compute API.
type ComputeError struct {
	Kind     ComputeErrorKind
	Op       string
	Kernel   string
	Resource string
	Message  string
	Err      error
}

func (err *ComputeError) Error() string {
	if err == nil {
		return "<nil>"
	}
	msg := err.Message
	if msg == "" {
		switch err.Kind {
		case ComputeErrorUnavailable:
			msg = "Metal compute is unavailable"
		case ComputeErrorClosed:
			msg = "compute session is closed"
		case ComputeErrorInvalidState:
			msg = "invalid compute state"
		case ComputeErrorInvalidDescriptor:
			msg = "invalid compute descriptor"
		case ComputeErrorUnsupportedPixelFormat:
			msg = "unsupported pixel format"
		case ComputeErrorInvalidBuffer:
			msg = "invalid compute buffer"
		case ComputeErrorBufferSizeMismatch:
			msg = "buffer size mismatch"
		case ComputeErrorInvalidAllocation:
			msg = "invalid compute allocation"
		case ComputeErrorMissingKernelBuffer:
			msg = "missing kernel buffer"
		case ComputeErrorInvalidKernelArgs:
			msg = "invalid kernel arguments"
		case ComputeErrorInvalidScalar:
			msg = "invalid kernel scalar"
		case ComputeErrorUnknownKernel:
			msg = "unknown compute kernel"
		case ComputeErrorInternal:
			msg = "internal compute error"
		default:
			msg = "compute error"
		}
	}
	if err.Err != nil {
		return "mlx: " + msg + ": " + err.Err.Error()
	}
	return "mlx: " + msg
}

func (err *ComputeError) Unwrap() error { return err.Err }

func (err *ComputeError) Is(target error) bool {
	want, ok := target.(*ComputeError)
	if !ok {
		return false
	}
	if want.Kind != "" && err.Kind != want.Kind {
		return false
	}
	if want.Op != "" && err.Op != want.Op {
		return false
	}
	if want.Kernel != "" && err.Kernel != want.Kernel {
		return false
	}
	if want.Resource != "" && err.Resource != want.Resource {
		return false
	}
	return true
}

func computeErr(kind ComputeErrorKind, op, kernel, resource, message string) error {
	return &ComputeError{
		Kind:     kind,
		Op:       op,
		Kernel:   kernel,
		Resource: resource,
		Message:  message,
	}
}

func computeWrap(kind ComputeErrorKind, op, kernel, resource, message string, err error) error {
	return &ComputeError{
		Kind:     kind,
		Op:       op,
		Kernel:   kernel,
		Resource: resource,
		Message:  message,
		Err:      err,
	}
}

// PixelFormat identifies the layout of a packed pixel buffer.
type PixelFormat string

const (
	PixelRGBA8    PixelFormat = "rgba8"
	PixelBGRA8    PixelFormat = "bgra8"
	PixelRGB565   PixelFormat = "rgb565"
	PixelXRGB8888 PixelFormat = "xrgb8888"
	PixelIndexed8 PixelFormat = "indexed8"
)

// BytesPerPixel reports the packed bytes-per-pixel for the format.
func (format PixelFormat) BytesPerPixel() int {
	switch format {
	case PixelRGBA8, PixelBGRA8, PixelXRGB8888:
		return 4
	case PixelRGB565:
		return 2
	case PixelIndexed8:
		return 1
	default:
		return 0
	}
}

// PixelBufferDesc describes one packed image buffer.
type PixelBufferDesc struct {
	Width  int
	Height int
	Stride int
	Format PixelFormat
}

// Validate checks whether the descriptor can back a packed pixel buffer.
func (desc PixelBufferDesc) Validate() error {
	if desc.Width <= 0 {
		return computeErr(ComputeErrorInvalidDescriptor, "validate_pixel_buffer", "", "width", "pixel buffer width must be positive")
	}
	if desc.Height <= 0 {
		return computeErr(ComputeErrorInvalidDescriptor, "validate_pixel_buffer", "", "height", "pixel buffer height must be positive")
	}
	if desc.Stride <= 0 {
		return computeErr(ComputeErrorInvalidDescriptor, "validate_pixel_buffer", "", "stride", "pixel buffer stride must be positive")
	}
	bytesPerPixel := desc.Format.BytesPerPixel()
	if bytesPerPixel == 0 {
		return computeErr(ComputeErrorUnsupportedPixelFormat, "validate_pixel_buffer", "", "format", "unsupported pixel format")
	}
	if desc.Stride < desc.Width*bytesPerPixel {
		return computeErr(ComputeErrorInvalidDescriptor, "validate_pixel_buffer", "", "stride", "pixel buffer stride is smaller than width * bytes_per_pixel")
	}
	return nil
}

// SizeBytes reports the total packed byte length of the buffer.
func (desc PixelBufferDesc) SizeBytes() int {
	return desc.Height * desc.Stride
}

// SessionOption configures a compute session.
type SessionOption func(*sessionConfig)

type sessionConfig struct {
	label           string
	verboseKernels  bool
	resetPeakMemory bool
}

func newSessionConfig(opts []SessionOption) sessionConfig {
	cfg := sessionConfig{resetPeakMemory: true}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}

// WithSessionLabel attaches a human-readable label to a compute session.
// The label is folded into compiled kernel names so verbose kernel logs can be
// tied back to a specific frame pipeline.
func WithSessionLabel(label string) SessionOption {
	return func(cfg *sessionConfig) {
		cfg.label = label
	}
}

// WithVerboseKernels enables verbose kernel compilation logging for the session.
func WithVerboseKernels(verbose bool) SessionOption {
	return func(cfg *sessionConfig) {
		cfg.verboseKernels = verbose
	}
}

// WithResetPeakMemory controls whether session creation resets the global MLX peak counter.
func WithResetPeakMemory(reset bool) SessionOption {
	return func(cfg *sessionConfig) {
		cfg.resetPeakMemory = reset
	}
}

// SessionMetrics reports coarse timing and memory figures for a compute session.
type SessionMetrics struct {
	Passes                int
	LastKernel            string
	LastDispatchDuration  time.Duration
	LastSyncDuration      time.Duration
	TotalDispatchDuration time.Duration
	TotalSyncDuration     time.Duration
	ActiveMemoryBytes     uint64
	PeakMemoryBytes       uint64
}

// FrameMetrics reports timing and memory figures for a single frame lifecycle.
type FrameMetrics struct {
	Frame             int
	Passes            int
	LastKernel        string
	DispatchDuration  time.Duration
	SyncDuration      time.Duration
	TotalDuration     time.Duration
	ActiveMemoryBytes uint64
	PeakMemoryBytes   uint64
}

func sanitizeComputeLabel(label string) string {
	label = strings.TrimSpace(label)
	if label == "" {
		return ""
	}

	var builder strings.Builder
	lastUnderscore := false
	for _, r := range label {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			builder.WriteRune(r)
			lastUnderscore = false
		case r >= 'A' && r <= 'Z':
			builder.WriteRune(unicode.ToLower(r))
			lastUnderscore = false
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			builder.WriteRune(unicode.ToLower(r))
			lastUnderscore = false
		default:
			if builder.Len() > 0 && !lastUnderscore {
				builder.WriteByte('_')
				lastUnderscore = true
			}
		}
	}

	return strings.Trim(builder.String(), "_")
}

func computeKernelRuntimeName(sessionLabel, kernelName string) string {
	label := sanitizeComputeLabel(sessionLabel)
	if label == "" {
		return kernelName
	}
	return "compute_" + label + "__" + kernelName
}

// Buffer is a device-resident compute buffer.
type Buffer interface {
	Size() int
	bufferHandle()
}

// PixelBuffer is a packed image buffer stored on the compute device.
type PixelBuffer interface {
	Buffer
	Descriptor() PixelBufferDesc
	Upload(data []byte) error
	Read() ([]byte, error)
}

// ByteBuffer is a generic device-resident byte buffer.
type ByteBuffer interface {
	Buffer
	Upload(data []byte) error
	Read() ([]byte, error)
}

// KernelArgs groups named inputs, outputs, and scalar parameters for a kernel dispatch.
type KernelArgs struct {
	Inputs  map[string]Buffer
	Outputs map[string]Buffer
	Scalars map[string]float64
}

// Compute is the public non-LLM Metal compute surface for frame workloads.
type Compute interface {
	Available() bool
	DeviceInfo() DeviceInfo
	NewSession(opts ...SessionOption) (Session, error)
}

// Session owns a set of device buffers and reusable kernel state.
type Session interface {
	Close() error
	BeginFrame() error
	FinishFrame() (FrameMetrics, error)
	NewPixelBuffer(desc PixelBufferDesc) (PixelBuffer, error)
	NewByteBuffer(size int) (ByteBuffer, error)
	Run(kernel string, args KernelArgs) error
	Sync() error
	Metrics() SessionMetrics
	FrameMetrics() FrameMetrics
}

const (
	KernelNearestScale      = "nearest_scale"
	KernelBilinearScale     = "bilinear_scale"
	KernelIntegerScale      = "integer_scale"
	KernelRGB565ToRGBA8     = "rgb565_to_rgba8"
	KernelRGBA8ToBGRA8      = "rgba8_to_bgra8"
	KernelBGRA8ToRGBA8      = "bgra8_to_rgba8"
	KernelXRGB8888ToRGBA8   = "xrgb8888_to_rgba8"
	KernelPaletteExpandRGBA = "palette_expand_rgba8"
	KernelScanlineFilter    = "scanline_filter"
	KernelCRTFilter         = "crt_filter"
	KernelSoftenFilter      = "soften_filter"
	KernelSharpenFilter     = "sharpen_filter"
)
