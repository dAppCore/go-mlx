package mlx

import (
	"errors"
	"time"
)

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
		return errors.New("mlx: pixel buffer width must be positive")
	}
	if desc.Height <= 0 {
		return errors.New("mlx: pixel buffer height must be positive")
	}
	if desc.Stride <= 0 {
		return errors.New("mlx: pixel buffer stride must be positive")
	}
	bytesPerPixel := desc.Format.BytesPerPixel()
	if bytesPerPixel == 0 {
		return errors.New("mlx: unsupported pixel format")
	}
	if desc.Stride < desc.Width*bytesPerPixel {
		return errors.New("mlx: pixel buffer stride is smaller than width * bytes_per_pixel")
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
	NewPixelBuffer(desc PixelBufferDesc) (PixelBuffer, error)
	NewByteBuffer(size int) (ByteBuffer, error)
	Run(kernel string, args KernelArgs) error
	Sync() error
	Metrics() SessionMetrics
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
)
