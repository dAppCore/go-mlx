//go:build darwin && arm64 && !nomlx

package mlx

import (
	"errors"
	"math"
	"sync"
	"time"

	"dappco.re/go/mlx/internal/metal"
)

var defaultComputeBackend Compute = computeBackend{}

// DefaultCompute returns the package's default Metal compute backend.
func DefaultCompute() Compute { return defaultComputeBackend }

// NewSession creates a compute session from the default Metal backend.
func NewSession(opts ...SessionOption) (Session, error) {
	return defaultComputeBackend.NewSession(opts...)
}

type computeBackend struct{}

func (computeBackend) Available() bool        { return MetalAvailable() }
func (computeBackend) DeviceInfo() DeviceInfo { return GetDeviceInfo() }

func (computeBackend) NewSession(opts ...SessionOption) (Session, error) {
	if !MetalAvailable() {
		return nil, errors.New("mlx: Metal compute is unavailable")
	}

	cfg := newSessionConfig(opts)
	if cfg.resetPeakMemory {
		metal.ResetPeakMemory()
	}

	return &computeSession{
		cfg:              cfg,
		kernels:          make(map[string]*metal.MetalKernel),
		buffers:          make(map[*bufferBase]struct{}),
		baseActiveMemory: metal.GetActiveMemory(),
		basePeakMemory:   metal.GetPeakMemory(),
	}, nil
}

type computeSession struct {
	mu               sync.Mutex
	cfg              sessionConfig
	kernels          map[string]*metal.MetalKernel
	buffers          map[*bufferBase]struct{}
	metrics          SessionMetrics
	baseActiveMemory uint64
	basePeakMemory   uint64
	closed           bool
}

type bufferBase struct {
	session *computeSession
	array   *metal.Array
	size    int
}

func (*bufferBase) bufferHandle() {}

func (base *bufferBase) Size() int { return base.size }

func (base *bufferBase) requireOpenLocked() error {
	if base == nil || base.session == nil {
		return errors.New("mlx: buffer is nil")
	}
	if base.session.closed {
		return errors.New("mlx: compute session is closed")
	}
	if base.array == nil {
		return errors.New("mlx: buffer has no backing storage")
	}
	return nil
}

func (base *bufferBase) replaceLocked(next *metal.Array) {
	if base.array != nil {
		metal.Free(base.array)
	}
	base.array = next
}

func (base *bufferBase) readLocked() ([]byte, error) {
	if err := base.requireOpenLocked(); err != nil {
		return nil, err
	}
	if err := base.session.syncLocked(); err != nil {
		return nil, err
	}
	return base.array.Bytes(), nil
}

type pixelBuffer struct {
	bufferBase
	desc PixelBufferDesc
}

func (buffer *pixelBuffer) Descriptor() PixelBufferDesc { return buffer.desc }

func (buffer *pixelBuffer) Upload(data []byte) error {
	buffer.session.mu.Lock()
	defer buffer.session.mu.Unlock()

	if err := buffer.requireOpenLocked(); err != nil {
		return err
	}
	if len(data) != buffer.size {
		return errors.New("mlx: pixel buffer upload size does not match descriptor")
	}
	next := metal.FromValues(data, buffer.desc.Height, buffer.desc.Stride)
	buffer.replaceLocked(next)
	return nil
}

func (buffer *pixelBuffer) Read() ([]byte, error) {
	buffer.session.mu.Lock()
	defer buffer.session.mu.Unlock()
	return buffer.readLocked()
}

type byteBuffer struct {
	bufferBase
}

func (buffer *byteBuffer) Upload(data []byte) error {
	buffer.session.mu.Lock()
	defer buffer.session.mu.Unlock()

	if err := buffer.requireOpenLocked(); err != nil {
		return err
	}
	if len(data) != buffer.size {
		return errors.New("mlx: byte buffer upload size does not match allocation")
	}
	next := metal.FromValues(data, len(data))
	buffer.replaceLocked(next)
	return nil
}

func (buffer *byteBuffer) Read() ([]byte, error) {
	buffer.session.mu.Lock()
	defer buffer.session.mu.Unlock()
	return buffer.readLocked()
}

func (session *computeSession) Close() error {
	session.mu.Lock()
	defer session.mu.Unlock()

	if session.closed {
		return nil
	}

	for base := range session.buffers {
		if base.array != nil {
			metal.Free(base.array)
			base.array = nil
		}
	}
	for name, kernel := range session.kernels {
		if kernel != nil {
			kernel.Free()
			session.kernels[name] = nil
		}
	}
	session.closed = true
	return nil
}

func (session *computeSession) NewPixelBuffer(desc PixelBufferDesc) (PixelBuffer, error) {
	if err := desc.Validate(); err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()

	if session.closed {
		return nil, errors.New("mlx: compute session is closed")
	}

	buffer := &pixelBuffer{
		bufferBase: bufferBase{
			session: session,
			array:   metal.Zeros([]int32{int32(desc.Height), int32(desc.Stride)}, metal.DTypeUint8),
			size:    desc.SizeBytes(),
		},
		desc: desc,
	}
	session.buffers[&buffer.bufferBase] = struct{}{}
	return buffer, nil
}

func (session *computeSession) NewByteBuffer(size int) (ByteBuffer, error) {
	if size <= 0 {
		return nil, errors.New("mlx: byte buffer size must be positive")
	}

	session.mu.Lock()
	defer session.mu.Unlock()

	if session.closed {
		return nil, errors.New("mlx: compute session is closed")
	}

	buffer := &byteBuffer{
		bufferBase: bufferBase{
			session: session,
			array:   metal.Zeros([]int32{int32(size)}, metal.DTypeUint8),
			size:    size,
		},
	}
	session.buffers[&buffer.bufferBase] = struct{}{}
	return buffer, nil
}

func (session *computeSession) Run(kernel string, args KernelArgs) error {
	session.mu.Lock()
	defer session.mu.Unlock()

	if session.closed {
		return errors.New("mlx: compute session is closed")
	}

	start := time.Now()
	err := session.runLocked(kernel, args)
	dispatchDuration := time.Since(start)
	if err != nil {
		return err
	}

	session.metrics.Passes++
	session.metrics.LastKernel = kernel
	session.metrics.LastDispatchDuration = dispatchDuration
	session.metrics.TotalDispatchDuration += dispatchDuration
	session.updateMemoryMetricsLocked()
	return nil
}

func (session *computeSession) Sync() error {
	session.mu.Lock()
	defer session.mu.Unlock()
	return session.syncLocked()
}

func (session *computeSession) Metrics() SessionMetrics {
	session.mu.Lock()
	defer session.mu.Unlock()
	return session.metrics
}

func (session *computeSession) syncLocked() error {
	if session.closed {
		return errors.New("mlx: compute session is closed")
	}
	start := time.Now()
	metal.Synchronize(metal.DefaultStream())
	syncDuration := time.Since(start)
	session.metrics.LastSyncDuration = syncDuration
	session.metrics.TotalSyncDuration += syncDuration
	session.updateMemoryMetricsLocked()
	return nil
}

func (session *computeSession) updateMemoryMetricsLocked() {
	active := metal.GetActiveMemory()
	peak := metal.GetPeakMemory()
	if active >= session.baseActiveMemory {
		session.metrics.ActiveMemoryBytes = active - session.baseActiveMemory
	}
	if peak >= session.basePeakMemory {
		session.metrics.PeakMemoryBytes = peak - session.basePeakMemory
	}
}

func (session *computeSession) runLocked(kernel string, args KernelArgs) error {
	switch kernel {
	case KernelNearestScale:
		return session.runNearestScaleLocked(args, false)
	case KernelIntegerScale:
		return session.runNearestScaleLocked(args, true)
	case KernelBilinearScale:
		return session.runBilinearScaleLocked(args)
	case KernelRGB565ToRGBA8:
		return session.runRGB565ToRGBA8Locked(args)
	case KernelRGBA8ToBGRA8, KernelBGRA8ToRGBA8:
		return session.runChannelSwizzleLocked(args)
	case KernelXRGB8888ToRGBA8:
		return session.runXRGB8888ToRGBA8Locked(args)
	case KernelPaletteExpandRGBA:
		return session.runPaletteExpandLocked(args)
	case KernelScanlineFilter:
		return session.runScanlineFilterLocked(args)
	case KernelCRTFilter:
		return session.runCRTFilterLocked(args)
	default:
		return errors.New("mlx: unknown compute kernel")
	}
}

type kernelSpec struct {
	inputNames  []string
	outputNames []string
	source      string
}

var computeKernelSpecs = map[string]kernelSpec{
	"frame_copy_scale": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint dst_x = thread_position_in_grid.x;
uint dst_y = thread_position_in_grid.y;
if (dst_x >= DST_WIDTH || dst_y >= DST_HEIGHT) {
    return;
}
uint src_x = (dst_x * SRC_WIDTH) / DST_WIDTH;
uint src_y = (dst_y * SRC_HEIGHT) / DST_HEIGHT;
uint src_index = src_y * SRC_STRIDE + src_x * BPP;
uint dst_index = dst_y * DST_STRIDE + dst_x * BPP;
for (int channel = 0; channel < BPP; channel++) {
    dst[dst_index + channel] = src[src_index + channel];
}`,
	},
	"frame_bilinear_rgba": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint dst_x = thread_position_in_grid.x;
uint dst_y = thread_position_in_grid.y;
if (dst_x >= DST_WIDTH || dst_y >= DST_HEIGHT) {
    return;
}
float src_x = ((float(dst_x) + 0.5f) * float(SRC_WIDTH) / float(DST_WIDTH)) - 0.5f;
float src_y = ((float(dst_y) + 0.5f) * float(SRC_HEIGHT) / float(DST_HEIGHT)) - 0.5f;
int x0 = int(metal::floor(src_x));
int y0 = int(metal::floor(src_y));
float tx = src_x - float(x0);
float ty = src_y - float(y0);
x0 = metal::clamp(x0, 0, SRC_WIDTH - 1);
y0 = metal::clamp(y0, 0, SRC_HEIGHT - 1);
int x1 = metal::clamp(x0 + 1, 0, SRC_WIDTH - 1);
int y1 = metal::clamp(y0 + 1, 0, SRC_HEIGHT - 1);
uint dst_index = dst_y * DST_STRIDE + dst_x * 4;
uint tl = uint(y0) * SRC_STRIDE + uint(x0) * 4;
uint tr = uint(y0) * SRC_STRIDE + uint(x1) * 4;
uint bl = uint(y1) * SRC_STRIDE + uint(x0) * 4;
uint br = uint(y1) * SRC_STRIDE + uint(x1) * 4;
for (int channel = 0; channel < 4; channel++) {
    float top = float(src[tl + uint(channel)]) + (float(src[tr + uint(channel)]) - float(src[tl + uint(channel)])) * tx;
    float bottom = float(src[bl + uint(channel)]) + (float(src[br + uint(channel)]) - float(src[bl + uint(channel)])) * tx;
    float value = top + (bottom - top) * ty;
    dst[dst_index + uint(channel)] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}`,
	},
	"frame_rgb565_to_rgba8": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 2;
ushort packed = ushort(src[src_index]) | (ushort(src[src_index + 1]) << 8);
uchar r = uchar((((packed >> 11) & 0x1F) * 255 + 15) / 31);
uchar g = uchar((((packed >> 5) & 0x3F) * 255 + 31) / 63);
uchar b = uchar(((packed & 0x1F) * 255 + 15) / 31);
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = r;
dst[dst_index + 1] = g;
dst[dst_index + 2] = b;
dst[dst_index + 3] = 255;`,
	},
	"frame_channel_swizzle": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 4;
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = src[src_index + 2];
dst[dst_index + 1] = src[src_index + 1];
dst[dst_index + 2] = src[src_index + 0];
dst[dst_index + 3] = src[src_index + 3];`,
	},
	"frame_xrgb8888_to_rgba8": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 4;
uint dst_index = y * DST_STRIDE + x * 4;
uchar b = src[src_index + 0];
uchar g = src[src_index + 1];
uchar r = src[src_index + 2];
dst[dst_index + 0] = r;
dst[dst_index + 1] = g;
dst[dst_index + 2] = b;
dst[dst_index + 3] = 255;`,
	},
	"frame_palette_expand_rgba8": {
		inputNames:  []string{"src", "palette"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x;
uint palette_index = uint(src[src_index]) * 4;
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = palette[palette_index + 0];
dst[dst_index + 1] = palette[palette_index + 1];
dst[dst_index + 2] = palette[palette_index + 2];
dst[dst_index + 3] = palette[palette_index + 3];`,
	},
	"frame_scanline_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
float scan = ((y & 1u) == 0u) ? 1.0f : (1.0f - float(STRENGTH) / 256.0f);
for (uint channel = 0; channel < 3; channel++) {
    float value = float(src[index + channel]) * scan;
    dst[index + channel] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}
dst[index + 3] = src[index + 3];`,
	},
	"frame_crt_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
uint r_index = BGRA_ORDER ? 2u : 0u;
uint g_index = 1u;
uint b_index = BGRA_ORDER ? 0u : 2u;
float scan = ((y & 1u) == 0u) ? 1.0f : (1.0f - float(SCANLINE_STRENGTH) / 256.0f);
float shadow = 1.0f - float(MASK_STRENGTH) / 256.0f;
float r_mask = shadow;
float g_mask = shadow;
float b_mask = shadow;
switch (x % 3u) {
case 0u:
    r_mask = 1.0f;
    break;
case 1u:
    g_mask = 1.0f;
    break;
default:
    b_mask = 1.0f;
    break;
}
float r = float(src[index + r_index]) * scan * r_mask;
float g = float(src[index + g_index]) * scan * g_mask;
float b = float(src[index + b_index]) * scan * b_mask;
dst[index + r_index] = uchar(metal::clamp(metal::rint(r), 0.0f, 255.0f));
dst[index + g_index] = uchar(metal::clamp(metal::rint(g), 0.0f, 255.0f));
dst[index + b_index] = uchar(metal::clamp(metal::rint(b), 0.0f, 255.0f));
dst[index + 3] = src[index + 3];`,
	},
}

const computeKernelHeader = "#include <metal_stdlib>\nusing namespace metal;\n"

func (session *computeSession) kernelLocked(name string) (*metal.MetalKernel, error) {
	if kernel := session.kernels[name]; kernel != nil {
		return kernel, nil
	}

	spec, ok := computeKernelSpecs[name]
	if !ok {
		return nil, errors.New("mlx: missing kernel spec")
	}

	kernel := metal.NewMetalKernel(name, spec.inputNames, spec.outputNames, spec.source, computeKernelHeader, true, false)
	session.kernels[name] = kernel
	return kernel, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func threadGroup(width, height int) (int, int) {
	return maxInt(1, minInt(width, 16)), maxInt(1, minInt(height, 16))
}

func (session *computeSession) pixelBufferLocked(value Buffer, role string) (*pixelBuffer, error) {
	buffer, ok := value.(*pixelBuffer)
	if !ok || buffer == nil {
		return nil, errors.New("mlx: " + role + " must be a pixel buffer")
	}
	if err := buffer.requireOpenLocked(); err != nil {
		return nil, err
	}
	return buffer, nil
}

func (session *computeSession) byteBufferLocked(value Buffer, role string) (*byteBuffer, error) {
	buffer, ok := value.(*byteBuffer)
	if !ok || buffer == nil {
		return nil, errors.New("mlx: " + role + " must be a byte buffer")
	}
	if err := buffer.requireOpenLocked(); err != nil {
		return nil, err
	}
	return buffer, nil
}

func requireBuffer(buffers map[string]Buffer, name string) (Buffer, error) {
	if buffers == nil {
		return nil, errors.New("mlx: kernel buffers are missing")
	}
	value, ok := buffers[name]
	if !ok || value == nil {
		return nil, errors.New("mlx: missing kernel buffer " + name)
	}
	return value, nil
}

func sameDimensions(a, b PixelBufferDesc) bool {
	return a.Width == b.Width && a.Height == b.Height
}

func unitScalar(args KernelArgs, name string, defaultValue float64) (int, error) {
	if args.Scalars == nil {
		return quantizeUnitScalar(defaultValue), nil
	}
	value, ok := args.Scalars[name]
	if !ok {
		return quantizeUnitScalar(defaultValue), nil
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, errors.New("mlx: kernel scalar " + name + " must be finite")
	}
	if value < 0 || value > 1 {
		return 0, errors.New("mlx: kernel scalar " + name + " must be between 0 and 1")
	}
	return quantizeUnitScalar(value), nil
}

func quantizeUnitScalar(value float64) int {
	return maxInt(0, minInt(256, int(math.Round(value*256.0))))
}

func validateFilterBuffers(src, dst *pixelBuffer, kernel string) error {
	if !sameDimensions(src.desc, dst.desc) {
		return errors.New("mlx: " + kernel + " requires matching source and destination dimensions")
	}
	if src.desc.Format != dst.desc.Format {
		return errors.New("mlx: " + kernel + " requires matching pixel formats")
	}
	if src.desc.Format != PixelRGBA8 && src.desc.Format != PixelBGRA8 {
		return errors.New("mlx: " + kernel + " requires rgba8 or bgra8 buffers")
	}
	return nil
}

func (session *computeSession) applyUnaryPixelKernelLocked(kernelName string, src *pixelBuffer, dst *pixelBuffer, addTemplates func(*metal.MetalKernelConfig)) error {
	kernel, err := session.kernelLocked(kernelName)
	if err != nil {
		return err
	}

	config := metal.NewMetalKernelConfig()
	defer config.Free()

	width, height := threadGroup(dst.desc.Width, dst.desc.Height)
	config.SetGrid(dst.desc.Width, dst.desc.Height, 1)
	config.SetThreadGroup(width, height, 1)
	config.SetVerbose(session.cfg.verboseKernels)
	config.AddOutputArg([]int32{int32(dst.desc.Height), int32(dst.desc.Stride)}, metal.DTypeUint8)
	if addTemplates != nil {
		addTemplates(config)
	}

	results, err := kernel.Apply(config, src.array)
	if err != nil {
		return err
	}
	dst.replaceLocked(results[0])
	return nil
}

func (session *computeSession) runNearestScaleLocked(args KernelArgs, requireIntegerScale bool) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if src.desc.Format != dst.desc.Format {
		return errors.New("mlx: nearest scaling requires matching pixel formats")
	}
	if requireIntegerScale {
		if dst.desc.Width%src.desc.Width != 0 || dst.desc.Height%src.desc.Height != 0 {
			return errors.New("mlx: integer scaling requires exact output multiples")
		}
		if dst.desc.Width/src.desc.Width != dst.desc.Height/src.desc.Height {
			return errors.New("mlx: integer scaling requires the same factor on both axes")
		}
	}
	bpp := src.desc.Format.BytesPerPixel()
	return session.applyUnaryPixelKernelLocked("frame_copy_scale", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("BPP", bpp)
		config.AddTemplateInt("SRC_WIDTH", src.desc.Width)
		config.AddTemplateInt("SRC_HEIGHT", src.desc.Height)
		config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
		config.AddTemplateInt("DST_WIDTH", dst.desc.Width)
		config.AddTemplateInt("DST_HEIGHT", dst.desc.Height)
		config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	})
}

func (session *computeSession) runBilinearScaleLocked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if src.desc.Format != dst.desc.Format {
		return errors.New("mlx: bilinear scaling requires matching pixel formats")
	}
	if src.desc.Format != PixelRGBA8 && src.desc.Format != PixelBGRA8 {
		return errors.New("mlx: bilinear scaling currently supports rgba8 and bgra8 only")
	}
	return session.applyUnaryPixelKernelLocked("frame_bilinear_rgba", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("SRC_WIDTH", src.desc.Width)
		config.AddTemplateInt("SRC_HEIGHT", src.desc.Height)
		config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
		config.AddTemplateInt("DST_WIDTH", dst.desc.Width)
		config.AddTemplateInt("DST_HEIGHT", dst.desc.Height)
		config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	})
}

func (session *computeSession) runRGB565ToRGBA8Locked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if src.desc.Format != PixelRGB565 {
		return errors.New("mlx: rgb565_to_rgba8 requires an rgb565 source buffer")
	}
	if dst.desc.Format != PixelRGBA8 {
		return errors.New("mlx: rgb565_to_rgba8 requires an rgba8 destination buffer")
	}
	if !sameDimensions(src.desc, dst.desc) {
		return errors.New("mlx: rgb565_to_rgba8 requires matching source and destination dimensions")
	}
	return session.applyUnaryPixelKernelLocked("frame_rgb565_to_rgba8", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("WIDTH", src.desc.Width)
		config.AddTemplateInt("HEIGHT", src.desc.Height)
		config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
		config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	})
}

func (session *computeSession) runChannelSwizzleLocked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if !sameDimensions(src.desc, dst.desc) {
		return errors.New("mlx: channel swizzle requires matching dimensions")
	}
	if src.desc.Format == PixelRGBA8 && dst.desc.Format != PixelBGRA8 {
		return errors.New("mlx: rgba8_to_bgra8 requires a bgra8 destination")
	}
	if src.desc.Format == PixelBGRA8 && dst.desc.Format != PixelRGBA8 {
		return errors.New("mlx: bgra8_to_rgba8 requires an rgba8 destination")
	}
	if src.desc.Format != PixelRGBA8 && src.desc.Format != PixelBGRA8 {
		return errors.New("mlx: channel swizzle requires an rgba8 or bgra8 source")
	}
	return session.applyUnaryPixelKernelLocked("frame_channel_swizzle", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("WIDTH", src.desc.Width)
		config.AddTemplateInt("HEIGHT", src.desc.Height)
		config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
		config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	})
}

func (session *computeSession) runXRGB8888ToRGBA8Locked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if src.desc.Format != PixelXRGB8888 {
		return errors.New("mlx: xrgb8888_to_rgba8 requires an xrgb8888 source buffer")
	}
	if dst.desc.Format != PixelRGBA8 {
		return errors.New("mlx: xrgb8888_to_rgba8 requires an rgba8 destination buffer")
	}
	if !sameDimensions(src.desc, dst.desc) {
		return errors.New("mlx: xrgb8888_to_rgba8 requires matching source and destination dimensions")
	}
	return session.applyUnaryPixelKernelLocked("frame_xrgb8888_to_rgba8", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("WIDTH", src.desc.Width)
		config.AddTemplateInt("HEIGHT", src.desc.Height)
		config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
		config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	})
}

func (session *computeSession) runPaletteExpandLocked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	paletteValue, err := requireBuffer(args.Inputs, "palette")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	palette, err := session.byteBufferLocked(paletteValue, "palette")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if src.desc.Format != PixelIndexed8 {
		return errors.New("mlx: palette_expand_rgba8 requires an indexed8 source buffer")
	}
	if dst.desc.Format != PixelRGBA8 {
		return errors.New("mlx: palette_expand_rgba8 requires an rgba8 destination buffer")
	}
	if !sameDimensions(src.desc, dst.desc) {
		return errors.New("mlx: palette expansion requires matching source and destination dimensions")
	}
	if palette.size < 256*4 {
		return errors.New("mlx: palette buffer must contain at least 256 RGBA entries")
	}

	kernel, err := session.kernelLocked("frame_palette_expand_rgba8")
	if err != nil {
		return err
	}

	config := metal.NewMetalKernelConfig()
	defer config.Free()

	width, height := threadGroup(dst.desc.Width, dst.desc.Height)
	config.SetGrid(dst.desc.Width, dst.desc.Height, 1)
	config.SetThreadGroup(width, height, 1)
	config.SetVerbose(session.cfg.verboseKernels)
	config.AddTemplateInt("WIDTH", src.desc.Width)
	config.AddTemplateInt("HEIGHT", src.desc.Height)
	config.AddTemplateInt("SRC_STRIDE", src.desc.Stride)
	config.AddTemplateInt("DST_STRIDE", dst.desc.Stride)
	config.AddOutputArg([]int32{int32(dst.desc.Height), int32(dst.desc.Stride)}, metal.DTypeUint8)

	results, err := kernel.Apply(config, src.array, palette.array)
	if err != nil {
		return err
	}
	dst.replaceLocked(results[0])
	return nil
}

func (session *computeSession) runScanlineFilterLocked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if err := validateFilterBuffers(src, dst, "scanline_filter"); err != nil {
		return err
	}
	strength, err := unitScalar(args, "strength", 0.35)
	if err != nil {
		return err
	}
	return session.applyUnaryPixelKernelLocked("frame_scanline_filter", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("WIDTH", src.desc.Width)
		config.AddTemplateInt("HEIGHT", src.desc.Height)
		config.AddTemplateInt("STRIDE", src.desc.Stride)
		config.AddTemplateInt("STRENGTH", strength)
	})
}

func (session *computeSession) runCRTFilterLocked(args KernelArgs) error {
	srcValue, err := requireBuffer(args.Inputs, "src")
	if err != nil {
		return err
	}
	dstValue, err := requireBuffer(args.Outputs, "dst")
	if err != nil {
		return err
	}
	src, err := session.pixelBufferLocked(srcValue, "src")
	if err != nil {
		return err
	}
	dst, err := session.pixelBufferLocked(dstValue, "dst")
	if err != nil {
		return err
	}
	if err := validateFilterBuffers(src, dst, "crt_filter"); err != nil {
		return err
	}
	scanlineStrength, err := unitScalar(args, "scanline_strength", 0.25)
	if err != nil {
		return err
	}
	maskStrength, err := unitScalar(args, "mask_strength", 0.35)
	if err != nil {
		return err
	}
	return session.applyUnaryPixelKernelLocked("frame_crt_filter", src, dst, func(config *metal.MetalKernelConfig) {
		config.AddTemplateInt("WIDTH", src.desc.Width)
		config.AddTemplateInt("HEIGHT", src.desc.Height)
		config.AddTemplateInt("STRIDE", src.desc.Stride)
		config.AddTemplateInt("SCANLINE_STRENGTH", scanlineStrength)
		config.AddTemplateInt("MASK_STRENGTH", maskStrength)
		config.AddTemplateBool("BGRA_ORDER", src.desc.Format == PixelBGRA8)
	})
}
