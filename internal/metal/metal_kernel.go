//go:build darwin && arm64 && !nomlx

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"runtime"
	"unsafe"

	"dappco.re/go/core"
)

// MetalKernel wraps a custom Metal shader kernel for GPU execution.
// It holds the compiled kernel handle and is released automatically by GC finaliser,
// or explicitly via Free.
//
//	source := "uint elem = thread_position_in_grid.x; T tmp = inp[elem]; out[elem] = metal::exp(tmp);"
//	kernel := metal.NewMetalKernel("myexp", []string{"inp"}, []string{"out"}, source, "", true, false)
//	defer kernel.Free()
//
//	cfg := metal.NewMetalKernelConfig()
//	cfg.AddTemplateDType("T", metal.DTypeFloat32)
//	cfg.SetGrid(input.Size(), 1, 1)
//	cfg.SetThreadGroup(256, 1, 1)
//	cfg.AddOutputArg(input.Shape(), input.Dtype())
//
//	results, err := kernel.Apply(cfg, input)
//	if err != nil { log.Fatal(err) }
//	output := results[0]
type MetalKernel struct {
	ctx C.mlx_fast_metal_kernel
}

// NewMetalKernel creates a custom Metal kernel from MSL source code.
//
// Parameters:
//   - name: unique identifier for the kernel (used for caching)
//   - inputNames: names for input arrays referenced in the source
//   - outputNames: names for output arrays referenced in the source
//   - source: Metal Shading Language kernel body
//   - header: additional MSL header code (pass "" for none)
//   - ensureRowContiguous: if true, inputs are made row-contiguous before dispatch
//   - atomicOutputs: if true, output buffers support atomic operations
//
//	kernel := metal.NewMetalKernel("myadd", []string{"a", "b"}, []string{"out"},
//	    "uint i = thread_position_in_grid.x; out[i] = a[i] + b[i];", "", true, false)
func NewMetalKernel(name string, inputNames, outputNames []string, source, header string, ensureRowContiguous, atomicOutputs bool) *MetalKernel {
	Init()

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(header)
	defer C.free(unsafe.Pointer(cHeader))

	inNames := C.mlx_vector_string_new()
	for _, n := range inputNames {
		cs := C.CString(n)
		C.mlx_vector_string_append_value(inNames, cs)
		C.free(unsafe.Pointer(cs))
	}

	outNames := C.mlx_vector_string_new()
	for _, n := range outputNames {
		cs := C.CString(n)
		C.mlx_vector_string_append_value(outNames, cs)
		C.free(unsafe.Pointer(cs))
	}

	k := &MetalKernel{
		ctx: C.mlx_fast_metal_kernel_new(
			cName, inNames, outNames, cSource, cHeader,
			C._Bool(ensureRowContiguous), C._Bool(atomicOutputs),
		),
	}

	C.mlx_vector_string_free(inNames)
	C.mlx_vector_string_free(outNames)

	runtime.SetFinalizer(k, finalizeMetalKernel)
	return k
}

// finalizeMetalKernel is called by Go GC to release the underlying C kernel handle.
func finalizeMetalKernel(k *MetalKernel) {
	if k != nil && k.ctx.ctx != nil {
		C.mlx_fast_metal_kernel_free(k.ctx)
		k.ctx.ctx = nil
	}
}

// Free explicitly releases the C kernel handle. Safe to call multiple times.
//
//	kernel.Free() // release immediately instead of waiting for GC
func (k *MetalKernel) Free() {
	if k != nil && k.ctx.ctx != nil {
		C.mlx_fast_metal_kernel_free(k.ctx)
		k.ctx.ctx = nil
		runtime.SetFinalizer(k, nil)
	}
}

// Apply executes the kernel with the given configuration and input arrays.
// Returns the output arrays produced by the kernel.
//
//	results, err := kernel.Apply(cfg, inputA, inputB)
//	if err != nil { return err }
//	output := results[0]
func (k *MetalKernel) Apply(config *MetalKernelConfig, inputs ...*Array) ([]*Array, error) {
	inputVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(inputVec)
	for _, a := range inputs {
		C.mlx_vector_array_append_value(inputVec, a.ctx)
	}

	outputVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputVec)

	rc := C.mlx_fast_metal_kernel_apply(&outputVec, k.ctx, inputVec, config.ctx, DefaultStream().ctx)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.MetalKernel.Apply", core.Sprintf("kernel apply failed (rc=%d)", rc), nil)
	}

	n := C.mlx_vector_array_size(outputVec)

	results := make([]*Array, int(n))
	for i := range results {
		out := newArray("METAL_KERNEL")
		C.mlx_vector_array_get(&out.ctx, outputVec, C.size_t(i))
		results[i] = out
	}
	return results, nil
}

// MetalKernelConfig holds dispatch parameters for a custom Metal kernel:
// grid dimensions, thread group dimensions, template arguments, and output shapes.
//
//	cfg := metal.NewMetalKernelConfig()
//	cfg.AddTemplateDType("T", metal.DTypeFloat32)
//	cfg.SetGrid(n, 1, 1)
//	cfg.SetThreadGroup(256, 1, 1)
//	cfg.AddOutputArg([]int32{4, 16}, metal.DTypeFloat32)
type MetalKernelConfig struct {
	ctx C.mlx_fast_metal_kernel_config
}

// NewMetalKernelConfig creates an empty kernel dispatch configuration.
//
//	cfg := metal.NewMetalKernelConfig()
func NewMetalKernelConfig() *MetalKernelConfig {
	Init()
	c := &MetalKernelConfig{
		ctx: C.mlx_fast_metal_kernel_config_new(),
	}
	runtime.SetFinalizer(c, finalizeMetalKernelConfig)
	return c
}

// finalizeMetalKernelConfig is called by Go GC to release the underlying C config handle.
func finalizeMetalKernelConfig(c *MetalKernelConfig) {
	if c != nil && c.ctx.ctx != nil {
		C.mlx_fast_metal_kernel_config_free(c.ctx)
		c.ctx.ctx = nil
	}
}

// Free explicitly releases the C config handle. Safe to call multiple times.
//
//	cfg.Free()
func (c *MetalKernelConfig) Free() {
	if c != nil && c.ctx.ctx != nil {
		C.mlx_fast_metal_kernel_config_free(c.ctx)
		c.ctx.ctx = nil
		runtime.SetFinalizer(c, nil)
	}
}

// SetGrid sets the compute grid dimensions (x, y, z) for kernel dispatch.
// Typically x = number of elements, y = 1, z = 1 for element-wise kernels.
//
//	cfg.SetGrid(input.Size(), 1, 1) // one thread per element
func (c *MetalKernelConfig) SetGrid(x, y, z int) {
	C.mlx_fast_metal_kernel_config_set_grid(c.ctx, C.int(x), C.int(y), C.int(z))
}

// SetThreadGroup sets the thread group dimensions (x, y, z) for kernel dispatch.
// Common values: 256 or 1024 for x, 1 for y and z.
//
//	cfg.SetThreadGroup(256, 1, 1) // 256 threads per threadgroup
func (c *MetalKernelConfig) SetThreadGroup(x, y, z int) {
	C.mlx_fast_metal_kernel_config_set_thread_group(c.ctx, C.int(x), C.int(y), C.int(z))
}

// AddTemplateDType adds a dtype template argument to the kernel.
// The name must match a template parameter in the MSL source.
//
//	cfg.AddTemplateDType("T", metal.DTypeFloat32) // template <typename T>
func (c *MetalKernelConfig) AddTemplateDType(name string, dtype DType) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_fast_metal_kernel_config_add_template_arg_dtype(c.ctx, cName, C.mlx_dtype(dtype))
}

// AddTemplateInt adds an integer template argument to the kernel.
//
//	cfg.AddTemplateInt("BLOCK_SIZE", 256)
func (c *MetalKernelConfig) AddTemplateInt(name string, value int) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_fast_metal_kernel_config_add_template_arg_int(c.ctx, cName, C.int(value))
}

// AddTemplateBool adds a boolean template argument to the kernel.
//
//	cfg.AddTemplateBool("USE_BIAS", true)
func (c *MetalKernelConfig) AddTemplateBool(name string, value bool) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_fast_metal_kernel_config_add_template_arg_bool(c.ctx, cName, C._Bool(value))
}

// AddOutputArg declares an output array with the given shape and dtype.
// Call once per output in the order matching outputNames from NewMetalKernel.
//
//	cfg.AddOutputArg([]int32{4, 16}, metal.DTypeFloat32)
func (c *MetalKernelConfig) AddOutputArg(shape []int32, dtype DType) {
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	C.mlx_fast_metal_kernel_config_add_output_arg(c.ctx, &cShape[0], C.size_t(len(cShape)), C.mlx_dtype(dtype))
}

// SetInitValue sets the initial value for output buffers before kernel dispatch.
//
//	cfg.SetInitValue(0.0) // zero-initialise outputs
func (c *MetalKernelConfig) SetInitValue(value float32) {
	C.mlx_fast_metal_kernel_config_set_init_value(c.ctx, C.float(value))
}

// SetVerbose enables verbose logging for kernel compilation and dispatch.
//
//	cfg.SetVerbose(true) // debug Metal shader compilation
func (c *MetalKernelConfig) SetVerbose(verbose bool) {
	C.mlx_fast_metal_kernel_config_set_verbose(c.ctx, C._Bool(verbose))
}
