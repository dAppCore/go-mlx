// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

// mlx_fast_metal_kernel_apply_inline collapses the
// (mlx_vector_array_new + N×mlx_vector_array_append_value + mlx_fast_metal_kernel_apply + mlx_vector_array_free)
// sequence into a single cgo crossing.  MLX's vector_array constructor + per-element
// append + free were each separate cgo entries; for a tiny MoE kernel call with
// 5 inputs that's 7 cgo crossings before the actual apply.  This wrapper takes a
// caller-owned mlx_array handle array (typically stack-allocated on Go side via
// the 8-slot scratch pool) and runs the whole sequence C-side, returning the rc
// from mlx_fast_metal_kernel_apply.  outVec is left to the caller — the per-call
// holder pool already pins it without allocation.
//
// Net effect on the expert_id matvec hot path (N=5 inputs, 1 output):
//   before: 11 cgo crossings (new + 5×append + free + apply + size + get)
//   after:  4 cgo crossings (apply_inline + size + get + holder.vec free)
static inline int mlx_fast_metal_kernel_apply_inline(
    mlx_vector_array* res,
    mlx_fast_metal_kernel kernel,
    const mlx_array* inputs, size_t input_num,
    mlx_fast_metal_kernel_config config,
    mlx_stream s) {
    mlx_vector_array inputVec = mlx_vector_array_new();
    for (size_t i = 0; i < input_num; ++i) {
        mlx_vector_array_append_value(inputVec, inputs[i]);
    }
    int rc = mlx_fast_metal_kernel_apply(res, kernel, inputVec, config, s);
    mlx_vector_array_free(inputVec);
    return rc;
}

// mlx_fast_metal_kernel_apply_one_inline pushes the single-output extraction
// across the same cgo crossing as apply.  12 of 14 production MetalKernel
// callers (expert_id_matvec, dense_matvec, gemma4_ffn_residual, jang_dequant,
// codebook_vq, dense gemma4 router) declare exactly one output via
// AddOutputArg and immediately index results[0].  Folding the
// mlx_vector_array_size + mlx_vector_array_get pair into the same C frame as
// the kernel apply eliminates two more cgo crossings per call and lets Go
// drop the []*Array result slice (no heap escape, no len()-1 ceremony).
//
// Returns: rc from mlx_fast_metal_kernel_apply (0 on success); on success the
// output count is written to *count and (if count==1) the first array handle
// is moved into *out.  Caller checks count==1 to confirm single-output before
// using *out; mismatched output arity reports the actual count for diagnostics.
static inline int mlx_fast_metal_kernel_apply_one_inline(
    mlx_array* out, size_t* count, mlx_vector_array* res,
    mlx_fast_metal_kernel kernel,
    const mlx_array* inputs, size_t input_num,
    mlx_fast_metal_kernel_config config,
    mlx_stream s) {
    mlx_vector_array inputVec = mlx_vector_array_new();
    for (size_t i = 0; i < input_num; ++i) {
        mlx_vector_array_append_value(inputVec, inputs[i]);
    }
    int rc = mlx_fast_metal_kernel_apply(res, kernel, inputVec, config, s);
    mlx_vector_array_free(inputVec);
    if (rc != 0) {
        *count = 0;
        return rc;
    }
    *count = mlx_vector_array_size(*res);
    if (*count == 1) {
        mlx_vector_array_get(out, *res, 0);
    }
    return 0;
}

// mlx_fast_metal_kernel_dispatch_one_inline collapses the entire kernel
// dispatch — fresh config creation, grid + thread-group + single output-arg
// configuration, apply + single-output extract, config free — into a single
// cgo crossing.  Every production single-output MetalKernel caller in this
// package follows the same pattern (fresh cfg per call, no reuse): 6 cgo
// crossings (config_new + set_grid + set_thread_group + add_output_arg +
// apply + config_free) collapse into 1.
//
// shape_in must point to an int32 array (in Go terms []int32 / []C.int32_t);
// shape_num is its length.  shape_buf is materialised on the C stack from
// shape_in to convert int32 → int as MLX's add_output_arg expects.  shape_num
// is bounded by the metal-kernel rank cap (8); larger ranks reject early on
// the Go side.
static inline int mlx_fast_metal_kernel_dispatch_one_inline(
    mlx_array* out, size_t* count, mlx_vector_array* res,
    mlx_fast_metal_kernel kernel,
    int grid_x, int grid_y, int grid_z,
    int tg_x, int tg_y, int tg_z,
    const int32_t* shape_in, size_t shape_num, mlx_dtype dtype,
    const mlx_array* inputs, size_t input_num,
    mlx_stream s) {
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_set_grid(cfg, grid_x, grid_y, grid_z);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, tg_x, tg_y, tg_z);
    if (shape_num == 0) {
        mlx_fast_metal_kernel_config_add_output_arg(cfg, NULL, 0, dtype);
    } else {
        int shape_buf[8];
        for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
        mlx_fast_metal_kernel_config_add_output_arg(cfg, shape_buf, shape_num, dtype);
    }
    mlx_vector_array inputVec = mlx_vector_array_new();
    for (size_t i = 0; i < input_num; ++i) {
        mlx_vector_array_append_value(inputVec, inputs[i]);
    }
    int rc = mlx_fast_metal_kernel_apply(res, kernel, inputVec, cfg, s);
    mlx_vector_array_free(inputVec);
    mlx_fast_metal_kernel_config_free(cfg);
    if (rc != 0) {
        *count = 0;
        return rc;
    }
    *count = mlx_vector_array_size(*res);
    if (*count == 1) {
        mlx_vector_array_get(out, *res, 0);
    }
    return 0;
}
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"

	"dappco.re/go"
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
//
//   - name: unique identifier for the kernel (used for caching)
//
//   - inputNames: names for input arrays referenced in the source
//
//   - outputNames: names for output arrays referenced in the source
//
//   - source: Metal Shading Language kernel body
//
//   - header: additional MSL header code (pass "" for none)
//
//   - ensureRowContiguous: if true, inputs are made row-contiguous before dispatch
//
//   - atomicOutputs: if true, output buffers support atomic operations
//
//     kernel := metal.NewMetalKernel("myadd", []string{"a", "b"}, []string{"out"},
//     "uint i = thread_position_in_grid.x; out[i] = a[i] + b[i];", "", true, false)
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

// metalKernelOutputVecHolder pins a mlx_vector_array struct so its address
// can be passed to cgo without forcing a fresh heap allocation each call.
// The holder is recycled via metalKernelOutputVecPool; the inner C handle
// is freed between uses so the holder always returns to the pool with a
// nil ctx, ready for the next caller's mlx_fast_metal_kernel_apply to
// either reuse or allocate the underlying std::vector.
//
// The count field is the output-count slot for the ApplyOne fast path —
// the inline-C wrapper writes the kernel's output count there, allowing
// the per-call `var count C.size_t` (which escapes via cgo &count) to
// move into the pooled holder and avoid the heap allocation.
type metalKernelOutputVecHolder struct {
	vec   C.mlx_vector_array
	count C.size_t
}

var metalKernelOutputVecPool = sync.Pool{
	New: func() any {
		return &metalKernelOutputVecHolder{}
	},
}

// metalKernelInputScratchRank caps the pooled input-handle scratch buffer used
// by Apply. Every current MetalKernel caller in this package passes between 1
// and 9 input arrays (expert_id_matvec tops out at 8 quantization factor sets;
// gemma4_ffn_residual passes 6).  Sized at 16 to comfortably cover that plus
// future split-quantization layouts.  Callers exceeding the cap fall back to a
// heap-allocated buffer.
const metalKernelInputScratchRank = 16

// metalKernelInputScratch is a sync.Pool of fixed-arity C.mlx_array buffers used
// by Apply as a handle-conversion scratch for mlx_fast_metal_kernel_apply_inline.
// The cgo trampoline forces any Go pointer passed across the boundary to escape,
// so a stack array does not actually stay on the stack; pooling lets us amortise
// the allocation across calls and keep the per-call alloc count at zero on the
// fast path.  Each entry is *[]C.mlx_array (16 slots).
var metalKernelInputScratch = sync.Pool{
	New: func() any {
		buf := make([]C.mlx_array, metalKernelInputScratchRank)
		return &buf
	},
}

// Apply executes the kernel with the given configuration and input arrays.
// Returns the output arrays produced by the kernel.
//
//	results, err := kernel.Apply(cfg, inputA, inputB)
//	if err != nil { return err }
//	output := results[0]
func (k *MetalKernel) Apply(config *MetalKernelConfig, inputs ...*Array) ([]*Array, error) {
	if k == nil || k.ctx.ctx == nil {
		return nil, core.E("mlx.MetalKernel.Apply", "kernel handle is nil", nil)
	}
	if config == nil || config.ctx.ctx == nil {
		return nil, core.E("mlx.MetalKernel.Apply", "kernel config handle is nil", nil)
	}
	for i, a := range inputs {
		if a == nil || !a.Valid() {
			return nil, core.E("mlx.MetalKernel.Apply", core.Sprintf("input %d handle is nil", i), nil)
		}
	}

	// Pooled holder pins the outputVec struct so taking its address for the
	// mlx_fast_metal_kernel_apply out-parameter does not allocate a fresh
	// 16-byte Go cell each call. mlx_fast_metal_kernel_apply lazily
	// allocates the underlying std::vector when ctx is nil, and reuses it
	// otherwise — both safe with a recycled holder.
	holder := metalKernelOutputVecPool.Get().(*metalKernelOutputVecHolder)
	defer func() {
		C.mlx_vector_array_free(holder.vec)
		holder.vec.ctx = nil
		metalKernelOutputVecPool.Put(holder)
	}()

	// Marshal input handles into a pooled fixed-arity scratch buffer and let
	// the inline-C wrapper materialise the input mlx_vector_array C-side. This
	// collapses (new + N×append + apply + free) into one cgo crossing — on the
	// 5-input expert_id matvec path, 7 cgo crossings become 1.
	var inputsPtr *C.mlx_array
	var bufPtr *[]C.mlx_array
	nInputs := len(inputs)
	if nInputs > 0 {
		if nInputs <= metalKernelInputScratchRank {
			bufPtr = metalKernelInputScratch.Get().(*[]C.mlx_array)
			buf := (*bufPtr)[:nInputs]
			for i, a := range inputs {
				buf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&buf[0]))
		} else {
			heapBuf := make([]C.mlx_array, nInputs)
			for i, a := range inputs {
				heapBuf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&heapBuf[0]))
		}
	}

	rc := C.mlx_fast_metal_kernel_apply_inline(&holder.vec, k.ctx, inputsPtr, C.size_t(nInputs), config.ctx, DefaultStream().ctx)
	if bufPtr != nil {
		metalKernelInputScratch.Put(bufPtr)
	}
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.MetalKernel.Apply", core.Sprintf("kernel apply failed (rc=%d)", rc), nil)
	}

	n := C.mlx_vector_array_size(holder.vec)

	results := make([]*Array, int(n))
	for i := range results {
		out := newArray("METAL_KERNEL")
		C.mlx_vector_array_get(&out.ctx, holder.vec, C.size_t(i))
		results[i] = out
	}
	return results, nil
}

// metalKernelGrid bundles grid + thread-group dimensions for the
// DispatchOne fast path.  Pairing the six ints keeps the call signature
// readable and prevents accidental swap between the two triples.
//
//	g := metal.MetalKernelGrid{GridX: n, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1}
type MetalKernelGrid struct {
	GridX, GridY, GridZ int
	TGX, TGY, TGZ       int
}

// DispatchOne is the all-in-one single-output dispatch path that obsoletes
// the (NewMetalKernelConfig + SetGrid + SetThreadGroup + AddOutputArg +
// ApplyOne + cfg.Free) call sequence.  Every single-output MetalKernel caller
// in this package follows the same pattern of a fresh cfg per call with no
// reuse; DispatchOne collapses the entire dispatch into a single cgo
// crossing via mlx_fast_metal_kernel_dispatch_one_inline.
//
// The MetalKernelConfig Go wrapper escapes to heap on every NewMetalKernelConfig
// call (the SetFinalizer triple plus the embedded C handle force it onto the
// heap regardless of escape analysis).  DispatchOne removes the wrapper
// entirely from the per-call path — the C config handle is born and freed
// inside the inline wrapper, leaving zero Go-side allocs on the dispatch frame.
//
// Per-call cgo savings on the expert_id_matvec hot path (5 inputs, 1 output):
//
//	  Before DispatchOne: 7 crossings (config_new, set_grid, set_thread_group,
//	    add_output_arg, apply_one_inline, free, holder free)
//	  After DispatchOne:  2 crossings (dispatch_one_inline, holder free)
//
//		out, err := kernel.DispatchOne(metal.MetalKernelGrid{GridX: n, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
//		    outShape, metal.DTypeFloat32, input, weight, scales, biases, expertIDs)
func (k *MetalKernel) DispatchOne(g MetalKernelGrid, outShape []int32, dtype DType, inputs ...*Array) (*Array, error) {
	if k == nil || k.ctx.ctx == nil {
		return nil, core.E("mlx.MetalKernel.DispatchOne", "kernel handle is nil", nil)
	}
	if len(outShape) > maxTensorRank {
		return nil, core.E("mlx.MetalKernel.DispatchOne",
			core.Sprintf("output shape rank %d exceeds maxTensorRank %d", len(outShape), maxTensorRank), nil)
	}
	for i, a := range inputs {
		if a == nil || !a.Valid() {
			return nil, core.E("mlx.MetalKernel.DispatchOne", core.Sprintf("input %d handle is nil", i), nil)
		}
	}

	holder := metalKernelOutputVecPool.Get().(*metalKernelOutputVecHolder)
	defer func() {
		C.mlx_vector_array_free(holder.vec)
		holder.vec.ctx = nil
		metalKernelOutputVecPool.Put(holder)
	}()

	var inputsPtr *C.mlx_array
	var bufPtr *[]C.mlx_array
	nInputs := len(inputs)
	if nInputs > 0 {
		if nInputs <= metalKernelInputScratchRank {
			bufPtr = metalKernelInputScratch.Get().(*[]C.mlx_array)
			buf := (*bufPtr)[:nInputs]
			for i, a := range inputs {
				buf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&buf[0]))
		} else {
			heapBuf := make([]C.mlx_array, nInputs)
			for i, a := range inputs {
				heapBuf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&heapBuf[0]))
		}
	}

	var shapePtr *C.int32_t
	if len(outShape) > 0 {
		shapePtr = (*C.int32_t)(unsafe.Pointer(&outShape[0]))
	}

	out := newArray("METAL_KERNEL")
	rc := C.mlx_fast_metal_kernel_dispatch_one_inline(
		&out.ctx, &holder.count, &holder.vec, k.ctx,
		C.int(g.GridX), C.int(g.GridY), C.int(g.GridZ),
		C.int(g.TGX), C.int(g.TGY), C.int(g.TGZ),
		shapePtr, C.size_t(len(outShape)), C.mlx_dtype(dtype),
		inputsPtr, C.size_t(nInputs),
		DefaultStream().ctx,
	)
	if bufPtr != nil {
		metalKernelInputScratch.Put(bufPtr)
	}
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.MetalKernel.DispatchOne", core.Sprintf("kernel apply failed (rc=%d)", rc), nil)
	}
	if holder.count != 1 {
		return nil, core.E("mlx.MetalKernel.DispatchOne",
			core.Sprintf("expected 1 output, got %d", int(holder.count)), nil)
	}
	return out, nil
}

// ApplyOne is the single-output fast path for MetalKernel.Apply.  Returns the
// kernel's sole output without allocating a []*Array slice or making the
// separate mlx_vector_array_size + mlx_vector_array_get cgo crossings — the
// inline-C wrapper extracts result[0] in the same frame as the apply.
//
// 12 of 14 production callers (expert_id matvec, dense_matvec,
// gemma4_ffn_residual, jang_dequant, codebook_vq, gemma4_router_topk single-
// output path) declare exactly one output via AddOutputArg and immediately
// pull results[0]; ApplyOne replaces that pattern at zero alloc cost.
//
// Returns an error if the kernel produces 0 or >1 outputs — caller mismatch
// against the cfg.AddOutputArg declaration is surfaced rather than silently
// swallowed.
//
//	out, err := kernel.ApplyOne(cfg, input, weight, scales, biases, expertIDs)
//	if err != nil { return err }
func (k *MetalKernel) ApplyOne(config *MetalKernelConfig, inputs ...*Array) (*Array, error) {
	if k == nil || k.ctx.ctx == nil {
		return nil, core.E("mlx.MetalKernel.ApplyOne", "kernel handle is nil", nil)
	}
	if config == nil || config.ctx.ctx == nil {
		return nil, core.E("mlx.MetalKernel.ApplyOne", "kernel config handle is nil", nil)
	}
	for i, a := range inputs {
		if a == nil || !a.Valid() {
			return nil, core.E("mlx.MetalKernel.ApplyOne", core.Sprintf("input %d handle is nil", i), nil)
		}
	}

	holder := metalKernelOutputVecPool.Get().(*metalKernelOutputVecHolder)
	defer func() {
		C.mlx_vector_array_free(holder.vec)
		holder.vec.ctx = nil
		metalKernelOutputVecPool.Put(holder)
	}()

	var inputsPtr *C.mlx_array
	var bufPtr *[]C.mlx_array
	nInputs := len(inputs)
	if nInputs > 0 {
		if nInputs <= metalKernelInputScratchRank {
			bufPtr = metalKernelInputScratch.Get().(*[]C.mlx_array)
			buf := (*bufPtr)[:nInputs]
			for i, a := range inputs {
				buf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&buf[0]))
		} else {
			heapBuf := make([]C.mlx_array, nInputs)
			for i, a := range inputs {
				heapBuf[i] = a.ctx
			}
			inputsPtr = (*C.mlx_array)(unsafe.Pointer(&heapBuf[0]))
		}
	}

	out := newArray("METAL_KERNEL")
	// holder.count is the output-count slot — pooled so the &count cgo pass
	// does not force a per-call heap allocation.
	rc := C.mlx_fast_metal_kernel_apply_one_inline(
		&out.ctx, &holder.count, &holder.vec, k.ctx,
		inputsPtr, C.size_t(nInputs), config.ctx,
		DefaultStream().ctx,
	)
	if bufPtr != nil {
		metalKernelInputScratch.Put(bufPtr)
	}
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.MetalKernel.ApplyOne", core.Sprintf("kernel apply failed (rc=%d)", rc), nil)
	}
	if holder.count != 1 {
		// Free the output vector handles so MLX does not leak the
		// arrays the caller cannot reach.  The holder.vec defer above
		// already frees the underlying vector itself; we just need to
		// avoid returning a partially-initialised out handle.
		return nil, core.E("mlx.MetalKernel.ApplyOne",
			core.Sprintf("expected 1 output, got %d", int(holder.count)), nil)
	}
	return out, nil
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

// metalKernelOutputArgScratchRank caps the pooled scratch buffer reused in
// AddOutputArg. Every current caller in this package emits a shape of rank
// 1, 2, or 3; arbitrary callers may pass an *Array's full shape, but MLX
// itself caps tensor rank well below this bound. Shapes that exceed the
// cap fall back to a heap-allocated buffer.
const metalKernelOutputArgScratchRank = 8

// metalKernelOutputArgScratch is a sync.Pool of fixed-rank C.int buffers used
// by AddOutputArg as a shape-conversion scratch. The cgo trampoline forces
// any Go pointer passed across the boundary to escape, so a stack array does
// not actually stay on the stack; pooling lets us amortise the allocation
// across calls and keep the per-call alloc count at zero on the fast path.
var metalKernelOutputArgScratch = sync.Pool{
	New: func() any {
		buf := make([]C.int, metalKernelOutputArgScratchRank)
		return &buf
	},
}

// AddOutputArg declares an output array with the given shape and dtype.
// Call once per output in the order matching outputNames from NewMetalKernel.
//
//	cfg.AddOutputArg([]int32{4, 16}, metal.DTypeFloat32)
func (c *MetalKernelConfig) AddOutputArg(shape []int32, dtype DType) {
	n := len(shape)
	if n == 0 {
		C.mlx_fast_metal_kernel_config_add_output_arg(c.ctx, nil, 0, C.mlx_dtype(dtype))
		return
	}
	if n <= metalKernelOutputArgScratchRank {
		// Pooled scratch fast path: the C callee copies the shape buffer
		// synchronously, so the same buffer can be returned to the pool
		// once the cgo call returns. This eliminates the per-call
		// make([]C.int, len(shape)) allocation on MoE-heavy hot paths.
		bufPtr := metalKernelOutputArgScratch.Get().(*[]C.int)
		buf := (*bufPtr)[:n]
		for i, s := range shape[:n] {
			buf[i] = C.int(s)
		}
		C.mlx_fast_metal_kernel_config_add_output_arg(c.ctx, &buf[0], C.size_t(n), C.mlx_dtype(dtype))
		metalKernelOutputArgScratch.Put(bufPtr)
		return
	}
	cShape := make([]C.int, n)
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	C.mlx_fast_metal_kernel_config_add_output_arg(c.ctx, &cShape[0], C.size_t(n), C.mlx_dtype(dtype))
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
