//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"

// Callback for gradient closures — same signature as goCompiledFunc.
extern int goGradFunc(mlx_vector_array *outputs, const mlx_vector_array inputs, void *payload);

// Destructor for closure payload to prevent leaks in gradFuncs.
extern void goGradDestructor(void *payload);

static mlx_closure new_grad_closure(void *payload) {
    return mlx_closure_new_func_payload(&goGradFunc, payload, &goGradDestructor);
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// --- Closure registry (separate from compile.go's registry) ---

var (
	gradFuncs  sync.Map
	gradNextID atomic.Uintptr
)

//export goGradFunc
func goGradFunc(outputs *C.mlx_vector_array, inputs C.mlx_vector_array, payload unsafe.Pointer) C.int {
	id := uintptr(payload)
	fnI, ok := gradFuncs.Load(id)
	if !ok {
		return 1
	}
	fn := fnI.(func([]*Array) []*Array)

	nInputs := int(C.mlx_vector_array_size(inputs))
	goInputs := make([]*Array, nInputs)
	for i := range nInputs {
		a := newArray("GRAD_INPUT")
		C.mlx_vector_array_get(&a.ctx, inputs, C.size_t(i))
		goInputs[i] = a
	}

	goOutputs := fn(goInputs)

	// The output vector arrives with ctx=nullptr (from internal mlx_vector_array_new_).
	// Create a valid vector, fill it, then copy to the output pointer.
	tmp := C.mlx_vector_array_new()
	for _, out := range goOutputs {
		if out != nil && out.Valid() {
			C.mlx_vector_array_append_value(tmp, out.ctx)
		}
	}
	C.mlx_vector_array_set(outputs, tmp)
	C.mlx_vector_array_free(tmp)
	return 0
}

//export goGradDestructor
func goGradDestructor(payload unsafe.Pointer) {
	id := uintptr(payload)
	gradFuncs.Delete(id)
}

// newClosure registers a Go function as an MLX closure for autograd.
func newClosure(fn func([]*Array) []*Array) C.mlx_closure {
	id := gradNextID.Add(1)
	gradFuncs.Store(id, fn)
	return C.new_grad_closure(unsafe.Pointer(id))
}

// --- VJP (Vector-Jacobian Product) — Reverse Mode ---

// VJP computes the vector-Jacobian product (reverse-mode autodiff).
// Given a function fn, input primals, and output cotangents (upstream gradients),
// returns (outputs, gradients) where gradients are w.r.t. the primals.
//
// This is the fundamental backward pass operation.
func VJP(fn func([]*Array) []*Array, primals []*Array, cotangents []*Array) (outputs []*Array, vjps []*Array, err error) {
	Init()

	closure := newClosure(fn)
	defer C.mlx_closure_free(closure)

	// Pack primals into vector
	primalsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(primalsVec)
	for _, p := range primals {
		C.mlx_vector_array_append_value(primalsVec, p.ctx)
	}

	// Pack cotangents into vector
	cotangentsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(cotangentsVec)
	for _, c := range cotangents {
		C.mlx_vector_array_append_value(cotangentsVec, c.ctx)
	}

	// Call mlx_vjp
	var outVec, vjpVec C.mlx_vector_array
	outVec = C.mlx_vector_array_new()
	vjpVec = C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	defer C.mlx_vector_array_free(vjpVec)

	rc := C.mlx_vjp(&outVec, &vjpVec, closure, primalsVec, cotangentsVec)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, nil, err
		}
		return nil, nil, fmt.Errorf("mlx: vjp failed (rc=%d)", rc)
	}

	outputs = vectorToArrays(outVec)
	vjps = vectorToArrays(vjpVec)
	return outputs, vjps, nil
}

// --- JVP (Jacobian-Vector Product) — Forward Mode ---

// JVP computes the Jacobian-vector product (forward-mode autodiff).
// Given a function fn, input primals, and input tangents (perturbation directions),
// returns (outputs, output_tangents).
//
// Useful for directional derivatives and Hessian-vector products.
func JVP(fn func([]*Array) []*Array, primals []*Array, tangents []*Array) (outputs []*Array, jvps []*Array, err error) {
	Init()

	closure := newClosure(fn)
	defer C.mlx_closure_free(closure)

	primalsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(primalsVec)
	for _, p := range primals {
		C.mlx_vector_array_append_value(primalsVec, p.ctx)
	}

	tangentsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(tangentsVec)
	for _, t := range tangents {
		C.mlx_vector_array_append_value(tangentsVec, t.ctx)
	}

	var outVec, jvpVec C.mlx_vector_array
	outVec = C.mlx_vector_array_new()
	jvpVec = C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	defer C.mlx_vector_array_free(jvpVec)

	rc := C.mlx_jvp(&outVec, &jvpVec, closure, primalsVec, tangentsVec)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, nil, err
		}
		return nil, nil, fmt.Errorf("mlx: jvp failed (rc=%d)", rc)
	}

	outputs = vectorToArrays(outVec)
	jvps = vectorToArrays(jvpVec)
	return outputs, jvps, nil
}

// --- ValueAndGrad — Combined Forward + Backward ---

// GradFn is a function that computes both the loss value and gradients
// with respect to specified arguments. Call it with inputs to get
// (values, gradients).
type GradFn struct {
	cls C.mlx_closure_value_and_grad
}

// ValueAndGrad creates a GradFn that computes both the function value and
// gradients with respect to the arguments at the given indices.
//
// The returned GradFn can be called repeatedly with different inputs.
// This is the primary API for training loops:
//
//	lossFn := func(params []*Array) []*Array { return []*Array{loss} }
//	grad := mlx.ValueAndGrad(lossFn, 0)  // differentiate w.r.t. first arg
//	values, grads := grad.Apply(params...)
func ValueAndGrad(fn func([]*Array) []*Array, argnums ...int) *GradFn {
	Init()

	closure := newClosure(fn)
	defer C.mlx_closure_free(closure)

	// Default: differentiate w.r.t. first argument
	if len(argnums) == 0 {
		argnums = []int{0}
	}

	cArgs := make([]C.int, len(argnums))
	for i, a := range argnums {
		cArgs[i] = C.int(a)
	}

	g := &GradFn{}
	C.mlx_value_and_grad(&g.cls, closure, &cArgs[0], C.size_t(len(cArgs)))
	return g
}

// Apply calls the gradient function with the given inputs.
// Returns (values, gradients) where values are the function outputs
// and gradients are w.r.t. the arguments specified in ValueAndGrad.
func (g *GradFn) Apply(inputs ...*Array) (values []*Array, grads []*Array, err error) {
	inputVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(inputVec)
	for _, in := range inputs {
		C.mlx_vector_array_append_value(inputVec, in.ctx)
	}

	var valVec, gradVec C.mlx_vector_array
	valVec = C.mlx_vector_array_new()
	gradVec = C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(valVec)
	defer C.mlx_vector_array_free(gradVec)

	rc := C.mlx_closure_value_and_grad_apply(&valVec, &gradVec, g.cls, inputVec)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, nil, err
		}
		return nil, nil, fmt.Errorf("mlx: value_and_grad apply failed (rc=%d)", rc)
	}

	values = vectorToArrays(valVec)
	grads = vectorToArrays(gradVec)
	return values, grads, nil
}

// Free releases the underlying C closure.
func (g *GradFn) Free() {
	if g.cls.ctx != nil {
		C.mlx_closure_value_and_grad_free(g.cls)
		g.cls.ctx = nil
	}
}

// --- Checkpoint — Memory-Efficient Gradient Recomputation ---

// Checkpoint wraps a function so that during backward pass, intermediate
// activations are recomputed rather than stored. Trades compute for memory.
//
// Use this for memory-constrained training (large models on limited VRAM).
func Checkpoint(fn func([]*Array) []*Array) func([]*Array) []*Array {
	Init()

	closure := newClosure(fn)
	// Do NOT free closure here, it's needed by mlx_checkpoint and its result.

	var checkpointed C.mlx_closure
	C.mlx_checkpoint(&checkpointed, closure)
	C.mlx_closure_free(closure) // checkpointed increments refcount if needed

	// Wrap in a Go struct to manage checkpointed closure lifetime
	type cpWrapper struct {
		cls C.mlx_closure
	}
	cp := &cpWrapper{cls: checkpointed}
	runtime.SetFinalizer(cp, func(c *cpWrapper) {
		C.mlx_closure_free(c.cls)
	})

	return func(inputs []*Array) []*Array {
		inputVec := C.mlx_vector_array_new()
		defer C.mlx_vector_array_free(inputVec)
		for _, in := range inputs {
			C.mlx_vector_array_append_value(inputVec, in.ctx)
		}

		outVec := C.mlx_vector_array_new()
		defer C.mlx_vector_array_free(outVec)

		C.mlx_closure_apply(&outVec, cp.cls, inputVec)
		return vectorToArrays(outVec)
	}
}

// --- Loss Functions ---

// CrossEntropyLoss computes cross-entropy loss between logits and integer targets.
// logits: [..., V] (raw model output, pre-softmax, last dim = vocab)
// targets: [...] (integer token IDs, same shape as logits minus last dim)
// Returns scalar loss averaged over all positions.
//
// Numerically stable: loss_i = logsumexp(logits_i) - logits_i[target_i]
func CrossEntropyLoss(logits, targets *Array) *Array {
	Init()
	// LogSumExp along last axis (vocab dimension) for numerical stability
	lse := LogSumExp(logits, -1, false)

	// Gather the logit at the target index: logits[..., target]
	// targets needs to be [..., 1] for take_along_axis
	tgtExpanded := ExpandDims(targets, -1)
	gathered := TakeAlongAxis(logits, tgtExpanded, -1)
	gathered = Squeeze(gathered, -1) // back to [...] shape

	// Per-position loss: logsumexp - logit_at_target
	perPos := Subtract(lse, gathered)

	// Mean over all positions
	return MeanAll(perPos)
}

// MaskedCrossEntropyLoss computes cross-entropy loss only on masked positions.
// logits: [B, L, V], targets: [B, L], mask: [B, L] (1.0 = compute loss, 0.0 = ignore)
// Returns scalar loss averaged over masked positions only.
func MaskedCrossEntropyLoss(logits, targets, mask *Array) *Array {
	Init()
	lse := LogSumExp(logits, -1, false)
	tgtExpanded := ExpandDims(targets, -1)
	gathered := TakeAlongAxis(logits, tgtExpanded, -1)
	gathered = Squeeze(gathered, -1)
	perPos := Subtract(lse, gathered) // [B, L]

	// Apply mask and average over masked positions
	masked := Mul(perPos, mask)
	return Divide(SumAll(masked), SumAll(mask))
}

// MSELoss computes the mean squared error loss: mean((predictions - targets)^2).
func MSELoss(predictions, targets *Array) *Array {
	diff := Subtract(predictions, targets)
	sq := Square(diff)
	return MeanAll(sq)
}

// --- Helpers ---

// vectorToArrays extracts all arrays from an mlx_vector_array.
func vectorToArrays(vec C.mlx_vector_array) []*Array {
	n := int(C.mlx_vector_array_size(vec))
	out := make([]*Array, n)
	for i := range n {
		a := newArray("VEC_OUT")
		C.mlx_vector_array_get(&a.ctx, vec, C.size_t(i))
		out[i] = a
	}
	return out
}

// Log returns element-wise natural logarithm.
func Log(a *Array) *Array {
	out := newArray("LOG", a)
	C.mlx_log(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// SumAll reduces by summation over all elements, returning a scalar.
func SumAll(a *Array) *Array {
	flat := Reshape(a, -1)
	return Sum(flat, 0, false)
}

// MeanAll reduces by averaging over all elements, returning a scalar.
func MeanAll(a *Array) *Array {
	flat := Reshape(a, -1)
	return Mean(flat, 0, false)
}

// OnesLike creates an array of ones with the same shape and type as the input.
func OnesLike(a *Array) *Array {
	out := newArray("ONES_LIKE", a)
	C.mlx_ones_like(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}
