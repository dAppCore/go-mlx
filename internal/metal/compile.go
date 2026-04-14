//go:build darwin && arm64 && !nomlx

package metal

import "sync"

// CompiledFunc wraps a function for efficient repeated execution.
// The function is called directly; MLX's lazy evaluation graph
// still deduplicates and optimises the underlying Metal operations.
type CompiledFunc struct {
	fn func([]*Array) []*Array
	mu sync.Mutex
}

// CompileShapeless wraps a function for repeated execution.
// The shapeless parameter is accepted for API compatibility but unused.
//
//	geluFn := metal.CompileShapeless(func(in []*Array) []*Array {
//	    return []*Array{geluApprox(in[0])}
//	}, true)
func CompileShapeless(fn func([]*Array) []*Array, shapeless bool) *CompiledFunc {
	return &CompiledFunc{fn: fn}
}

// Call executes the function with the given inputs.
//
//	result := geluFn.Call(gateProj)[0] // fused GELU on gate projection
func (cf *CompiledFunc) Call(inputs ...*Array) []*Array {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	return cf.fn(inputs)
}
