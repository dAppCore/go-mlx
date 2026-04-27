// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/array.h"

// mlx_array_detach breaks an evaluated array's graph connections.
// ctx is a mlx::core::array* — we call detach() via a C++ helper.
void mlx_array_detach_impl(mlx_array arr);
*/
import "C"

// Detach breaks an array's graph connections after evaluation.
// This allows Metal memory from parent operations to be freed.
//
//	metal.Detach(logits)              // after Eval — release prefill graph
//	for _, c := range caches { c.Detach() } // break cache graph chains each step
func Detach(arrays ...*Array) {
	for _, a := range arrays {
		if a != nil && a.ctx.ctx != nil {
			C.mlx_array_detach_impl(a.ctx)
		}
	}
}
