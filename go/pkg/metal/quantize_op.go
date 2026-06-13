// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"unsafe"

	core "dappco.re/go"
)

// Quantize packs a dense weight into an mlx quantized form, returning the
// quantized weight, per-group scales, and (for affine) per-group biases.
//
// The array count depends on the mode: the default group-affine mode returns the
// (w, scales, biases) triple QuantizedMatmul consumes — matching mlx-lm and the
// mlx-community quants. The scale-only FP4 modes ("mxfp4", "nvfp4") carry block
// scales with no zero-point, so mlx_quantize returns just (w, scales) and biases
// is nil — the dequantize / quantized-matmul paths already nil-tolerate it
// (optionalArray). groupSize + bits set the grouping along the last dim. Backed
// by the mlx_quantize C op — a plain binding, no C++ shim. This is the op behind
// affineQuant.Quantize, the quantize verb, and the SSD/P-phase fuse path.
func Quantize(w *Array, groupSize, bits int, mode string) (wq, scales, biases *Array, err error) {
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	cMode := C.CString(NormalizeQuantizationMode(mode))
	defer C.free(unsafe.Pointer(cMode))

	res := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(res)
	if rc := C.mlx_quantize(&res, w.ctx, gs, b, cMode, optionalArray(nil), DefaultStream().ctx); rc != 0 {
		if e := LastError(); e != nil {
			return nil, nil, nil, e
		}
		return nil, nil, nil, core.E("metal.Quantize", core.Sprintf("mlx_quantize failed (rc=%d)", int(rc)), nil)
	}
	out := vectorToArrays(res)
	switch len(out) {
	case 3: // affine: weight, scales, biases (zero-point)
		return out[0], out[1], out[2], nil
	case 2: // scale-only FP4 (mxfp4/nvfp4): weight, scales — no zero-point
		return out[0], out[1], nil, nil
	default:
		return nil, nil, nil, core.E("metal.Quantize", core.Sprintf("mlx_quantize returned %d arrays, want 2 (w, scales) or 3 (w, scales, biases)", len(out)), nil)
	}
}
