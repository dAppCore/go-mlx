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

// Quantize packs a dense weight into mlx group-affine form, returning the
// quantized weight, per-group scales, and biases — exactly the (w, scales,
// biases) triple QuantizedMatmul consumes, so the output matches what mlx-lm and
// the mlx-community quants produce. mode is the mlx quantization mode
// ("affine"); groupSize + bits set the grouping along the last dim. Backed by
// the mlx_quantize C op — the vendored mlx-c already exposes it, so this is a
// plain binding, no C++ shim. This is the op behind affineQuant.Quantize, the
// quantize verb, and the SSD/P-phase fuse-to-q4 path.
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
	if len(out) != 3 {
		return nil, nil, nil, core.E("metal.Quantize", core.Sprintf("mlx_quantize returned %d arrays, want 3 (w, scales, biases)", len(out)), nil)
	}
	return out[0], out[1], out[2], nil
}
