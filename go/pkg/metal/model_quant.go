// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// affineModelQMV is the metal (mlx-c) backend's entry in pkg/model's quant
// cross-section. The contract (model.QuantMatVec) speaks raw BYTES; metal computes
// on *Array, so MatVec reconstructs the mlx Arrays from the packed bytes and runs
// QuantizedMatmul. Registered "metal"/"affine" alongside pkg/native's
// "native"/"affine" — the two coexist in the (backend,kind) registry so the engine
// picks by the backend it loaded, the same answer either way.
type affineModelQMV struct{}

func (affineModelQMV) Kind() string { return "affine" }
func (affineModelQMV) Bits() int    { return 0 } // the model's config declares the width

func (affineModelQMV) MatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if outDim == 0 {
		return nil, core.NewError("metal.affineModelQMV: outDim is 0")
	}
	// Reconstruct the row-major Arrays. The last dim is derived from the byte
	// length (÷outDim), so the exact packing (8×4-bit per uint32, etc.) need not be
	// hardcoded — whatever mlx_quantize produced round-trips through its bytes.
	wq := FromRawBytes(packed, []int{outDim, len(packed) / 4 / outDim}, DTypeUint32)
	sc := FromRawBytes(scales, []int{outDim, len(scales) / 2 / outDim}, DTypeBFloat16)
	bi := FromRawBytes(biases, []int{outDim, len(biases) / 2 / outDim}, DTypeBFloat16)
	xa := FromRawBytes(x, []int{1, inDim}, DTypeBFloat16)
	res := QuantizedMatmul(xa, wq, sc, bi, true, groupSize, bits)
	if res == nil {
		Free(wq, sc, bi, xa)
		return nil, core.NewError("metal.affineModelQMV: QuantizedMatmul returned nil")
	}
	Materialize(res)
	out := append([]byte(nil), res.RawBytes()...)
	Free(wq, sc, bi, xa, res)
	return out, nil
}

func init() { model.RegisterBackendQuant("metal", affineModelQMV{}) }
