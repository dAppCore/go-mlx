// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// QuantCompute is the driver-side contract a weight-quant scheme fulfils on the
// metal Engine: run the packed matmul, and (for the quantize verb / SSD fuse)
// pack a dense weight. The engine resolves it with scheme.QuantFor(kind); a new
// format — q4_0, mxfp4, nvfp4 — registers in its own file and implements this,
// with no engine edit.
type QuantCompute interface {
	scheme.QuantScheme // Kind() + Bits()
	// Matmul runs y = x · dequant(w)ᵀ entirely in the scheme's packed form.
	Matmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int) *Array
	// Quantize packs a dense weight into the scheme — the quantize verb's op.
	Quantize(dense *Array, groupSize, bits int) (w, scales, biases *Array, err error)
}

// QuantComputeFor resolves a registered weight-quant scheme and asserts it
// carries the metal compute surface; a metadata-only catalogue entry reports
// (nil,false) so the engine refuses it cleanly rather than calling a stub.
func QuantComputeFor(kind string) (QuantCompute, bool) {
	q, ok := scheme.QuantFor(kind)
	if !ok {
		return nil, false
	}
	qc, ok := q.(QuantCompute)
	return qc, ok
}

// affineQuant is mlx group-affine — the engine's default weight quant and the
// reference QuantCompute the format tasks mirror. Matmul rides the existing
// quantizedMatmulMode (the kind IS the mlx mode); Quantize awaits the mlx
// weight-quantize cgo binding (task #5) and reports unimplemented until it lands.
type affineQuant struct{}

func (affineQuant) Kind() string { return "affine" }
func (affineQuant) Bits() int    { return 0 } // 0 = the model's config declares the width (4/6/8)

func (affineQuant) Matmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int) *Array {
	return quantizedMatmulMode(x, w, scales, biases, transpose, groupSize, bits, "affine")
}

func (affineQuant) Quantize(dense *Array, groupSize, bits int) (*Array, *Array, *Array, error) {
	return nil, nil, nil, errAffineQuantizeUnbound
}

var errAffineQuantizeUnbound = core.NewError("mlx: affine Quantize needs the mlx weight-quantize cgo binding (task #5)")

func init() { scheme.RegisterQuant(affineQuant{}) }

// compile-time proof affineQuant is a full metal.QuantCompute.
var _ QuantCompute = affineQuant{}
