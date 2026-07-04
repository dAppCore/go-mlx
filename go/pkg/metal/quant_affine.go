// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// affine is the engine's default weight-quant format (mlx group-affine). It
// self-registers its loader so the quant registry always carries it; q4_0,
// mxfp4, nvfp4 register their own loaders in their own files (#33), and a model
// whose config declares that kind resolves it here with no engine edit. The
// loader assembles the quantized Linear from the checkpoint tensors; the affine
// matmul mode is applied later at Forward.
func init() {
	RegisterQuantLoader("affine", func(t QuantTensors) (*Linear, error) {
		return &Linear{
			Weight:           t.Weight,
			Scales:           t.Scales,
			Biases:           t.Biases,
			Bias:             t.Bias,
			GroupSize:        t.GroupSize,
			Bits:             t.Bits,
			QuantizationMode: "affine",
		}, nil
	})
}
