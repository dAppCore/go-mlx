// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// quant_fp4.go registers the scale-only FP4 weight-quant formats — mxfp4 (the MX
// format: an E8M0 power-of-two block scale over E2M1 elements) and nvfp4 (the
// NVIDIA FP4 format). Unlike affine they carry block scales with NO zero-point,
// so a checkpoint has no biases tensor and mlx_quantize returns just (w, scales);
// the dequantize + quantized-matmul paths read the mode and nil-tolerate biases.
// Registering them makes the engine's supported-quant set explicit — the
// NewQuantizedLinearWithMode dispatch resolves the config's mode through the
// registry instead of the generic fallback — and gives each format a home for
// any future format-specific assembly. The mode string drives the MLX kernel; a
// loader's job is only to assemble the *Linear (the affine pattern).
func init() {
	RegisterQuantLoader("mxfp4", fp4Loader("mxfp4"))
	RegisterQuantLoader("nvfp4", fp4Loader("nvfp4"))
}

// fp4Loader builds a loader that assembles a scale-only FP4 Linear for mode:
// weight + scales come from the checkpoint, biases is legitimately nil.
func fp4Loader(mode string) QuantLoader {
	return func(t QuantTensors) (*Linear, error) {
		return &Linear{
			Weight:           t.Weight,
			Scales:           t.Scales,
			Biases:           t.Biases, // nil for scale-only FP4
			Bias:             t.Bias,
			GroupSize:        t.GroupSize,
			Bits:             t.Bits,
			QuantizationMode: mode,
		}, nil
	}
}
