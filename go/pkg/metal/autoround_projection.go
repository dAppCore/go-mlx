// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/quant/autoround"
)

// AutoRoundPackedProjectionLinearFused executes a loaded AutoRound packed
// projection using the native fused Metal kernel.
func AutoRoundPackedProjectionLinearFused(input *Array, projection autoround.PackedProjection) (*Array, error) {
	if len(projection.Weights.Packed) == 0 {
		return nil, core.NewError("mlx: AutoRound packed projection requires packed weights")
	}
	if len(projection.Weights.Scales) == 0 || len(projection.Weights.ZeroPoints) == 0 {
		return nil, core.NewError("mlx: AutoRound packed projection requires scales and zero-points")
	}
	packed := FromValues(projection.Weights.Packed, len(projection.Weights.Packed))
	scales := FromValues(projection.Weights.Scales, len(projection.Weights.Scales))
	zeroPoints := FromValues(projection.Weights.ZeroPoints, len(projection.Weights.ZeroPoints))
	var bias *Array
	if len(projection.Bias) > 0 {
		bias = FromValues(projection.Bias, len(projection.Bias))
	}
	defer Free(packed, scales, zeroPoints, bias)
	return AutoRoundPackedLinearFused(
		input,
		packed,
		scales,
		zeroPoints,
		bias,
		projection.Weights.Shape,
		projection.Weights.GroupSize,
		projection.Weights.Bits,
		projection.Weights.QMin,
	)
}
