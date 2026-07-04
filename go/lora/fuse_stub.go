// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package lora

import (
	"context"

	core "dappco.re/go"
)

// errFuseUnsupported is the sentinel returned by the non-native stub
// when FuseIntoPack is called on a platform without native MLX support.
// Hoisted to a package var so the stub matches the sentinel-error
// pattern used by the native fuse.go path.
var errFuseUnsupported = core.NewError("mlx: LoRA pack fusion requires darwin/arm64 native MLX support")

// FuseIntoPack requires native MLX safetensors support.
func FuseIntoPack(_ context.Context, _ FuseOptions) (*FuseResult, error) {
	return nil, errFuseUnsupported
}
