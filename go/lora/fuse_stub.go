// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package lora

import (
	"context"

	core "dappco.re/go"
)

// FuseIntoPack requires native MLX safetensors support.
func FuseIntoPack(_ context.Context, _ FuseOptions) (*FuseResult, error) {
	return nil, core.NewError("mlx: LoRA pack fusion requires darwin/arm64 native MLX support")
}
