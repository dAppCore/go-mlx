// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
)

// FuseLoRAIntoModelPack requires native MLX safetensors support.
func FuseLoRAIntoModelPack(_ context.Context, _ FuseLoRAOptions) (*FuseLoRAResult, error) {
	return nil, core.NewError("mlx: LoRA pack fusion requires darwin/arm64 native MLX support")
}
