// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// g4Assemble runs the engine's generic assembler with gemma4's weight layout — gemma4 no longer owns an
// Assemble (model.Assemble does), so the native tests that build a gemma4 LoadedModel from a synthetic
// tensor set go through this.
func g4Assemble(ts map[string]safetensors.Tensor, arch model.Arch) (*model.LoadedModel, error) {
	return model.Assemble(ts, arch, model.StandardWeightNames())
}
