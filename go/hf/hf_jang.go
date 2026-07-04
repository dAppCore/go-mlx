// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	sharedhf "dappco.re/go/inference/hf"
	"dappco.re/go/inference/quant/jang"
)

// InferJANG classifies a model's JANG/JANGTQ quantisation profile from its
// Hugging Face metadata (id, tags, filenames) when the metadata itself
// carries no explicit JANG block. Returns nil for ordinary (non-JANG)
// weights.
//
// Delegates to the engine-agnostic implementation in
// dappco.re/go/inference/hf, shared by every LEM Engine (mlx, rocm, cpu) —
// the id/tag/filename needle scan carries no MLX-specific state.
//
//	info := hf.InferJANG(meta)
func InferJANG(meta ModelMetadata) *jang.Info {
	return sharedhf.InferJANG(meta)
}
