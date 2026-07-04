// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// MoERouter is the model-family neutral router weight set for a sparse
// mixture-of-experts layer. Qwen3, Mixtral, GPT-OSS, and Kimi all build this
// same hidden -> expert-score projection; only their loaders differ in which
// checkpoint weight names they probe. The per-token routing algorithm that
// consumes it lives in moe_router.go (projection + top-k selection) and
// moe_expert.go (selected-expert SwiGLU dispatch).
type MoERouter struct {
	Weight    *Array
	Scales    *Array
	Biases    *Array
	GroupSize int
	Bits      int
}
