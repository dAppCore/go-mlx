// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ExampleGemma4Model_ForwardLastTokenLogits shows the memory-bounded prefill
// contract: only the final position's logits are projected, so the result is
// [1,1,vocab] instead of the full [1,L,vocab].
func ExampleGemma4Model_ForwardLastTokenLogits() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.ForwardLastTokenLogits(tokens, nil, caches)
	metal.Free(tokens, logits)
	metal.FreeCaches(caches)
}

// ExampleGemma4Model_ForwardGreedyToken shows the fused greedy-decode contract:
// the call returns the next token directly, skipping the softcapped logits
// tensor (greedy selection is monotonic under final logit softcapping).
func ExampleGemma4Model_ForwardGreedyToken() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	next := model.ForwardGreedyToken(tokens, nil, caches)
	metal.Free(tokens, next)
	metal.FreeCaches(caches)
}

// ExampleGemma4Model_ForwardGreedyTokenWithSuppression shows masking chat-template
// or modality token ids out of the greedy argmax — the suppressed ids can never
// be selected as the next token.
func ExampleGemma4Model_ForwardGreedyTokenWithSuppression() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	// Suppress the image and audio placeholder ids during a text turn.
	suppress := []int32{model.Cfg.ImageTokenID, model.Cfg.AudioTokenID}
	next := model.ForwardGreedyTokenWithSuppression(tokens, nil, caches, suppress)
	metal.Free(tokens, next)
	metal.FreeCaches(caches)
}

// ExampleGemma4Model_ImagePlaceholderBlock renders the HF processor convention
// for one image: BOI, one image token per soft token, then EOI. The block is
// pure string assembly, so it runs without a loaded model.
func ExampleGemma4Model_ImagePlaceholderBlock() {
	model := &Gemma4Model{}
	core.Println(model.ImagePlaceholderBlock(2))
	// Output: <|image><|image|><|image|><image|>
}
