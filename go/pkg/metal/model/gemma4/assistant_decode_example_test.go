// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import core "dappco.re/go"

func ExampleGemma4AssistantPair_DraftStep() {
	var pair *Gemma4AssistantPair
	_, err := pair.DraftStep(1, nil, nil)
	core.Println(err.Error())
	// Output: gemma4.assistant draft step requires a validated pair
}

func ExampleGemma4AssistantDraftStepResult_Close() {
	result := &Gemma4AssistantDraftStepResult{}
	result.Close()
	core.Println(result.Logits == nil, result.Token == nil, result.Hidden == nil)
	// Output: true true true
}

func ExampleGemma4AssistantPair_DraftBlock() {
	var pair *Gemma4AssistantPair
	_, err := pair.DraftBlock(1, nil, nil, 0)
	core.Println(err.Error())
	// Output: gemma4.assistant draft block maxDraftTokens must be > 0
}

func ExampleGemma4AssistantDraftBlockResult_Close() {
	result := &Gemma4AssistantDraftBlockResult{Tokens: []int32{1, 2}}
	result.Close()
	core.Println(result.Tokens == nil, result.Hidden == nil)
	// Output: true true
}

func ExampleGemma4AssistantPair_VerifyDraftBlock() {
	var pair *Gemma4AssistantPair
	_, err := pair.VerifyDraftBlock(nil, []int32{1}, nil)
	core.Println(err.Error())
	// Output: gemma4.assistant verify requires a target model
}

func ExampleGemma4AssistantVerifyResult_Close() {
	result := &Gemma4AssistantVerifyResult{
		DraftedTokens:  []int32{1, 2},
		TargetTokens:   []int32{1},
		AcceptedTokens: []int32{1},
		RejectedTokens: []int32{2},
	}
	result.Close()
	core.Println(result.DraftedTokens == nil, result.TargetTokens == nil, result.AcceptedTokens == nil, result.RejectedTokens == nil)
	// Output: true true true true
}
