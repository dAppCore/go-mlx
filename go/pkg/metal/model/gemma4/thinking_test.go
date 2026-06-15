// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "testing"

// TestThinking_ThinkingChannelTokens_Bad pins the guards that keep the decode
// loop's thinking budget from acting on a model that cannot delimit a thought
// channel: a nil model, a model with no tokenizer, and a real loaded model whose
// minimal tokenizer lacks the <|channel>/<channel|> markers all report ok=false.
func TestThinking_ThinkingChannelTokens_Bad(t *testing.T) {
	var nilModel *Gemma4Model
	if _, _, ok := nilModel.ThinkingChannelTokens(); ok {
		t.Fatal("ThinkingChannelTokens(nil) ok = true, want false")
	}

	noTok := &Gemma4Model{}
	if _, _, ok := noTok.ThinkingChannelTokens(); ok {
		t.Fatal("ThinkingChannelTokens(no tokenizer) ok = true, want false")
	}

	// A loaded model whose minimal tokenizer has no channel markers: the
	// markers do not encode to a single token, so resolution fails cleanly.
	model := loadGemma4DenseTestModel(t)
	open, close, ok := model.ThinkingChannelTokens()
	if ok {
		t.Fatalf("ThinkingChannelTokens ok = true (open=%d close=%d) for a tokenizer without channel markers, want false", open, close)
	}
	if open != 0 || close != 0 {
		t.Fatalf("ThinkingChannelTokens not-ok returned (open=%d close=%d), want (0, 0)", open, close)
	}
}
