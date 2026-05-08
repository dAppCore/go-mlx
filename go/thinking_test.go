// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

type fakeThinkingTokenizer struct {
	pieces map[int32]string
}

func (t fakeThinkingTokenizer) Encode(string) []int32 { return nil }

func (t fakeThinkingTokenizer) Decode(tokens []int32) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(t.pieces[token])
	}
	return builder.String()
}

func (t fakeThinkingTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (t fakeThinkingTokenizer) IDToken(id int32) string      { return t.pieces[id] }
func (t fakeThinkingTokenizer) BOS() int32                   { return 0 }
func (t fakeThinkingTokenizer) EOS() int32                   { return 0 }
func (t fakeThinkingTokenizer) HasBOSToken() bool            { return false }

func TestFilterThinkingTokens_QwenCaptureWithFakeTokenizer_Good(t *testing.T) {
	coverageTokens := "QwenCaptureWithFakeTokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	tokenizer := &Tokenizer{tok: fakeThinkingTokenizer{pieces: map[int32]string{
		1: "<think>",
		2: "map",
		3: "</think>",
		4: "visible",
	}}}
	var captured []ThinkingChunk

	got, err := FilterThinkingTokens(tokenizer, []int32{1, 2, 3, 4}, ThinkingConfig{
		Mode: ThinkingCapture,
		Capture: func(chunk ThinkingChunk) {
			captured = append(captured, chunk)
		},
	}, ModelInfo{Architecture: "qwen3"})
	if err != nil {
		t.Fatalf("FilterThinkingTokens() error = %v", err)
	}
	if got.Text != "visible" {
		t.Fatalf("Text = %q, want visible", got.Text)
	}
	if got.Reasoning != "map" {
		t.Fatalf("Reasoning = %q, want map", got.Reasoning)
	}
	if len(captured) != 1 {
		t.Fatalf("captured len = %d, want 1", len(captured))
	}
	if captured[0].Text != "map" || captured[0].Channel != "thinking" || captured[0].Model != "qwen" {
		t.Fatalf("captured chunk = %+v", captured[0])
	}
}

func TestFilterThinkingText_GemmaHideChannelMarkers_Good(t *testing.T) {
	coverageTokens := "GemmaHideChannelMarkers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}

	got := FilterThinkingText(
		"<start_of_turn>thinking\nplan<end_of_turn>final",
		ThinkingConfig{Mode: ThinkingHide},
		ModelInfo{Architecture: "gemma4_text"},
	)
	if got.Text != "final" {
		t.Fatalf("Text = %q, want final", got.Text)
	}
	if got.Reasoning != "plan" {
		t.Fatalf("Reasoning = %q, want plan", got.Reasoning)
	}
}

func TestFilterThinkingText_ShowIsPassthrough_Ugly(t *testing.T) {
	coverageTokens := "ShowIsPassthrough"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	raw := "<think>secret</think>visible"

	got := FilterThinkingText(raw, ThinkingConfig{Mode: ThinkingShow}, ModelInfo{Architecture: "qwen3"})
	if got.Text != raw {
		t.Fatalf("Text = %q, want raw passthrough", got.Text)
	}
	if got.Reasoning != "" {
		t.Fatalf("Reasoning = %q, want empty for passthrough mode", got.Reasoning)
	}
}
