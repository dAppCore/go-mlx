// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference/decode"
)

// These tests cover the mlx-side shim around go-inference/decode/.
// Algorithmic coverage lives in go-inference/decode/decode_test.go; here
// we only verify the boundary converters + legacy-alias surface.

func TestRunSpeculativeDecode_Mlx_AcceptsAndRejectsDraftTokens_Good(t *testing.T) {
	target := func(_ context.Context, _ string, cfg GenerateConfig) (DecodeGeneration, error) {
		if cfg.MaxTokens != 3 {
			t.Fatalf("target MaxTokens = %d, want 3 (clamped from cfg.MaxTokens=3)", cfg.MaxTokens)
		}
		return DecodeGeneration{
			Tokens:  []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 4, Text: "D"}},
			Metrics: Metrics{GeneratedTokens: 3},
		}, nil
	}
	draft := func(context.Context, string, GenerateConfig) (DecodeGeneration, error) {
		return DecodeGeneration{Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 3, Text: "C"}}}, nil
	}
	result, err := RunSpeculativeDecode(context.Background(), SpeculativeDecodeConfig{
		Prompt:         "p",
		MaxTokens:      3,
		DraftTokens:    3,
		TargetGenerate: target,
		DraftGenerate:  draft,
	})
	if err != nil {
		t.Fatalf("RunSpeculativeDecode() error = %v", err)
	}
	if result.Mode != DecodeModeSpeculative {
		t.Fatalf("Mode = %q, want %q", result.Mode, DecodeModeSpeculative)
	}
	if result.Text != "ABD" {
		t.Fatalf("Text = %q, want ABD", result.Text)
	}
	if result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 {
		t.Fatalf("metrics = %+v, want 2 accepted + 1 rejected", result.Metrics)
	}
}

func TestRunPromptLookupDecode_Mlx_AcceptsRepeatedContextTokens_Good(t *testing.T) {
	target := func(context.Context, string, GenerateConfig) (DecodeGeneration, error) {
		return DecodeGeneration{Tokens: []Token{{ID: 10, Text: "go"}, {ID: 11, Text: "-"}, {ID: 12, Text: "mlx"}}}, nil
	}
	result, err := RunPromptLookupDecode(context.Background(), PromptLookupDecodeConfig{
		Prompt:         "go-mlx go-mlx",
		MaxTokens:      3,
		TargetGenerate: target,
		LookupTokens:   []Token{{ID: 10, Text: "go"}, {ID: 99, Text: "?"}, {ID: 12, Text: "mlx"}},
	})
	if err != nil {
		t.Fatalf("RunPromptLookupDecode() error = %v", err)
	}
	if result.Mode != DecodeModePromptLookup {
		t.Fatalf("Mode = %q, want %q", result.Mode, DecodeModePromptLookup)
	}
	if result.Text != "go-mlx" {
		t.Fatalf("Text = %q, want go-mlx", result.Text)
	}
}

func TestRunSpeculativeDecode_Mlx_RequiresTargetAndDraft_Bad(t *testing.T) {
	if _, err := RunSpeculativeDecode(context.Background(), SpeculativeDecodeConfig{}); err == nil {
		t.Fatal("RunSpeculativeDecode() error = nil, want missing-target")
	}
}

func TestRunPromptLookupDecode_Mlx_RequiresTarget_Bad(t *testing.T) {
	if _, err := RunPromptLookupDecode(context.Background(), PromptLookupDecodeConfig{}); err == nil {
		t.Fatal("RunPromptLookupDecode() error = nil, want missing-target")
	}
}

func TestMlxDecodeGenToDecode_NilFunc_Ugly(t *testing.T) {
	if got := mlxDecodeGenToDecode(nil); got != nil {
		t.Fatalf("mlxDecodeGenToDecode(nil) = non-nil, want nil")
	}
}

func TestMlxDecodeGenToDecode_ConvertsCallback_Good(t *testing.T) {
	gotMlxCfg := GenerateConfig{}
	src := func(_ context.Context, prompt string, cfg GenerateConfig) (DecodeGeneration, error) {
		gotMlxCfg = cfg
		return DecodeGeneration{Text: prompt + "!", Tokens: []Token{{ID: 7, Text: "x"}}}, nil
	}
	wrapped := mlxDecodeGenToDecode(src)
	out, err := wrapped(context.Background(), "hi", decode.GenerateConfig{MaxTokens: 9})
	if err != nil {
		t.Fatalf("wrapped() error = %v", err)
	}
	if gotMlxCfg.MaxTokens != 9 {
		t.Fatalf("inner mlx cfg MaxTokens = %d, want 9", gotMlxCfg.MaxTokens)
	}
	if out.Text != "hi!" {
		t.Fatalf("out.Text = %q, want hi!", out.Text)
	}
	if len(out.Tokens) != 1 || out.Tokens[0].ID != 7 || out.Tokens[0].Text != "x" {
		t.Fatalf("out.Tokens = %+v", out.Tokens)
	}
}

func TestMlxTokensToDecode_RoundTrip_Good(t *testing.T) {
	src := []Token{{ID: 1, Text: "a", Value: "alpha"}, {ID: 2, Text: "b"}}
	dec := mlxTokensToDecode(src)
	back := decodeTokensToMlx(dec)
	if len(back) != len(src) {
		t.Fatalf("round-trip length mismatch: %d vs %d", len(back), len(src))
	}
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("round-trip token[%d] = %+v, want %+v", i, back[i], src[i])
		}
	}
}

func TestMlxTokensToDecode_NilInNilOut_Ugly(t *testing.T) {
	if got := mlxTokensToDecode(nil); got != nil {
		t.Fatalf("mlxTokensToDecode(nil) = %v, want nil", got)
	}
	if got := decodeTokensToMlx(nil); got != nil {
		t.Fatalf("decodeTokensToMlx(nil) = %v, want nil", got)
	}
}

func TestDecodeTokensText_RendersFromMlxTokens_Good(t *testing.T) {
	got := decodeTokensText([]Token{{Text: "go"}, {Value: "-"}, {Text: "mlx"}})
	if got != "go-mlx" {
		t.Fatalf("decodeTokensText = %q, want go-mlx", got)
	}
}
