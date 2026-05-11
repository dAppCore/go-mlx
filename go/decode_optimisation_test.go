// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"
)

func TestRunSpeculativeDecode_Good_AcceptsAndRejectsDraftTokens(t *testing.T) {
	targetCalls := 0
	draftCalls := 0
	target := func(context.Context, string, GenerateConfig) (DecodeGeneration, error) {
		targetCalls++
		return DecodeGeneration{
			Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 4, Text: "D"}},
			Metrics: Metrics{
				GeneratedTokens:     3,
				DecodeDuration:      30 * time.Millisecond,
				DecodeTokensPerSec:  100,
				PrefillTokensPerSec: 200,
			},
		}, nil
	}
	draft := func(context.Context, string, GenerateConfig) (DecodeGeneration, error) {
		draftCalls++
		return DecodeGeneration{
			Tokens:  []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 3, Text: "C"}},
			Metrics: Metrics{GeneratedTokens: 3, DecodeDuration: 5 * time.Millisecond},
		}, nil
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
	if result.Text != "ABD" {
		t.Fatalf("Text = %q, want ABD", result.Text)
	}
	if result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 || result.Metrics.AcceptanceRate != 2.0/3.0 {
		t.Fatalf("metrics = %+v, want two accepted and one rejected draft token", result.Metrics)
	}
	if result.Metrics.TargetCalls != 1 || result.Metrics.DraftCalls != 1 || targetCalls != 1 || draftCalls != 1 {
		t.Fatalf("calls = metrics:%+v target:%d draft:%d, want one target and draft call", result.Metrics, targetCalls, draftCalls)
	}
}

func TestRunPromptLookupDecode_Good_AcceptsRepeatedContextTokens(t *testing.T) {
	target := func(context.Context, string, GenerateConfig) (DecodeGeneration, error) {
		return DecodeGeneration{
			Tokens: []Token{{ID: 10, Text: "go"}, {ID: 11, Text: "-"}, {ID: 12, Text: "mlx"}},
		}, nil
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
	if result.Text != "go-mlx" {
		t.Fatalf("Text = %q, want go-mlx", result.Text)
	}
	if result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 || result.Metrics.LookupTokens != 3 {
		t.Fatalf("metrics = %+v, want two lookup accepts, one rejection", result.Metrics)
	}
}

func TestRunSpeculativeDecode_Bad_RequiresTargetAndDraft(t *testing.T) {
	_, err := RunSpeculativeDecode(context.Background(), SpeculativeDecodeConfig{})
	if err == nil {
		t.Fatal("RunSpeculativeDecode() error = nil, want missing runner error")
	}
}
