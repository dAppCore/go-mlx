// SPDX-Licence-Identifier: EUPL-1.2

package speculative

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/decode"
	mlx "dappco.re/go/mlx"
)

func TestRunPairProfile_Good(t *testing.T) {
	pair := &fakeProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 7, Text: "G"}, {ID: 8, Text: "o"}},
			Text:   "Go",
		},
		metrics: mlx.Metrics{
			GeneratedTokens:            2,
			DecodeDuration:             20 * time.Millisecond,
			DecodeTokensPerSec:         100,
			PeakMemoryBytes:            2048,
			ActiveMemoryBytes:          1024,
			CacheMemoryBytes:           512,
			PromptCacheRestoreDuration: 3 * time.Millisecond,
			MTP: &mlx.MTPMetrics{
				DraftTokenSchedule:     []int{2},
				ProposedTokens:         2,
				AcceptedTokens:         2,
				TargetVerifyCalls:      1,
				DraftCalls:             1,
				AcceptanceRate:         1,
				VisibleTokensPerSec:    90,
				TargetTokensPerSec:     120,
				WarmDecodeTokensPerSec: 100,
			},
		},
	}

	run, err := RunPairProfile(context.Background(), pair, ProfileConfig{
		Prompt:        "prompt",
		MaxTokens:     2,
		DraftTokens:   2,
		IncludeOutput: true,
	})
	if err != nil {
		t.Fatalf("RunPairProfile() error = %v", err)
	}
	if pair.prompt != "prompt" || pair.config.MaxTokens != 2 || pair.config.DraftTokens != 2 {
		t.Fatalf("Generate args prompt=%q cfg=%+v, want prompt/max=2/draft=2", pair.prompt, pair.config)
	}
	if run.Output != "Go" || run.VisibleTokens != 2 {
		t.Fatalf("run output = %q tokens=%d, want Go/2", run.Output, run.VisibleTokens)
	}
	if len(run.SampledTokenIDs) != 2 || run.SampledTokenIDs[0] != 7 || run.SampledTokenTexts[1] != "o" {
		t.Fatalf("sampled tokens = ids:%v text:%v, want copied token sample", run.SampledTokenIDs, run.SampledTokenTexts)
	}
	if run.Metrics.MTP == nil || run.Metrics.MTP.ProposedTokens != 2 || run.Metrics.MTP.WarmDecodeTokensPerSec != 100 {
		t.Fatalf("run metrics = %+v, want MTP counters from target", run.Metrics.MTP)
	}
}

func TestRunPairProfile_ChatControls_Good(t *testing.T) {
	pair := &fakeProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 9, Text: "A"}},
			Text:   "A",
		},
	}

	_, err := RunPairProfile(context.Background(), pair, ProfileConfig{
		Prompt:       "state smoke",
		MaxTokens:    1,
		DraftTokens:  1,
		Chat:         true,
		Architecture: "gemma4_text",
		GenerateConfig: mlx.GenerateConfig{
			StopTokens:     []int32{1, 106},
			SuppressTokens: []int32{0, 2, 3},
		},
	})
	if err != nil {
		t.Fatalf("RunPairProfile() error = %v", err)
	}
	if !core.Contains(pair.prompt, "<|turn>user\nstate smoke<turn|>\n<|turn>model\n") {
		t.Fatalf("prompt = %q, want Gemma 4 chat-formatted prompt", pair.prompt)
	}
	if got := pair.config.GenerateConfig.StopTokens; len(got) != 2 || got[0] != 1 || got[1] != 106 {
		t.Fatalf("StopTokens = %v, want [1 106]", got)
	}
	if got := pair.config.GenerateConfig.SuppressTokens; len(got) != 3 || got[0] != 0 || got[2] != 3 {
		t.Fatalf("SuppressTokens = %v, want [0 2 3]", got)
	}
}

func TestRunPairProfile_DefaultDraftTokensPolicy_Good(t *testing.T) {
	pair := &fakeProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 10, Text: "D"}},
			Text:   "D",
		},
	}

	_, err := RunPairProfile(context.Background(), pair, ProfileConfig{
		Prompt:    "prompt",
		MaxTokens: 1,
	})
	if err != nil {
		t.Fatalf("RunPairProfile() error = %v", err)
	}
	if pair.config.DraftTokens != mlx.ProductionMTPDefaultDraftTokens {
		t.Fatalf("DraftTokens = %d, want production default %d", pair.config.DraftTokens, mlx.ProductionMTPDefaultDraftTokens)
	}
}

func TestRunPairProfile_Bad(t *testing.T) {
	if _, err := RunPairProfile(context.Background(), nil, ProfileConfig{}); err == nil {
		t.Fatal("RunPairProfile(nil) error = nil, want guard")
	}
}

type fakeProfilePair struct {
	prompt  string
	config  mlx.SpeculativeDecodeConfig
	result  mlx.SpeculativeDecodeResult
	metrics mlx.Metrics
	err     error
}

func (pair *fakeProfilePair) Generate(_ context.Context, prompt string, cfg mlx.SpeculativeDecodeConfig) (mlx.SpeculativeDecodeResult, error) {
	pair.prompt = prompt
	pair.config = cfg
	return pair.result, pair.err
}

func (pair *fakeProfilePair) Metrics() mlx.Metrics {
	return pair.metrics
}

func (pair *fakeProfilePair) Err() error {
	return pair.err
}
