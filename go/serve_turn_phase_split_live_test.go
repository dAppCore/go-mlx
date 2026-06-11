// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestServeTurnPhaseSplit_LiveModel is the #74 instrument: the serve chat
// lane driven in-process across multi-turn growth, each turn split into the
// phases the HTTP wall conflates — acquire+prefill (the Chat call), first
// token, the decode stream, and finishTurn (the sleep). The engine's own
// Metrics cross-check the external clocks. Numbers, then fixes.
//
//	go test -tags model_eval -run TestServeTurnPhaseSplit_LiveModel -count=1 dappco.re/go/mlx
//	MLX_PHASE_SPLIT_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit go test ... (bigger models)
func TestServeTurnPhaseSplit_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	model := core.Getenv("MLX_PHASE_SPLIT_MODEL")
	if model == "" {
		model = "mlx-community/gemma-4-e2b-it-4bit"
	}
	dir := metaltest.HFModelPath(t, model)
	m, err := LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	store := state.NewInMemoryStore(nil)
	continuity, err := NewConversationContinuity(m, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity: %v", err)
	}
	ctx := context.Background()
	off := false

	messages := []inference.Message{{Role: "user", Content: "Begin a story about a glassblower. Around three hundred words."}}
	const turns = 6
	for turn := 1; turn <= turns; turn++ {
		chatStart := time.Now()
		seq, ok := continuity.Chat(ctx, messages,
			inference.WithMaxTokens(420), inference.WithTemperature(0.8), inference.WithEnableThinking(&off))
		if !ok {
			t.Fatalf("turn %d: continuity declined", turn)
		}
		chatDur := time.Since(chatStart)

		var firstTok, lastTok time.Time
		tokens := 0
		reply := core.NewBuilder()
		drainStart := time.Now()
		for token := range seq {
			if tokens == 0 {
				firstTok = time.Now()
			}
			lastTok = time.Now()
			tokens++
			reply.WriteString(token.Text)
		}
		seqExit := time.Now()
		if tokens == 0 {
			t.Fatalf("turn %d generated nothing", turn)
		}

		first := firstTok.Sub(drainStart)
		decode := lastTok.Sub(firstTok)
		finish := seqExit.Sub(lastTok)
		decodeRate := float64(tokens-1) / decode.Seconds()
		metrics := m.Metrics()
		t.Logf("turn %d │ chat(acquire+prefill) %6.0fms │ first-tok %6.0fms │ decode %5.2fs %5.1f tok/s (%d toks) │ finish(sleep) %6.0fms │ engine: prefill %4.0fms hit %4d/%4d lane=%s %5.1f tok/s",
			turn, chatDur.Seconds()*1000, first.Seconds()*1000, decode.Seconds(), decodeRate, tokens,
			finish.Seconds()*1000, metrics.PrefillDuration.Seconds()*1000,
			metrics.PromptCacheHitTokens, metrics.PromptTokens, metrics.DecodeLane, metrics.DecodeTokensPerSec)
		if metrics.DecodeLane != "pipelined" {
			t.Errorf("turn %d lane = %q (%s) — want pipelined", turn, metrics.DecodeLane, metrics.DecodeLaneReason)
		}

		messages = append(messages,
			inference.Message{Role: "assistant", Content: reply.String()},
			inference.Message{Role: "user", Content: "Continue the story."})
	}
}
