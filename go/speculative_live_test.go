// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/inference/memory"
)

// TestSpeculativeServeStreaming_LiveModel proves the serve-shaped MTP lane
// streams: tokens must arrive incrementally as verify rounds land, not as one
// burst after the loop completes (the pre-GenerateWithSink behaviour). The
// gate: the first token lands in the first third of the run and the arrivals
// span most of it.
//
//	go test -tags model_eval -run TestSpeculativeServeStreaming_LiveModel -count=1 dappco.re/go/mlx
func TestSpeculativeServeStreaming_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; cache gemma-4-e2b-it-4bit + its assistant drafter")
	}
	targetDir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	draftDir := metaltest.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	tm, err := LoadSpeculativePairAsTextModel(targetDir, draftDir,
		WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadSpeculativePairAsTextModel: %v", err)
	}
	defer func() {
		if closeErr := resultError(tm.Close()); closeErr != nil {
			t.Errorf("Close: %v", closeErr)
		}
	}()

	messages := []inference.Message{{
		Role:    "user",
		Content: "Write a Go function that parses a CSV file into a slice of Person structs (Name string, Age int, Email string), with full error handling and a doc comment.",
	}}

	start := time.Now()
	var arrivals []time.Duration
	for range tm.Chat(context.Background(), messages, inference.WithMaxTokens(200)) {
		arrivals = append(arrivals, time.Since(start))
	}
	if err := resultError(tm.Err()); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(arrivals) < 30 {
		t.Fatalf("expected a real generation, got %d tokens", len(arrivals))
	}
	total := arrivals[len(arrivals)-1]
	first := arrivals[0]
	t.Logf("tokens %d · first %.0fms · total %.0fms", len(arrivals), first.Seconds()*1000, total.Seconds()*1000)
	if first > total/3 {
		t.Errorf("first token at %.0fms of %.0fms — not streaming (one-shot burst)", first.Seconds()*1000, total.Seconds()*1000)
	}
	spread := total - first
	if spread < total/2 {
		t.Errorf("arrivals span %.0fms of %.0fms — tokens arrived as a burst, not incrementally", spread.Seconds()*1000, total.Seconds()*1000)
	}
}
