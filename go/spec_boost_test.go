// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/internal/metaltest"
	gemma4chat "dappco.re/go/mlx/pkg/metal/model/gemma4/chat"
)

// TestSpeculativeBoost_Repro gates batched MTP verify: it MUST reproduce the
// target's plain greedy output exactly (speculative decoding is greedy-exact),
// and it reports the decode speedup + accept rate + target-call count.
func TestSpeculativeBoost_Repro(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval diagnostic; -tags model_eval + cached target/assistant")
	}
	targetRepo := core.Getenv("GO_MLX_SPEC_TARGET")
	if targetRepo == "" {
		targetRepo = "mlx-community/gemma-4-e2b-it-4bit"
	}
	draftRepo := core.Getenv("GO_MLX_SPEC_DRAFT")
	if draftRepo == "" {
		draftRepo = "mlx-community/gemma-4-E2B-it-assistant-bf16"
	}
	large := core.Getenv("GO_MLX_SPEC_LARGE") != ""
	targetPath := metaltest.HFModelPath(t, targetRepo)
	draftPath := metaltest.HFModelPath(t, draftRepo)
	t.Logf("target=%s draft=%s", targetRepo, draftRepo)

	pair, err := LoadSpeculativePair(targetPath, draftPath, SpeculativePairConfig{})
	if err != nil {
		t.Fatalf("LoadSpeculativePair: %v", err)
	}
	defer func() { _ = pair.Close() }()

	formatted := gemma4chat.Format(
		[]chat.Message{{Role: "user", Content: "Write a short, vivid story about a lighthouse keeper and the deep ocean."}},
		chat.Config{Architecture: "gemma4", LargeVariant: large},
	)
	const maxTok = 200

	// Plain greedy reference from the SAME target model.
	pstart := time.Now()
	plainText, perr := pair.Target.Generate(formatted, WithMaxTokens(maxTok))
	if perr != nil {
		t.Fatalf("plain greedy: %v", perr)
	}
	plainTokPerSec := float64(maxTok) / time.Since(pstart).Seconds()

	// Warm the MTP kernels, then time it.
	if _, err := pair.Generate(context.Background(), formatted, SpeculativeDecodeConfig{MaxTokens: 8, GenerateConfig: GenerateConfig{MaxTokens: 8}}); err != nil {
		t.Fatalf("warm: %v", err)
	}
	mstart := time.Now()
	res, err := pair.Generate(context.Background(), formatted, SpeculativeDecodeConfig{MaxTokens: maxTok, GenerateConfig: GenerateConfig{MaxTokens: maxTok}})
	mdur := time.Since(mstart)
	if err != nil {
		t.Fatalf("mtp generate: %v", err)
	}
	mtpTokPerSec := float64(len(res.Tokens)) / mdur.Seconds()

	t.Logf("plain=%.1f tok/s  mtp=%.1f tok/s  (%.2fx)  accept=%.3f  targetCalls=%d draftCalls=%d",
		plainTokPerSec, mtpTokPerSec, mtpTokPerSec/plainTokPerSec,
		res.Metrics.AcceptanceRate, res.Metrics.TargetCalls, res.Metrics.DraftCalls)

	// CORRECTNESS GATE — speculative decode must be greedy-exact.
	if res.Text != plainText {
		pn, mn := min(160, len(plainText)), min(160, len(res.Text))
		t.Errorf("MTP output != plain greedy — speculative correctness BROKEN\nplain: %q\nmtp:   %q", plainText[:pn], res.Text[:mn])
	}
}
