// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"strconv"
	"strings"
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
	draftTokens := 0 // 0 -> assistant default (2)
	if v := core.Getenv("GO_MLX_SPEC_DRAFTTOKENS"); v != "" {
		if n, perr := strconv.Atoi(v); perr == nil {
			draftTokens = n
		}
	}
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
	if _, err := pair.Generate(context.Background(), formatted, SpeculativeDecodeConfig{MaxTokens: 8, DraftTokens: draftTokens, GenerateConfig: GenerateConfig{MaxTokens: 8}}); err != nil {
		t.Fatalf("warm: %v", err)
	}
	mstart := time.Now()
	res, err := pair.Generate(context.Background(), formatted, SpeculativeDecodeConfig{MaxTokens: maxTok, DraftTokens: draftTokens, GenerateConfig: GenerateConfig{MaxTokens: maxTok}})
	mdur := time.Since(mstart)
	if err != nil {
		t.Fatalf("mtp generate: %v", err)
	}
	mtpTokPerSec := float64(len(res.Tokens)) / mdur.Seconds()

	t.Logf("plain=%.1f tok/s  mtp=%.1f tok/s  (%.2fx)  accept=%.3f  targetCalls=%d draftCalls=%d",
		plainTokPerSec, mtpTokPerSec, mtpTokPerSec/plainTokPerSec,
		res.Metrics.AcceptanceRate, res.Metrics.TargetCalls, res.Metrics.DraftCalls)
	m := res.Metrics
	var draftPerCall, verifyPerCall float64
	if m.DraftCalls > 0 {
		draftPerCall = m.DraftDuration.Seconds() * 1000 / float64(m.DraftCalls)
	}
	if m.TargetCalls > 0 {
		verifyPerCall = m.TargetDuration.Seconds() * 1000 / float64(m.TargetCalls)
	}
	t.Logf("  split: draft=%v (%.2f ms/block over %d) verify=%v (%.2f ms/call over %d)",
		m.DraftDuration.Round(time.Millisecond), draftPerCall, m.DraftCalls,
		m.TargetDuration.Round(time.Millisecond), verifyPerCall, m.TargetCalls)

	// CORRECTNESS GATE — speculative decode must be greedy-exact.
	if res.Text != plainText {
		pn, mn := min(160, len(plainText)), min(160, len(res.Text))
		t.Errorf("MTP output != plain greedy — speculative correctness BROKEN\nplain: %q\nmtp:   %q", plainText[:pn], res.Text[:mn])
	}
}

// TestSpeculativeSampling_Repro exercises the temperature>0 speculative-SAMPLING
// path (option B). It must actually run the sampled lane (not fall back to plain
// or error), produce valid non-empty output, engage the drafter (DraftCalls>0),
// and be reproducible under a fixed seed — the proof that the accept-coin + draft
// RNG threading is correct. Output-distribution equivalence to plain sampling is
// argued from the unit-tested accept/residual maths; this is the integration +
// determinism check.
func TestSpeculativeSampling_Repro(t *testing.T) {
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
	targetPath := metaltest.HFModelPath(t, targetRepo)
	draftPath := metaltest.HFModelPath(t, draftRepo)

	pair, err := LoadSpeculativePair(targetPath, draftPath, SpeculativePairConfig{})
	if err != nil {
		t.Fatalf("LoadSpeculativePair: %v", err)
	}
	defer func() { _ = pair.Close() }()

	formatted := gemma4chat.Format(
		[]chat.Message{{Role: "user", Content: "Write a short Go function that sums a slice of ints."}},
		chat.Config{Architecture: "gemma4"},
	)
	const maxTok = 80
	mkCfg := func() SpeculativeDecodeConfig {
		return SpeculativeDecodeConfig{
			MaxTokens:      maxTok,
			DraftTokens:    0,
			GenerateConfig: GenerateConfig{MaxTokens: maxTok, Temperature: 1.0, TopP: 0.95, TopK: 64, Seed: 42, SeedSet: true},
		}
	}

	res1, err := pair.Generate(context.Background(), formatted, mkCfg())
	if err != nil {
		if strings.Contains(err.Error(), "logits-exposing drafter") {
			t.Skipf("drafter %s is ordered-embedding (sparse q) — needs the dense-q scatter to do sampled MTP: %v", draftRepo, err)
		}
		t.Fatalf("sampled generate: %v", err)
	}
	if len(res1.Tokens) == 0 || res1.Text == "" {
		t.Fatalf("sampled output empty (Tokens=%d Text=%q)", len(res1.Tokens), res1.Text)
	}
	if res1.Metrics.DraftCalls == 0 {
		t.Fatalf("sampled path did not engage the drafter (DraftCalls=0) — fell back to plain?")
	}
	t.Logf("sampled: %d tok  accept=%.3f  draftCalls=%d  text=%q",
		len(res1.Tokens), res1.Metrics.AcceptanceRate, res1.Metrics.DraftCalls, res1.Text[:min(80, len(res1.Text))])

	res2, err := pair.Generate(context.Background(), formatted, mkCfg())
	if err != nil {
		t.Fatalf("sampled generate (repro): %v", err)
	}
	if res2.Text != res1.Text {
		t.Errorf("seeded sampling not reproducible:\nrun1: %q\nrun2: %q",
			res1.Text[:min(120, len(res1.Text))], res2.Text[:min(120, len(res2.Text))])
	}
}
