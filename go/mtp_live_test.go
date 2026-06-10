// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/memory"
)

// mtpBaselinePairs are the target/drafter pairings the baseline measurement
// covers. One pair loads at a time; comment pairs in or out to scope a run.
//
// Measured 2026-06-10 (e2b, story prompt): plain pipelined 161 tok/s vs MTP
// best 73 tok/s at acceptance 0.35 (bf16 drafter; qat 0.31) — the pipelined
// plain loop already amortises the host tax MTP used to win back, so small
// host-bound models don't benefit. The MTP lane is the GPU-bound big models,
// where one weight-read verifying 1+α·k tokens is the only lever.
var mtpBaselinePairs = []struct {
	name   string
	target string
	draft  string
}{
	// Measured 2026-06-10, 31b-qat4: plain 5.1 tok/s (wide-head global layers
	// scan the full capacity band under the pipeline's wide gate — fill-band
	// attention slicing is the fix) vs MTP 6.2 at acceptance 0.42: the
	// relative MTP win on a GPU-bound model is real; absolute 31B perf is
	// blocked on the wide-read fix.
	// {"e2b-bf16", "mlx-community/gemma-4-e2b-it-4bit", "mlx-community/gemma-4-E2B-it-assistant-bf16"},
	{"31b-qat4", "mlx-community/gemma-4-31b-it-qat-4bit", "mlx-community/gemma-4-31B-it-qat-assistant-4bit"},
	// {"26b-qat4", "mlx-community/gemma-4-26B-A4B-it-qat-4bit", "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit"},
}

// TestMTPPair_Baseline_LiveModel measures the native Gemma 4 MTP speculative
// lane against a plain pipelined session on the same loaded target. This is
// the measurement that sizes the encode-amortisation work: what the MTP loop
// delivers today, where its time goes (target verify vs draft), and the
// acceptance rate that bounds the theoretical speedup.
//
//	go test -tags model_eval -run TestMTPPair_Baseline_LiveModel -count=1 dappco.re/go/mlx
func TestMTPPair_Baseline_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache the target + assistant pairs")
	}
	// Acceptance is prompt-class dependent: creative free text has
	// high-entropy continuations (the drafter's guesses miss), while code is
	// dense with low-entropy tokens — syntax, boilerplate, predictable
	// identifiers — where the drafter states the obvious and the obvious is
	// most of the text. Both classes run so the numbers bound the range.
	prompts := []struct{ name, text string }{
		{"story", "Write a long, detailed story about a clockmaker who repairs time itself."},
		{"code", "Write a Go function that parses a CSV file into a slice of Person structs (Name string, Age int, Email string), with full error handling and a doc comment."},
	}
	ctx := context.Background()

	for _, tc := range mtpBaselinePairs {
		t.Run(tc.name, func(t *testing.T) {
			targetDir := metaltest.HFModelPath(t, tc.target)
			draftDir := metaltest.HFModelPath(t, tc.draft)

			pair, err := LoadSpeculativePair(targetDir, draftDir, SpeculativePairConfig{
				TargetOptions: []LoadOption{WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096)},
			})
			if err != nil {
				t.Fatalf("LoadSpeculativePair: %v", err)
			}
			defer pair.Target.Close()
			if pair.Draft != nil {
				defer pair.Draft.Close()
			}
			if pair.Gemma4Assistant == nil {
				t.Fatalf("pair did not attach the native gemma4 assistant (layout: %+v)", pair.Report.AssistantLayout)
			}
			t.Logf("assistant layout: %+v", pair.Report.AssistantLayout)

			for _, p := range prompts {
				// Plain pipelined session on the same target — the number
				// MTP has to beat.
				sess, err := pair.Target.NewSession()
				if err != nil {
					t.Fatalf("%s: plain session: %v", p.name, err)
				}
				if err := sess.Prefill(p.text); err != nil {
					t.Fatalf("%s: plain prefill: %v", p.name, err)
				}
				tokens := 0
				start := time.Now()
				for range sess.GenerateStream(ctx, WithMaxTokens(200), WithTemperature(0)) {
					tokens++
				}
				if err := sess.Err(); err != nil {
					t.Fatalf("%s: plain generate: %v", p.name, err)
				}
				plainRate := float64(tokens) / time.Since(start).Seconds()
				sess.Close()
				t.Logf("%s · plain pipelined session: %.1f tok/s (%d tok)", p.name, plainRate, tokens)

				for _, draftTokens := range []int{2, 4} {
					res, err := pair.Generate(ctx, p.text, SpeculativeDecodeConfig{
						MaxTokens:   200,
						DraftTokens: draftTokens,
						GenerateConfig: GenerateConfig{
							MaxTokens:   200,
							Temperature: 0,
						},
					})
					if err != nil {
						t.Fatalf("%s: pair.Generate (draft=%d): %v", p.name, draftTokens, err)
					}
					m := res.Metrics
					rate := 0.0
					if m.Duration > 0 {
						rate = float64(m.EmittedTokens) / m.Duration.Seconds()
					}
					t.Logf("%s · MTP draft=%d: %.1f tok/s overall · emitted %d · accept %.2f (acc %d / rej %d / draft %d) · target %d calls %dms · draft %d calls %dms · total %dms",
						p.name, draftTokens, rate, m.EmittedTokens, m.AcceptanceRate,
						m.AcceptedTokens, m.RejectedTokens, m.DraftTokens,
						m.TargetCalls, m.TargetDuration.Milliseconds(),
						m.DraftCalls, m.DraftDuration.Milliseconds(),
						m.Duration.Milliseconds())
				}
			}
		})
	}
}
