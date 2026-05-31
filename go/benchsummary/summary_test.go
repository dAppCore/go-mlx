// SPDX-Licence-Identifier: EUPL-1.2

package benchsummary

import (
	"bytes"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
)

func TestBenchSummary_WriteMTPMetrics_Good(t *testing.T) {
	coverageTokens := "BenchSummary WriteMTPMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	var out bytes.Buffer
	Write(&out, &bench.Report{
		ModelPath: "/models/gemma-4-e2b",
		Generation: bench.GenerationSummary{
			PrefillTokensPerSec: 1500,
			DecodeTokensPerSec:  120,
			PeakMemoryBytes:     8 << 30,
			ActiveMemoryBytes:   6 << 30,
		},
		SpeculativeDecode: bench.DecodeOptimisationReport{
			Attempted: true,
			Result: bench.DecodeOptimisationResult{
				Mode: "mtp",
			},
			Metrics: bench.DecodeOptimisationMetrics{
				DraftTokens:         4,
				AcceptedTokens:      3,
				RejectedTokens:      1,
				AcceptanceRate:      0.75,
				VisibleTokensPerSec: 132,
				TargetTokensPerSec:  180,
				DraftTokensPerSec:   320,
				TargetCalls:         2,
				DraftCalls:          2,
				Duration:            time.Second,
			},
		},
	})
	got := out.String()
	if !core.Contains(got, "target-only: prefill 1500.0 tok/s, raw decode 120.0 tok/s") {
		t.Fatalf("summary = %q, want target-only raw decode line", got)
	}
	if !core.Contains(got, "mtp: 75.0% accepted (4 proposed, 3 accepted, 1 rejected), 132.0 visible tok/s, wall 1s") {
		t.Fatalf("summary = %q, want MTP proposed/accepted/rejected line", got)
	}
	if !core.Contains(got, "target: 180.0 tok/s across 2 calls, draft: 320.0 tok/s across 2 calls") {
		t.Fatalf("summary = %q, want target/draft throughput line", got)
	}
}

func TestBenchSummary_WriteNil_Ugly(t *testing.T) {
	coverageTokens := "BenchSummary WriteNil"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	var out bytes.Buffer
	Write(&out, nil)
	if out.String() != "" {
		t.Fatalf("summary = %q, want empty nil report output", out.String())
	}
}
