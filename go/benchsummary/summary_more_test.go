// SPDX-Licence-Identifier: EUPL-1.2

package benchsummary

import (
	"bytes"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
)

// TestBenchSummary_WriteAllOptionalSections_Good drives every optional report
// section (prompt cache, KV restore, state bundle, probes) with Attempted set,
// and an empty-Mode speculative decode so the default "speculative" label
// branch is taken.
func TestBenchSummary_WriteAllOptionalSections_Good(t *testing.T) {
	var out bytes.Buffer
	Write(&out, &bench.Report{
		ModelPath: "/models/gemma-4-e2b",
		Generation: bench.GenerationSummary{
			PrefillTokensPerSec: 900,
			DecodeTokensPerSec:  88,
			PeakMemoryBytes:     4 << 30,
			ActiveMemoryBytes:   3 << 30,
		},
		PromptCache: bench.PromptCacheReport{
			Attempted: true,
			Hits:      3,
			Misses:    1,
			HitRate:   0.5,
		},
		KVRestore: bench.LatencyReport{
			Attempted: true,
			Duration:  2 * time.Millisecond,
		},
		StateBundle: bench.StateBundleReport{
			Attempted: true,
			Bytes:     512,
			Duration:  3 * time.Millisecond,
		},
		Probes: bench.ProbeReport{
			Attempted:     true,
			EventCount:    10,
			OverheadRatio: 0.025,
		},
		SpeculativeDecode: bench.DecodeOptimisationReport{
			Attempted: true,
			Result: bench.DecodeOptimisationResult{
				Mode: "",
			},
			Metrics: bench.DecodeOptimisationMetrics{
				DraftTokens:         4,
				AcceptedTokens:      3,
				RejectedTokens:      1,
				AcceptanceRate:      0.75,
				VisibleTokensPerSec: 96,
				Duration:            time.Second,
			},
		},
	})
	got := out.String()

	for _, want := range []string{
		"  prompt cache: 50% hit rate (3 hit, 1 miss)\n",
		"  KV restore: 2ms\n",
		"  state bundle: 512 bytes, 3ms round trip\n",
		"  probes: 10 events, 2.5% overhead\n",
		"speculative: 75.0% accepted (4 proposed, 3 accepted, 1 rejected), 96.0 visible tok/s, wall 1s",
	} {
		if !core.Contains(got, want) {
			t.Fatalf("summary = %q, want substring %q", got, want)
		}
	}
}

// TestBenchSummary_WriteSpeculativeNoCallMetrics_Good exercises the speculative
// decode path when no target/draft throughput is recorded, so the optional
// "target: ... draft: ..." line is omitted.
func TestBenchSummary_WriteSpeculativeNoCallMetrics_Good(t *testing.T) {
	var out bytes.Buffer
	Write(&out, &bench.Report{
		ModelPath: "/models/gemma-4-e2b",
		SpeculativeDecode: bench.DecodeOptimisationReport{
			Attempted: true,
			Result:    bench.DecodeOptimisationResult{Mode: "mtp"},
			Metrics: bench.DecodeOptimisationMetrics{
				AcceptanceRate:      0.5,
				VisibleTokensPerSec: 100,
				Duration:            time.Second,
			},
		},
	})
	got := out.String()
	if !core.Contains(got, "mtp: 50.0% accepted") {
		t.Fatalf("summary = %q, want speculative line", got)
	}
	if core.Contains(got, "target:") {
		t.Fatalf("summary = %q, want no target/draft line when call metrics absent", got)
	}
}
