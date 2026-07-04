// SPDX-Licence-Identifier: EUPL-1.2

// Package benchsummary renders concise human summaries for fast-eval reports.
package benchsummary

import (
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
)

// Write prints a compact fast-eval report for CLI users.
//
//	benchsummary.Write(stdout, report)
func Write(stdout io.Writer, report *bench.Report) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("fast eval: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  target-only: prefill %.1f tok/s, raw decode %.1f tok/s\n", report.Generation.PrefillTokensPerSec, report.Generation.DecodeTokensPerSec))
	core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, active memory: %d MB\n", report.Generation.PeakMemoryBytes/1024/1024, report.Generation.ActiveMemoryBytes/1024/1024))
	if report.PromptCache.Attempted {
		core.WriteString(stdout, core.Sprintf("  prompt cache: %.0f%% hit rate (%d hit, %d miss)\n", report.PromptCache.HitRate*100, report.PromptCache.Hits, report.PromptCache.Misses))
	}
	if report.KVRestore.Attempted {
		core.WriteString(stdout, core.Sprintf("  KV restore: %s\n", report.KVRestore.Duration))
	}
	if report.StateBundle.Attempted {
		core.WriteString(stdout, core.Sprintf("  state bundle: %d bytes, %s round trip\n", report.StateBundle.Bytes, report.StateBundle.Duration))
	}
	if report.Probes.Attempted {
		core.WriteString(stdout, core.Sprintf("  probes: %d events, %.1f%% overhead\n", report.Probes.EventCount, report.Probes.OverheadRatio*100))
	}
	if report.SpeculativeDecode.Attempted {
		metrics := report.SpeculativeDecode.Metrics
		core.WriteString(stdout, core.Sprintf("  %s: %.1f%% accepted (%d proposed, %d accepted, %d rejected), %.1f visible tok/s, wall %s\n",
			decodeOptimisationLabel(report.SpeculativeDecode.Result.Mode),
			metrics.AcceptanceRate*100,
			metrics.DraftTokens,
			metrics.AcceptedTokens,
			metrics.RejectedTokens,
			metrics.VisibleTokensPerSec,
			metrics.Duration,
		))
		if metrics.TargetTokensPerSec > 0 || metrics.DraftTokensPerSec > 0 || metrics.TargetCalls > 0 || metrics.DraftCalls > 0 {
			core.WriteString(stdout, core.Sprintf("    target: %.1f tok/s across %d calls, draft: %.1f tok/s across %d calls\n",
				metrics.TargetTokensPerSec,
				metrics.TargetCalls,
				metrics.DraftTokensPerSec,
				metrics.DraftCalls,
			))
		}
	}
}

func decodeOptimisationLabel(mode string) string {
	if mode == "" {
		return "speculative"
	}
	return mode
}
