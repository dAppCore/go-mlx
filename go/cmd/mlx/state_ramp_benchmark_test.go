// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"
	"time"

	mlx "dappco.re/go/mlx"
)

var (
	stateRampBenchmarkString string
	stateRampBenchmarkTokens []int32
	stateRampBenchmarkReport stateRampProfileSummary
	stateRampBenchmarkInt    int
)

func benchmarkStateRampMaterial() string {
	return `Review the retained state-ramp-profile implementation against GOAL.md.

Focus on:
- whether append/generate turns keep the model inside the accepted workload;
- whether output-length failures show runner drift rather than only speed;
- whether the report separates raw decode, wall time, memory, and energy;
- whether the next action is runner anchors or long-context degradation work.

Use the retained project context and write a concrete engineering verdict.`
}

func BenchmarkStateRampProfileTurnPrompt_Gemma4WholeTurn(b *testing.B) {
	material := benchmarkStateRampMaterial()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateRampBenchmarkString = stateRampProfileTurnPrompt("gemma4", material, false)
	}
}

func BenchmarkStateRampProfileVisibleOutput_Gemma4ThoughtBlock(b *testing.B) {
	output := "<|channel>thought\nDrafting private notes that should not be retained.<channel|>" +
		"The implementation should keep the folded state compact and continue from it.<turn|>"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateRampBenchmarkString = stateRampProfileVisibleOutput("gemma4", output)
	}
}

func BenchmarkRepeatedStateRampTokens_Append4096Contiguous(b *testing.B) {
	source := make([]int32, 27303)
	for i := range source {
		source[i] = int32(i % 262144)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateRampBenchmarkTokens = repeatedStateRampTokens(source, 4096, 4096)
	}
}

func BenchmarkRepeatedStateRampTokens_Append4096Wrapped(b *testing.B) {
	source := make([]int32, 27303)
	for i := range source {
		source[i] = int32(i % 262144)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateRampBenchmarkTokens = repeatedStateRampTokens(source, len(source)-128, 4096)
	}
}

func BenchmarkForEachRepeatedStateRampTokenSpan_Append4096Wrapped(b *testing.B) {
	source := make([]int32, 27303)
	for i := range source {
		source[i] = int32(i % 262144)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		total := 0
		if _, err := forEachRepeatedStateRampTokenSpan(source, len(source)-128, 4096, func(tokens []int32) error {
			total += len(tokens)
			return nil
		}); err != nil {
			b.Fatalf("forEachRepeatedStateRampTokenSpan: %v", err)
		}
		stateRampBenchmarkInt = total
	}
}

func BenchmarkSummariseStateRampProfileTurns_TenTurns(b *testing.B) {
	turns := make([]stateRampProfileTurn, 10)
	for i := range turns {
		turns[i] = stateRampProfileTurn{
			Index:               i + 1,
			TokensBeforeAppend:  30000 + i*3000,
			AppendedTokens:      2730,
			TokensAfterAppend:   32730 + i*3000,
			TokensAfterGenerate: 33500 + i*3000,
			TurnCloseTokens:     2,
			AppendDuration:      1500 * time.Millisecond,
			Duration:            11 * time.Second,
			VisibleTokens:       625,
			Metrics: mlx.Metrics{
				GeneratedTokens:            625,
				DecodeDuration:             8 * time.Second,
				PeakMemoryBytes:            3600 << 20,
				ActiveMemoryBytes:          3200 << 20,
				CacheMemoryBytes:           6200 << 20,
				ProcessVirtualMemoryBytes:  590 << 30,
				ProcessResidentMemoryBytes: 3300 << 20,
				ProcessPeakResidentBytes:   3300 << 20,
			},
		}
	}
	opts := stateRampProfileOptions{
		TargetTokens:              70000,
		CompactionThresholdTokens: 70000,
		CompactionTailTokens:      8192,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateRampBenchmarkReport = summariseStateRampProfileTurns(11*time.Second, 30000, turns, opts)
	}
}
