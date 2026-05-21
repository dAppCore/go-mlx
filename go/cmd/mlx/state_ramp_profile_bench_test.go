// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"
	"time"

	mlx "dappco.re/go/mlx"
)

var (
	benchStateRampStringSink  string
	benchStateRampIntSink     int
	benchStateRampSummarySink stateRampProfileSummary
)

const benchStateRampTurnMaterial = `User turn 7:
Review the retained-state benchmark and identify the exact point where
long-context content quality stops matching the runner parity target. Include
the concrete memory metric, decode speed, and next validation step.`

func BenchmarkStateRampProfileTurnPrompt_Gemma4VisibleFloor(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStateRampStringSink = stateRampProfileTurnPrompt("gemma4", benchStateRampTurnMaterial, false, 256)
	}
}

func BenchmarkStateRampProfileVisibleOutput_Gemma4LongThoughtBlock(b *testing.B) {
	output := "Visible preamble.\n<|channel>thought\nhidden scratchpad that must not be retained<channel|>\nVisible final answer.\n<turn|>"

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStateRampStringSink = stateRampProfileVisibleOutput("gemma4", output)
	}
}

func BenchmarkStateRampProfileOutputIssues_FullResponse(b *testing.B) {
	output := "The retained run is not yet production-ready because turn 17 fell below the floor.\n\n" +
		"The next validation step is to fold the State and resume from the compacted summary."

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStateRampIntSink = len(stateRampProfileOutputIssues(output))
	}
}

func BenchmarkStateRampProfileTurnAppendSource_DelimitedSections(b *testing.B) {
	sections := benchStateRampSections(32, 1024)
	opts := stateRampProfileOptions{
		AppendTokens:              4096,
		TargetTokens:              100000,
		CompactionThresholdTokens: 100000,
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, count := stateRampProfileTurnAppendSource(nil, sections, i, 50000, i+1, opts)
		benchStateRampIntSink = count
	}
}

func BenchmarkStateRampProfileTurnAppendSource_FixedWrap(b *testing.B) {
	source := benchStateRampTokenSource(8192)
	opts := stateRampProfileOptions{
		AppendTokens:              4096,
		TargetTokens:              100000,
		CompactionThresholdTokens: 100000,
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, count := stateRampProfileTurnAppendSource(source, nil, 6144+i, 50000, i+1, opts)
		benchStateRampIntSink = count
	}
}

func BenchmarkSummariseStateRampProfileTurns_LongRamp(b *testing.B) {
	turns := make([]stateRampProfileTurn, 100)
	for i := range turns {
		turns[i] = stateRampProfileTurn{
			Index:               i + 1,
			AppendedTokens:      2048,
			TokensAfterAppend:   30000 + ((i + 1) * 2048),
			TokensAfterGenerate: 31024 + ((i + 1) * 2048),
			AppendDuration:      300 * time.Millisecond,
			Duration:            10 * time.Second,
			VisibleTokens:       1024,
			Metrics: mlx.Metrics{
				GeneratedTokens:            1024,
				DecodeDuration:             10 * time.Second,
				PeakMemoryBytes:            uint64(3+i%8) << 30,
				ActiveMemoryBytes:          uint64(2+i%6) << 30,
				CacheMemoryBytes:           uint64(5+i%4) << 30,
				ProcessVirtualMemoryBytes:  uint64(600+i) << 30,
				ProcessResidentMemoryBytes: uint64(3+i%3) << 30,
			},
		}
	}
	opts := stateRampProfileOptions{
		TargetTokens:              100000,
		CompactionThresholdTokens: 100000,
		CompactionTailTokens:      8192,
		TurnMinTokens:             256,
		TurnMinTokensPolicy:       "mark",
		FoldOnDegradation:         true,
		DegradationMinConsecutive: 2,
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStateRampSummarySink = summariseStateRampProfileTurns(30*time.Second, 30000, turns, opts)
	}
}

func benchStateRampTokenSource(count int) []int32 {
	tokens := make([]int32, count)
	for i := range tokens {
		tokens[i] = int32(1000 + (i % 2048))
	}
	return tokens
}

func benchStateRampSections(sectionCount, sectionTokens int) [][]int32 {
	sections := make([][]int32, sectionCount)
	for i := range sections {
		sections[i] = benchStateRampTokenSource(sectionTokens)
	}
	return sections
}
