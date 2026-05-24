// SPDX-Licence-Identifier: EUPL-1.2

package main

import "testing"

var stateWakeBenchDelta *stateWakeMemoryDelta
var stateWakeBenchSample stateWakeMemorySample

func BenchmarkStateWakeMemoryDeltaBetween_ProfilePhases(b *testing.B) {
	before := stateWakeMemorySample{
		goHeapAllocBytes:     4096,
		goHeapObjects:        30,
		goTotalAllocBytes:    8192,
		goMallocs:            100,
		goFrees:              40,
		activeMemoryBytes:    20_000,
		cacheMemoryBytes:     4_000,
		peakMemoryBytes:      50_000,
		processVirtualBytes:  100_000,
		processResidentBytes: 20_000,
		processPeakResident:  25_000,
	}
	after := stateWakeMemorySample{
		goHeapAllocBytes:     2048,
		goHeapObjects:        25,
		goTotalAllocBytes:    12288,
		goMallocs:            112,
		goFrees:              47,
		activeMemoryBytes:    24_000,
		cacheMemoryBytes:     2_000,
		peakMemoryBytes:      55_000,
		processVirtualBytes:  98_000,
		processResidentBytes: 21_024,
		processPeakResident:  27_000,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateWakeBenchDelta = stateWakeMemoryDeltaBetween(before, after)
	}
}

func BenchmarkStateWakeMemoryNow_ProfilePhaseSample(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateWakeBenchSample = stateWakeMemoryNow()
	}
}
