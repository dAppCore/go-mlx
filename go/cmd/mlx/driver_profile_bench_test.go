// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	mlx "dappco.re/go/mlx"
)

var benchDriverProfileIntSink int

func BenchmarkApplyGemma4FastLaneDefaults_DefaultDriverProfile(b *testing.B) {
	visited := map[string]bool{}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		contextLen := 0
		cacheMode := ""
		prefillChunkSize := 0
		promptChunkBytes := 0
		restores := applyGemma4FastLaneDefaults(visited, &contextLen, &cacheMode, &prefillChunkSize, &promptChunkBytes, mlx.ProductionLaneContextLength)
		benchDriverProfileIntSink += len(restores) + contextLen + len(cacheMode) + prefillChunkSize + promptChunkBytes
		for j := len(restores) - 1; j >= 0; j-- {
			restores[j]()
		}
	}
}

func BenchmarkApplyGemma4FastLaneDefaults_HyperLongDriverProfile(b *testing.B) {
	visited := map[string]bool{}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		contextLen := 0
		cacheMode := ""
		prefillChunkSize := 0
		promptChunkBytes := 0
		restores := applyGemma4FastLaneDefaults(visited, &contextLen, &cacheMode, &prefillChunkSize, &promptChunkBytes, mlx.ProductionLaneHyperLongContextLength)
		benchDriverProfileIntSink += len(restores) + contextLen + len(cacheMode) + prefillChunkSize + promptChunkBytes
		for j := len(restores) - 1; j >= 0; j-- {
			restores[j]()
		}
	}
}
