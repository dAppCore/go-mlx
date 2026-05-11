// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"

	"dappco.re/go/mlx/memory"
)

func TestBench_CompareModesRanksMemoryAndUseCase_Good(t *testing.T) {
	coverageTokens := "CompareModesRanksMemoryAndUseCase"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}

	report := CompareModes(BenchConfig{
		ContextLength: 32768,
		NumLayers:     32,
		HiddenSize:    3072,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged},
	})

	if len(report.Modes) != 4 {
		t.Fatalf("modes len = %d, want 4", len(report.Modes))
	}
	fp16 := report.ByMode(memory.KVCacheModeFP16)
	q8 := report.ByMode(memory.KVCacheModeQ8)
	asym := report.ByMode(memory.KVCacheModeKQ8VQ4)
	paged := report.ByMode(memory.KVCacheModePaged)
	if fp16.StorageBytes == 0 || q8.StorageBytes == 0 || asym.StorageBytes == 0 || paged.StorageBytes == 0 {
		t.Fatalf("storage bytes not populated: %+v", report.Modes)
	}
	if !(asym.StorageBytes < q8.StorageBytes && q8.StorageBytes < fp16.StorageBytes) {
		t.Fatalf("storage order = fp16 %d q8 %d asym %d, want asym < q8 < fp16", fp16.StorageBytes, q8.StorageBytes, asym.StorageBytes)
	}
	if q8.WinsWhen == "" || asym.WinsWhen == "" || paged.WinsWhen == "" {
		t.Fatalf("wins_when missing: %+v", report.Modes)
	}
	if report.RecommendedMode != memory.KVCacheModeQ8 {
		t.Fatalf("RecommendedMode = %q, want q8 for 32GB-class context", report.RecommendedMode)
	}
}
