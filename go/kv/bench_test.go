// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"

	"dappco.re/go/mlx/memory"
)

func TestBench_CompareModesRanksMemoryAndUseCase_Good(t *testing.T) {
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

// TestBench_ByMode_Ugly covers the not-found branch of BenchReport.ByMode
// (bench.go:73 — the `return ModeBench{}` fallthrough). A report built for a
// single mode is queried for a mode it does not contain, so ByMode must return
// a zero ModeBench rather than a populated row.
func TestBench_ByMode_Ugly(t *testing.T) {
	report := CompareModes(BenchConfig{
		ContextLength: 4096,
		NumLayers:     4,
		HiddenSize:    512,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16},
	})

	missing := report.ByMode(memory.KVCacheModeKQ8VQ4)
	if missing != (ModeBench{}) {
		t.Fatalf("ByMode(missing) = %+v, want zero ModeBench", missing)
	}
	// Sanity: a mode that IS present returns a populated row, so the zero
	// above is genuinely the not-found path and not an empty report.
	if present := report.ByMode(memory.KVCacheModeFP16); present.StorageBytes == 0 {
		t.Fatalf("ByMode(present) = %+v, want populated row", present)
	}
}

// TestBench_CompareModesAllDefaults_Good covers normalizeBenchConfig's
// zero-field default branches (bench.go:77,80,83,89 — ContextLength, NumLayers,
// HiddenSize and Modes) in one pass by passing an all-zero BenchConfig. The
// resulting default shape (ctx 131072, 32 layers, hidden 3072, dtype 2) yields
// fp16 storage well above the 20 GiB threshold, so it also exercises the
// recommendMode KQ8VQ4 branch (bench.go:166).
func TestBench_CompareModesAllDefaults_Good(t *testing.T) {
	report := CompareModes(BenchConfig{})

	if report.Config.ContextLength != defaultBenchContextLength {
		t.Fatalf("Config.ContextLength = %d, want default %d", report.Config.ContextLength, defaultBenchContextLength)
	}
	if report.Config.NumLayers != 32 || report.Config.HiddenSize != 3072 || report.Config.DTypeBytes != 2 {
		t.Fatalf("Config defaults = layers %d hidden %d dtype %d, want 32/3072/2", report.Config.NumLayers, report.Config.HiddenSize, report.Config.DTypeBytes)
	}
	if len(report.Config.Modes) != 4 || len(report.Modes) != 4 {
		t.Fatalf("Config.Modes = %d, Modes = %d, want 4 default modes each", len(report.Config.Modes), len(report.Modes))
	}
	// NumLayers/HiddenSize are non-zero after normalize, so the shape-fallback
	// note must NOT be present here.
	for _, note := range report.Notes {
		if note == "using shape fallback; pass model metadata for sharper cache estimates" {
			t.Fatalf("Notes = %v, want no shape-fallback note for a full default shape", report.Notes)
		}
	}
	if report.RecommendedMode != memory.KVCacheModeKQ8VQ4 {
		t.Fatalf("RecommendedMode = %q, want kq8vq4 for >=20GiB default-context shape", report.RecommendedMode)
	}
}

// TestBench_RecommendModeSmallShape_Good covers the recommendMode FP16 branch
// (bench.go:170 — the `default` arm below the 2 GiB threshold). A tiny model
// shape keeps fp16 storage under 2 GiB so FP16 is recommended and no quantised
// mode is suggested.
func TestBench_RecommendModeSmallShape_Good(t *testing.T) {
	report := CompareModes(BenchConfig{
		ContextLength: 4096,
		NumLayers:     4,
		HiddenSize:    512,
		DTypeBytes:    2,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModeQ8},
	})

	if report.RecommendedMode != memory.KVCacheModeFP16 {
		t.Fatalf("RecommendedMode = %q, want fp16 for sub-2GiB shape", report.RecommendedMode)
	}
}
