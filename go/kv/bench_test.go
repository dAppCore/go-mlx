// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"

	"dappco.re/go/mlx/memory"
)

// TestBench_CompareModes_Good exercises CompareModes across its normal,
// all-default and small-shape inputs. Each sub-case asserts a distinct property
// of the returned BenchReport: ranking order + recommendation for a 32GB-class
// context, the zero-field default fill in normalizeBenchConfig, and the FP16
// recommendation for a sub-2GiB shape.
func TestBench_CompareModes_Good(t *testing.T) {
	t.Run("RanksMemoryAndUseCase", func(t *testing.T) {
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
	})

	t.Run("AllDefaults", func(t *testing.T) {
		// An all-zero BenchConfig exercises normalizeBenchConfig's zero-field
		// default branches (ContextLength, NumLayers, HiddenSize, Modes) in one
		// pass. The resulting default shape (ctx 131072, 32 layers, hidden 3072,
		// dtype 2) yields fp16 storage above the 20 GiB threshold, so it also
		// drives the recommendMode KQ8VQ4 branch.
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
	})

	t.Run("RecommendModeSmallShape", func(t *testing.T) {
		// A tiny model shape keeps fp16 storage under 2 GiB so the recommendMode
		// FP16 default arm is taken and no quantised mode is suggested.
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
	})
}

// TestBench_CompareModes_Bad feeds CompareModes a hostile shape: a mid-size
// context paired with an unrecognised KV cache mode. CompareModes must not panic
// on the unknown mode — modeStorageBytes/modeBits/modeWinsWhen all fall through
// to their default arms — and the report must still rank the recognised FP16
// baseline correctly while the unknown mode lands on a fp16-equivalent estimate.
func TestBench_CompareModes_Bad(t *testing.T) {
	const unknownMode memory.KVCacheMode = "totally-not-a-real-mode"
	report := CompareModes(BenchConfig{
		ContextLength: 16384,
		NumLayers:     16,
		HiddenSize:    1024,
		DTypeBytes:    2,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, unknownMode},
	})

	if len(report.Modes) != 2 {
		t.Fatalf("modes len = %d, want 2 (fp16 + unknown)", len(report.Modes))
	}
	fp16 := report.ByMode(memory.KVCacheModeFP16)
	unknown := report.ByMode(unknownMode)
	if fp16.StorageBytes == 0 {
		t.Fatalf("fp16 storage not populated: %+v", report.Modes)
	}
	// The unknown mode hits every default arm: bits = dtypeBytes*8, storage =
	// elements*dtypeBytes, penalty 0, wins_when the quality/speed string. That
	// makes it fp16-equivalent in storage and bit-width.
	if unknown.StorageBytes != fp16.StorageBytes {
		t.Fatalf("unknown mode storage = %d, want fp16-equivalent %d", unknown.StorageBytes, fp16.StorageBytes)
	}
	if unknown.KeyBits != 16 || unknown.ValueBits != 16 {
		t.Fatalf("unknown mode bits = %d/%d, want 16/16 (dtype default arm)", unknown.KeyBits, unknown.ValueBits)
	}
	if unknown.EstimatedDecodePenalty != 0 {
		t.Fatalf("unknown mode penalty = %v, want 0 (default arm)", unknown.EstimatedDecodePenalty)
	}
}

// TestBench_CompareModes_Ugly drives CompareModes with a degenerate config:
// every shape field negative or zero AND an explicitly empty Modes slice. The
// negatives must be clamped to defaults by normalizeBenchConfig and the empty
// Modes slice replaced by the four-mode default set, so even nonsense input
// yields a fully populated four-row report.
func TestBench_CompareModes_Ugly(t *testing.T) {
	report := CompareModes(BenchConfig{
		ContextLength: -100,
		NumLayers:     -7,
		HiddenSize:    -1,
		DTypeBytes:    -4,
		Modes:         []memory.KVCacheMode{},
	})

	if report.Config.ContextLength != defaultBenchContextLength {
		t.Fatalf("Config.ContextLength = %d, want negative clamped to default %d", report.Config.ContextLength, defaultBenchContextLength)
	}
	if report.Config.NumLayers != 32 || report.Config.HiddenSize != 3072 || report.Config.DTypeBytes != 2 {
		t.Fatalf("clamped defaults = layers %d hidden %d dtype %d, want 32/3072/2", report.Config.NumLayers, report.Config.HiddenSize, report.Config.DTypeBytes)
	}
	if len(report.Modes) != 4 {
		t.Fatalf("modes len = %d, want 4 default modes for empty input slice", len(report.Modes))
	}
	for _, bench := range report.Modes {
		if bench.StorageBytes == 0 {
			t.Fatalf("mode %q storage not populated under degenerate input: %+v", bench.Mode, report.Modes)
		}
	}
}

// TestBench_BenchReport_ByMode_Good asserts BenchReport.ByMode returns the
// populated comparison row for a mode that is present in the report.
func TestBench_BenchReport_ByMode_Good(t *testing.T) {
	report := CompareModes(BenchConfig{
		ContextLength: 8192,
		NumLayers:     8,
		HiddenSize:    1024,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModeQ8},
	})

	q8 := report.ByMode(memory.KVCacheModeQ8)
	if q8.Mode != memory.KVCacheModeQ8 {
		t.Fatalf("ByMode(q8).Mode = %q, want q8", q8.Mode)
	}
	if q8.StorageBytes == 0 {
		t.Fatalf("ByMode(q8) = %+v, want populated storage", q8)
	}
	if q8.KeyBits != 8 || q8.ValueBits != 8 {
		t.Fatalf("ByMode(q8) bits = %d/%d, want 8/8", q8.KeyBits, q8.ValueBits)
	}
}

// TestBench_BenchReport_ByMode_Bad queries BenchReport.ByMode on a report that
// holds exactly one mode for a different mode. The lookup misses every row and
// must return the zero ModeBench fallthrough rather than a partially-populated
// row from the wrong mode.
func TestBench_BenchReport_ByMode_Bad(t *testing.T) {
	report := CompareModes(BenchConfig{
		ContextLength: 4096,
		NumLayers:     4,
		HiddenSize:    512,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16},
	})

	missing := report.ByMode(memory.KVCacheModeQ8)
	if missing != (ModeBench{}) {
		t.Fatalf("ByMode(absent) = %+v, want zero ModeBench", missing)
	}
}

// TestBench_BenchReport_ByMode_Ugly covers the not-found branch of
// BenchReport.ByMode (the `return ModeBench{}` fallthrough). A report built for a
// single mode is queried for a mode it does not contain, so ByMode must return
// a zero ModeBench rather than a populated row.
func TestBench_BenchReport_ByMode_Ugly(t *testing.T) {
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
