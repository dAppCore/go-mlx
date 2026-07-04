// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// TestSplitFFNTune_CacheLayers_Good — a comma list parses to ints,
// trimming whitespace around each entry.
func TestSplitFFNTune_CacheLayers_Good(t *testing.T) {
	got, err := cliSplitFFNCacheLayers(" 0, 2 ,4 ")
	if err != nil {
		t.Fatalf("cliSplitFFNCacheLayers: %v", err)
	}
	want := []int{0, 2, 4}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("layer[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestSplitFFNTune_CacheLayers_Bad — a non-numeric entry is a hard error,
// not a silently dropped layer.
func TestSplitFFNTune_CacheLayers_Bad(t *testing.T) {
	if _, err := cliSplitFFNCacheLayers("2,oops,4"); err == nil {
		t.Fatal("expected parse error for non-numeric layer, got nil")
	}
}

// TestSplitFFNTune_CacheLayers_Ugly — empty input and a list of only
// separators/blanks both yield nil (no caches requested) rather than a
// zero-length-but-non-nil slice or an error.
func TestSplitFFNTune_CacheLayers_Ugly(t *testing.T) {
	if got, err := cliSplitFFNCacheLayers("   "); err != nil || got != nil {
		t.Fatalf("blank input = (%v, %v), want (nil, nil)", got, err)
	}
	if got, err := cliSplitFFNCacheLayers(", ,"); err != nil || len(got) != 0 {
		t.Fatalf("separators-only = (%v, %v), want empty no-error", got, err)
	}
	// Negative cache layers are legal (streaming mode) — must parse, not reject.
	got, err := cliSplitFFNCacheLayers("-1")
	if err != nil || len(got) != 1 || got[0] != -1 {
		t.Fatalf("negative layer = (%v, %v), want [-1]", got, err)
	}
}

func ffnEstimate(cache int, peak, resident int64, loads int) cliSplitFFNEstimate {
	return cliSplitFFNEstimate{
		cache: cache,
		report: mlx.CPUSplitFFNMemoryReport{
			PeakResidentBytes: peak,
			ResidentBytes:     resident,
			LayerLoads:        loads,
		},
	}
}

// TestSplitFFNTune_SortEstimates_Good — estimates sort ascending by peak
// resident bytes (lowest memory footprint first = the preferred recipe).
func TestSplitFFNTune_SortEstimates_Good(t *testing.T) {
	est := []cliSplitFFNEstimate{
		ffnEstimate(0, 900, 800, 10),
		ffnEstimate(2, 300, 200, 4),
		ffnEstimate(4, 600, 500, 7),
	}
	cliSortSplitFFNEstimates(est)
	if est[0].report.PeakResidentBytes != 300 || est[2].report.PeakResidentBytes != 900 {
		t.Fatalf("sorted peaks = [%d ... %d], want ascending from 300 to 900",
			est[0].report.PeakResidentBytes, est[2].report.PeakResidentBytes)
	}
}

// TestSplitFFNTune_SortEstimates_Bad — already-sorted and single-element
// inputs are stable no-ops (the insertion sort must not corrupt them).
func TestSplitFFNTune_SortEstimates_Bad(t *testing.T) {
	single := []cliSplitFFNEstimate{ffnEstimate(1, 100, 50, 2)}
	cliSortSplitFFNEstimates(single)
	if single[0].cache != 1 {
		t.Fatalf("single-element sort mutated value: %#v", single[0])
	}
	cliSortSplitFFNEstimates(nil) // must not panic
}

// TestSplitFFNTune_EstimateLess_Ugly — the tie-break ladder: equal peak
// falls to resident, equal resident falls to layer loads, equal loads
// falls to the cache count.
func TestSplitFFNTune_EstimateLess_Ugly(t *testing.T) {
	// Equal peak; a has fewer resident bytes → a < b.
	a := ffnEstimate(8, 500, 100, 9)
	b := ffnEstimate(2, 500, 200, 3)
	if !cliSplitFFNEstimateLess(a, b) {
		t.Fatal("equal peak: lower resident should rank first")
	}
	// Equal peak + resident; a has fewer layer loads → a < b.
	a = ffnEstimate(8, 500, 100, 2)
	b = ffnEstimate(2, 500, 100, 9)
	if !cliSplitFFNEstimateLess(a, b) {
		t.Fatal("equal peak+resident: fewer layer loads should rank first")
	}
	// All equal except cache; lower cache → a < b.
	a = ffnEstimate(1, 500, 100, 5)
	b = ffnEstimate(7, 500, 100, 5)
	if !cliSplitFFNEstimateLess(a, b) {
		t.Fatal("all-equal: lower cache count should rank first")
	}
}

// TestSplitFFNTune_BaseCandidate_Good — an existing candidate for the
// requested workload is reused as the base (its fields seed the derived
// split candidate).
func TestSplitFFNTune_BaseCandidate_Good(t *testing.T) {
	plan := inference.TuningPlan{
		Candidates: []inference.TuningCandidate{
			{ID: "chat:base", Workload: inference.TuningWorkloadChat, ContextLength: 4096},
		},
	}
	base := cliBaseCandidateForWorkload(plan, inference.TuningWorkloadChat)
	if base.ID != "chat:base" || base.ContextLength != 4096 {
		t.Fatalf("base = %#v, want the chat candidate", base)
	}
}

// TestSplitFFNTune_BaseCandidate_Bad — no candidate matches the workload,
// so a fresh candidate carrying the plan's model + runtime is synthesised.
func TestSplitFFNTune_BaseCandidate_Bad(t *testing.T) {
	plan := inference.TuningPlan{
		Model:   inference.ModelIdentity{Path: "/m"},
		Runtime: inference.RuntimeIdentity{},
	}
	base := cliBaseCandidateForWorkload(plan, inference.TuningWorkloadChat)
	if base.Workload != inference.TuningWorkloadChat {
		t.Fatalf("synthesised base workload = %q, want chat", base.Workload)
	}
	if base.Model.Path != "/m" {
		t.Fatalf("synthesised base model = %q, want plan model", base.Model.Path)
	}
}

// TestSplitFFNTune_Labels_Good — the label map carries the split marker,
// rank, and the memory-report numbers the UI surfaces, while preserving
// the base labels it was cloned from.
func TestSplitFFNTune_Labels_Good(t *testing.T) {
	base := map[string]string{"existing": "kept"}
	est := cliSplitFFNEstimate{
		cache: 2,
		report: mlx.CPUSplitFFNMemoryReport{
			TotalLayers:       40,
			LoadedLayers:      40,
			ResidentBytes:     1024,
			PeakResidentBytes: 2048,
		},
	}
	labels := cliSplitFFNLabels(base, est, 1)
	if labels["existing"] != "kept" {
		t.Fatalf("base label dropped: %#v", labels)
	}
	if labels["split"] != "cpu_ffn" || labels["rank"] != "1" {
		t.Fatalf("split/rank labels = %q/%q, want cpu_ffn/1", labels["split"], labels["rank"])
	}
	if labels["cpu_ffn_cache_layers"] != "2" || labels["cpu_ffn_peak_resident_bytes"] != "2048" {
		t.Fatalf("memory labels wrong: %#v", labels)
	}
}

// TestSplitFFNTune_Labels_Ugly — labelling with a nil base map must not
// panic; it produces a fresh map with just the split fields.
func TestSplitFFNTune_Labels_Ugly(t *testing.T) {
	labels := cliSplitFFNLabels(nil, cliSplitFFNEstimate{cache: 0}, 3)
	if labels["estimated"] != "true" || labels["rank"] != "3" {
		t.Fatalf("nil-base labels = %#v, want split fields present", labels)
	}
}

// TestSplitFFNTune_Reason_Good — a positive cache count produces a
// "keeps up to N layers resident" reason plus the peak-resident line.
func TestSplitFFNTune_Reason_Good(t *testing.T) {
	reasons := cliSplitFFNReason(ffnEstimate(3, 4096, 2048, 5))
	if len(reasons) != 2 {
		t.Fatalf("reasons = %v, want 2 lines", reasons)
	}
	if reasons[0] == "" || !strings.Contains(reasons[0], "3 layers resident") {
		t.Fatalf("reason[0] = %q, want resident-cap phrasing", reasons[0])
	}
}

// TestSplitFFNTune_Reason_Ugly — a negative cache means streaming (no
// resident cache); the phrasing must flip accordingly.
func TestSplitFFNTune_Reason_Ugly(t *testing.T) {
	streaming := cliSplitFFNReason(ffnEstimate(-1, 1024, 0, 40))
	if !strings.Contains(streaming[0], "without retaining a resident cache") {
		t.Fatalf("negative-cache reason[0] = %q, want streaming phrasing", streaming[0])
	}
	zero := cliSplitFFNReason(ffnEstimate(0, 1024, 1024, 40))
	if !strings.Contains(zero[0], "caches all layers after first load") {
		t.Fatalf("zero-cache reason[0] = %q, want cache-all phrasing", zero[0])
	}
}

// TestSplitFFNTune_CloneStringLabels_Good — the clone is independent of
// the source map.
func TestSplitFFNTune_CloneStringLabels_Good(t *testing.T) {
	src := map[string]string{"a": "1"}
	clone := cliCloneStringLabels(src)
	clone["a"] = "mutated"
	if src["a"] != "1" {
		t.Fatalf("source mutated through clone: %q", src["a"])
	}
}

// TestSplitFFNTune_CloneStringLabels_Ugly — cloning a nil map yields a
// usable empty map, not nil (callers immediately write into it).
func TestSplitFFNTune_CloneStringLabels_Ugly(t *testing.T) {
	clone := cliCloneStringLabels(nil)
	if clone == nil {
		t.Fatal("clone of nil map is nil; callers write into it without a guard")
	}
	clone["k"] = "v" // must not panic
}
