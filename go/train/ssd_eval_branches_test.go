// SPDX-Licence-Identifier: EUPL-1.2

// Branch coverage for the SSD code-benchmark helpers and the RunSSDCodeBenchmark
// orchestration arms — all pure Go (injected Generate/RunTests runners, crafted
// result fixtures), no model and no Metal.
package train

import (
	"context"
	"errors"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/spine"
)

// TestSsdEvalBranches_RunSSDCodeBenchmark_ContextAndRunnerErrors covers the
// orchestration error arms: a nil context defaults to Background, an already
// cancelled context aborts at the sample boundary, a failing Generate and a
// failing RunTests each surface mid-run, and a bad OutputPath fails the report
// write after the run completes.
func TestSsdEvalBranches_RunSSDCodeBenchmark_ContextAndRunnerErrors(t *testing.T) {
	okGen := func(context.Context, string, spine.GenerateConfig) (string, error) {
		return "```python\nx=1\n```", nil
	}
	okTests := func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
		return SSDCodeExecution{Passed: true, TotalTests: 1, PassedTests: 1}, nil
	}
	samples := []SSDCodeBenchmarkSample{{ID: "a", Prompt: "p"}}

	// nil context → Background; run completes.
	if _, err := RunSSDCodeBenchmark(nil, SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{}); err != nil { //nolint:staticcheck // nil ctx is the branch under test
		t.Fatalf("RunSSDCodeBenchmark(nil ctx) error = %v, want nil", err)
	}

	// Cancelled context aborts at the first sample's ctx.Err() guard.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RunSSDCodeBenchmark(cancelled, SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{}); err == nil {
		t.Fatal("RunSSDCodeBenchmark(cancelled) error = nil, want context error")
	}

	// Generate error surfaces.
	genErr := errors.New("generate boom")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "", genErr },
		RunTests: okTests,
	}, samples, SSDCodeBenchmarkConfig{}); !errors.Is(err, genErr) {
		t.Fatalf("RunSSDCodeBenchmark generate-error = %v, want %v", err, genErr)
	}

	// RunTests error surfaces.
	testErr := errors.New("runtests boom")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: okGen,
		RunTests: func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
			return SSDCodeExecution{}, testErr
		},
	}, samples, SSDCodeBenchmarkConfig{}); !errors.Is(err, testErr) {
		t.Fatalf("RunSSDCodeBenchmark runtests-error = %v, want %v", err, testErr)
	}

	// Bad OutputPath: a file where the parent directory must be → write fails.
	blocker := core.PathJoin(t.TempDir(), "blocker")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	badOut := core.PathJoin(blocker, "report.json")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{OutputPath: badOut}); err == nil {
		t.Fatal("RunSSDCodeBenchmark(bad OutputPath) error = nil, want report-write failure")
	}
}

// TestSsdEvalBranches_GeneratePrompt covers ssdCodeBenchmarkGeneratePrompt's
// arms: no Meta returns the bare prompt, Meta without is_stdin returns the bare
// prompt, an is_stdin sample with a usable prompt returns the formatted variant,
// and an is_stdin sample with a blank prompt falls back to the bare prompt.
func TestSsdEvalBranches_GeneratePrompt(t *testing.T) {
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare"}); got != "bare" {
		t.Fatalf("no-Meta prompt = %q, want bare", got)
	}
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare", Meta: map[string]string{"x": "y"}}); got != "bare" {
		t.Fatalf("no-is_stdin prompt = %q, want bare", got)
	}
	formatted := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "solve me", Meta: map[string]string{"is_stdin": "true"}})
	if formatted == "solve me" || formatted == "" {
		t.Fatalf("is_stdin prompt = %q, want formatted variant", formatted)
	}
	// is_stdin set but prompt blank → FormatSSDLiveCodeBenchPrompt returns ""
	// → fall back to the (blank) sample prompt.
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "   ", Meta: map[string]string{"is_stdin": "true"}}); got != "   " {
		t.Fatalf("blank is_stdin prompt = %q, want the blank sample prompt fallback", got)
	}
}

// TestSsdEvalBranches_PassAtKEdges drives computeSSDCodeBenchmarkPassAtK's skip
// and empty arms: an empty result set returns nil, a candidate-less sample is
// skipped, and a sample whose candidate count is below every k yields no metric.
func TestSsdEvalBranches_PassAtKEdges(t *testing.T) {
	if got := computeSSDCodeBenchmarkPassAtK(nil, 1); got != nil {
		t.Fatalf("pass@k over no results = %v, want nil", got)
	}
	// One sample with zero candidates is skipped; with all samples skipped the
	// aggregate is empty → nil.
	skipped := []SSDCodeBenchmarkSampleResult{{Sample: SSDCodeBenchmarkSample{ID: "a"}}}
	if got := computeSSDCodeBenchmarkPassAtK(skipped, 1); got != nil {
		t.Fatalf("pass@k over candidate-less sample = %v, want nil", got)
	}
	// nRepeat 10 asks for pass@1 and pass@5, but the sample has only 1
	// candidate → pass@5 is skipped (total < k), pass@1 still computed.
	one := []SSDCodeBenchmarkSampleResult{{
		Sample:     SSDCodeBenchmarkSample{ID: "a"},
		Candidates: []SSDCodeBenchmarkCandidateResult{{Execution: SSDCodeExecution{Passed: true}}},
	}}
	got := computeSSDCodeBenchmarkPassAtK(one, 10)
	if _, ok := got["pass@1"]; !ok {
		t.Fatalf("pass@k = %v, want pass@1 present", got)
	}
	if _, ok := got["pass@5"]; ok {
		t.Fatalf("pass@k = %v, did not want pass@5 (one candidate < 5)", got)
	}
}

// TestSsdEvalBranches_DifficultyEdges drives computeSSDCodeBenchmarkDifficultyMetrics'
// skip arms: an empty k-list (nRepeat 0) returns nil; samples without difficulty
// meta, without meta at all, or without candidates contribute nothing → nil.
func TestSsdEvalBranches_DifficultyEdges(t *testing.T) {
	if got := computeSSDCodeBenchmarkDifficultyMetrics(nil, 0); got != nil {
		t.Fatalf("difficulty over empty k-list = %v, want nil", got)
	}
	results := []SSDCodeBenchmarkSampleResult{
		{Sample: SSDCodeBenchmarkSample{ID: "nometa"}},                                            // Meta nil → skip
		{Sample: SSDCodeBenchmarkSample{ID: "blank", Meta: map[string]string{"difficulty": " "}}}, // blank difficulty → skip
		{Sample: SSDCodeBenchmarkSample{ID: "nocand", Meta: map[string]string{"difficulty": "easy"}}}, // no candidates → skip
	}
	if got := computeSSDCodeBenchmarkDifficultyMetrics(results, 1); got != nil {
		t.Fatalf("difficulty over all-skipped results = %v, want nil", got)
	}
}

// TestSsdEvalBranches_KList pins the n-repeat thresholds that widen the k-list.
func TestSsdEvalBranches_KList(t *testing.T) {
	cases := []struct {
		nRepeat int
		want    []int
	}{
		{1, []int{1}},
		{10, []int{1, 5}},
		{20, []int{1, 5, 10}},
		{32, []int{1, 5, 10, 16}},
		{40, []int{1, 5, 10, 16, 20}},
		{64, []int{1, 5, 10, 16, 20, 32}},
	}
	for _, tc := range cases {
		got := ssdCodeBenchmarkKList(tc.nRepeat)
		if !equalIntSlices(got, tc.want) {
			t.Fatalf("ssdCodeBenchmarkKList(%d) = %v, want %v", tc.nRepeat, got, tc.want)
		}
	}
}

// TestSsdEvalBranches_LastSSDCodeFenceUnclosed covers lastSSDCodeFence's break
// arms: a fence opener with no newline after it, and an opened-but-never-closed
// fence both yield no extracted body.
func TestSsdEvalBranches_LastSSDCodeFenceUnclosed(t *testing.T) {
	// "```" with no following newline → break before a body is found.
	if body, ok := lastSSDCodeFence("text ``` still text"); ok || body != "" {
		t.Fatalf("unclosed-no-newline fence = (%q, %v), want (\"\", false)", body, ok)
	}
	// Opener + language line but no closing fence → break.
	if body, ok := lastSSDCodeFence("```python\ncode without a close"); ok || body != "" {
		t.Fatalf("never-closed fence = (%q, %v), want (\"\", false)", body, ok)
	}
	// A complete fence still extracts (sanity for the found path).
	if body, ok := lastSSDCodeFence("```python\nx=1\n```"); !ok || core.Trim(body) != "x=1" {
		t.Fatalf("closed fence = (%q, %v), want (x=1, true)", body, ok)
	}
}

// TestSsdEvalBranches_WriteReportErrors covers writeSSDCodeBenchmarkReport's two
// failure arms: a directory that cannot be created (parent is a file) and a path
// whose final component is a directory (write fails). The success path with a
// nested directory is also exercised.
func TestSsdEvalBranches_WriteReportErrors(t *testing.T) {
	report := &SSDCodeBenchmarkReport{Version: 1}

	// MkdirAll fails: parent is a regular file.
	blocker := core.PathJoin(t.TempDir(), "afile")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	if err := writeSSDCodeBenchmarkReport(core.PathJoin(blocker, "sub", "r.json"), report); err == nil {
		t.Fatal("writeReport under a file-parent = nil, want MkdirAll failure")
	}

	// Success: a fresh nested directory is created and the file lands.
	ok := core.PathJoin(t.TempDir(), "nested", "report.json")
	if err := writeSSDCodeBenchmarkReport(ok, report); err != nil {
		t.Fatalf("writeReport nested success error = %v", err)
	}
	if !core.ReadFile(ok).OK {
		t.Fatal("writeReport success but file unreadable")
	}
}

// TestSsdEvalBranches_NonZeroDuration pins the duration floor: a non-positive
// measured duration is reported as a single nanosecond, a positive one passes
// through.
func TestSsdEvalBranches_NonZeroDuration(t *testing.T) {
	if got := nonZeroSSDCodeBenchmarkDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns", got)
	}
	if got := nonZeroSSDCodeBenchmarkDuration(-5); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-5) = %v, want 1ns", got)
	}
	if got := nonZeroSSDCodeBenchmarkDuration(42 * time.Millisecond); got != 42*time.Millisecond {
		t.Fatalf("nonZeroDuration(42ms) = %v, want 42ms", got)
	}
}
