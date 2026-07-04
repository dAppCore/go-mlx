// SPDX-Licence-Identifier: EUPL-1.2

package specprofile

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/decode"
	mlx "dappco.re/go/mlx"
)

// coverProfilePair is a target/draft stub with independently settable Generate
// and Err failures, so the post-decode guard ladder (Generate error, pair.Err,
// ctx.Err) can each be reached in isolation. The shared fakeProfilePair returns
// one error for both hooks, which cannot separate the pair.Err branch.
type coverProfilePair struct {
	result  mlx.SpeculativeDecodeResult
	metrics mlx.Metrics
	genErr  error
	errErr  error
}

func (pair *coverProfilePair) Generate(_ context.Context, _ string, _ mlx.SpeculativeDecodeConfig) (mlx.SpeculativeDecodeResult, error) {
	return pair.result, pair.genErr
}

func (pair *coverProfilePair) Metrics() mlx.Metrics { return pair.metrics }

func (pair *coverProfilePair) Err() error { return pair.errErr }

// RunPairProfile defaults a nil context and a non-positive MaxTokens rather than
// trusting the caller, so a no-context single-token request still produces a
// usable run. One call exercises both the ctx==nil and the MaxTokens<=0 guards.
func TestRunPairProfile_NilCtxAndMaxTokensFloor_Cover(t *testing.T) {
	pair := &coverProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 11, Text: "Z"}},
			Text:   "Z",
		},
	}

	//nolint:staticcheck // SA1012: passing a nil context is the behaviour under test.
	run, err := RunPairProfile(nil, pair, ProfileConfig{MaxTokens: 0})
	if err != nil {
		t.Fatalf("RunPairProfile(nil ctx) error = %v", err)
	}
	if run.VisibleTokens != 1 {
		t.Fatalf("VisibleTokens = %d, want 1 from the single produced token", run.VisibleTokens)
	}
	if run.Duration <= 0 {
		t.Fatalf("Duration = %v, want a positive elapsed time", run.Duration)
	}
}

// A Generate error is returned verbatim after the run snapshot is captured, so
// callers still see metrics and sampled tokens alongside the failure.
func TestRunPairProfile_GenerateError_Cover(t *testing.T) {
	wantErr := core.NewError("specprofile: generate boom")
	pair := &coverProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 12, Text: "Y"}},
			Text:   "Y",
		},
		genErr: wantErr,
	}

	run, err := RunPairProfile(context.Background(), pair, ProfileConfig{Prompt: "p", MaxTokens: 1})
	if err == nil {
		t.Fatal("RunPairProfile() error = nil, want the Generate failure surfaced")
	}
	if run.VisibleTokens != 1 {
		t.Fatalf("VisibleTokens = %d, want the snapshot captured before the error", run.VisibleTokens)
	}
}

// After a clean Generate, RunPairProfile still consults pair.Err so a deferred
// runtime fault reported by the pair is not swallowed.
func TestRunPairProfile_PairErr_Cover(t *testing.T) {
	pair := &coverProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 13, Text: "X"}},
			Text:   "X",
		},
		errErr: core.NewError("specprofile: pair faulted"),
	}

	if _, err := RunPairProfile(context.Background(), pair, ProfileConfig{Prompt: "p", MaxTokens: 1}); err == nil {
		t.Fatal("RunPairProfile() error = nil, want the pair.Err() fault surfaced")
	}
}

// Generate runs before the context check, so an already-cancelled context with
// a clean pair still reaches the ctx.Err guard and reports cancellation.
func TestRunPairProfile_CtxCancelled_Cover(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pair := &coverProfilePair{
		result: mlx.SpeculativeDecodeResult{
			Tokens: []decode.Token{{ID: 14, Text: "W"}},
			Text:   "W",
		},
	}

	if _, err := RunPairProfile(ctx, pair, ProfileConfig{Prompt: "p", MaxTokens: 1}); err == nil {
		t.Fatal("RunPairProfile() error = nil, want the cancelled-context guard")
	}
}

// sampledTokenIDs returns nil rather than an empty slice when there is nothing
// to sample, covering both the no-tokens and the non-positive-limit guards.
func TestSampledTokenIDs_EmptyAndZeroLimit_Cover(t *testing.T) {
	if got := sampledTokenIDs(nil, 32); got != nil {
		t.Fatalf("sampledTokenIDs(nil) = %v, want nil for an empty token slice", got)
	}
	tokens := []decode.Token{{ID: 1, Text: "a"}}
	if got := sampledTokenIDs(tokens, 0); got != nil {
		t.Fatalf("sampledTokenIDs(limit=0) = %v, want nil for a non-positive limit", got)
	}
}

// nonZeroProfileDuration floors a non-positive elapsed measurement to one
// nanosecond so a sub-tick run never reports a zero duration.
func TestNonZeroProfileDuration_Floor_Cover(t *testing.T) {
	if got := nonZeroProfileDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroProfileDuration(0) = %v, want 1ns floor", got)
	}
	if got := nonZeroProfileDuration(-5); got != time.Nanosecond {
		t.Fatalf("nonZeroProfileDuration(-5) = %v, want 1ns floor for a negative input", got)
	}
	if got := nonZeroProfileDuration(7 * time.Millisecond); got != 7*time.Millisecond {
		t.Fatalf("nonZeroProfileDuration(7ms) = %v, want the positive value passed through", got)
	}
}
