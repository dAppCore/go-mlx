// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

func TestArchSessionPrefillAppendGenerateFromCache(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	full := append(append([]int32{}, prefix...), suffix...)

	retained := newSessionStateFixture(t)
	if err := retained.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if retained.Pos() != len(prefix) {
		t.Fatalf("Pos after PrefillTokens = %d, want %d", retained.Pos(), len(prefix))
	}
	if !idsEqual(retained.cachedIDs, prefix) {
		t.Fatalf("cached ids after PrefillTokens = %v, want %v", retained.cachedIDs, prefix)
	}
	if err := retained.AppendTokens(suffix); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}
	if retained.Pos() != len(full) {
		t.Fatalf("Pos after AppendTokens = %d, want %d", retained.Pos(), len(full))
	}
	if !idsEqual(retained.cachedIDs, full) {
		t.Fatalf("cached ids after AppendTokens = %v, want %v", retained.cachedIDs, full)
	}

	got, err := retained.GenerateFromCache(4, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(full, 4, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache = %v, want cold retained-state continuation %v", got, want)
	}
	if retained.Pos() != len(full)+len(got) {
		t.Fatalf("Pos after GenerateFromCache = %d, want %d", retained.Pos(), len(full)+len(got))
	}
	if !idsEqual(retained.cachedIDs, append(append([]int32{}, full...), got...)) {
		t.Fatalf("cached ids after GenerateFromCache = %v, want full prompt plus generated %v", retained.cachedIDs, got)
	}
}

func TestArchSessionPrefillTokensResetsRetainedState(t *testing.T) {
	requireNativeRuntime(t)

	retained := newSessionStateFixture(t)
	if _, err := retained.Generate([]int32{9, 8, 7}, 2, -1); err != nil {
		t.Fatalf("seed Generate: %v", err)
	}
	prompt := []int32{1, 2, 3, 4}
	if err := retained.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens reset: %v", err)
	}
	got, err := retained.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after reset: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache after reset = %v, want cold prompt continuation %v", got, want)
	}
}

func TestArchSessionRestoreStatePreservesGenerateFromCacheBoundary(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreState: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestArchSessionGenerateRecordsResidentIDs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3}

	sess := newSessionStateFixture(t)
	got, err := sess.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	wantResident := append(append([]int32(nil), prompt...), got...)
	if sess.Pos() != len(wantResident) {
		t.Fatalf("Pos after generate = %d, want %d", sess.Pos(), len(wantResident))
	}
	if !idsEqual(sess.cachedIDs, wantResident) {
		t.Fatalf("cached ids after generate = %v, want prompt plus generated %v", sess.cachedIDs, wantResident)
	}
}

func TestArchSessionGenerateSampledEachRecordsResidentIDs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}

	sess := newSessionStateFixture(t)
	got, err := sess.GenerateSampledEach(prompt, 3, nil, model.NewSampler(17), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	wantResident := append(append([]int32(nil), prompt...), got...)
	if sess.Pos() != len(wantResident) {
		t.Fatalf("Pos after sampled generate = %d, want %d", sess.Pos(), len(wantResident))
	}
	if !idsEqual(sess.cachedIDs, wantResident) {
		t.Fatalf("cached ids after sampled generate = %v, want prompt plus generated %v", sess.cachedIDs, wantResident)
	}
}
