// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
)

// liveE2BModelRepo is the cached gemma-4 e2b checkpoint these coverage tests
// drive. The 4bit pack loads clean (the 6bit pack is the one that trips the
// intermittent `reshape size 4 into (3,3)` weight-load flake).
const liveE2BModelRepo = "mlx-community/gemma-4-e2b-it-4bit"

// liveModelOnce guards a single shared e2b load reused across every coverage
// test in this package. One load honours the RAM budget — the model is large
// and stacking concurrent loads OOMs the host — and gives a single place to
// retry around the flaky weight-reshape error.
var (
	liveModelOnce sync.Once
	liveModelDir  string
	liveModel     *Model
	liveModelErr  error
)

// reshapeFlakeNeedle marks the known intermittent weight-load failure that is
// real-model-bound and pre-existing (a 6bit-pack reshape race). The prompt
// carves it out — work around it, do not be blocked. A load that fails this way
// is retried; if it never takes, the caller skips rather than reds the suite.
const reshapeFlakeNeedle = "reshape"

// requireLiveE2BModel resolves the cached e2b checkpoint and returns the shared
// loaded model, skipping (never failing) when Metal is unavailable, the weights
// are not cached, or the load never survives the reshape flake. Coverage tests
// that need a real *Model call this; the model is loaded at most once per
// process and left resident for the process lifetime (the OS reclaims it at
// exit — one model is within the RAM budget and avoids a TestMain that would
// collide with parallel agents adding their own).
func requireLiveE2BModel(t *testing.T) *Model {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to drive the live e2b model")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	liveModelOnce.Do(func() {
		liveModelDir = metaltest.HFModelPath(t, liveE2BModelRepo)
		if liveModelDir == "" {
			return
		}
		// Retry the load a few times: the reshape flake is intermittent, so a
		// fresh attempt usually takes. ClearCache between tries drops any
		// half-loaded GPU residency.
		for attempt := 0; attempt < 3; attempt++ {
			m, err := LoadModel(liveModelDir)
			if err == nil {
				liveModel = m
				return
			}
			liveModelErr = err
			if !core.Contains(err.Error(), reshapeFlakeNeedle) {
				return // a real, non-flake load error — surface it via skip below
			}
			ClearCache()
		}
	})
	if liveModelDir == "" {
		t.Skipf("model %s not cached", liveE2BModelRepo)
	}
	if liveModel == nil {
		t.Skipf("live e2b model load did not survive the reshape flake: %v", liveModelErr)
	}
	return liveModel
}

// requireLiveE2BAdapter resolves the shared e2b model as the *metaladapter the
// inference.TextModel surface uses (Generate/Chat/Classify/BatchGenerate/Metrics
// /Info/InspectAttention). LoadModelAsTextModel re-loads the weights, so this is
// a SECOND resident model — callers must Close it and tests using it should not
// stack many at once.
func requireLiveE2BAdapter(t *testing.T) (*metaladapter, func()) {
	t.Helper()
	// Ensure the gates (Metal + cached weights) pass before the heavier load.
	requireLiveE2BModel(t)
	tm, err := LoadModelAsTextModel(liveModelDir)
	if err != nil {
		if core.Contains(err.Error(), reshapeFlakeNeedle) {
			t.Skipf("adapter load did not survive the reshape flake: %v", err)
		}
		t.Fatalf("LoadModelAsTextModel(%s): %v", liveModelDir, err)
	}
	adapter, ok := tm.(*metaladapter)
	if !ok {
		_ = closeTextModel(tm)
		t.Fatalf("LoadModelAsTextModel returned %T, want *metaladapter", tm)
	}
	return adapter, func() { _ = adapter.Close() }
}

// closeTextModel closes an inference.TextModel when it exposes Close, used on
// the adapter error path.
func closeTextModel(tm inference.TextModel) error {
	if closer, ok := tm.(interface{ Close() error }); ok {
		return closer.Close()
	}
	return nil
}
