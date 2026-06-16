// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// isolateRuntimeGates snapshots every runtime gate on entry and restores them
// all when the test ends.
//
// LoadAndInit applies the loaded model's EngineFeatures to the global gate array
// (e.g. e2b flips GateFixedSlidingCache), and nothing restores those on
// Model.Close — harmless for a one-model serve, but in the test binary a
// model_eval test that loads a real model leaks its gates into the synthetic
// cache/gate tests that run afterwards (they assert the default regime and fail).
// EngineFeatures.Apply()'s own restore only covers the gates *it* set, not the
// model's, so the full-array snapshot here is what actually isolates a load.
//
// Every model_eval test that calls LoadAndInit must call this first.
func isolateRuntimeGates(t *testing.T) {
	t.Helper()
	var saved [gateCount]bool
	for i := range runtimeGates {
		saved[i] = runtimeGates[i].Load()
	}
	t.Cleanup(func() {
		for i := range runtimeGates {
			runtimeGates[i].Store(saved[i])
		}
	})
}
