// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/spine"
)

// TestModel_CaptureKVAndRestoreLiveModel drives the Model-level KV capture and
// restore facade on the shared model: CaptureKVWithOptions (the options arm),
// NewSessionFromKV, and NewSessionFromBundle (via a bundle built from the
// snapshot). These bodies past the nil guard need a real native model.
func TestModel_CaptureKVAndRestoreLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)

	const prompt = "The vault combination is seven, four, nine."

	// CaptureKVWithOptions — the explicit-options capture arm.
	snapshot, err := m.CaptureKVWithOptions(prompt, kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot == nil || len(snapshot.Layers) == 0 {
		t.Fatalf("CaptureKVWithOptions returned an empty snapshot")
	}
	t.Logf("captured snapshot: %d layers, %d tokens", len(snapshot.Layers), len(snapshot.Tokens))

	// NewSessionFromKV — restore a session directly from the snapshot.
	sessFromKV, err := m.NewSessionFromKV(snapshot)
	if err != nil {
		t.Fatalf("NewSessionFromKV: %v", err)
	}
	sessFromKV.Close()

	// NewSessionFromBundle — wrap the snapshot in a compatible bundle, then
	// restore. The bundle's Source must match the model for the compat check.
	b, err := bundle.New(snapshot, bundle.Options{
		Source: spine.ModelInfoToBundle(m.Info()),
		Prompt: prompt,
	})
	if err != nil {
		t.Fatalf("bundle.New: %v", err)
	}
	sessFromBundle, err := m.NewSessionFromBundle(b)
	if err != nil {
		t.Fatalf("NewSessionFromBundle: %v", err)
	}
	sessFromBundle.Close()
}

// TestModel_CaptureKVChunksLiveModel drives CaptureKVChunksWithOptions — the
// streaming-chunk capture arm — on the shared model.
func TestModel_CaptureKVChunksLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)
	ctx := context.Background()

	chunks := func(yield func(string) bool) {
		for _, c := range []string{"The keeper is named Wren. ", "The lamp burns amber."} {
			if !yield(c) {
				return
			}
		}
	}
	snapshot, err := m.CaptureKVChunksWithOptions(ctx, chunks, kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVChunksWithOptions: %v", err)
	}
	if snapshot == nil || len(snapshot.Layers) == 0 {
		t.Fatal("CaptureKVChunksWithOptions returned an empty snapshot")
	}
	t.Logf("chunk-captured snapshot: %d layers, %d tokens", len(snapshot.Layers), len(snapshot.Tokens))
}

// TestModel_WarmPromptCacheFromStateBlocksLiveModel drives the prompt-cache warm
// path: a session is prefilled and its KV blocks saved to a store, then the
// model warms its prompt cache from those durable blocks.
func TestModel_WarmPromptCacheFromStateBlocksLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)
	ctx := context.Background()

	src, err := m.NewSession()
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer src.Close()
	if err := src.Prefill("The cartographer hid the map behind the third stone."); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	store := state.NewInMemoryStore(nil)
	bundleRef, err := src.SaveKVBlocksToState(ctx, store, kv.StateBlockOptions{
		URI:   "mlx://test/warmcache",
		Title: "warmcache",
	})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState: %v", err)
	}

	if err := m.WarmPromptCacheFromStateBlocks(ctx, store, bundleRef, bundleRef.TokenCount); err != nil {
		t.Fatalf("WarmPromptCacheFromStateBlocks: %v", err)
	}
}

// TestModel_MergeLoRALiveModel drives MergeLoRA's non-nil arm — applying a LoRA
// on a fresh model load then merging it in place. A fresh load (not the shared
// model) because the merge mutates the weights.
func TestModel_MergeLoRALiveModel(t *testing.T) {
	if !requireLiveModelGate(t) {
		return
	}
	m, err := loadFreshLiveE2B(t)
	if err != nil {
		t.Skipf("fresh load did not survive the reshape flake: %v", err)
	}
	defer m.Close()

	adapter := NewLoRA(m, &LoRAConfig{Rank: 2, Alpha: 4, TargetKeys: []string{"q_proj"}})
	if adapter == nil {
		t.Fatal("NewLoRA returned nil")
	}
	merged := m.MergeLoRA(adapter)
	if merged == nil {
		t.Fatal("MergeLoRA returned nil model")
	}

	// Nil-adapter arm returns the model unchanged.
	if got := m.MergeLoRA(nil); got != m {
		t.Error("MergeLoRA(nil) should return the same model")
	}
}

// requireLiveModelGate runs the Metal + cached-weights gate and returns false
// (after skipping) when the live model is unavailable.
func requireLiveModelGate(t *testing.T) bool {
	t.Helper()
	requireLiveE2BModel(t) // skips if Metal off / uncached / flake
	return true
}

// loadFreshLiveE2B loads a brand-new, independent e2b *Model (separate from the
// shared one) for tests that mutate the weights, retrying the reshape flake.
func loadFreshLiveE2B(t *testing.T) (*Model, error) {
	t.Helper()
	var (
		m   *Model
		err error
	)
	for attempt := 0; attempt < 3; attempt++ {
		m, err = LoadModel(liveModelDir)
		if err == nil {
			return m, nil
		}
		if !core.Contains(err.Error(), reshapeFlakeNeedle) {
			t.Fatalf("LoadModel(%s): %v", liveModelDir, err)
		}
		ClearCache()
	}
	return nil, err
}
