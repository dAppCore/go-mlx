// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestSession_SnapshotRestoreFork_Eval drives the stateful session lifecycle on a
// real model: prefill, generate, multi-turn append, CaptureKV → snapshot, range
// the snapshot's blocks, fork the session, restore the snapshot into a fresh
// session and resume, and feed the snapshot through the prompt-cache restore
// path. The synthetic suite never captures/restores real KV state, so this whole
// cluster (CaptureKV, RestoreKV, Fork, RangeKVBlocks + the snapshot
// shape-inference/validation helpers, and prompt_cache's RestorePromptCacheFromKV)
// is otherwise dark. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestSession_SnapshotRestoreFork_Eval ./pkg/metal/
func TestSession_SnapshotRestoreFork_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
	isolateRuntimeGates(t)
	restore := DefaultEngineFeatures().Apply()
	defer restore()

	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "google/gemma-4-e2b-it"
	}
	dir := metaltest.HFModelPath(t, repo)
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 4096})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()
	ctx := context.Background()

	s, ok := model.NewSession().(*ModelSession)
	if !ok {
		t.Fatal("NewSession did not return *ModelSession")
	}
	defer s.Close()

	if err := s.Prefill(ctx, "Write a long, detailed story about a lighthouse keeper and the deep ocean."); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	gen := 0
	for range s.Generate(ctx, GenerateConfig{MaxTokens: 16}) {
		gen++
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if gen == 0 {
		t.Fatal("session generated no tokens")
	}
	if err := s.AppendPrompt(ctx, " Tell me about its history."); err != nil {
		t.Fatalf("AppendPrompt: %v", err)
	}

	// Capture the live KV/logit state into a snapshot.
	snap, err := s.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snap == nil {
		t.Fatal("CaptureKV returned a nil snapshot")
	}

	// Range the snapshot's KV blocks (the block-streaming capture path).
	blocks := 0
	if err := s.RangeKVBlocks(ctx, 256, KVSnapshotCaptureOptions{}, func(KVSnapshotBlock) (bool, error) {
		blocks++
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}

	// Fork the session — a second handle sharing the prefix state.
	forked, err := s.Fork(ctx)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	defer forked.Close()

	// Restore the snapshot into a fresh session and resume generation.
	s2, ok := model.NewSession().(*ModelSession)
	if !ok {
		t.Fatal("NewSession did not return *ModelSession")
	}
	defer s2.Close()
	if err := s2.RestoreKV(ctx, snap); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	resumed := 0
	for range s2.Generate(ctx, GenerateConfig{MaxTokens: 8}) {
		resumed++
	}
	if err := s2.Err(); err != nil {
		t.Fatalf("restored session Generate: %v", err)
	}

	// Bonus: the same snapshot feeds the model's prompt-cache restore path.
	if err := model.RestorePromptCacheFromKV(ctx, snap); err != nil {
		t.Fatalf("RestorePromptCacheFromKV: %v", err)
	}
}
