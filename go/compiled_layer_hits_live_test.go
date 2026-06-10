// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// compiledHitsProbeModel selects the model for the per-token compiled-layer
// coverage probe — in-code knob, point it at whichever build is in question.
const compiledHitsProbeModel = "mlx-community/gemma-4-e2b-it-4bit"

// TestCompiledLayerHits_LiveModel reports how many layer steps per decoded
// token run through the compiled closure on the probed model — the first
// question whenever a model's host encode looks too big for its layer count
// (a declining layer runs the loose op-by-op graph and shows up here, not in
// the output, which stays correct either way).
//
//	go test -tags model_eval -run TestCompiledLayerHits_LiveModel -count=1 dappco.re/go/mlx
func TestCompiledLayerHits_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache the probed model")
	}
	dir := metaltest.HFModelPath(t, compiledHitsProbeModel)
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	info := m.Info()

	sess, err := m.NewSession()
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer sess.Close()
	if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	const decodeTokens = 24
	hitsBefore := gemma4.CompiledLayerDecodeHits()
	tokens := 0
	ctx := context.Background()
	for range sess.GenerateStream(ctx, WithMaxTokens(decodeTokens), WithTemperature(0)) {
		tokens++
	}
	if err := sess.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	hits := gemma4.CompiledLayerDecodeHits() - hitsBefore
	perToken := float64(hits) / float64(tokens)
	t.Logf("%s: %d tokens · %d compiled layer steps · %.1f/token (ctx %d)",
		compiledHitsProbeModel, tokens, hits, perToken, info.ContextLength)

	// What the caches actually store — the KV storage dtype follows the
	// arriving activation dtype unless a storage dtype was set, so read the
	// truth off the live session rather than assuming the parse default.
	snapshot, err := sess.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	for i, layer := range snapshot.Layers {
		if i >= 2 && i != len(snapshot.Layers)-1 {
			continue
		}
		t.Logf("cache %d: keys dtype %q · shape %v", i, layer.KeyDType, layer.KeyShape)
	}
}
