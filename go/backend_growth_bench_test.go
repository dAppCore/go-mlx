// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/adapter"
	"dappco.re/go/mlx/internal/metaltest"
)

// BenchmarkBackend_ContextGrowth is the serve-path twin of
// BenchmarkGenerate_ContextGrowth (pkg/metal). The raw decode loop
// (model.Generate) is leak-free; this drives the SAME growth sweep through the
// inference-layer path the serve actually uses — NewMLXBackend → adapter.Generate
// → the inference.TextModel — to localise the serve's per-token memory leak. A
// climbing resid_mb here (where the raw loop stayed flat) puts the leak in the
// inference/adapter wrapper, not the engine core.
//
//	go test -tags 'metal_runtime model_eval' -run '^$' \
//	  -bench BenchmarkBackend_ContextGrowth -benchtime=1x dappco.re/go/mlx/
func BenchmarkBackend_ContextGrowth(b *testing.B) {
	if !metaltest.RunModelEvalTests {
		b.Skip("model-eval benchmark; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	backend, err := NewMLXBackend(dir)
	if err != nil {
		b.Fatalf("NewMLXBackend: %v", err)
	}

	const prompt = "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	for _, length := range []int{512, 1024, 2048} {
		b.Run(core.Sprintf("tokens_%d", length), func(b *testing.B) {
			before := GetActiveMemory()
			for b.Loop() {
				if _, err := backend.Generate(context.Background(), prompt, adapter.GenOpts{MaxTokens: length}); err != nil {
					b.Fatalf("Generate: %v", err)
				}
			}
			b.ReportMetric(float64(GetActiveMemory()-before)/(1<<20), "resid_mb")
		})
	}
}
