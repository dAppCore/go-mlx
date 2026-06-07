// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// BenchmarkGenerate_ContextGrowth is the AX-11 (RFC-CORE-008 §11) instrument for
// the decode hot path: it measures GPU memory and throughput as the generated
// sequence grows. It generates increasing token counts on a real E2B-4bit pack
// and reports, per length, the peak and residual active-memory delta plus decode
// throughput.
//
// Reading the output: peak_mb that stays ~flat across lengths means decode holds
// a bounded working set (correct). peak_mb that climbs with the token count is a
// per-token memory leak — the engine retains live allocations across the decode
// loop instead of freeing them each step. This benchmark is the regression gate
// for that leak and the baseline any fix iterates against.
//
//	go test -tags 'metal_runtime model_eval' -run '^$' \
//	  -bench BenchmarkGenerate_ContextGrowth -benchmem dappco.re/go/mlx/pkg/metal/
func BenchmarkGenerate_ContextGrowth(b *testing.B) {
	if !metaltest.RunModelEvalTests {
		b.Skip("model-eval benchmark; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	model, err := LoadAndInit(dir, LoadConfig{})
	if err != nil {
		b.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()

	const prompt = "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	for _, length := range []int{128, 512, 1024, 2048} {
		b.Run(core.Sprintf("tokens_%d", length), func(b *testing.B) {
			ResetPeakMemory()
			before := GetActiveMemory()
			for b.Loop() {
				for range model.Generate(context.Background(), prompt, GenerateConfig{MaxTokens: length}) {
				}
			}
			mb := func(bytes uint64) float64 { return float64(bytes) / (1 << 20) }
			b.ReportMetric(mb(GetPeakMemory()-before), "peak_mb")
			b.ReportMetric(mb(GetActiveMemory()-before), "resid_mb")
			b.ReportMetric(float64(length)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
