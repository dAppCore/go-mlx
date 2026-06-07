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
// sequence grows, and it pins down the serve's memory leak.
//
// Reading the output: peak_mb/resid_mb that stay flat across lengths means decode
// holds a bounded working set (correct); values that climb with the token count
// are a per-token leak.
//
// What it found: the raw decode loop is leak-free under the default cache, but
// the serve's memory-plan config selects KVCacheMode "paged" (PagedKVCache), and
// that path leaks ~per token once a real context length is set — resid climbs
// 1.4 → 4.3 → 8+ GB across 512/1024/2048 tokens on E2B-4bit. The default
// (non-paged) cache stays flat at ~160 MB. So this benchmark loads with the
// paged config to reproduce the production leak and gate against its return.
//
//	go test -tags 'metal_runtime model_eval' -run '^$' \
//	  -bench BenchmarkGenerate_ContextGrowth -benchtime=1x dappco.re/go/mlx/pkg/metal/
func BenchmarkGenerate_ContextGrowth(b *testing.B) {
	if !metaltest.RunModelEvalTests {
		b.Skip("model-eval benchmark; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	// Paged KV cache — the mode the memory planner selects for the serve, and the
	// one that leaks. Swap KVCacheMode to "" to confirm the default cache is flat.
	model, err := LoadAndInit(dir, LoadConfig{
		ContextLen:  32768,
		CachePolicy: "rotating",
		KVCacheMode: "paged",
	})
	if err != nil {
		b.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()

	const prompt = "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	thinkOn := true
	configs := []struct {
		name string
		cfg  GenerateConfig
	}{
		{"greedy", GenerateConfig{}},
		{"sampled_think", GenerateConfig{Temperature: 0.8, TopP: 0.95, EnableThinking: &thinkOn}},
	}
	mb := func(bytes uint64) float64 { return float64(bytes) / (1 << 20) }
	for _, length := range []int{512, 1024, 2048} {
		for _, variant := range configs {
			b.Run(core.Sprintf("%s/tokens_%d", variant.name, length), func(b *testing.B) {
				cfg := variant.cfg
				cfg.MaxTokens = length
				ResetPeakMemory()
				before := GetActiveMemory()
				for b.Loop() {
					for range model.Generate(context.Background(), prompt, cfg) {
					}
				}
				b.ReportMetric(mb(GetPeakMemory()-before), "peak_mb")
				b.ReportMetric(mb(GetActiveMemory()-before), "resid_mb")
				b.ReportMetric(float64(length)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
			})
		}
	}
}
