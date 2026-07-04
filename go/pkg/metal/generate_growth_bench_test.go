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
// Reading the output: rss_mb is real process resident memory; cache_mb is the
// MLX allocator's freed-buffer pool. cache_mb that climbs without bound across
// lengths (and never falls back for short prompts) is the allocator hoarding
// buffers under size-diverse prompts — now bounded by the auto-derived cache
// limit in LoadAndInit. (The former peak_mb/resid_mb read mlx_get_active_memory,
// which over-counts — it can exceed RSS and only grows — so it masked the cache.)
//
// What it found: the broken PagedKVCache leaked ~per token (resid climbed
// 1.4 → 4.3 → 8+ GB across 512/1024/2048 on E2B-4bit); the leak fix routed the
// planner off paged onto the default (rotating) cache, flat at ~160 MB. This
// benchmark now loads the DEFAULT cache — the real serve path — so it doubles as
// the decode-throughput (tok/s) baseline for the perf campaign (target 100 tok/s+
// at q4/q6). E2B-4bit measures ≈ 110-115 tok/s on M3 Ultra.
//
//	go test -tags 'metal_runtime model_eval' -run '^$' \
//	  -bench BenchmarkGenerate_ContextGrowth -benchtime=1x dappco.re/go/mlx/pkg/metal/
func BenchmarkGenerate_ContextGrowth(b *testing.B) {
	if !metaltest.RunModelEvalTests {
		b.Skip("model-eval benchmark; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	// Apply the model's accepted fast-path gates (q6 bitstream matvec, MLP/Linear/
	// attention matvec, direct-greedy, async prefetch) — the SAME set the serve
	// enables at boot. Without this the bench measures the gate-off floor, which
	// badly under-reports q6 (its bitstream kernel is gated, so q6 falls back to
	// the slow generic matmul and lands below q8 — backwards).
	restore := DefaultEngineFeatures().Apply()
	defer restore()
	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "mlx-community/gemma-4-e2b-it-4bit"
	}
	dir := metaltest.HFModelPath(b, repo)
	// Default (rotating) cache — the real serve path post leak-fix; the decode
	// throughput + bounded-memory baseline. The broken paged cache is retired.
	model, err := LoadAndInit(dir, LoadConfig{
		ContextLen:  32768,
		CachePolicy: "rotating",
		KVCacheMode: "",
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
				for b.Loop() {
					for range model.Generate(context.Background(), prompt, cfg) {
					}
				}
				// Report honest memory: real process RSS plus the MLX allocator
				// cache — the freed-buffer pool that balloons under size-diverse
				// prompts when no cache limit is set. The former peak_mb/resid_mb
				// read mlx_get_active_memory, which over-counts (it can exceed RSS
				// and climbs monotonically), masking the cache as the real signal.
				b.ReportMetric(mb(GetProcessMemory().ResidentMemoryBytes), "rss_mb")
				b.ReportMetric(mb(GetCacheMemory()), "cache_mb")
				b.ReportMetric(float64(length)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
			})
		}
	}
}
