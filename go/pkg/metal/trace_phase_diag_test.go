// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"sort"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestTrace_DecodePhaseBreakdown_Diag dumps the steady-state per-token phase
// breakdown for GO_MLX_BENCH_MODEL (default e2b q6) so the per-token overhead
// has a target. Run:
//
//	GO_MLX_BENCH_MODEL=mlx-community/gemma-4-e2b-it-6bit go test -tags \
//	  'metal_runtime model_eval' -run TestTrace_DecodePhaseBreakdown_Diag -v ...
func TestTrace_DecodePhaseBreakdown_Diag(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval diagnostic; build with -tags model_eval")
	}
	isolateRuntimeGates(t)
	restore := DefaultEngineFeatures().Apply()
	defer restore()
	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "mlx-community/gemma-4-e2b-it-6bit"
	}
	dir := metaltest.HFModelPath(t, repo)
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32768, CachePolicy: "rotating"})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()

	const prompt = "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	cfg := GenerateConfig{MaxTokens: 160, TraceTokenPhases: true}
	for range model.Generate(context.Background(), prompt, cfg) {
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) < 40 {
		t.Fatalf("too few phase traces: %d", len(phases))
	}
	steady := phases[16:] // drop warmup/prefill-adjacent steps
	sums := map[string]time.Duration{}
	var total time.Duration
	for _, p := range steady {
		sums["forward"] += p.ForwardDuration
		sums["logits"] += p.LogitsDuration
		sums["sample"] += p.SampleDuration
		sums["sampleEval"] += p.SampleEvalDuration
		sums["tokenRead"] += p.TokenReadDuration
		sums["decodeText"] += p.DecodeTextDuration
		sums["probeToken"] += p.ProbeTokenDuration
		sums["yield"] += p.YieldDuration
		sums["nextInput"] += p.NextInputDuration
		sums["prefetch"] += p.PrefetchDuration
		sums["prefetchLogits"] += p.PrefetchLogitsDuration
		sums["prefetchCache"] += p.PrefetchCacheDuration
		sums["materialize"] += p.MaterializeDuration
		sums["detach"] += p.DetachDuration
		sums["cacheProbe"] += p.CacheProbeDuration
		sums["other"] += p.OtherDuration
		total += p.TotalDuration
	}
	n := time.Duration(len(steady))
	type row struct {
		name string
		mean time.Duration
	}
	var rows []row
	for k, v := range sums {
		rows = append(rows, row{k, v / n})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].mean > rows[j].mean })
	meanTotal := total / n
	t.Logf("%s steady-state: %d tokens, mean total %.3f ms (%.1f tok/s)",
		repo, len(steady), float64(meanTotal)/float64(time.Millisecond), float64(time.Second)/float64(meanTotal))
	for _, r := range rows {
		if r.mean > 0 {
			t.Logf("  %-16s %7.3f ms  (%4.1f%%)", r.name,
				float64(r.mean)/float64(time.Millisecond), 100*float64(r.mean)/float64(meanTotal))
		}
	}
}
