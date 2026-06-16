// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestSample_SamplerVariants_Eval drives the real sampler chain on a model: the
// synthetic/greedy path (temperature 0) never builds the top-k/top-p chain, the
// min-p filter, the repeat-penalty pass, or the token-suppression cache, so those
// branches are dark. Each case sets temperature > 0 to force the sampler off the
// greedy fast-path. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestSample_SamplerVariants_Eval ./pkg/metal/
func TestSample_SamplerVariants_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
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
	prompt := "Write a long, detailed story about a lighthouse keeper and the deep ocean."

	cases := []struct {
		name string
		cfg  GenerateConfig
	}{
		{"topK+topP", GenerateConfig{MaxTokens: 16, Temperature: 0.8, TopP: 0.95, TopK: 40}},
		{"minP", GenerateConfig{MaxTokens: 16, Temperature: 0.8, MinP: 0.05}},
		{"repeatPenalty", GenerateConfig{MaxTokens: 16, Temperature: 0.8, RepeatPenalty: 1.3}},
		{"suppressTokens", GenerateConfig{MaxTokens: 16, Temperature: 0.8, TopK: 40, SuppressTokens: []int32{1000, 2000, 3000}}},
		{"temperatureOnly", GenerateConfig{MaxTokens: 16, Temperature: 1.0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := 0
			for range model.Generate(ctx, prompt, tc.cfg) {
				n++
			}
			if err := model.Err(); err != nil {
				t.Fatalf("%s generation: %v", tc.name, err)
			}
			if n == 0 {
				t.Fatalf("%s produced no tokens", tc.name)
			}
		})
	}
}
