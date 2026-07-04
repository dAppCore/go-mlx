// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/memory"
	"dappco.re/go/mlx/spine"
)

func ExampleGC() {
	GC()
}

func ExampleSeedRandom() {
	if err := SeedRandom(42); err != nil {
		panic(err)
	}
}

func ExampleAttentionSnapshot_HasQueries() {
	snapshot := AttentionSnapshot{Queries: [][][]float32{{{1}}}}
	core.Println(snapshot.HasQueries())
	// Output: true
}

func ExampleDefaultGenerateConfig() {
	cfg := DefaultGenerateConfig()
	core.Println(cfg.MaxTokens, cfg.Temperature)
	// Output: 0 0
}

func ExampleWithMaxTokens() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithMaxTokens(2048)})
	core.Println(cfg.MaxTokens)
	// Output: 2048
}

func ExampleWithTemperature() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithTemperature(0.7)})
	core.Println(cfg.Temperature)
	// Output: 0.7
}

func ExampleWithTopK() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithTopK(40)})
	core.Println(cfg.TopK)
	// Output: 40
}

func ExampleWithTopP() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithTopP(0.95)})
	core.Println(cfg.TopP)
	// Output: 0.95
}

func ExampleWithMinP() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithMinP(0.05)})
	core.Println(cfg.MinP)
	// Output: 0.05
}

func ExampleWithSeed() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithSeed(1234)})
	core.Println(cfg.SeedSet, cfg.Seed)
	// Output: true 1234
}

func ExampleWithLogits() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithLogits()})
	core.Println(cfg.ReturnLogits)
	// Output: true
}

func ExampleWithReturnLogits() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithReturnLogits()})
	core.Println(cfg.ReturnLogits)
	// Output: true
}

func ExampleWithStopTokens() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithStopTokens(1, 2)})
	core.Println(len(cfg.StopTokens), cfg.StopTokens[0], cfg.StopTokens[1])
	// Output: 2 1 2
}

func ExampleWithMinTokensBeforeStop() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithMinTokensBeforeStop(8)})
	core.Println(cfg.MinTokensBeforeStop)
	// Output: 8
}

func ExampleWithRepeatPenalty() {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{WithRepeatPenalty(1.1)})
	core.Println(cfg.RepeatPenalty)
	// Output: 1.1
}

func ExampleDefaultLoadConfig() {
	cfg := DefaultLoadConfig()
	core.Println(cfg.ContextLength, cfg.ParallelSlots, cfg.PromptCache, cfg.PromptCacheMinTokens, cfg.Device, cfg.AutoMemoryPlan)
	// Output: 0 1 true 2048 gpu true
}

func ExampleWithContextLength() {
	cfg := applyLoadOptions([]LoadOption{WithContextLength(131072)})
	core.Println(cfg.ContextLength)
	// Output: 131072
}

func ExampleWithParallelSlots() {
	cfg := applyLoadOptions([]LoadOption{WithParallelSlots(2)})
	core.Println(cfg.ParallelSlots)
	// Output: 2
}

func ExampleWithPromptCache() {
	cfg := applyLoadOptions([]LoadOption{WithPromptCache(false)})
	core.Println(cfg.PromptCache)
	// Output: false
}

func ExampleWithPromptCacheMinTokens() {
	cfg := applyLoadOptions([]LoadOption{WithPromptCacheMinTokens(4096)})
	core.Println(cfg.PromptCacheMinTokens)
	// Output: 4096
}

func ExampleWithQuantization() {
	cfg := applyLoadOptions([]LoadOption{WithQuantization(6)})
	core.Println(cfg.Quantization)
	// Output: 6
}

func ExampleWithDevice() {
	cfg := applyLoadOptions([]LoadOption{WithDevice("cpu")})
	core.Println(cfg.Device)
	// Output: cpu
}

func ExampleWithAdapterPath() {
	cfg := applyLoadOptions([]LoadOption{WithAdapterPath("/models/gemma4-domain-adapter")})
	core.Println(cfg.AdapterPath)
	// Output: /models/gemma4-domain-adapter
}

func ExampleWithMedium() {
	medium := coreio.NewMemoryMedium()
	cfg := applyLoadOptions([]LoadOption{WithMedium(medium)})
	core.Println(cfg.Medium != nil)
	// Output: true
}

func ExampleWithAutoMemoryPlan() {
	cfg := applyLoadOptions([]LoadOption{WithAutoMemoryPlan(false)})
	core.Println(cfg.AutoMemoryPlan)
	// Output: false
}

func ExampleWithMemoryPlan() {
	plan := memory.Plan{ContextLength: 8192, CachePolicy: memory.KVCacheRotating}
	cfg := applyLoadOptions([]LoadOption{WithMemoryPlan(plan)})
	core.Println(cfg.AutoMemoryPlan, cfg.MemoryPlan.ContextLength, cfg.MemoryPlan.CachePolicy)
	// Output: false 8192 rotating
}

func ExampleWithCachePolicy() {
	cfg := applyLoadOptions([]LoadOption{WithCachePolicy(memory.KVCacheFull)})
	core.Println(cfg.CachePolicy)
	// Output: full
}

func ExampleWithBatchSize() {
	cfg := applyLoadOptions([]LoadOption{WithBatchSize(4)})
	core.Println(cfg.BatchSize)
	// Output: 4
}

func ExampleWithPrefillChunkSize() {
	cfg := applyLoadOptions([]LoadOption{WithPrefillChunkSize(1024)})
	core.Println(cfg.PrefillChunkSize)
	// Output: 1024
}

func ExampleWithAllocatorLimits() {
	cfg := applyLoadOptions([]LoadOption{WithAllocatorLimits(16<<30, 4<<30, 2<<30)})
	core.Println(cfg.MemoryLimitBytes, cfg.CacheLimitBytes, cfg.WiredLimitBytes)
	// Output: 17179869184 4294967296 2147483648
}
