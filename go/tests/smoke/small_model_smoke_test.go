// SPDX-Licence-Identifier: EUPL-1.2

package smoke

import (
	mlx "dappco.re/go/mlx"
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"testing"
	"time"
)

func TestSmallModelSmokeBudget_Q4Under26GiB_Good(t *testing.T) {
	budget := EvaluateSmallModelSmokeBudget(mp.ModelPack{
		Path:           "/models/gemma-small-q4",
		QuantBits:      4,
		WeightBytes:    5 * memory.GiB,
		NativeLoadable: true,
		OK:             true,
	}, SmallModelSmokeConfig{})

	if !budget.SafeToLoad {
		t.Fatalf("SafeToLoad = false, want true: %+v", budget)
	}
	if budget.MaxWeightBytes != 26*memory.GiB || budget.RequiredQuantization != 4 {
		t.Fatalf("defaults = max:%d quant:%d, want 26GiB/q4", budget.MaxWeightBytes, budget.RequiredQuantization)
	}
}

func TestSmallModelSmokeBudget_RejectsOversizeQ4_Bad(t *testing.T) {
	budget := EvaluateSmallModelSmokeBudget(mp.ModelPack{
		Path:           "/models/qwen-large-q4",
		QuantBits:      4,
		WeightBytes:    27 * memory.GiB,
		NativeLoadable: true,
		OK:             true,
	}, SmallModelSmokeConfig{})

	if budget.SafeToLoad {
		t.Fatal("SafeToLoad = true, want oversize q4 model rejected")
	}
	if budget.Reason == "" {
		t.Fatalf("Reason is empty, want budget explanation: %+v", budget)
	}
}

func TestSmallModelSmokeBudget_RejectsNonQ4_Bad(t *testing.T) {
	budget := EvaluateSmallModelSmokeBudget(mp.ModelPack{
		Path:           "/models/gemma-small-bf16",
		QuantBits:      16,
		WeightBytes:    8 * memory.GiB,
		NativeLoadable: true,
		OK:             true,
	}, SmallModelSmokeConfig{})

	if budget.SafeToLoad {
		t.Fatal("SafeToLoad = true, want non-q4 model rejected by default")
	}
	if budget.RequiredQuantization != 4 {
		t.Fatalf("RequiredQuantization = %d, want q4 default", budget.RequiredQuantization)
	}
}

func TestSmallModelSmokeBudget_RejectsUnsafeMetadata_Bad(t *testing.T) {
	cases := []struct {
		name string
		pack mp.ModelPack
		want string
	}{
		{
			name: "invalid pack",
			pack: mp.ModelPack{OK: false, NativeLoadable: true, WeightBytes: memory.GiB, QuantBits: 4},
			want: "validation",
		},
		{
			name: "not native loadable",
			pack: mp.ModelPack{OK: true, NativeLoadable: false, WeightBytes: memory.GiB, QuantBits: 4},
			want: "native-loadable",
		},
		{
			name: "unknown weights",
			pack: mp.ModelPack{OK: true, NativeLoadable: true, WeightBytes: 0, QuantBits: 4},
			want: "unknown",
		},
		{
			name: "unknown quantization",
			pack: mp.ModelPack{OK: true, NativeLoadable: true, WeightBytes: memory.GiB, QuantBits: 0},
			want: "quantization is unknown",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			budget := EvaluateSmallModelSmokeBudget(tc.pack, SmallModelSmokeConfig{})
			if budget.SafeToLoad || !core.Contains(budget.Reason, tc.want) {
				t.Fatalf("budget = %+v, want unsafe reason containing %q", budget, tc.want)
			}
		})
	}
}

func TestPlanSmallModelSmoke_CapsContextForAppleSmoke_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	plan, err := PlanSmallModelSmoke(dir, SmallModelSmokeConfig{
		Device: mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
	})
	if err != nil {
		t.Fatalf("PlanSmallModelSmoke() error = %v", err)
	}
	if !plan.Budget.SafeToLoad {
		t.Fatalf("SafeToLoad = false, want true: %+v", plan.Budget)
	}
	if plan.Load.ContextLength != 8192 {
		t.Fatalf("smoke context length = %d, want 8192", plan.Load.ContextLength)
	}
	if plan.MemoryPlan.ContextLength <= plan.Load.ContextLength {
		t.Fatalf("memory plan context = %d, want larger than smoke cap %d", plan.MemoryPlan.ContextLength, plan.Load.ContextLength)
	}
	if !smallModelSmokeHasNote(plan, "context capped") {
		t.Fatalf("notes = %+v, want context cap note", plan.Notes)
	}
}

func TestDefaultSmallModelSmokeConfig_UsesCapturedMemvidPrefix_Good(t *testing.T) {
	cfg := DefaultSmallModelSmokeConfig()

	if !cfg.Workload.FastEval.IncludeMemvidKVBlockWarm {
		t.Fatal("IncludeMemvidKVBlockWarm = false, want memvid KV warmup covered by smoke")
	}
	if cfg.Workload.FastEval.MemvidKVPrefixTokens != 0 {
		t.Fatalf("MemvidKVPrefixTokens = %d, want 0 so short prompts use captured token length", cfg.Workload.FastEval.MemvidKVPrefixTokens)
	}
}

func TestPlanSmallModelSmoke_RedactsChatTemplateByDefault_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")
	writeModelPackFile(t, core.PathJoin(dir, "chat_template.jinja"), "large-template-body")

	plan, err := PlanSmallModelSmoke(dir, SmallModelSmokeConfig{
		Device: mlx.DeviceInfo{MemorySize: 16 * memory.GiB},
	})
	if err != nil {
		t.Fatalf("PlanSmallModelSmoke() error = %v", err)
	}
	if !plan.Pack.HasChatTemplate || plan.Pack.ChatTemplateSource != mp.ModelPackChatTemplateJinja {
		t.Fatalf("chat template metadata = has:%v source:%q", plan.Pack.HasChatTemplate, plan.Pack.ChatTemplateSource)
	}
	if plan.Pack.ChatTemplate != "" {
		t.Fatalf("ChatTemplate = %q, want redacted report body", plan.Pack.ChatTemplate)
	}
}

func TestRunSmallModelSmoke_Bad_SkipsUnsafePackWithoutLoading(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 262208,
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"max_position_embeddings": 8192,
		"quantization_config": {"bits": 8, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	report, err := RunSmallModelSmoke(nil, SmallModelSmokeConfig{ModelPath: dir})

	if err != nil {
		t.Fatalf("RunSmallModelSmoke() error = %v", err)
	}
	if report == nil || !report.Skipped || report.SkipReason == "" || report.Bench != nil {
		t.Fatalf("report = %+v, want skipped unsafe pack without bench", report)
	}
}

func TestSmallModelSmokeHelpers_Good(t *testing.T) {
	cfg := normalizeSmallModelSmokeConfig(SmallModelSmokeConfig{
		RequiredQuantization: 8,
		MaxContextLength:     4096,
		MaxBatchSize:         2,
		MaxPrefillChunkSize:  128,
		Workload: mlx.WorkloadBenchConfig{
			FastEval: bench.Config{Prompt: "custom", MaxTokens: 2},
		},
	})
	if cfg.RequiredQuantization != 8 || cfg.MaxContextLength != 4096 || cfg.MaxBatchSize != 2 || cfg.MaxPrefillChunkSize != 128 {
		t.Fatalf("normalised config = %+v, want caller numeric caps retained", cfg)
	}
	if len(smallModelSmokePackOptions(cfg)) != 2 {
		t.Fatalf("pack options len = %d, want chat-template option plus quantization", len(smallModelSmokePackOptions(cfg)))
	}
	load := smallModelSmokeLoadPlan(memory.Plan{
		ContextLength:        16384,
		ParallelSlots:        3,
		PromptCache:          true,
		BatchSize:            8,
		PrefillChunkSize:     1024,
		MemoryLimitBytes:     10,
		CacheLimitBytes:      5,
		WiredLimitBytes:      3,
		PromptCacheMinTokens: 0,
	}, cfg)
	if load.ContextLength != 4096 || load.BatchSize != 2 || load.PrefillChunkSize != 128 || load.PromptCacheMinTokens != DefaultSmallModelSmokePromptCacheMinSize {
		t.Fatalf("load plan = %+v, want capped smoke shape", load)
	}
	opts := smallModelSmokeLoadOptions(SmallModelSmokePlan{MemoryPlan: memory.Plan{}, Load: load}, SmallModelSmokeConfig{
		AdditionalLoadOptions: []mlx.LoadOption{mlx.WithDevice("cpu")},
	})
	if len(opts) != 13 {
		t.Fatalf("load options len = %d, want base options plus additional option", len(opts))
	}
}

func TestPlanSmallModelSmoke_Bad_RequiresModelPath(t *testing.T) {
	if _, err := PlanSmallModelSmoke("", SmallModelSmokeConfig{}); err == nil {
		t.Fatal("PlanSmallModelSmoke(empty path) error = nil")
	}
}

func smallModelSmokeHasNote(plan SmallModelSmokePlan, fragment string) bool {
	for _, note := range plan.Notes {
		if core.Contains(note, fragment) {
			return true
		}
	}
	return false
}

func TestRunSmallModelSmoke_ForwardsBudgetedLoadOptions_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var got metal.LoadConfig
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		got = cfg
		return &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture:  "gemma4_text",
				ContextLength: 8192,
				NumLayers:     26,
				HiddenSize:    2048,
				QuantBits:     4,
			},
			tokens: []metal.Token{{ID: 1, Text: "ok"}},
			metrics: metal.Metrics{
				PromptTokens:               4,
				GeneratedTokens:            1,
				PrefillTokensPerSec:        200,
				DecodeTokensPerSec:         40,
				TotalDuration:              time.Millisecond,
				PromptCacheHits:            1,
				PromptCacheHitTokens:       4,
				PromptCacheRestoreDuration: time.Millisecond,
			},
		}, nil
	}

	report, err := RunSmallModelSmoke(context.Background(), SmallModelSmokeConfig{
		ModelPath: dir,
		Device: mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Workload: mlx.WorkloadBenchConfig{
			FastEval: bench.Config{
				Prompt:             "hi",
				CachePrompt:        "hi",
				MaxTokens:          1,
				Runs:               1,
				IncludePromptCache: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("RunSmallModelSmoke() error = %v", err)
	}
	if report == nil || report.Skipped || report.Bench == nil {
		t.Fatalf("report = %+v, want loaded bench", report)
	}
	if got.ContextLen != 8192 || got.ExpectedQuantization != 4 {
		t.Fatalf("load context/quant = %d/q%d, want 8192/q4", got.ContextLen, got.ExpectedQuantization)
	}
	if got.BatchSize != 1 || got.PrefillChunkSize > 1024 {
		t.Fatalf("load shape = batch:%d prefill:%d, want small smoke shape", got.BatchSize, got.PrefillChunkSize)
	}
	if got.MemoryLimitBytes == 0 || got.CacheLimitBytes == 0 || got.WiredLimitBytes == 0 {
		t.Fatalf("allocator limits not forwarded: %+v", got)
	}
	if report.Bench.Summary.PrefillTokensPerSec != 200 || report.Bench.Summary.DecodeTokensPerSec != 40 {
		t.Fatalf("bench summary = %+v, want fake metrics", report.Bench.Summary)
	}
}
