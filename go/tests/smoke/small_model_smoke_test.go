// SPDX-Licence-Identifier: EUPL-1.2

package smoke

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"testing"
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

func TestPlanSmallModelSmoke_GemmaQwenCoverageMatrix_Good(t *testing.T) {
	for _, tc := range []struct {
		name         string
		modelType    string
		architecture string
		template     string
	}{
		{name: "gemma4", modelType: "gemma4_text", architecture: "gemma4_text", template: "gemma4"},
		{name: "qwen2", modelType: "qwen2", architecture: "qwen2", template: "qwen"},
		{name: "qwen3", modelType: "qwen3", architecture: "qwen3", template: "qwen"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeGoodSafetensorsPack(t, dir, tc.modelType)

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
				t.Fatalf("SafeToLoad = false, want true for %s: %+v", tc.architecture, plan.Budget)
			}
			if plan.Pack.Architecture != tc.architecture || !plan.Pack.NativeLoadable || plan.Pack.ChatTemplateSource != mp.ModelPackChatTemplateNative {
				t.Fatalf("pack = arch:%q native:%v template_source:%q, want %s native template", plan.Pack.Architecture, plan.Pack.NativeLoadable, plan.Pack.ChatTemplateSource, tc.architecture)
			}
			if plan.Pack.ChatTemplate != "" {
				t.Fatalf("ChatTemplate = %q, want redacted body in smoke report", plan.Pack.ChatTemplate)
			}
			if plan.Load.ContextLength != DefaultSmallModelSmokeMaxContextLength || plan.Load.BatchSize != DefaultSmallModelSmokeMaxBatchSize || plan.Load.PrefillChunkSize > DefaultSmallModelSmokeMaxPrefillChunk {
				t.Fatalf("load = %+v, want shared small-model smoke shape", plan.Load)
			}
			if !plan.Load.PromptCache || plan.Load.PromptCacheMinTokens <= 0 {
				t.Fatalf("prompt cache load = %+v, want shared state-smoke cache settings", plan.Load)
			}
			if !DefaultSmallModelSmokeConfig().Workload.FastEval.IncludeStateKVBlockWarm {
				t.Fatal("default smoke workload should include State KV warmup across model families")
			}
		})
	}
}

func TestRunSmallModelSmoke_GemmaQwenPublicContracts_Good(t *testing.T) {
	originalLoadAndBench := runSmallModelSmokeLoadAndBench
	t.Cleanup(func() { runSmallModelSmokeLoadAndBench = originalLoadAndBench })

	expected := map[string]string{}
	seen := map[string]bool{}
	runSmallModelSmokeLoadAndBench = func(ctx context.Context, modelPath string, opts []mlx.LoadOption, workload mlx.WorkloadBenchConfig, includeBench bool) (*mlx.WorkloadBenchReport, error) {
		architecture := expected[modelPath]
		if architecture == "" {
			t.Fatalf("unexpected model path loaded: %q", modelPath)
		}
		if !includeBench {
			t.Fatalf("%s includeBench = false, want workload bench generation path", architecture)
		}
		got := mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&got)
		}
		if got.ContextLength != DefaultSmallModelSmokeMaxContextLength || got.BatchSize != DefaultSmallModelSmokeMaxBatchSize {
			t.Fatalf("%s load config = %+v, want shared smoke load shape", architecture, got)
		}
		if workload.FastEval.MaxTokens != DefaultSmallModelSmokeMaxTokens {
			t.Fatalf("%s max tokens = %d, want shared smoke generation cap", architecture, workload.FastEval.MaxTokens)
		}
		seen[architecture] = true
		return &mlx.WorkloadBenchReport{
			Summary: mlx.WorkloadBenchSummary{
				PrefillTokensPerSec: 200,
				DecodeTokensPerSec:  40,
			},
		}, nil
	}

	for _, tc := range []struct {
		name         string
		modelType    string
		architecture string
	}{
		{name: "gemma4", modelType: "gemma4_text", architecture: "gemma4_text"},
		{name: "qwen2", modelType: "qwen2", architecture: "qwen2"},
		{name: "qwen3", modelType: "qwen3", architecture: "qwen3"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeGoodSafetensorsPack(t, dir, tc.modelType)
			expected[dir] = tc.architecture

			report, err := RunSmallModelSmoke(context.Background(), SmallModelSmokeConfig{
				ModelPath: dir,
				Device: mlx.DeviceInfo{
					Architecture:                 "apple9",
					MemorySize:                   96 * memory.GiB,
					MaxRecommendedWorkingSetSize: 90 * memory.GiB,
				},
			})

			if err != nil {
				t.Fatalf("RunSmallModelSmoke() error = %v", err)
			}
			if report == nil || report.Skipped || report.Bench == nil {
				t.Fatalf("report = %+v, want same load plus generation bench path", report)
			}
			if report.Plan.Pack.Architecture != tc.architecture {
				t.Fatalf("architecture = %q, want %q", report.Plan.Pack.Architecture, tc.architecture)
			}
			if report.Bench.Summary.DecodeTokensPerSec != 40 {
				t.Fatalf("bench summary = %+v, want fake generation metrics", report.Bench.Summary)
			}
		})
	}
	for _, architecture := range []string{"gemma4_text", "qwen2", "qwen3"} {
		if !seen[architecture] {
			t.Fatalf("architecture %s did not reach public load/generate contract path", architecture)
		}
	}
}

func TestPlanSmallModelSmoke_Qwen36FallbackSkipsNativeLoad_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"text_config": {
			"model_type": "qwen3_5_text",
			"vocab_size": 248320,
			"hidden_size": 5120,
			"num_hidden_layers": 64,
			"max_position_embeddings": 262144,
			"layer_types": ["linear_attention", "full_attention"]
		},
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	plan, err := PlanSmallModelSmoke(dir, SmallModelSmokeConfig{
		Device: mlx.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 90 * memory.GiB},
	})

	if err != nil {
		t.Fatalf("PlanSmallModelSmoke() error = %v", err)
	}
	if plan.Pack.Architecture != "qwen3_6" || !plan.Pack.SupportedArchitecture || !plan.Pack.NativeLoadable {
		t.Fatalf("pack = arch:%q supported:%v native:%v, want native staged qwen3_6", plan.Pack.Architecture, plan.Pack.SupportedArchitecture, plan.Pack.NativeLoadable)
	}
	if plan.Pack.HiddenSize != 5120 || plan.Pack.NumLayers != 64 || plan.Pack.ContextLength != 262144 {
		t.Fatalf("shape metadata = hidden:%d layers:%d ctx:%d, want Qwen 3.6 text_config shape", plan.Pack.HiddenSize, plan.Pack.NumLayers, plan.Pack.ContextLength)
	}
	if !plan.Budget.SafeToLoad {
		t.Fatalf("budget = %+v, want safe native load for staged Qwen 3.6", plan.Budget)
	}
}

func TestDefaultSmallModelSmokeConfig_UsesCapturedStatePrefix_Good(t *testing.T) {
	cfg := DefaultSmallModelSmokeConfig()

	if !cfg.Workload.FastEval.IncludeStateKVBlockWarm {
		t.Fatal("IncludeStateKVBlockWarm = false, want State KV warmup covered by smoke")
	}
	if cfg.Workload.FastEval.StateKVPrefixTokens != 0 {
		t.Fatalf("StateKVPrefixTokens = %d, want 0 so short prompts use captured token length", cfg.Workload.FastEval.StateKVPrefixTokens)
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

	originalLoadAndBench := runSmallModelSmokeLoadAndBench
	t.Cleanup(func() { runSmallModelSmokeLoadAndBench = originalLoadAndBench })

	var gotPath string
	var got mlx.LoadConfig
	runSmallModelSmokeLoadAndBench = func(ctx context.Context, modelPath string, opts []mlx.LoadOption, workload mlx.WorkloadBenchConfig, includeBench bool) (*mlx.WorkloadBenchReport, error) {
		gotPath = modelPath
		got = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&got)
		}
		return &mlx.WorkloadBenchReport{
			Summary: mlx.WorkloadBenchSummary{
				PrefillTokensPerSec: 200,
				DecodeTokensPerSec:  40,
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
	if gotPath != dir {
		t.Fatalf("model path = %q, want %q", gotPath, dir)
	}
	if got.ContextLength != 8192 || got.ExpectedQuantization != 4 {
		t.Fatalf("load context/quant = %d/q%d, want 8192/q4", got.ContextLength, got.ExpectedQuantization)
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
