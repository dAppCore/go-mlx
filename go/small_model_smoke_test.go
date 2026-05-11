// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestSmallModelSmokeBudget_Q4Under26GiB_Good(t *testing.T) {
	budget := EvaluateSmallModelSmokeBudget(ModelPack{
		Path:           "/models/gemma-small-q4",
		QuantBits:      4,
		WeightBytes:    5 * MemoryGiB,
		NativeLoadable: true,
		OK:             true,
	}, SmallModelSmokeConfig{})

	if !budget.SafeToLoad {
		t.Fatalf("SafeToLoad = false, want true: %+v", budget)
	}
	if budget.MaxWeightBytes != 26*MemoryGiB || budget.RequiredQuantization != 4 {
		t.Fatalf("defaults = max:%d quant:%d, want 26GiB/q4", budget.MaxWeightBytes, budget.RequiredQuantization)
	}
}

func TestSmallModelSmokeBudget_RejectsOversizeQ4_Bad(t *testing.T) {
	budget := EvaluateSmallModelSmokeBudget(ModelPack{
		Path:           "/models/qwen-large-q4",
		QuantBits:      4,
		WeightBytes:    27 * MemoryGiB,
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
	budget := EvaluateSmallModelSmokeBudget(ModelPack{
		Path:           "/models/gemma-small-bf16",
		QuantBits:      16,
		WeightBytes:    8 * MemoryGiB,
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
		pack ModelPack
		want string
	}{
		{
			name: "invalid pack",
			pack: ModelPack{OK: false, NativeLoadable: true, WeightBytes: MemoryGiB, QuantBits: 4},
			want: "validation",
		},
		{
			name: "not native loadable",
			pack: ModelPack{OK: true, NativeLoadable: false, WeightBytes: MemoryGiB, QuantBits: 4},
			want: "native-loadable",
		},
		{
			name: "unknown weights",
			pack: ModelPack{OK: true, NativeLoadable: true, WeightBytes: 0, QuantBits: 4},
			want: "unknown",
		},
		{
			name: "unknown quantization",
			pack: ModelPack{OK: true, NativeLoadable: true, WeightBytes: MemoryGiB, QuantBits: 0},
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
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 90 * MemoryGiB,
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
		Device: DeviceInfo{MemorySize: 16 * MemoryGiB},
	})
	if err != nil {
		t.Fatalf("PlanSmallModelSmoke() error = %v", err)
	}
	if !plan.Pack.HasChatTemplate || plan.Pack.ChatTemplateSource != ModelPackChatTemplateJinja {
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
		Workload: WorkloadBenchConfig{
			FastEval: FastEvalConfig{Prompt: "custom", MaxTokens: 2},
		},
	})
	if cfg.RequiredQuantization != 8 || cfg.MaxContextLength != 4096 || cfg.MaxBatchSize != 2 || cfg.MaxPrefillChunkSize != 128 {
		t.Fatalf("normalised config = %+v, want caller numeric caps retained", cfg)
	}
	if len(smallModelSmokePackOptions(cfg)) != 2 {
		t.Fatalf("pack options len = %d, want chat-template option plus quantization", len(smallModelSmokePackOptions(cfg)))
	}
	load := smallModelSmokeLoadPlan(MemoryPlan{
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
	opts := smallModelSmokeLoadOptions(SmallModelSmokePlan{MemoryPlan: MemoryPlan{}, Load: load}, SmallModelSmokeConfig{
		AdditionalLoadOptions: []LoadOption{WithDevice("cpu")},
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
