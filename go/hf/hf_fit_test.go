// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
)

// TestHfFit_PlanFits_Good is the canonical happy-path triplet entry: a single
// supported model from an injected source plans to a loadable, inference-ready
// fit with a real memory estimate. The named TestHfFit_PlanFits_* variants
// below exercise the many architecture-specific shapes; this one pins the
// minimal "it fits and runs" contract. No real network.
func TestHfFit_PlanFits_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"Qwen/Qwen3-0.6B": {
				ID: "Qwen/Qwen3-0.6B",
				Config: ModelConfig{
					ModelType:             "qwen3",
					HiddenSize:            1024,
					NumHiddenLayers:       28,
					NumAttentionHeads:     16,
					NumKeyValueHeads:      8,
					MaxPositionEmbeddings: 40960,
					Quantization:          &QuantizationConfig{Bits: 4, GroupSize: 64},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 420 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"Qwen/Qwen3-0.6B"},
		Device: memory.DeviceInfo{
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.ModelID != "Qwen/Qwen3-0.6B" || plan.Architecture != "qwen3" {
		t.Fatalf("plan identity = %q/%q, want Qwen/Qwen3-0.6B/qwen3", plan.ModelID, plan.Architecture)
	}
	if !plan.SupportedArchitecture || !plan.NativeLoadable || !plan.InferenceFits {
		t.Fatalf("fit flags = supported:%v native:%v inference:%v, want all true", plan.SupportedArchitecture, plan.NativeLoadable, plan.InferenceFits)
	}
	if plan.QuantBits != 4 || plan.WeightBytes == 0 || plan.ExpectedKVBytes == 0 {
		t.Fatalf("sizing = quant:%d weight:%d kv:%d, want quant + memory estimates", plan.QuantBits, plan.WeightBytes, plan.ExpectedKVBytes)
	}
}

// TestHfFit_PlanFits_Bad is the canonical error-path triplet entry: PlanFits
// with no metadata sources at all must fail rather than return an empty report.
// The named TestHfFit_PlanFits_*_Bad variants below cover the specific
// guard/propagation paths. No network.
func TestHfFit_PlanFits_Bad(t *testing.T) {
	if _, err := PlanFits(context.Background(), FitConfig{}); err == nil {
		t.Fatal("PlanFits(no sources) error = nil, want a no-metadata error")
	}
	// A model-id lookup with no Source is a misconfiguration, not an empty result.
	_, err := PlanFits(context.Background(), FitConfig{ModelIDs: []string{"org/model"}})
	if err == nil || !core.Contains(err.Error(), "source") {
		t.Fatalf("PlanFits(ids, no source) error = %v, want a missing-source error", err)
	}
}

// TestHfFit_PlanFits_Ugly is the canonical advisory-path triplet entry: an
// architecture the native loaders don't recognise still receives a memory
// estimate, but PlanFits flags it unsupported / non-loadable and attaches an
// explanatory note. The named TestHfFit_PlanFits_UnsupportedArchitecture_Ugly
// variant carries the oversized-device case; this one pins the core advisory
// contract. No network.
func TestHfFit_PlanFits_Ugly(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"future/arch-model": {
				ID: "future/arch-model",
				Config: ModelConfig{
					ModelType:             "totally_future_arch",
					HiddenSize:            2048,
					NumHiddenLayers:       24,
					NumAttentionHeads:     16,
					MaxPositionEmbeddings: 8192,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 1 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"future/arch-model"},
		Device:   memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 86 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	plan := report.Models[0]
	if plan.SupportedArchitecture || plan.NativeLoadable || plan.InferenceFits {
		t.Fatalf("unrecognised arch flagged loadable: supported:%v native:%v inference:%v", plan.SupportedArchitecture, plan.NativeLoadable, plan.InferenceFits)
	}
	if plan.WeightBytes == 0 {
		t.Fatal("WeightBytes = 0, want a memory estimate even for an unsupported arch")
	}
	if len(plan.Notes) == 0 {
		t.Fatal("Notes empty, want an explanatory note for the unsupported arch")
	}
}

func TestHfFit_PlanFits_InjectedSearch_Good(t *testing.T) {
	source := &fakeHFModelSource{
		search: []ModelMetadata{{
			ID: "Qwen/Qwen3-0.6B",
			Config: ModelConfig{
				ModelType:             "qwen3",
				HiddenSize:            1024,
				NumHiddenLayers:       28,
				NumAttentionHeads:     16,
				NumKeyValueHeads:      8,
				MaxPositionEmbeddings: 40960,
				Quantization:          &QuantizationConfig{Bits: 4, GroupSize: 64},
			},
			Files: []ModelFile{
				{Name: "model.safetensors", Size: 420 * 1024 * 1024},
				{Name: "tokenizer.json", Size: 4 * 1024 * 1024},
			},
		}},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		Query:      "qwen 0.6b",
		MaxResults: 5,
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m3-ultra",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if !source.searchCalled {
		t.Fatal("SearchModels was not called")
	}
	if report.DeviceClass != memory.ClassApple96GB || report.MemoryPlan.ContextLength != 131072 {
		t.Fatalf("device plan = %+v class=%s", report.MemoryPlan, report.DeviceClass)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.ModelID != "Qwen/Qwen3-0.6B" || plan.Architecture != "qwen3" || !plan.SupportedArchitecture {
		t.Fatalf("plan identity = %+v", plan)
	}
	if plan.QuantBits != 4 || plan.WeightBytes == 0 || plan.ExpectedKVBytes == 0 {
		t.Fatalf("sizing = %+v, want quant and memory estimates", plan)
	}
	if !plan.InferenceFits || !plan.Training.LoRAFeasible || plan.Training.FullFineTuneFeasible {
		t.Fatalf("fit/training = inference:%v training:%+v", plan.InferenceFits, plan.Training)
	}
	if plan.ContextRecommendation != 40960 {
		t.Fatalf("ContextRecommendation = %d, want %d", plan.ContextRecommendation, 40960)
	}
}

func TestHfFit_PlanFits_LocalCache_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--mlx-community--gemma-4-e2b-it-4bit")
	dir := core.PathJoin(cacheRoot, "snapshots", "abc123")
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", dir, result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	report, err := PlanFits(context.Background(), FitConfig{
		LocalPaths: []string{cacheRoot},
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m1-pro",
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 13 * memory.GiB,
		},
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.ModelID != "mlx-community/gemma-4-e2b-it-4bit" {
		t.Fatalf("ModelID = %q", plan.ModelID)
	}
	if plan.Source != SourceLocal || plan.LocalPath != dir {
		t.Fatalf("source/path = %q %q", plan.Source, plan.LocalPath)
	}
	if plan.Architecture != "gemma4_text" || !plan.SupportedArchitecture {
		t.Fatalf("architecture support = %q %v", plan.Architecture, plan.SupportedArchitecture)
	}
	if plan.ContextRecommendation != 94208 || plan.MemoryPlan.CachePolicy != memory.KVCacheRotating {
		t.Fatalf("context/cache = rec:%d policy:%q, want rec 94208 (e2b on 16GB derives 94208 from truth — memory bounds it below the 131072 model max; the old 8192 was the RAM-class cap) + rotating", plan.ContextRecommendation, plan.MemoryPlan.CachePolicy)
	}
	if plan.ExpectedKVBytes == 0 {
		t.Fatal("ExpectedKVBytes = 0, want estimate")
	}
}

func TestHfFit_PlanFits_QwenNextNestedTextConfig_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"Qwen/Qwen3.5-0.8B-Base": {
				ID: "Qwen/Qwen3.5-0.8B-Base",
				Config: ModelConfig{
					ModelType: "qwen3_5",
					TextConfig: &ModelConfig{
						ModelType:             "qwen3_next",
						HiddenSize:            1536,
						NumHiddenLayers:       28,
						NumAttentionHeads:     16,
						NumKeyValueHeads:      8,
						MaxPositionEmbeddings: 98304,
						QuantizationConfig:    &QuantizationConfig{Bits: 4, GroupSize: 64},
					},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 900 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"Qwen/Qwen3.5-0.8B-Base"},
		Device:   memory.DeviceInfo{MemorySize: 24 * memory.GiB, MaxRecommendedWorkingSetSize: 20 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "qwen3_next" || !plan.SupportedArchitecture || !plan.NativeLoadable {
		t.Fatalf("architecture/loadable = %q supported=%v native=%v", plan.Architecture, plan.SupportedArchitecture, plan.NativeLoadable)
	}
	// Qwen3-Next is an other-model arch not yet updated to declare its KV dims;
	// its context recommendation now derives from truth (model max ∩ memory)
	// instead of the old machine-class cap. Assert a positive derived
	// recommendation, not a fixed number that pins an incomplete-config artifact.
	if plan.ContextRecommendation <= 0 {
		t.Fatalf("ContextRecommendation = %d, want a positive derived recommendation", plan.ContextRecommendation)
	}
}

func TestHfFit_PlanFits_Gemma4AssistantUsesOuterArchitecture_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"google/gemma-4-E2B-it-assistant": {
				ID: "google/gemma-4-E2B-it-assistant",
				Config: ModelConfig{
					ModelType:     "gemma4_assistant",
					Architectures: []string{"Gemma4AssistantForCausalLM"},
					TextConfig: &ModelConfig{
						ModelType:             "gemma4_text",
						VocabSize:             262144,
						HiddenSize:            256,
						NumHiddenLayers:       4,
						NumAttentionHeads:     4,
						NumKeyValueHeads:      1,
						MaxPositionEmbeddings: 131072,
						QuantizationConfig:    &QuantizationConfig{Bits: 16, GroupSize: 64},
					},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 2 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"google/gemma-4-E2B-it-assistant"},
		Device:   memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 86 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "gemma4_assistant" || !plan.SupportedArchitecture || plan.NativeLoadable || plan.InferenceFits {
		t.Fatalf("assistant plan = arch:%q supported:%v native:%v inference:%v, want attachable-only assistant", plan.Architecture, plan.SupportedArchitecture, plan.NativeLoadable, plan.InferenceFits)
	}
	if plan.ContextLimit != 131072 || plan.QuantBits != 16 {
		t.Fatalf("assistant metadata = ctx:%d quant:%d, want text_config metadata retained", plan.ContextLimit, plan.QuantBits)
	}
	noteText := core.Join("\n", plan.Notes...)
	if !core.Contains(noteText, "attached MTP drafter") || !core.Contains(noteText, "LoadSpeculativePair") {
		t.Fatalf("assistant notes = %q, want attached drafter guidance", noteText)
	}
}

func TestHfFit_PlanFits_Gemma412BUnifiedPreservesArchitecture_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"google/gemma-4-12B-it": {
				ID: "google/gemma-4-12B-it",
				Config: ModelConfig{
					ModelType:     "gemma4_unified",
					Architectures: []string{"Gemma4UnifiedForConditionalGeneration"},
					TextConfig: &ModelConfig{
						ModelType:             "gemma4_unified_text",
						VocabSize:             262144,
						HiddenSize:            3840,
						NumHiddenLayers:       48,
						NumAttentionHeads:     16,
						NumKeyValueHeads:      8,
						MaxPositionEmbeddings: 262144,
						QuantizationConfig:    &QuantizationConfig{Bits: 6, GroupSize: 64},
					},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 12 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"google/gemma-4-12B-it"},
		Device:   memory.DeviceInfo{MemorySize: 128 * memory.GiB, MaxRecommendedWorkingSetSize: 112 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "gemma4_unified" || !plan.SupportedArchitecture || !plan.NativeLoadable {
		t.Fatalf("plan architecture = %q supported=%v native=%v, want native Gemma 4 12B Unified", plan.Architecture, plan.SupportedArchitecture, plan.NativeLoadable)
	}
	if plan.ContextLimit != 262144 || plan.ContextRecommendation != 61440 || plan.QuantBits != 6 || plan.QuantGroup != 64 {
		t.Fatalf("plan metadata = ctx:%d rec:%d quant:%d/%d, want 262144 ctx + rec 61440 (12B-unified weights leave 61440 of its 256K window — derived from truth, not the old 131072 RAM-class cap) + q6/g64", plan.ContextLimit, plan.ContextRecommendation, plan.QuantBits, plan.QuantGroup)
	}
	if plan.ExpectedKVBytes == 0 {
		t.Fatal("ExpectedKVBytes = 0, want generation KV estimate for Unified decoder")
	}
}

func TestHfFit_PlanFits_BertEmbeddingUsesEncoderMemoryPlan_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"BAAI/bge-small-en-v1.5": {
				ID:          "BAAI/bge-small-en-v1.5",
				PipelineTag: "feature-extraction",
				Config: ModelConfig{
					ModelType:             "bert",
					Architectures:         []string{"BertModel"},
					HiddenSize:            384,
					NumHiddenLayers:       12,
					MaxPositionEmbeddings: 512,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 130 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"BAAI/bge-small-en-v1.5"},
		Device:   memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "bert" || !plan.SupportedArchitecture {
		t.Fatalf("architecture support = %q %v", plan.Architecture, plan.SupportedArchitecture)
	}
	if !plan.Embeddings || plan.Rerank {
		t.Fatalf("task flags = embeddings:%v rerank:%v, want embedding encoder fit plan", plan.Embeddings, plan.Rerank)
	}
	if plan.ExpectedKVBytes != 0 || plan.MemoryPlan.CacheMode != memory.KVCacheModeDefault || plan.MemoryPlan.PromptCache {
		t.Fatalf("encoder memory = kv:%d plan:%+v, want no generation KV cache", plan.ExpectedKVBytes, plan.MemoryPlan)
	}
	if plan.ContextRecommendation != 512 {
		t.Fatalf("ContextRecommendation = %d, want 512", plan.ContextRecommendation)
	}
}

func TestHfFit_PlanFits_BertRerankUsesScorerMemoryPlan_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"BAAI/bge-reranker-base": {
				ID:          "BAAI/bge-reranker-base",
				PipelineTag: "text-classification",
				Config: ModelConfig{
					ModelType:             "bert",
					Architectures:         []string{"BertForSequenceClassification"},
					HiddenSize:            768,
					NumHiddenLayers:       12,
					MaxPositionEmbeddings: 512,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 280 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"BAAI/bge-reranker-base"},
		Device:   memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "bert_rerank" || !plan.SupportedArchitecture {
		t.Fatalf("architecture support = %q %v", plan.Architecture, plan.SupportedArchitecture)
	}
	if plan.Embeddings || !plan.Rerank {
		t.Fatalf("task flags = embeddings:%v rerank:%v, want rerank scorer fit plan", plan.Embeddings, plan.Rerank)
	}
	if plan.ExpectedKVBytes != 0 || plan.MemoryPlan.PromptCache {
		t.Fatalf("rerank memory = kv:%d plan:%+v, want no generation KV cache", plan.ExpectedKVBytes, plan.MemoryPlan)
	}
}

func TestHfFit_PlanFits_MiniMaxJANGTQMemoryFit_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"dealignai/MiniMax-M2.7-JANGTQ-CRACK": {
				ID:   "dealignai/MiniMax-M2.7-JANGTQ-CRACK",
				Tags: []string{"mlx", "jang", "jangtq", "minimax_m2"},
				Config: ModelConfig{
					ModelType:             "minimax_m2",
					Architectures:         []string{"MiniMaxM2ForCausalLM"},
					HiddenSize:            3072,
					NumHiddenLayers:       62,
					NumAttentionHeads:     48,
					NumKeyValueHeads:      8,
					HeadDim:               128,
					MaxPositionEmbeddings: 196608,
					Quantization:          &QuantizationConfig{Bits: 8, GroupSize: 64, Type: "affine"},
				},
				Files: []ModelFile{
					{Name: "model-00001-of-00061.safetensors", Size: 60 * memory.GiB},
					{Name: "jangtq_runtime.safetensors", Size: 20 * 1024},
					{Name: "chat_template.jinja", Size: 6 * 1024},
				},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"dealignai/MiniMax-M2.7-JANGTQ-CRACK"},
		Device: memory.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	plan := report.Models[0]
	if plan.Architecture != "minimax_m2" || !plan.SupportedArchitecture {
		t.Fatalf("architecture support = %q/%v", plan.Architecture, plan.SupportedArchitecture)
	}
	if plan.QuantBits != 2 || plan.QuantType != "jangtq" || plan.QuantFamily != "jang" {
		t.Fatalf("quantization = bits:%d type:%q family:%q", plan.QuantBits, plan.QuantType, plan.QuantFamily)
	}
	if plan.NativeLoadable || !plan.MemoryFits || plan.InferenceFits {
		t.Fatalf("fit flags = native:%v memory:%v inference:%v, want staged native pack that still blocks standalone inference", plan.NativeLoadable, plan.MemoryFits, plan.InferenceFits)
	}
	// MiniMax M2 is an other-model arch not yet updated to declare its KV dims;
	// its context now derives from truth (the 60GB pack on the test box lands
	// below the 32768 arch cap via the hidden-size KV fallback). Assert a
	// positive derived context and the forced batch 1, not the old fixed cap.
	if plan.ContextRecommendation <= 0 || plan.MemoryPlan.BatchSize != 1 {
		t.Fatalf("context/batch = %d/%d, want a positive derived context and batch 1", plan.ContextRecommendation, plan.MemoryPlan.BatchSize)
	}
	if !hfFitPlanHasNote(plan, "staged") {
		t.Fatalf("Notes = %+v, want staged MiniMax M2 note", plan.Notes)
	}
}

// TestHfFit_PlanFits_FilenameQuantNotConsulted_Good: a misleading filename must
// NOT set quantisation. Quant is read from the model's declared config (or,
// post-download, the packed-tensor geometry) — never guessed from the file
// name. A base model that merely has "q4" in a filename is full precision until
// its config says otherwise.
func TestHfFit_PlanFits_FilenameQuantNotConsulted_Good(t *testing.T) {
	source := &fakeHFModelSource{
		search: []ModelMetadata{{
			ID: "Example/Base-Model",
			Config: ModelConfig{
				ModelType:             "qwen3",
				HiddenSize:            1024,
				NumHiddenLayers:       28,
				NumAttentionHeads:     16,
				NumKeyValueHeads:      8,
				MaxPositionEmbeddings: 40960,
				// No Quantization block — a full-precision base model.
			},
			Files: []ModelFile{
				{Name: "model-q4.safetensors", Size: 420 * 1024 * 1024},
				{Name: "tokenizer.json", Size: 4 * 1024 * 1024},
			},
		}},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		Query:      "qwen 0.6b",
		MaxResults: 5,
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m3-ultra",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	if got := report.Models[0].QuantBits; got != 0 {
		t.Fatalf("QuantBits = %d from a 'q4' filename, want 0 — the filename must not be consulted", got)
	}
}

// TestHfFit_PlanFits_MixedSourcesContextHint_Good combines a local-cache path
// with a remote model-id lookup in one PlanFits call and caps the context with
// ContextHint. Exercises the multi-source collectFitEntries merge plus the
// planFit ContextHint clamp (cfg.ContextHint < memoryPlan recommendation),
// which the single-source tests don't reach.
func TestHfFit_PlanFits_MixedSourcesContextHint_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--mlx-community--gemma-4-e2b-it-4bit")
	dir := core.PathJoin(cacheRoot, "snapshots", "snap1")
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", dir, result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"Qwen/Qwen3-0.6B": {
				ID: "Qwen/Qwen3-0.6B",
				Config: ModelConfig{
					ModelType:             "qwen3",
					HiddenSize:            1024,
					NumHiddenLayers:       28,
					NumAttentionHeads:     16,
					NumKeyValueHeads:      8,
					MaxPositionEmbeddings: 40960,
					Quantization:          &QuantizationConfig{Bits: 4, GroupSize: 64},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 420 * 1024 * 1024}},
			},
		},
	}

	const hint = 8192
	report, err := PlanFits(context.Background(), FitConfig{
		LocalPaths:  []string{cacheRoot},
		ModelIDs:    []string{"Qwen/Qwen3-0.6B"},
		ContextHint: hint,
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m3-ultra",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 2 {
		t.Fatalf("models = %d, want 2 (one local + one remote)", len(report.Models))
	}
	var sawLocal, sawRemote bool
	for _, plan := range report.Models {
		switch plan.Source {
		case SourceLocal:
			sawLocal = true
			if plan.ModelID != "mlx-community/gemma-4-e2b-it-4bit" {
				t.Fatalf("local ModelID = %q", plan.ModelID)
			}
		case SourceRemote:
			sawRemote = true
			if plan.ModelID != "Qwen/Qwen3-0.6B" {
				t.Fatalf("remote ModelID = %q", plan.ModelID)
			}
		}
		// ContextHint clamps every plan's recommendation down to the hint
		// (both models' native windows are far larger than 8192).
		if plan.ContextRecommendation != hint {
			t.Fatalf("%s ContextRecommendation = %d, want %d (ContextHint clamp)", plan.Source, plan.ContextRecommendation, hint)
		}
	}
	if !sawLocal || !sawRemote {
		t.Fatalf("entries = local:%v remote:%v, want both sources merged", sawLocal, sawRemote)
	}
}

// TestHfFit_PlanFits_QuerySearch_Good drives the Query-search path through the
// public PlanFits API — the ID-injected tests leave collectFitEntries'
// SearchModels branch otherwise uncovered. The injected source returns one
// model for the fixed "qwen 0.6b" query; no network.
func TestHfFit_PlanFits_QuerySearch_Good(t *testing.T) {
	source := &fakeHFModelSource{
		search: []ModelMetadata{
			{
				ID: "Qwen/Qwen3-0.6B",
				Config: ModelConfig{
					ModelType:             "qwen3",
					HiddenSize:            1024,
					NumHiddenLayers:       28,
					NumAttentionHeads:     16,
					NumKeyValueHeads:      8,
					MaxPositionEmbeddings: 40960,
					QuantizationConfig:    &QuantizationConfig{Bits: 4, GroupSize: 64},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 420 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		Query:      "qwen 0.6b",
		MaxResults: 5,
		Device:     memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 86 * memory.GiB},
		Source:     source,
	})
	if err != nil {
		t.Fatalf("PlanFits(query) error = %v", err)
	}
	if !source.searchCalled {
		t.Fatal("PlanFits(query) did not invoke SearchModels")
	}
	if len(report.Models) != 1 || report.Models[0].Source != SourceRemote {
		t.Fatalf("query report = %+v, want one remote model", report.Models)
	}
}

func TestHfFit_PlanFits_RequiresSourceForQuery_Bad(t *testing.T) {
	_, err := PlanFits(context.Background(), FitConfig{Query: "gemma"})
	if err == nil {
		t.Fatal("expected missing source error")
	}
	if !core.Contains(err.Error(), "source") {
		t.Fatalf("error = %v, want source context", err)
	}
}

func TestHfFit_PlanFits_ErrorPaths_Bad(t *testing.T) {
	if _, err := PlanFits(context.Background(), FitConfig{}); err == nil {
		t.Fatal("expected no metadata error")
	}
	if _, err := PlanFits(context.Background(), FitConfig{ModelIDs: []string{"qwen/model"}}); err == nil || !core.Contains(err.Error(), "source") {
		t.Fatalf("missing source error = %v", err)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := PlanFits(cancelled, FitConfig{LocalPaths: []string{t.TempDir()}})
	if err != context.Canceled {
		t.Fatalf("PlanFits(cancelled local) = %v, want context.Canceled", err)
	}

	badLocal := t.TempDir()
	writeModelPackFile(t, core.PathJoin(badLocal, "config.json"), "{")
	if _, err := PlanFits(context.Background(), FitConfig{LocalPaths: []string{badLocal}}); err == nil {
		t.Fatal("expected bad local config error")
	}
}

// TestHfFit_PlanFits_MissingLocalConfig_Bad asserts the local-inspect error
// path: a cache root whose snapshot directory exists but holds no config.json
// surfaces a read error rather than a partial plan.
func TestHfFit_PlanFits_MissingLocalConfig_Bad(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--no-config")
	snap := core.PathJoin(cacheRoot, "snapshots", "x")
	if result := core.MkdirAll(snap, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", snap, result.Value)
	}
	// Snapshot has a weight file but no config.json.
	writeModelPackFile(t, core.PathJoin(snap, "model.safetensors"), "stub")

	_, err := PlanFits(context.Background(), FitConfig{
		LocalPaths: []string{cacheRoot},
		Device:     memory.DeviceInfo{MemorySize: 16 * memory.GiB},
	})
	if err == nil {
		t.Fatal("expected a missing-config.json error for a config-less snapshot")
	}
	if !core.Contains(err.Error(), "config.json") {
		t.Fatalf("error = %v, want config.json context", err)
	}
}

// TestHfFit_PlanFits_QuerySearchError_Bad covers the SearchModels-error path in
// collectFitEntries: a query the injected source rejects propagates as a
// PlanFits error.
func TestHfFit_PlanFits_QuerySearchError_Bad(t *testing.T) {
	source := &fakeHFModelSource{} // SearchModels rejects anything but "qwen 0.6b"
	_, err := PlanFits(context.Background(), FitConfig{
		Query:  "unexpected query",
		Device: memory.DeviceInfo{MemorySize: 16 * memory.GiB},
		Source: source,
	})
	if err == nil {
		t.Fatal("PlanFits(bad query) error = nil, want SearchModels error propagated")
	}
}

func TestHfFit_PlanFits_UnsupportedArchitecture_Ugly(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"future/model": {
				ID: "future/model",
				Config: ModelConfig{
					ModelType:             "future_arch",
					HiddenSize:            4096,
					NumHiddenLayers:       32,
					NumAttentionHeads:     32,
					MaxPositionEmbeddings: 32768,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 30 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"future/model"},
		Device:   memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 12 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	plan := report.Models[0]
	if plan.SupportedArchitecture || plan.NativeLoadable {
		t.Fatalf("unsupported model marked loadable: %+v", plan)
	}
	if plan.InferenceFits {
		t.Fatalf("InferenceFits = true for oversized unsupported model: %+v", plan)
	}
	if len(plan.Notes) == 0 {
		t.Fatal("expected explanatory notes for unsupported/oversized model")
	}
}

func TestHFModelFitHelpers_Ugly(t *testing.T) {
	files := []ModelFile{
		{Name: "model-q4.gguf", Size: 10},
		{RFilename: "model.safetensors", SizeBytes: 20},
		{Name: "pytorch_model.bin", Size: 30},
	}
	format, bytes := weightFormatAndBytes(files)
	if format != string(mp.ModelPackFormatMixed) || bytes != 60 {
		t.Fatalf("weightFormatAndBytes = %q/%d, want mixed/60", format, bytes)
	}
	config := ModelConfig{HiddenSize: 128, NumHiddenLayers: 2, NumAttentionHeads: 4, NumKeyValueHeads: 2}
	if got := estimateModelKVBytes(config, 16, 2, 2); got != 16384 {
		t.Fatalf("estimateModelKVBytes(GQA) = %d, want 16384", got)
	}
	if got := estimateModelKVBytes(ModelConfig{HiddenSize: 128, NumHiddenLayers: 2}, 16, 0, 0); got != 16384 {
		t.Fatalf("estimateModelKVBytes(hidden fallback) = %d, want 16384", got)
	}
	if got := estimateModelKVBytes(ModelConfig{}, 16, 1, 2); got != 0 {
		t.Fatalf("estimateModelKVBytes(empty) = %d, want 0", got)
	}
	if got := estimateRuntimeOverheadBytes(0); got != 0 {
		t.Fatalf("estimateRuntimeOverheadBytes(0) = %d, want 0", got)
	}
	if got := estimateRuntimeOverheadBytes(2 * memory.GiB); got != memory.GiB {
		t.Fatalf("estimateRuntimeOverheadBytes(small) = %d, want 1GiB", got)
	}

	plan := FitPlan{
		NativeLoadable:       true,
		InferenceFits:        true,
		QuantBits:            16,
		WeightBytes:          100,
		ExpectedKVBytes:      10,
		ExpectedRuntimeBytes: 10,
		ExpectedTotalBytes:   120,
	}
	fit := estimateTrainingFit(ModelConfig{HiddenSize: 8, NumHiddenLayers: 2}, plan, 0, -1)
	if !fit.LoRAFeasible || !fit.FullFineTuneFeasible || fit.RecommendedLoRARank != 16 {
		t.Fatalf("training fit = %+v", fit)
	}
	if got := positiveInt(-3); got != 0 {
		t.Fatalf("positiveInt(-3) = %d, want 0", got)
	}
	if err := fitResultError(core.Result{Value: "bad", OK: false}); err == nil || !core.Contains(err.Error(), "core result failed") {
		t.Fatalf("fitResultError(non-error) = %v", err)
	}
}

// TestWeightFormatAndBytes_Good covers weightFormatAndBytes across its format
// branches: a pure safetensors set, a pure GGUF set, a mixed set (safetensors
// + gguf collapses to "mixed"), a .bin set, and the empty-input early return.
// byteSize sums only recognised weight files; the RFilename fallback in
// filename() is exercised by a sibling-only entry.
func TestWeightFormatAndBytes_Good(t *testing.T) {
	safet := []ModelFile{
		{Name: "model-00001-of-00002.safetensors", Size: 100},
		{RFilename: "model-00002-of-00002.safetensors", SizeBytes: 200},
	}
	if format, total := weightFormatAndBytes(safet); format != "safetensors" || total != 300 {
		t.Fatalf("safetensors = %q/%d, want safetensors/300 (RFilename + SizeBytes fallbacks)", format, total)
	}

	ggufFiles := []ModelFile{{Name: "model.Q4_K_M.gguf", Size: 500}}
	if format, total := weightFormatAndBytes(ggufFiles); format != "gguf" || total != 500 {
		t.Fatalf("gguf = %q/%d, want gguf/500", format, total)
	}

	mixed := []ModelFile{
		{Name: "model.safetensors", Size: 10},
		{Name: "model.gguf", Size: 20},
	}
	if format, total := weightFormatAndBytes(mixed); format != "mixed" || total != 30 {
		t.Fatalf("mixed = %q/%d, want mixed/30", format, total)
	}

	binFiles := []ModelFile{{Name: "pytorch_model.bin", Size: 42}}
	if format, total := weightFormatAndBytes(binFiles); format != "bin" || total != 42 {
		t.Fatalf("bin = %q/%d, want bin/42", format, total)
	}

	if format, total := weightFormatAndBytes(nil); format != "" || total != 0 {
		t.Fatalf("empty = %q/%d, want empty/0", format, total)
	}
}

// TestFitNotes_Branches_Ugly covers the fitNotes advisory branches that the
// PlanFits integration tests do not all reach: the gemma4_assistant and
// minimax_m2 non-standalone-native messages, the unknown-weight-bytes note,
// the over-budget note, and the context-capped note. fitNotes is package-
// internal so we drive it with hand-built FitPlans rather than a full plan.
func TestFitNotes_Branches_Ugly(t *testing.T) {
	hasNote := func(notes []string, fragment string) bool {
		for _, n := range notes {
			if core.Contains(n, fragment) {
				return true
			}
		}
		return false
	}

	// gemma4_assistant attached-drafter note (nonStandaloneNative=true).
	assistant := FitPlan{SupportedArchitecture: true, Architecture: "gemma4_assistant", WeightBytes: 100}
	notes := fitNotes(assistant, 0, true, true)
	if !hasNote(notes, "LoadSpeculativePair") {
		t.Fatalf("gemma4_assistant notes = %v, want attached-drafter guidance", notes)
	}

	// minimax_m2 staged-loader note.
	minimax := FitPlan{SupportedArchitecture: true, Architecture: "minimax_m2", WeightBytes: 100}
	if notes := fitNotes(minimax, 0, true, true); !hasNote(notes, "JANGTQ/MXTQ") {
		t.Fatalf("minimax_m2 notes = %v, want staged-loader note", notes)
	}

	// default non-standalone-native note for an arch with no special case.
	other := FitPlan{SupportedArchitecture: true, Architecture: "bert_rerank", WeightBytes: 100}
	if notes := fitNotes(other, 0, true, true); !hasNote(notes, "not a standalone generation target") {
		t.Fatalf("default native-asset notes = %v, want generic non-standalone note", notes)
	}

	// unknown weight bytes + over budget + context capped, all at once.
	multi := FitPlan{
		SupportedArchitecture: true,
		WeightBytes:           0,
		ExpectedTotalBytes:    200,
		ContextLimit:          8192,
		ContextRecommendation: 4096,
	}
	notes = fitNotes(multi, 100, true, false)
	if !hasNote(notes, "weight byte size is unknown") {
		t.Fatalf("notes = %v, want unknown-weight note", notes)
	}
	if !hasNote(notes, "exceeds local working-set budget") {
		t.Fatalf("notes = %v, want over-budget note", notes)
	}
	if !hasNote(notes, "capped by local machine class") {
		t.Fatalf("notes = %v, want context-capped note", notes)
	}

	// Zero notes → nil (no advisory conditions).
	clean := FitPlan{SupportedArchitecture: true, WeightBytes: 100}
	if notes := fitNotes(clean, 0, true, false); notes != nil {
		t.Fatalf("clean plan notes = %v, want nil", notes)
	}
}

// TestLocalModelFiles_SyntheticDir_Good covers localModelFiles and
// isLocalModelFileName against a synthetic snapshot directory: it surfaces
// safetensors/gguf/bin weights and the two tokenizer files, skips
// sub-directories and unrelated files, and reads each entry's size — no
// network, fixtures via t.TempDir().
func TestLocalModelFiles_SyntheticDir_Good(t *testing.T) {
	root := t.TempDir()
	writeModelPackFile(t, core.PathJoin(root, "model.safetensors"), "weights")
	writeModelPackFile(t, core.PathJoin(root, "model.gguf"), "gg")
	writeModelPackFile(t, core.PathJoin(root, "pytorch_model.bin"), "bin")
	writeModelPackFile(t, core.PathJoin(root, "tokenizer.json"), "{}")
	writeModelPackFile(t, core.PathJoin(root, "tokenizer_config.json"), "{}")
	writeModelPackFile(t, core.PathJoin(root, "README.md"), "ignored")
	writeModelPackFile(t, core.PathJoin(root, "config.json"), "{}") // not a weight/tokenizer name
	if result := core.MkdirAll(core.PathJoin(root, "subdir"), 0o755); !result.OK {
		t.Fatalf("mkdir subdir: %v", result.Value)
	}

	files := localModelFiles(root)
	got := make(map[string]uint64, len(files))
	for _, f := range files {
		got[f.Name] = f.Size
	}
	for _, want := range []string{"model.safetensors", "model.gguf", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"} {
		if _, ok := got[want]; !ok {
			t.Fatalf("localModelFiles missing %q; got %v", want, got)
		}
	}
	if _, ok := got["README.md"]; ok {
		t.Fatal("localModelFiles surfaced README.md, want it skipped")
	}
	if _, ok := got["config.json"]; ok {
		t.Fatal("localModelFiles surfaced config.json, want it skipped (not a weight/tokenizer name)")
	}
	if got["model.safetensors"] != uint64(len("weights")) {
		t.Fatalf("model.safetensors size = %d, want %d", got["model.safetensors"], len("weights"))
	}
}

// TestLocalModelFiles_MissingDir_Bad covers the ReadDir-failure early return:
// a non-existent root yields an empty (non-nil) slice rather than an error.
func TestLocalModelFiles_MissingDir_Bad(t *testing.T) {
	files := localModelFiles(core.PathJoin(t.TempDir(), "does-not-exist"))
	if len(files) != 0 {
		t.Fatalf("localModelFiles(missing) = %v, want empty", files)
	}
}

// TestLocalModelID_FromCacheLayout_Good covers localModelID: the
// HuggingFace `models--org--name` cache-directory convention decodes to
// `org/name`, walking up from the input path when the root itself is not the
// models-- directory.
func TestLocalModelID_FromCacheLayout_Good(t *testing.T) {
	base := t.TempDir()
	cacheRoot := core.PathJoin(base, "models--mlx-community--gemma-4-e2b-it-4bit")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "abc123")
	if got := localModelID(snapshot, cacheRoot); got != "mlx-community/gemma-4-e2b-it-4bit" {
		t.Fatalf("localModelID = %q, want mlx-community/gemma-4-e2b-it-4bit", got)
	}

	// No models-- segment anywhere → fall back to the root's base name.
	plain := core.PathJoin(base, "my-local-model")
	if got := localModelID(plain, plain); got != "my-local-model" {
		t.Fatalf("localModelID(no cache prefix) = %q, want my-local-model", got)
	}
}
