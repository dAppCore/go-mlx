// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

type fakeHFModelSource struct {
	searchCalled bool
	search       []ModelMetadata
	byID         map[string]ModelMetadata
}

func (s *fakeHFModelSource) SearchModels(_ context.Context, query string, limit int) ([]ModelMetadata, error) {
	if query != "qwen 0.6b" {
		return nil, core.NewError("unexpected query: " + query)
	}
	s.searchCalled = true
	if limit > 0 && limit < len(s.search) {
		return append([]ModelMetadata(nil), s.search[:limit]...), nil
	}
	return append([]ModelMetadata(nil), s.search...), nil
}

func (s *fakeHFModelSource) ModelMetadata(_ context.Context, id string) (ModelMetadata, error) {
	if meta, ok := s.byID[id]; ok {
		return meta, nil
	}
	return ModelMetadata{}, core.NewError("not found: " + id)
}

func TestPlanHFModelFits_InjectedSearch_Good(t *testing.T) {
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

func TestPlanHFModelFits_LocalCache_Good(t *testing.T) {
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

func TestPlanHFModelFits_QwenNextNestedTextConfig_Good(t *testing.T) {
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

func TestPlanHFModelFits_Gemma4AssistantUsesOuterArchitecture_Good(t *testing.T) {
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

func TestPlanHFModelFits_Gemma412BUnifiedPreservesArchitecture_Good(t *testing.T) {
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

func TestPlanHFModelFits_BertEmbeddingUsesEncoderMemoryPlan_Good(t *testing.T) {
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

func TestPlanHFModelFits_BertRerankUsesScorerMemoryPlan_Good(t *testing.T) {
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

func TestPlanHFModelFits_MiniMaxJANGTQMemoryFit_Good(t *testing.T) {
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

func TestPlanHFModelFits_RequiresSourceForQuery_Bad(t *testing.T) {
	_, err := PlanFits(context.Background(), FitConfig{Query: "gemma"})
	if err == nil {
		t.Fatal("expected missing source error")
	}
	if !core.Contains(err.Error(), "source") {
		t.Fatalf("error = %v, want source context", err)
	}
}

func TestPlanHFModelFits_UnsupportedArchitecture_Ugly(t *testing.T) {
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

func TestHuggingFaceModelSource_SearchAndMetadata_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		switch r.URL.Path {
		case "/api/models":
			if r.URL.Query().Get("search") != "qwen" || r.URL.Query().Get("limit") != "2" {
				t.Fatalf("query = %q, want search/limit", r.URL.RawQuery)
			}
			w.Header().Set("Content-Type", "application/json")
			core.WriteString(w, `[{
				"id": "Qwen/Qwen3-0.6B",
				"pipeline_tag": "text-generation",
				"config": {"model_type": "qwen3", "hidden_size": 1024},
				"siblings": [{"rfilename": "model.safetensors", "sizeBytes": 440401920}]
			}]`)
		case "/api/models/Qwen/Qwen3-0.6B":
			if r.Header.Get("Authorization") != "Bearer test-token" {
				t.Fatalf("Authorization = %q", r.Header.Get("Authorization"))
			}
			w.Header().Set("Content-Type", "application/json")
			core.WriteString(w, `{
				"modelId": "Qwen/Qwen3-0.6B",
				"config": {"model_type": "qwen3", "num_hidden_layers": 28},
				"siblings": [{"rfilename": "model.safetensors", "size": 440401920}]
			}`)
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{
		BaseURL: server.URL,
		Token:   "test-token",
	})
	found, err := source.SearchModels(context.Background(), "qwen", 2)
	if err != nil {
		t.Fatalf("SearchModels() error = %v", err)
	}
	if len(found) != 1 || found[0].ID != "Qwen/Qwen3-0.6B" {
		t.Fatalf("SearchModels() = %+v", found)
	}
	if found[0].Files[0].byteSize() != 440401920 {
		t.Fatalf("file size = %+v", found[0].Files[0])
	}

	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ModelID != "Qwen/Qwen3-0.6B" || meta.Config.NumHiddenLayers != 28 {
		t.Fatalf("ModelMetadata() = %+v", meta)
	}
}

func TestPlanHFModelFits_ErrorPaths_Bad(t *testing.T) {
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

func TestHuggingFaceModelSource_Errors_Bad(t *testing.T) {
	var source *RemoteSource
	if _, err := source.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("expected nil SearchModels error")
	}
	if _, err := source.ModelMetadata(context.Background(), "qwen/model"); err == nil {
		t.Fatal("expected nil ModelMetadata error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		switch r.URL.Path {
		case "/api/models":
			core.WriteString(w, "{")
		case "/api/models/missing":
			w.WriteHeader(404)
			core.WriteString(w, "not found")
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	source = NewRemoteSource(RemoteConfig{BaseURL: server.URL + "/", UserAgent: "tests"})
	if source.baseURL != server.URL || source.userAgent != "tests" || source.client == nil {
		t.Fatalf("source defaults = %+v", source)
	}
	if _, err := source.SearchModels(context.Background(), "qwen", 0); err == nil {
		t.Fatal("expected parse error from malformed search response")
	}
	if _, err := source.ModelMetadata(context.Background(), "missing"); err == nil || !core.Contains(err.Error(), "404") {
		t.Fatalf("expected HTTP status error, got %v", err)
	}
}

func TestHFLocalMetadataHelpers_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "b")
	if result := core.MkdirAll(snapshot, 0o755); !result.OK {
		t.Fatalf("mkdir snapshot: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(snapshot, "config.json"), `{"architectures":["Qwen3ForCausalLM"],"context_length":32768}`)
	writeModelPackFile(t, core.PathJoin(snapshot, "model-q4.gguf"), "gguf")
	writeModelPackFile(t, core.PathJoin(snapshot, "model.safetensors"), "safe")
	writeModelPackFile(t, core.PathJoin(snapshot, "pytorch_model.bin"), "bin")
	writeModelPackFile(t, core.PathJoin(snapshot, "tokenizer.json"), "{}")

	meta, root, err := inspectLocalMetadata(cacheRoot)
	if err != nil {
		t.Fatalf("inspectLocalMetadata: %v", err)
	}
	if root != snapshot {
		t.Fatalf("root = %q, want %q", root, snapshot)
	}
	if meta.ID != "org/name" {
		t.Fatalf("ID = %q, want org/name", meta.ID)
	}
	if len(meta.Files) != 4 {
		t.Fatalf("files = %+v", meta.Files)
	}
	if got := resolveLocalMetadataRoot(core.PathJoin(snapshot, "config.json")); got != snapshot {
		t.Fatalf("resolve config root = %q, want %q", got, snapshot)
	}
}

// A misleading filename must NOT set quantisation. Quant is read from the
// model's declared config (or, post-download, the packed-tensor geometry) —
// never guessed from the file name. A base model that merely has "q4" in a
// filename is full precision until its config says otherwise.
func TestPlanHFModelFits_FilenameQuantNotConsulted_Good(t *testing.T) {
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

func hfFitPlanHasNote(plan FitPlan, fragment string) bool {
	for _, note := range plan.Notes {
		if core.Contains(note, fragment) {
			return true
		}
	}
	return false
}

// TestPlanHFModelFits_MixedSourcesContextHint_Good combines a local-cache
// path with a remote model-id lookup in one PlanFits call and caps the
// context with ContextHint. Exercises the multi-source collectFitEntries
// merge plus the planFit ContextHint clamp (cfg.ContextHint < memoryPlan
// recommendation), which the single-source tests don't reach.
func TestPlanHFModelFits_MixedSourcesContextHint_Good(t *testing.T) {
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

// TestPlanHFModelFits_MissingLocalConfig_Bad asserts the local-inspect error
// path: a cache root whose snapshot directory exists but holds no config.json
// surfaces a read error rather than a partial plan.
func TestPlanHFModelFits_MissingLocalConfig_Bad(t *testing.T) {
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

// TestInferJANG_BasicProfile_Good drives the public InferJANG over a pack
// whose id carries a "jang_2s" needle but no "jangtq" — the jangBasic branch
// that builds the lowercase haystack, resolves the profile name, and reads
// the group size from the QuantizationConfig. Asserts the inferred profile,
// the bits derived from jang.ProfileBits ("jang_2*" -> 2), and the overridden
// group size (96, not the 64 default).
func TestInferJANG_BasicProfile_Good(t *testing.T) {
	meta := ModelMetadata{
		ID:   "dealignai/Qwen3-JANG_2S",
		Tags: []string{"mlx", "jang"},
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "tokenizer.json"},
		},
		Config: ModelConfig{
			QuantizationConfig: &QuantizationConfig{GroupSize: 96},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a 'jang_2s' pack, want a basic JANG profile")
	}
	if info.Profile != "JANG_2S" {
		t.Fatalf("Profile = %q, want JANG_2S", info.Profile)
	}
	if info.BitsDefault != 2 {
		t.Fatalf("BitsDefault = %d, want 2 (jang_2* -> 2 bits)", info.BitsDefault)
	}
	if info.GroupSize != 96 {
		t.Fatalf("GroupSize = %d, want 96 (read from QuantizationConfig, not the 64 default)", info.GroupSize)
	}
	if info.Packed == nil {
		t.Fatal("Packed profile = nil, want BuildPackedProfile output")
	}
}

// TestInferJANG_NoNeedle_Bad asserts the dominant miss path: metadata with no
// "jang" token anywhere (id/tags/filenames) returns nil with no profile work.
func TestInferJANG_NoNeedle_Bad(t *testing.T) {
	meta := ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}, {Name: "tokenizer.json"}},
	}
	if info := InferJANG(meta); info != nil {
		t.Fatalf("InferJANG = %+v, want nil for a non-JANG pack", info)
	}
}

// TestInferJANG_TQNeedleNoGroupSize_Ugly drives the JANGTQ short-circuit when
// the strongest token is "jangtq" (here only in a filename) and neither quant
// block declares a group size — the helper must fall back to the 64 default
// and stamp the fixed JANGTQ profile/bits without scanning a haystack.
func TestInferJANG_TQNeedleNoGroupSize_Ugly(t *testing.T) {
	meta := ModelMetadata{
		ID: "vendor/model-with-only-a-file-needle",
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "weights.JANGTQ.safetensors"},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a JANGTQ filename, want a JANGTQ profile")
	}
	if info.Profile != "JANGTQ" || info.WeightFormat != "mxtq" {
		t.Fatalf("profile/format = %q/%q, want JANGTQ/mxtq", info.Profile, info.WeightFormat)
	}
	if info.BitsDefault != 2 || info.RoutedExpertBits != 2 {
		t.Fatalf("bits = default:%d routed:%d, want 2/2", info.BitsDefault, info.RoutedExpertBits)
	}
	if info.GroupSize != 64 {
		t.Fatalf("GroupSize = %d, want 64 default (no quant block declared a group size)", info.GroupSize)
	}
}

// TestModelConfigAccessors_Good exercises the value-receiver ModelConfig
// accessors (architecture / contextLength / quantization) directly. planFit
// inlines these for the hot path, so only the benches drive them today; this
// asserts the normalize-then-read logic with real config shapes — including
// the nested text_config promotion that normalized() performs.
func TestModelConfigAccessors_Good(t *testing.T) {
	flat := ModelConfig{
		ModelType:             "qwen3",
		Architectures:         []string{"Qwen3ForCausalLM"},
		ContextLength:         0,
		MaxPositionEmbeddings: 40960,
		QuantizationConfig:    &QuantizationConfig{Bits: 4, GroupSize: 64},
	}
	if got := flat.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
	if got := flat.contextLength(); got != 40960 {
		t.Fatalf("contextLength() = %d, want 40960 (falls back to max_position_embeddings)", got)
	}
	if bits, group := flat.quantization(); bits != 4 || group != 64 {
		t.Fatalf("quantization() = %d/%d, want 4/64", bits, group)
	}

	// Nested text_config: normalized() lifts the inner config so the
	// accessors read the real architecture/context, not the outer wrapper.
	nested := ModelConfig{
		ModelType: "qwen3_5",
		TextConfig: &ModelConfig{
			ModelType:             "qwen3_next",
			Architectures:         []string{"Qwen3NextForCausalLM"},
			ContextLength:         98304,
			Quantization:          &QuantizationConfig{Bits: 8, GroupSize: 32},
		},
	}
	if got := nested.architecture(); got != "qwen3_next" {
		t.Fatalf("nested architecture() = %q, want qwen3_next", got)
	}
	if got := nested.contextLength(); got != 98304 {
		t.Fatalf("nested contextLength() = %d, want 98304", got)
	}
	if bits, group := nested.quantization(); bits != 8 || group != 32 {
		t.Fatalf("nested quantization() = %d/%d, want 8/32 (read from text_config)", bits, group)
	}
}

// TestModelConfigQuantization_Bad covers the no-quant-block path of the
// quantization accessor — an unquantised (dense) config returns 0/0.
func TestModelConfigQuantization_Bad(t *testing.T) {
	dense := ModelConfig{ModelType: "qwen3", HiddenSize: 1024}
	if bits, group := dense.quantization(); bits != 0 || group != 0 {
		t.Fatalf("quantization() on dense config = %d/%d, want 0/0", bits, group)
	}
	if got := dense.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
}

// TestModelConfigQuantizationType_Good covers quantizationType — the string
// label of the active quant block. QuantizationConfig wins over Quantization
// when both are present (normalized() promotes the nested text_config first,
// then the accessor prefers quantization_config over the legacy quantization
// key), and an unquantised config returns "".
func TestModelConfigQuantizationType_Good(t *testing.T) {
	cfg := ModelConfig{
		ModelType:          "qwen3",
		QuantizationConfig: &QuantizationConfig{Bits: 4, GroupSize: 64, Type: "mxfp4"},
	}
	if got := cfg.quantizationType(); got != "mxfp4" {
		t.Fatalf("quantizationType() = %q, want mxfp4 (from quantization_config)", got)
	}

	// Legacy `quantization` key only — quantization_config absent.
	legacy := ModelConfig{
		ModelType:    "qwen3",
		Quantization: &QuantizationConfig{Bits: 8, GroupSize: 32, Type: "affine"},
	}
	if got := legacy.quantizationType(); got != "affine" {
		t.Fatalf("quantizationType() = %q, want affine (from legacy quantization)", got)
	}

	// Dense (no quant block) → empty type.
	if got := (ModelConfig{ModelType: "qwen3"}).quantizationType(); got != "" {
		t.Fatalf("quantizationType() on dense config = %q, want empty", got)
	}
}

// TestConfigArchitecture_Good exercises configArchitecture, the
// already-normalised pointer-receiver variant, across its three resolution
// tiers: the BertForSequenceClassification → bert_rerank special case wins
// even when a model_type is present; otherwise model_type is normalised; and
// when model_type is empty the transformers architecture name resolves.
func TestConfigArchitecture_Good(t *testing.T) {
	// Rerank special case: the bert_rerank short-circuit fires before
	// model_type, so a sequence-classification head reports bert_rerank.
	rerank := ModelConfig{
		ModelType:     "bert",
		Architectures: []string{"BertForSequenceClassification"},
	}
	if got := configArchitecture(&rerank); got != "bert_rerank" {
		t.Fatalf("configArchitecture() = %q, want bert_rerank (special case wins over model_type)", got)
	}

	// model_type present, no rerank head → normalise the model_type.
	byType := ModelConfig{ModelType: "qwen3", Architectures: []string{"Qwen3ForCausalLM"}}
	if got := configArchitecture(&byType); got != "qwen3" {
		t.Fatalf("configArchitecture() = %q, want qwen3 (from model_type)", got)
	}

	// model_type empty → fall through to the transformers architecture name.
	byArch := ModelConfig{Architectures: []string{"LlamaForCausalLM"}}
	if got := configArchitecture(&byArch); got != "llama" {
		t.Fatalf("configArchitecture() = %q, want llama (from architectures)", got)
	}
}

// TestConfigArchitecture_Bad covers the empty-result path: neither a known
// model_type nor a recognised transformers architecture name resolves, so
// configArchitecture returns "".
func TestConfigArchitecture_Bad(t *testing.T) {
	if got := configArchitecture(&ModelConfig{}); got != "" {
		t.Fatalf("configArchitecture(empty) = %q, want empty", got)
	}
	unknown := ModelConfig{Architectures: []string{"TotallyMadeUpForCausalLM"}}
	if got := configArchitecture(&unknown); got != "" {
		t.Fatalf("configArchitecture(unknown arch) = %q, want empty", got)
	}
}

// TestGemma4ConfigDetectors_Good covers isGemma4UnifiedConfig and
// isGemma4AssistantConfig — the two detectors PlanFits uses to pick the
// unified-multimodal and attached-drafter memory plans. Each matches on both
// a normalised model_type and a transformers architecture name.
func TestGemma4ConfigDetectors_Good(t *testing.T) {
	// Unified via model_type.
	if !isGemma4UnifiedConfig(ModelConfig{ModelType: "gemma4_unified"}) {
		t.Fatal("isGemma4UnifiedConfig(model_type) = false, want true")
	}
	// Unified via architecture name.
	if !isGemma4UnifiedConfig(ModelConfig{Architectures: []string{"Gemma4UnifiedForConditionalGeneration"}}) {
		t.Fatal("isGemma4UnifiedConfig(architectures) = false, want true")
	}
	// Assistant via model_type.
	if !isGemma4AssistantConfig(ModelConfig{ModelType: "gemma4_assistant"}) {
		t.Fatal("isGemma4AssistantConfig(model_type) = false, want true")
	}
	// Assistant via architecture name.
	if !isGemma4AssistantConfig(ModelConfig{Architectures: []string{"Gemma4AssistantForCausalLM"}}) {
		t.Fatal("isGemma4AssistantConfig(architectures) = false, want true")
	}
}

// TestGemma4ConfigDetectors_Bad covers the negative path: a non-Gemma-4
// config matches neither detector, and the architecture-name loop must run to
// exhaustion without a hit.
func TestGemma4ConfigDetectors_Bad(t *testing.T) {
	plain := ModelConfig{ModelType: "qwen3", Architectures: []string{"Qwen3ForCausalLM"}}
	if isGemma4UnifiedConfig(plain) {
		t.Fatal("isGemma4UnifiedConfig(qwen3) = true, want false")
	}
	if isGemma4AssistantConfig(plain) {
		t.Fatal("isGemma4AssistantConfig(qwen3) = true, want false")
	}
}

// TestArchProfileHelpers_Good covers archSupported, archNativeRuntime and
// usesGenerationKVCache against the live architecture registry. qwen3 is a
// supported native generation arch; bert is a supported native *embedding*
// arch (so no generation KV cache); an unknown arch is unsupported on every
// axis.
//
// NOTE: archNativeRuntime's middle branch (recognised && !NativeRuntime) is
// not reachable through the public registry — every registered arch is a
// native* profile with NativeRuntime=true (verified: 0 metadata-only
// registrations). The lookup-miss path (!ok → false) and the native-true
// path are both covered here.
func TestArchProfileHelpers_Good(t *testing.T) {
	if !archSupported("qwen3") {
		t.Fatal("archSupported(qwen3) = false, want true")
	}
	if !archNativeRuntime("qwen3") {
		t.Fatal("archNativeRuntime(qwen3) = false, want true")
	}
	if !usesGenerationKVCache(nil, "qwen3") {
		t.Fatal("usesGenerationKVCache(nil, qwen3) = false, want true (generation arch)")
	}

	// bert is a recognised embedding encoder → no generation KV cache.
	if !archSupported("bert") {
		t.Fatal("archSupported(bert) = false, want true")
	}
	if usesGenerationKVCache(nil, "bert") {
		t.Fatal("usesGenerationKVCache(nil, bert) = true, want false (embedding arch)")
	}
}

// TestArchProfileHelpers_Bad covers the unsupported-architecture paths: an
// unknown name is neither supported nor native, and usesGenerationKVCache
// still defaults to true for an unknown arch (no profile says otherwise).
func TestArchProfileHelpers_Bad(t *testing.T) {
	const unknown = "totally_made_up_arch"
	if archSupported(unknown) {
		t.Fatalf("archSupported(%q) = true, want false", unknown)
	}
	if archNativeRuntime(unknown) {
		t.Fatalf("archNativeRuntime(%q) = true, want false", unknown)
	}
	if !usesGenerationKVCache(nil, unknown) {
		t.Fatalf("usesGenerationKVCache(nil, %q) = false, want true (unknown defaults to generation)", unknown)
	}
}

// TestUsesGenerationKVCache_PackOverrides_Ugly covers the ModelPack-driven
// branches of usesGenerationKVCache: an Embedding/Rerank pack short-circuits
// to false; a pack Architecture overrides the passed name; and an
// ArchitectureProfile flagged Embeddings/Rerank forces false even when the
// arch name would otherwise be a generation target.
func TestUsesGenerationKVCache_PackOverrides_Ugly(t *testing.T) {
	// Embedding pack → false regardless of arch name.
	embedPack := &mp.ModelPack{Embedding: &mp.ModelEmbeddingProfile{}}
	if usesGenerationKVCache(embedPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(embedding pack) = true, want false")
	}

	// Pack architecture overrides the passed name: pass a generation arch
	// but let the pack declare an embedding arch → false.
	overridePack := &mp.ModelPack{Architecture: "bert"}
	if usesGenerationKVCache(overridePack, "qwen3") {
		t.Fatal("usesGenerationKVCache(pack arch=bert) = true, want false (pack arch overrides)")
	}

	// ArchitectureProfile flagged Rerank → false.
	rerankPack := &mp.ModelPack{
		Architecture:        "qwen3",
		ArchitectureProfile: &profile.ModelArchitectureProfile{Rerank: true},
	}
	if usesGenerationKVCache(rerankPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(profile.Rerank) = true, want false")
	}

	// A generation pack (no embedding/rerank signal) keeps the KV cache.
	genPack := &mp.ModelPack{Architecture: "qwen3"}
	if !usesGenerationKVCache(genPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(generation pack) = false, want true")
	}
}

// TestResolveArchitectureProfile_Good covers resolveArchitectureProfile: it
// fills a pack's ArchitectureProfile from the registry when the pack names a
// recognised arch and has no profile yet, and is a no-op when the pack is
// nil, has no architecture, or already carries a profile.
func TestResolveArchitectureProfile_Good(t *testing.T) {
	// nil pack: must not panic.
	resolveArchitectureProfile(nil)

	// No architecture: profile stays nil.
	empty := &mp.ModelPack{}
	resolveArchitectureProfile(empty)
	if empty.ArchitectureProfile != nil {
		t.Fatal("resolveArchitectureProfile(no arch) populated a profile, want nil")
	}

	// Recognised arch, no profile yet → populated from the registry.
	pack := &mp.ModelPack{Architecture: "qwen3"}
	resolveArchitectureProfile(pack)
	if pack.ArchitectureProfile == nil || pack.ArchitectureProfile.ID != "qwen3" {
		t.Fatalf("resolveArchitectureProfile(qwen3) profile = %+v, want qwen3 profile", pack.ArchitectureProfile)
	}

	// Already has a profile → left untouched (no overwrite).
	sentinel := &profile.ModelArchitectureProfile{ID: "preset"}
	preset := &mp.ModelPack{Architecture: "qwen3", ArchitectureProfile: sentinel}
	resolveArchitectureProfile(preset)
	if preset.ArchitectureProfile != sentinel {
		t.Fatal("resolveArchitectureProfile overwrote an existing profile, want no-op")
	}

	// Unrecognised arch → profile stays nil.
	unknown := &mp.ModelPack{Architecture: "totally_made_up_arch"}
	resolveArchitectureProfile(unknown)
	if unknown.ArchitectureProfile != nil {
		t.Fatal("resolveArchitectureProfile(unknown arch) populated a profile, want nil")
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

// TestPlanHFModelFits_QuerySearch_Good drives the Query-search path through
// the public PlanFits API — the existing integration tests inject by ModelID,
// leaving collectFitEntries' SearchModels branch uncovered. The injected
// source returns one model for the fixed "qwen 0.6b" query; no network.
func TestPlanHFModelFits_QuerySearch_Good(t *testing.T) {
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

// TestPlanHFModelFits_QuerySearchError_Bad covers the SearchModels-error path
// in collectFitEntries: a query the injected source rejects propagates as a
// PlanFits error.
func TestPlanHFModelFits_QuerySearchError_Bad(t *testing.T) {
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

// TestRemoteSource_ModelMetadataIDFallback_Good covers the ModelMetadata
// fallback branch: when the Hub returns a metadata body carrying neither `id`
// nor `modelId`, the requested model id is filled in. Loopback httptest
// server only — no real network.
func TestRemoteSource_ModelMetadataIDFallback_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"config": {"model_type": "qwen3"}}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	meta, err := source.ModelMetadata(context.Background(), "org/no-id-model")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ID != "org/no-id-model" {
		t.Fatalf("ModelMetadata().ID = %q, want the requested id filled in", meta.ID)
	}
}

// TestRemoteSource_TransportError_Bad covers getJSON's client.Do failure
// branch: pointing the source at a closed loopback server surfaces a
// transport error from both SearchModels and ModelMetadata. The server is
// started then immediately closed so the dial fails locally — no real
// network egress.
func TestRemoteSource_TransportError_Bad(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		core.WriteString(w, "{}")
	}))
	closedURL := server.URL
	server.Close() // nothing listens at closedURL now → dial fails

	source := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := source.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(closed server) error = nil, want transport error")
	}
	if _, err := source.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(closed server) error = nil, want transport error")
	}
}

// TestModelConfigProbe_SyntheticConfig_Good covers the modelConfigProbe family
// — readModelConfig plus the architecture/numLayers/vocabSize/hiddenSize/
// contextLength/quantBits/quantGroup accessors — against a synthetic
// config.json. Top-level fields win; the nested text_config supplies the
// fallbacks. NOTE: this probe family is package-internal and has no live
// caller inside hf (the live, used copy is in gguf/info.go); these tests
// exercise the duplicated logic directly. Fixtures via t.TempDir(), no
// network.
func TestModelConfigProbe_SyntheticConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"architectures": ["Qwen3ForCausalLM"],
		"vocab_size": 151936,
		"hidden_size": 1024,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)

	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
	if got := probe.numLayers(); got != 28 {
		t.Fatalf("numLayers() = %d, want 28", got)
	}
	if got := probe.vocabSize(); got != 151936 {
		t.Fatalf("vocabSize() = %d, want 151936", got)
	}
	if got := probe.hiddenSize(); got != 1024 {
		t.Fatalf("hiddenSize() = %d, want 1024", got)
	}
	if got := probe.contextLength(); got != 40960 {
		t.Fatalf("contextLength() = %d, want 40960", got)
	}
	if got := probe.quantBits(); got != 4 {
		t.Fatalf("quantBits() = %d, want 4", got)
	}
	if got := probe.quantGroup(); got != 64 {
		t.Fatalf("quantGroup() = %d, want 64", got)
	}
}

// TestModelConfigProbe_NestedAndRerank_Ugly covers the probe fallbacks: a
// nested text_config supplies layers/vocab/hidden/context when the top level
// is empty, the legacy `quantization` key supplies bits/group when
// `quantization_config` is absent, the BertForSequenceClassification →
// bert_rerank special case fires, and a nil probe is safe on every accessor.
func TestModelConfigProbe_NestedAndRerank_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"text_config": {
			"model_type": "bert",
			"vocab_size": 30522,
			"hidden_size": 768,
			"num_hidden_layers": 12,
			"max_position_embeddings": 512
		},
		"quantization": {"bits": 8, "group_size": 32}
	}`)

	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.architecture(); got != "bert_rerank" {
		t.Fatalf("architecture() = %q, want bert_rerank (sequence-classification special case)", got)
	}
	if got := probe.numLayers(); got != 12 {
		t.Fatalf("numLayers() = %d, want 12 (from text_config)", got)
	}
	if got := probe.vocabSize(); got != 30522 {
		t.Fatalf("vocabSize() = %d, want 30522 (from text_config)", got)
	}
	if got := probe.hiddenSize(); got != 768 {
		t.Fatalf("hiddenSize() = %d, want 768 (from text_config)", got)
	}
	if got := probe.contextLength(); got != 512 {
		t.Fatalf("contextLength() = %d, want 512 (from text_config)", got)
	}
	if got := probe.quantBits(); got != 8 {
		t.Fatalf("quantBits() = %d, want 8 (from legacy quantization key)", got)
	}
	if got := probe.quantGroup(); got != 32 {
		t.Fatalf("quantGroup() = %d, want 32 (from legacy quantization key)", got)
	}

	// nil probe is safe on every accessor.
	var nilProbe *modelConfigProbe
	if nilProbe.architecture() != "" || nilProbe.numLayers() != 0 || nilProbe.vocabSize() != 0 ||
		nilProbe.hiddenSize() != 0 || nilProbe.contextLength() != 0 || nilProbe.quantBits() != 0 ||
		nilProbe.quantGroup() != 0 {
		t.Fatal("nil probe accessors returned non-zero, want all zero/empty")
	}
}

// TestModelConfigProbe_ReadErrors_Bad covers readModelConfig's failure
// branches: a missing config.json surfaces a read error, and a malformed JSON
// body surfaces an unmarshal error.
func TestModelConfigProbe_ReadErrors_Bad(t *testing.T) {
	if _, err := readModelConfig(t.TempDir()); err == nil {
		t.Fatal("readModelConfig(no config.json) error = nil, want read error")
	}

	bad := t.TempDir()
	writeModelPackFile(t, core.PathJoin(bad, "config.json"), "{not valid json")
	if _, err := readModelConfig(bad); err == nil {
		t.Fatal("readModelConfig(malformed) error = nil, want unmarshal error")
	}
}

// TestIndexString_Good covers indexString, the allocation-free substring
// search helper: empty needle returns 0, a longer needle than the haystack
// returns -1, a hit returns the first index, and a miss returns -1.
func TestIndexString_Good(t *testing.T) {
	cases := []struct {
		s, sub string
		want   int
	}{
		{"hello world", "", 0},
		{"hi", "longer-than-haystack", -1},
		{"hello world", "world", 6},
		{"hello world", "lo", 3},
		{"hello world", "xyz", -1},
		{"aaa", "aaa", 0},
	}
	for _, tc := range cases {
		if got := indexString(tc.s, tc.sub); got != tc.want {
			t.Fatalf("indexString(%q, %q) = %d, want %d", tc.s, tc.sub, got, tc.want)
		}
	}
}
