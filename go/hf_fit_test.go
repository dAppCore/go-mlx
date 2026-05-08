// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
)

type fakeHFModelSource struct {
	searchCalled bool
	search       []HFModelMetadata
	byID         map[string]HFModelMetadata
}

func (s *fakeHFModelSource) SearchModels(_ context.Context, query string, limit int) ([]HFModelMetadata, error) {
	if query != "qwen 0.6b" {
		return nil, core.NewError("unexpected query: " + query)
	}
	s.searchCalled = true
	if limit > 0 && limit < len(s.search) {
		return append([]HFModelMetadata(nil), s.search[:limit]...), nil
	}
	return append([]HFModelMetadata(nil), s.search...), nil
}

func (s *fakeHFModelSource) ModelMetadata(_ context.Context, id string) (HFModelMetadata, error) {
	if meta, ok := s.byID[id]; ok {
		return meta, nil
	}
	return HFModelMetadata{}, core.NewError("not found: " + id)
}

func TestPlanHFModelFits_InjectedSearch_Good(t *testing.T) {
	source := &fakeHFModelSource{
		search: []HFModelMetadata{{
			ID: "Qwen/Qwen3-0.6B",
			Config: HFModelConfig{
				ModelType:             "qwen3",
				HiddenSize:            1024,
				NumHiddenLayers:       28,
				NumAttentionHeads:     16,
				NumKeyValueHeads:      8,
				MaxPositionEmbeddings: 40960,
				Quantization:          &HFQuantizationConfig{Bits: 4, GroupSize: 64},
			},
			Files: []HFModelFile{
				{Name: "model.safetensors", Size: 420 * 1024 * 1024},
				{Name: "tokenizer.json", Size: 4 * 1024 * 1024},
			},
		}},
	}

	report, err := PlanHFModelFits(context.Background(), HFModelFitConfig{
		Query:      "qwen 0.6b",
		MaxResults: 5,
		Device: DeviceInfo{
			Architecture:                 "apple-m3-ultra",
			MemorySize:                   96 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 86 * MemoryGiB,
		},
		Source: source,
	})
	if err != nil {
		t.Fatalf("PlanHFModelFits() error = %v", err)
	}
	if !source.searchCalled {
		t.Fatal("SearchModels was not called")
	}
	if report.DeviceClass != MemoryClassApple96GB || report.MemoryPlan.ContextLength != DefaultLocalContextLength {
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

	report, err := PlanHFModelFits(context.Background(), HFModelFitConfig{
		LocalPaths: []string{cacheRoot},
		Device: DeviceInfo{
			Architecture:                 "apple-m1-pro",
			MemorySize:                   16 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 13 * MemoryGiB,
		},
	})
	if err != nil {
		t.Fatalf("PlanHFModelFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.ModelID != "mlx-community/gemma-4-e2b-it-4bit" {
		t.Fatalf("ModelID = %q", plan.ModelID)
	}
	if plan.Source != HFModelSourceLocal || plan.LocalPath != dir {
		t.Fatalf("source/path = %q %q", plan.Source, plan.LocalPath)
	}
	if plan.Architecture != "gemma4_text" || !plan.SupportedArchitecture {
		t.Fatalf("architecture support = %q %v", plan.Architecture, plan.SupportedArchitecture)
	}
	if plan.ContextRecommendation != 8192 || plan.MemoryPlan.CachePolicy != KVCacheRotating {
		t.Fatalf("context/cache plan = %+v", plan.MemoryPlan)
	}
	if plan.ExpectedKVBytes == 0 {
		t.Fatal("ExpectedKVBytes = 0, want estimate")
	}
}

func TestPlanHFModelFits_QwenNextNestedTextConfig_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]HFModelMetadata{
			"Qwen/Qwen3.5-0.8B-Base": {
				ID: "Qwen/Qwen3.5-0.8B-Base",
				Config: HFModelConfig{
					ModelType: "qwen3_5",
					TextConfig: &HFModelConfig{
						ModelType:             "qwen3_next",
						HiddenSize:            1536,
						NumHiddenLayers:       28,
						NumAttentionHeads:     16,
						NumKeyValueHeads:      8,
						MaxPositionEmbeddings: 65536,
						QuantizationConfig:    &HFQuantizationConfig{Bits: 4, GroupSize: 64},
					},
				},
				Files: []HFModelFile{{Name: "model.safetensors", Size: 900 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanHFModelFits(context.Background(), HFModelFitConfig{
		ModelIDs: []string{"Qwen/Qwen3.5-0.8B-Base"},
		Device:   DeviceInfo{MemorySize: 24 * MemoryGiB, MaxRecommendedWorkingSetSize: 20 * MemoryGiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanHFModelFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.Architecture != "qwen3_next" || !plan.SupportedArchitecture || !plan.NativeLoadable {
		t.Fatalf("architecture/loadable = %q supported=%v native=%v", plan.Architecture, plan.SupportedArchitecture, plan.NativeLoadable)
	}
	if plan.ContextRecommendation != 16384 {
		t.Fatalf("ContextRecommendation = %d, want machine-class cap 16384", plan.ContextRecommendation)
	}
}

func TestPlanHFModelFits_RequiresSourceForQuery_Bad(t *testing.T) {
	_, err := PlanHFModelFits(context.Background(), HFModelFitConfig{Query: "gemma"})
	if err == nil {
		t.Fatal("expected missing source error")
	}
	if !core.Contains(err.Error(), "source") {
		t.Fatalf("error = %v, want source context", err)
	}
}

func TestPlanHFModelFits_UnsupportedArchitecture_Ugly(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]HFModelMetadata{
			"future/model": {
				ID: "future/model",
				Config: HFModelConfig{
					ModelType:             "future_arch",
					HiddenSize:            4096,
					NumHiddenLayers:       32,
					NumAttentionHeads:     32,
					MaxPositionEmbeddings: 32768,
				},
				Files: []HFModelFile{{Name: "model.safetensors", Size: 30 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanHFModelFits(context.Background(), HFModelFitConfig{
		ModelIDs: []string{"future/model"},
		Device:   DeviceInfo{MemorySize: 16 * MemoryGiB, MaxRecommendedWorkingSetSize: 12 * MemoryGiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanHFModelFits() error = %v", err)
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

	source := NewHuggingFaceModelSource(HuggingFaceModelSourceConfig{
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
	if _, err := PlanHFModelFits(context.Background(), HFModelFitConfig{}); err == nil {
		t.Fatal("expected no metadata error")
	}
	if _, err := PlanHFModelFits(context.Background(), HFModelFitConfig{ModelIDs: []string{"qwen/model"}}); err == nil || !core.Contains(err.Error(), "source") {
		t.Fatalf("missing source error = %v", err)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := PlanHFModelFits(cancelled, HFModelFitConfig{LocalPaths: []string{t.TempDir()}})
	if err != context.Canceled {
		t.Fatalf("PlanHFModelFits(cancelled local) = %v, want context.Canceled", err)
	}

	badLocal := t.TempDir()
	writeModelPackFile(t, core.PathJoin(badLocal, "config.json"), "{")
	if _, err := PlanHFModelFits(context.Background(), HFModelFitConfig{LocalPaths: []string{badLocal}}); err == nil {
		t.Fatal("expected bad local config error")
	}
}

func TestHuggingFaceModelSource_Errors_Bad(t *testing.T) {
	var source *HuggingFaceModelSource
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

	source = NewHuggingFaceModelSource(HuggingFaceModelSourceConfig{BaseURL: server.URL + "/", UserAgent: "tests"})
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

	meta, root, err := inspectLocalHFModelMetadata(cacheRoot)
	if err != nil {
		t.Fatalf("inspectLocalHFModelMetadata: %v", err)
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
	if got := resolveLocalHFMetadataRoot(core.PathJoin(snapshot, "config.json")); got != snapshot {
		t.Fatalf("resolve config root = %q, want %q", got, snapshot)
	}
}

func TestHFModelFitHelpers_Ugly(t *testing.T) {
	files := []HFModelFile{
		{Name: "model-q4.gguf", Size: 10},
		{RFilename: "model.safetensors", SizeBytes: 20},
		{Name: "pytorch_model.bin", Size: 30},
	}
	format, bytes := hfWeightFormatAndBytes(files)
	if format != string(ModelPackFormatMixed) || bytes != 60 {
		t.Fatalf("hfWeightFormatAndBytes = %q/%d, want mixed/60", format, bytes)
	}
	if bits := inferHFQuantBits([]HFModelFile{{Name: "model-8bit.safetensors"}}); bits != 8 {
		t.Fatalf("inferHFQuantBits(8bit) = %d", bits)
	}
	for name, want := range map[string]int{
		"q2.gguf":       2,
		"q3.gguf":       3,
		"4-bit.gguf":    4,
		"q5.gguf":       5,
		"q6.gguf":       6,
		"fp16.bin":      16,
		"unknown.model": 0,
	} {
		if got := inferHFQuantBits([]HFModelFile{{Name: name}}); got != want {
			t.Fatalf("inferHFQuantBits(%q) = %d, want %d", name, got, want)
		}
	}

	config := HFModelConfig{HiddenSize: 128, NumHiddenLayers: 2, NumAttentionHeads: 4, NumKeyValueHeads: 2}
	if got := estimateHFModelKVBytes(config, 16, 2, 2); got != 16384 {
		t.Fatalf("estimateHFModelKVBytes(GQA) = %d, want 16384", got)
	}
	if got := estimateHFModelKVBytes(HFModelConfig{HiddenSize: 128, NumHiddenLayers: 2}, 16, 0, 0); got != 16384 {
		t.Fatalf("estimateHFModelKVBytes(hidden fallback) = %d, want 16384", got)
	}
	if got := estimateHFModelKVBytes(HFModelConfig{}, 16, 1, 2); got != 0 {
		t.Fatalf("estimateHFModelKVBytes(empty) = %d, want 0", got)
	}
	if got := estimateRuntimeOverheadBytes(0); got != 0 {
		t.Fatalf("estimateRuntimeOverheadBytes(0) = %d, want 0", got)
	}
	if got := estimateRuntimeOverheadBytes(2 * MemoryGiB); got != MemoryGiB {
		t.Fatalf("estimateRuntimeOverheadBytes(small) = %d, want 1GiB", got)
	}

	plan := HFModelFitPlan{
		NativeLoadable:       true,
		InferenceFits:        true,
		QuantBits:            16,
		WeightBytes:          100,
		ExpectedKVBytes:      10,
		ExpectedRuntimeBytes: 10,
		ExpectedTotalBytes:   120,
	}
	fit := estimateHFTrainingFit(HFModelConfig{HiddenSize: 8, NumHiddenLayers: 2}, plan, 0, -1)
	if !fit.LoRAFeasible || !fit.FullFineTuneFeasible || fit.RecommendedLoRARank != 16 {
		t.Fatalf("training fit = %+v", fit)
	}
	if got := positiveInt(-3); got != 0 {
		t.Fatalf("positiveInt(-3) = %d, want 0", got)
	}
	if err := hfFitResultError(core.Result{Value: "bad", OK: false}); err == nil || !core.Contains(err.Error(), "core result failed") {
		t.Fatalf("hfFitResultError(non-error) = %v", err)
	}
}
